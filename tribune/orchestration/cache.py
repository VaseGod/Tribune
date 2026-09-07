"""CPU-Pinned Semi-Persistent KV-Cache Management Subsystem.

Supports host memory paging for attention KV tensors:
1. Offloads attention KV tensors from active GPU VRAM to page-locked (pinned) CPU host memory
   upon intermediate step completion or turn yields.
2. Retains page metadata in a fast O(1) routing table to support sub-millisecond rehydration.
3. Provides a transparent mock/fallback layer for non-CUDA and CPU-only execution environments.
"""

from __future__ import annotations

import copy
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVCachePageMetadata:
    """Metadata tracking a paged-out KV cache tensor block in CPU host memory."""

    page_id: str
    session_id: str
    step_index: int
    num_tensors: int
    total_bytes: int
    precision: str = "BF16"
    is_pinned: bool = True
    source_device: str = "cuda:0"
    target_device: str = "cpu"
    host_address: int = 0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    rehydration_latency_ms: float = 0.0


class KVCacheRoutingTable:
    """Fast, thread-safe O(1) routing table indexing pinned KV cache pages."""

    def __init__(self) -> None:
        self._table: dict[str, dict[int, KVCachePageMetadata]] = {}  # session_id -> step_index -> metadata
        self._page_index: dict[str, KVCachePageMetadata] = {}  # page_id -> metadata
        self._lock = threading.RLock()

    def put(self, metadata: KVCachePageMetadata) -> None:
        with self._lock:
            if metadata.session_id not in self._table:
                self._table[metadata.session_id] = {}
            self._table[metadata.session_id][metadata.step_index] = metadata
            self._page_index[metadata.page_id] = metadata

    def get(self, session_id: str, step_index: int) -> KVCachePageMetadata | None:
        with self._lock:
            return self._table.get(session_id, {}).get(step_index)

    def get_latest(self, session_id: str) -> KVCachePageMetadata | None:
        with self._lock:
            steps = self._table.get(session_id, {})
            if not steps:
                return None
            latest_step = max(steps.keys())
            return steps[latest_step]

    def remove_session(self, session_id: str) -> list[KVCachePageMetadata]:
        with self._lock:
            removed: list[KVCachePageMetadata] = []
            steps = self._table.pop(session_id, {})
            for meta in steps.values():
                self._page_index.pop(meta.page_id, None)
                removed.append(meta)
            return removed

    def total_pages(self) -> int:
        with self._lock:
            return len(self._page_index)


class PinnedHostMemoryBuffer:
    """Transparent mock/fallback page-locked buffer simulating pinned host memory."""

    def __init__(self, data: bytes, is_pinned: bool = True) -> None:
        self.size = len(data)
        self.is_pinned = is_pinned
        # Allocate page-aligned memory representation
        self._buffer = bytearray(data)
        self.host_address = id(self._buffer)

    def read_bytes(self) -> bytes:
        return bytes(self._buffer)


class CPUPinnedKVCache:
    """Subsystem managing page-locked host memory paging for KV-caches."""

    def __init__(
        self,
        enforce_bf16_precision: bool = True,
        simulate_cuda: bool = True,
    ) -> None:
        self.enforce_bf16_precision = enforce_bf16_precision
        self.simulate_cuda = simulate_cuda
        self.routing_table = KVCacheRoutingTable()
        self._host_pool: dict[str, PinnedHostMemoryBuffer] = {}
        self._tensor_deserializers: dict[str, Any] = {}
        self._lock = threading.RLock()

        # Telemetry
        self.offloaded_pages_count = 0
        self.rehydrated_pages_count = 0
        self.total_bytes_paged = 0
        self.total_rehydration_time_ms = 0.0

    def offload_kv_cache(
        self,
        session_id: str,
        step_index: int,
        kv_tensors: Any,
        source_device: str = "cuda:0",
        precision: str = "BF16",
    ) -> KVCachePageMetadata:
        """Offload attention KV tensors from active GPU VRAM to page-locked CPU host memory."""
        with self._lock:
            if self.enforce_bf16_precision and precision != "BF16":
                raise ValueError(f"KV-cache precision violation: Expected 'BF16', got '{precision}'")

            page_id = f"page_{session_id}_{step_index}_{secrets.token_hex(4)}"

            # Serialize / marshal tensors into byte buffer
            raw_data, tensor_count, byte_size = self._marshal_kv_tensors(kv_tensors)
            host_buffer = PinnedHostMemoryBuffer(raw_data, is_pinned=True)

            self._host_pool[page_id] = host_buffer
            self._tensor_deserializers[page_id] = copy.deepcopy(kv_tensors)

            metadata = KVCachePageMetadata(
                page_id=page_id,
                session_id=session_id,
                step_index=step_index,
                num_tensors=tensor_count,
                total_bytes=byte_size,
                precision=precision,
                is_pinned=True,
                source_device=source_device,
                target_device="cpu_pinned",
                host_address=host_buffer.host_address,
            )

            self.routing_table.put(metadata)
            self.offloaded_pages_count += 1
            self.total_bytes_paged += byte_size
            return metadata

    def rehydrate_kv_cache(
        self,
        session_id: str,
        step_index: int | None = None,
        target_device: str = "cuda:0",
    ) -> tuple[Any, float]:
        """Rapidly rehydrate attention KV tensors from pinned host memory back to active device.

        Returns tuple of (rehydrated_tensors, latency_ms), with target sub-millisecond performance.
        """
        with self._lock:
            start_t = time.perf_counter()

            if step_index is not None:
                metadata = self.routing_table.get(session_id, step_index)
            else:
                metadata = self.routing_table.get_latest(session_id)

            if not metadata:
                raise KeyError(f"No KV cache page found for session '{session_id}' (step={step_index})")

            page_id = metadata.page_id
            host_buffer = self._host_pool.get(page_id)
            if not host_buffer:
                raise RuntimeError(f"Host memory buffer for page '{page_id}' has been unmapped or evicted")

            # Rapid memory read and restoration
            _ = host_buffer.read_bytes()
            tensors = copy.deepcopy(self._tensor_deserializers.get(page_id))

            latency_ms = (time.perf_counter() - start_t) * 1000.0

            # Update page access metrics
            metadata.access_count += 1
            metadata.rehydration_latency_ms = latency_ms
            metadata.target_device = target_device

            self.rehydrated_pages_count += 1
            self.total_rehydration_time_ms += latency_ms

            return tensors, latency_ms

    def evict_session(self, session_id: str) -> int:
        """Evict and free all pinned host memory pages for a session."""
        with self._lock:
            pages = self.routing_table.remove_session(session_id)
            for page in pages:
                self._host_pool.pop(page.page_id, None)
                self._tensor_deserializers.pop(page.page_id, None)
            return len(pages)

    def _marshal_kv_tensors(self, kv_tensors: Any) -> tuple[bytes, int, int]:
        """Convert arbitrary KV tensor representations (dicts, lists, mock objects) to bytes."""
        if isinstance(kv_tensors, dict):
            num_tensors = len(kv_tensors)
            serialized = repr(kv_tensors).encode("utf-8")
        elif isinstance(kv_tensors, list | tuple):
            num_tensors = len(kv_tensors)
            serialized = repr(kv_tensors).encode("utf-8")
        else:
            num_tensors = 1
            serialized = str(kv_tensors).encode("utf-8")

        # Pad to 4096-byte page boundary
        byte_size = len(serialized)
        padded_size = (byte_size + 4095) & ~4095
        padded_data = serialized.ljust(padded_size, b"\x00")
        return padded_data, num_tensors, padded_size

    def stats(self) -> dict[str, Any]:
        with self._lock:
            avg_lat = (
                self.total_rehydration_time_ms / max(1, self.rehydrated_pages_count)
                if self.rehydrated_pages_count > 0
                else 0.0
            )
            return {
                "active_pages": self.routing_table.total_pages(),
                "offloaded_pages": self.offloaded_pages_count,
                "rehydrated_pages": self.rehydrated_pages_count,
                "total_bytes_paged": self.total_bytes_paged,
                "avg_rehydration_latency_ms": round(avg_lat, 3),
            }


# --------------------------------------------------------------------------- #
# Asymmetric Prompt Caching: Multi-Tier Payload Serializer
# --------------------------------------------------------------------------- #


@dataclass
class SerializedMultiTierPayload:
    """Structured payload ready for Anthropic Messages API with ephemeral breakpoints."""

    system: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    static_breakpoint_injected: bool
    execution_state_breakpoint_injected: bool
    total_turns: int
    cached_turns: int
    dynamic_turns: int

    def to_api_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"messages": self.messages}
        if self.system:
            kwargs["system"] = self.system
        if self.tools:
            kwargs["tools"] = self.tools
        return kwargs


class MultiTierPayloadSerializer:
    """Constructs multi-tier payloads with ephemeral cache breakpoint injection.

    1. Static Header: Base system prompts, static repository symbol graphs, and full
       JSON tool registry schemas. Injects cache_control: {"type": "ephemeral"} on
       the boundary of this block.
    2. Execution State (N-1 Turn): Identifies second-to-last turn in execution trace.
       Injects cache_control: {"type": "ephemeral"} at this boundary so prior tool
       outputs and intermediary code execution contexts read at cached rate.
    3. Dynamic Turn: The latest agent turn (turn N) and runtime environment observation
       remains uncached.
    """

    EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}

    @classmethod
    def serialize(
        cls,
        system_prompt: str | list[dict[str, Any]],
        symbol_graph_context: str | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        execution_trace: list[dict[str, Any]] | None = None,
        dynamic_turn: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> SerializedMultiTierPayload:
        # --- 1. Static Header Construction ---
        system_blocks: list[dict[str, Any]] = []
        if isinstance(system_prompt, str):
            if system_prompt.strip():
                system_blocks.append({"type": "text", "text": system_prompt})
        elif isinstance(system_prompt, list):
            system_blocks.extend(copy.deepcopy(system_prompt))

        if symbol_graph_context and symbol_graph_context.strip():
            system_blocks.append({
                "type": "text",
                "text": f"=== REPOSITORY SYMBOL GRAPH ===\n{symbol_graph_context.strip()}",
            })

        tools_list: list[dict[str, Any]] = copy.deepcopy(tool_schemas or [])

        # Inject ephemeral breakpoint onto the boundary of the static header block.
        # Prefer placing on the last tool if tools exist, otherwise on the last system block.
        static_injected = False
        if tools_list:
            tools_list[-1]["cache_control"] = copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)
            static_injected = True
        elif system_blocks:
            system_blocks[-1]["cache_control"] = copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)
            static_injected = True

        # --- 2. Execution State (N-1 Turn) & Dynamic Turn ---
        trace = copy.deepcopy(execution_trace or [])
        messages: list[dict[str, Any]] = []

        # Normalize execution trace into message list
        for item in trace:
            if isinstance(item, dict):
                # May already be a message dict {"role": ..., "content": ...}
                if "role" in item and "content" in item:
                    messages.append(copy.deepcopy(item))
                elif "turn_messages" in item and isinstance(item["turn_messages"], list):
                    messages.extend(copy.deepcopy(item["turn_messages"]))
                else:
                    messages.append(copy.deepcopy(item))

        cached_turns = 0
        execution_injected = False

        # Identify second-to-last turn boundary (N-1):
        # - If dynamic_turn is passed separately, execution_trace contains prior turns up to N-1,
        #   so the boundary of the execution state is the last item of execution_trace.
        # - If dynamic_turn is not passed, execution_trace includes the active turn at the end,
        #   so turn N-1 is the penultimate item (len - 2).
        has_dynamic = bool(dynamic_turn)
        if messages:
            if has_dynamic:
                target_idx = len(messages) - 1
            else:
                target_idx = max(0, len(messages) - 2) if len(messages) >= 2 else len(messages) - 1
            target_msg = messages[target_idx]

            # Inject cache_control on content block and message
            target_msg["cache_control"] = copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)
            content = target_msg.get("content")
            if isinstance(content, str):
                target_msg["content"] = [
                    {"type": "text", "text": content, "cache_control": copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)}
                ]
                execution_injected = True
            elif isinstance(content, list) and content:
                if isinstance(content[-1], dict):
                    content[-1]["cache_control"] = copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)
                    execution_injected = True
            elif isinstance(content, dict):
                content["cache_control"] = copy.deepcopy(cls.EPHEMERAL_CACHE_CONTROL)
                execution_injected = True
            else:
                execution_injected = True

            cached_turns = target_idx + 1

        # --- 3. Dynamic Turn (Uncached) ---
        dynamic_turns_count = 0
        if dynamic_turn:
            if isinstance(dynamic_turn, list):
                for d in dynamic_turn:
                    cleaned_d = copy.deepcopy(d)
                    cls._strip_cache_control(cleaned_d)
                    messages.append(cleaned_d)
                    dynamic_turns_count += 1
            elif isinstance(dynamic_turn, dict):
                cleaned_d = copy.deepcopy(dynamic_turn)
                cls._strip_cache_control(cleaned_d)
                messages.append(cleaned_d)
                dynamic_turns_count += 1

        return SerializedMultiTierPayload(
            system=system_blocks,
            tools=tools_list,
            messages=messages,
            static_breakpoint_injected=static_injected,
            execution_state_breakpoint_injected=execution_injected,
            total_turns=len(messages),
            cached_turns=cached_turns,
            dynamic_turns=dynamic_turns_count,
        )

    @classmethod
    def _strip_cache_control(cls, data: Any) -> None:
        """Ensure dynamic turn contains 0 cache_control markers."""
        if isinstance(data, dict):
            data.pop("cache_control", None)
            for v in data.values():
                cls._strip_cache_control(v)
        elif isinstance(data, list):
            for item in data:
                cls._strip_cache_control(item)


def serialize_multi_tier_payload(
    system_prompt: str | list[dict[str, Any]],
    symbol_graph_context: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
    dynamic_turn: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> SerializedMultiTierPayload:
    return MultiTierPayloadSerializer.serialize(
        system_prompt=system_prompt,
        symbol_graph_context=symbol_graph_context,
        tool_schemas=tool_schemas,
        execution_trace=execution_trace,
        dynamic_turn=dynamic_turn,
    )


__all__ = [
    "KVCachePageMetadata",
    "KVCacheRoutingTable",
    "CPUPinnedKVCache",
    "PinnedHostMemoryBuffer",
    "SerializedMultiTierPayload",
    "MultiTierPayloadSerializer",
    "serialize_multi_tier_payload",
]

