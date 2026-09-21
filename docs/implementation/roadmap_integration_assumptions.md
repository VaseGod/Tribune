# Tribune Architectural Implementation Roadmap Integration Assumptions

*Authored 2026-09-20 by Tribune AI Systems & Engineering.*

---

## 1. Context Graph Modernization via Calibrated Decision Routing

### Assumptions:
1. **Categorical Taxonomy Completeness:**
   The edge taxonomy `EdgeClass ∈ { Contradicts, Extends, TemporalFollowup, Irrelevant }` captures all actionable multi-agent and evidentiary relationships necessary for statutory adjudication. Relationships that do not match this taxonomy are mapped to `Irrelevant` or placed on the escalation queue for caseworker review.
2. **Probability Calibration Model:**
   In local and offline environments, calibrated probabilities are generated via Platt-scaled deterministic feature kernels and cross-encoder logit approximations. The production RLCD classifier stub (`ModelBackedRLCDClassifier`) uses temperature-scaled cross-entropy logits, ready for integration with hosted transformer models.
3. **Strict Escalation Boundary:**
   The threshold `0.85` is strictly enforced. Any candidate relationship with confidence `< 0.85` is prevented from directly mutating the graph and is enqueued into `EscalationQueue`.

---

## 2. Calibrated Memory Management & Node Eviction

### Assumptions:
1. **Multi-Dimensional Utility Function:**
   Node utility is defined over seven weighted dimensions: recency, frequency of use, graph centrality, task relevance, semantic relevance, escalation status, and downstream reference count.
2. **Determinism:**
   When provided fixed random seeds, step indices, and identical inputs, eviction ordering is 100% deterministic, breaking ties by last access step and node ID.
3. **Non-Blocking Summarization:**
   Subgraph summarization upon eviction is non-blocking. If an LLM or summarization service is delayed or unavailable, the nodes are evicted immediately to protect token budgets, and summaries are updated asynchronously.

---

## 3. Streaming Full-Duplex Acoustic Pipeline

### Assumptions:
1. **Transport Abstraction:**
   In local developer workstations and CI environments without running WebRTC media servers or LiveKit rooms, `LocalLoopbackTransport` provides in-memory packet buffering and validates real-time streaming semantics.
2. **Barge-In Flush Ceiling:**
   Fast interruption flushing is targeted at `<50ms`. On local loopback and simulated transports, flush latency executes in `<1ms`, and production WebRTC socket drains operate well within the 50ms budget.
3. **Background Tool Execution:**
   Spoken dialogues triggering tool calls (such as database lookups or statutory checks) run on asynchronous background worker tasks without blocking the audio loop or stuttering outbound speech synthesis.

---

## 4. Coarse-to-Fine Dynamic Adapter Gating / MoVA Integration

### Assumptions:
1. **Unimodal Text Bypass:**
   Pure text queries do not require cross-modal attention adapters. Bypassing MoVA adapters directly to the LLM trunk saves 100% of adapter KV-cache allocation (0 MB) and eliminates cross-modal interference.
2. **KV-Cache Budget Enforcement:**
   When multimodal queries activate multiple experts, the total allocated KV-cache must not exceed `cache_budget_mb` (default 1024 MB). If the budget is exceeded, lowest-confidence experts are pruned.
3. **Static Fallback:**
   For regression testing and hardware without dynamic MoVA head routing, a static adapter mode (`force_static=True` or `enable_dynamic_routing=False`) remains available.

---

## 5. Sandboxed Shell Execution over Static Typed Tool Catalogs

### Assumptions:
1. **Shell-First Security Posture:**
   Standard agent tasks execute in an isolated ephemeral Bash shell with process timeouts, memory limits, output caps, and strict deny patterns (`sudo`, `su`, `chmod +s`, `rm -rf /`, `/etc/shadow`, exfiltration loops).
2. **Typed Catalog Boundary:**
   Static programmatic tool catalogs are reserved for audited enterprise actions (e.g. querying canonical statutory tables, verifying compliance disclaimers).
3. **Automatic Quarantine:**
   Any detected terminal exploit loop (e.g. identical command repeated N times, rapid no-op bursts, environment variable exfiltration) triggers automated session quarantine and bounded evidence logging.
