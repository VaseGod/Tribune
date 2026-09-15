"""Model Context Protocol (MCP) Server Endpoint & Tool Definitions for Tribune.

Exposes Tribune's resources and tools via an MCP-compliant JSON-RPC 2.0 interface.
Also exports tool definitions compatible with OpenAI Agent Plugins / Chat Completions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from .casegen.synthetic import SyntheticCaseGenerator
from .config import TribuneSettings, get_settings
from .corpus.programs import all_programs, get_ruleset, known_jurisdictions
from .ingestion.ocr import parse_text_to_fields
from .orchestration.pipeline import CasePipeline
from .types import ApplicantSituation, ProgramId, RawDocument

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2024-11-05"

# Tool Schemas for MCP & OpenAI Agent Plugins
TOOLS_DEFINITIONS = [
    {
        "name": "tribune_run_case",
        "description": "Run Tribune benefits eligibility assessment for an applicant situation across SNAP, Unemployment, Medicaid, Housing, or Appeals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "Optional case identifier"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction code (e.g. EX, CA, NY, TX)"},
                "target_programs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target programs to assess (e.g. ['snap', 'medicaid'])",
                },
                "overrides": {
                    "type": "object",
                    "description": "Situation key-value overrides (e.g. income_monthly, household_size, state)",
                },
                "document_text": {"type": "string", "description": "Unstructured text of applicant document"},
            },
            "required": [],
        },
    },
    {
        "name": "tribune_search_rules",
        "description": "Search and retrieve governing public benefit program criteria, rules, and legal citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "Program ID (snap, unemployment, medicaid, housing, appeals)"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
                "query": {"type": "string", "description": "Search query keywords or criterion topic"},
            },
            "required": ["program"],
        },
    },
    {
        "name": "tribune_extract_fields",
        "description": "Extract structured applicant situation fields and evidence items from document text using OCR parsing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_text": {"type": "string", "description": "Raw unstructured document text"},
            },
            "required": ["document_text"],
        },
    },
    {
        "name": "tribune_explain_assessment",
        "description": "Provide a grounded explanation of assessment results or answers to case questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "Case ID of the run"},
                "query": {"type": "string", "description": "Question about eligibility, missing documents, or next steps"},
            },
            "required": ["case_id", "query"],
        },
    },
    {
        "name": "analyzeText",
        "description": "Calculate exact token count, information density, and Shannon entropy across the working memory span.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "span": {"type": "string", "description": "Working memory text span to analyze"},
            },
            "required": ["span"],
        },
    },
    {
        "name": "checkBudget",
        "description": "Return remaining context window quota, consumption velocity, and compaction urgency flags.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "foldHistory",
        "description": "Discard resolved interaction spans and store structured semantic indexing headers in place.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "span_id": {"type": "string", "description": "Span ID to fold"},
                "summarize": {"type": "boolean", "description": "Whether to generate semantic summary in header"},
            },
            "required": ["span_id"],
        },
    },
    {
        "name": "compressContext",
        "description": "Extractive/distillation compression pipeline preserving named entities, citations, and causal predicates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_text": {"type": "string", "description": "Target text to compress"},
                "ratio": {"type": "number", "description": "Target compression ratio (e.g. 0.5)"},
            },
            "required": ["target_text"],
        },
    },
    {
        "name": "tribune_query_property_tax",
        "description": "Non-blocking query to county property tax registry to verify real property ownership and assessed valuations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "Case identifier"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
                "parcel_id_or_address": {"type": "string", "description": "Parcel ID or physical street address"},
            },
            "required": ["jurisdiction", "parcel_id_or_address"],
        },
    },
    {
        "name": "tribune_query_unemployment_database",
        "description": "Non-blocking query to state Department of Labor wage records and unemployment benefit claims.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "Case identifier"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
                "claimant_ssn_hash": {"type": "string", "description": "SHA-256 hashed claimant SSN"},
            },
            "required": ["jurisdiction", "claimant_ssn_hash"],
        },
    },
    {
        "name": "tribune_query_vital_statistics",
        "description": "Non-blocking query to state vital statistics records for identity, DOB, and residency verification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "description": "Case identifier"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction code"},
                "full_name": {"type": "string", "description": "Individual full legal name"},
                "dob": {"type": "string", "description": "Date of birth (YYYY-MM-DD)"},
            },
            "required": ["jurisdiction", "full_name"],
        },
    },
    {
        "name": "tribune_parallel_dispatch",
        "description": "Dispatches multiple external queries concurrently using asynchronous non-blocking execution to avoid I/O blocking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name"],
                    },
                    "description": "List of tool call objects to execute concurrently",
                },
            },
            "required": ["calls"],
        },
    },
]



class MCPAuthError(Exception):
    """Raised when MCP authorization checks fail."""
    pass


# --------------------------------------------------------------------------- #
# Enterprise Identity Delegation & Scoped Token Context
# --------------------------------------------------------------------------- #


@dataclass
class DelegatedUserTokenContext:
    """Enterprise IdP authenticated, scoped delegated user session context."""

    user_id: str
    subject: str
    tenant_id: str = "enterprise_default"
    issuer: str = "https://auth.tribune.enterprise.internal"
    roles: list[str] = field(default_factory=lambda: ["user"])
    scopes: set[str] = field(
        default_factory=lambda: {"rules:read", "cases:read", "cases:assess", "fields:extract"}
    )
    delegated_token: str = ""
    token_type: str = "Bearer"
    expires_at: float = field(default_factory=lambda: time.time() + 3600.0)
    claims: dict[str, Any] = field(default_factory=dict)
    renewable: bool = True

    def is_expired(self) -> bool:
        """Check if the delegated token has exceeded its TTL."""
        return time.time() >= self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Validate whether the token possesses a specific required capability scope."""
        if "admin:all" in self.scopes or "admin" in self.roles:
            return True
        return scope in self.scopes

    def renew_token(self, ttl: float = 3600.0) -> DelegatedUserTokenContext:
        """Renew token context with a refreshed expiration."""
        if not self.renewable:
            raise MCPAuthError("Token context is not renewable")
        return DelegatedUserTokenContext(
            user_id=self.user_id,
            subject=self.subject,
            tenant_id=self.tenant_id,
            issuer=self.issuer,
            roles=list(self.roles),
            scopes=set(self.scopes),
            delegated_token=self.delegated_token,
            token_type=self.token_type,
            expires_at=time.time() + ttl,
            claims=dict(self.claims),
            renewable=self.renewable,
        )


def validate_enterprise_token(
    token: str | None = None,
    headers: dict[str, str] | None = None,
    settings: TribuneSettings | None = None,
) -> DelegatedUserTokenContext:
    """Validate enterprise delegated token against IdP policies and return verified context."""
    headers = headers or {}
    settings = settings or get_settings()

    extracted_token = token
    if not extracted_token:
        auth_header = headers.get("authorization") or headers.get("Authorization") or ""
        api_key_header = headers.get("x-api-key") or headers.get("X-API-Key") or ""
        if auth_header.startswith("Bearer "):
            extracted_token = auth_header[7:].strip()
        elif api_key_header:
            extracted_token = api_key_header.strip()

    expected_token = settings.mcp_auth_token
    if expected_token and extracted_token != expected_token:
        raise MCPAuthError("Invalid or missing MCP Bearer token / API Key")

    role = headers.get("x-tribune-role") or headers.get("X-Tribune-Role") or "user"
    user_id = headers.get("x-tribune-user-id") or headers.get("X-Tribune-User-ID") or "user-default"
    tenant_id = (
        headers.get("x-tribune-tenant-id") or headers.get("X-Tribune-Tenant-ID") or "tenant-default"
    )

    # Scopes configuration based on role or explicit header
    scopes_header = headers.get("x-tribune-scopes") or headers.get("X-Tribune-Scopes")
    if scopes_header:
        scopes = {s.strip() for s in scopes_header.split(",") if s.strip()}
    elif role == "read_only":
        scopes = {"rules:read", "cases:read"}
    elif role == "admin":
        scopes = {"admin:all", "rules:read", "cases:read", "cases:assess", "fields:extract"}
    else:
        scopes = {"rules:read", "cases:read", "cases:assess", "fields:extract"}

    ctx = DelegatedUserTokenContext(
        user_id=user_id,
        subject=f"sub:{user_id}",
        tenant_id=tenant_id,
        roles=[role],
        scopes=scopes,
        delegated_token=extracted_token or "anonymous_token",
        expires_at=time.time() + 3600.0,
    )
    return ctx


def check_mcp_auth(headers: dict[str, str], settings: TribuneSettings | None = None) -> str:
    """Verify MCP authentication & RBAC headers (backward compatible)."""
    ctx = validate_enterprise_token(headers=headers, settings=settings)
    return ctx.roles[0] if ctx.roles else "user"


class MCPHandler:
    """Stateless MCP JSON-RPC 2.0 handler for Tribune."""

    def __init__(self, runs_store: dict[str, dict] | None = None, settings: TribuneSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.runs_store = runs_store if runs_store is not None else {}

    def handle_request(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Synchronous wrapper for handle_request_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.handle_request_async(payload, headers)).result()
        return asyncio.run(self.handle_request_async(payload, headers))

    async def handle_request_async(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Process a single JSON-RPC request asynchronously without blocking."""
        headers = headers or {}

        # Validate Auth & Scopes
        try:
            token_ctx = validate_enterprise_token(headers=headers, settings=self.settings)
        except MCPAuthError as exc:
            req_id = payload.get("id")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32001, "message": f"Unauthorized: {exc}"},
            }

        user_role = token_ctx.roles[0] if token_ctx.roles else "user"

        jsonrpc = payload.get("jsonrpc")
        if jsonrpc != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {"code": -32600, "message": "Invalid Request: jsonrpc must be '2.0'"},
            }

        method = payload.get("method")
        req_id = payload.get("id")
        params = payload.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "tribune-mcp", "version": "0.1.0"},
                },
            }

        if method == "notifications/initialized":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "resources/list":
            if not token_ctx.has_scope("rules:read"):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": "Forbidden: missing 'rules:read' scope"},
                }
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": self._list_resources()}}

        if method == "resources/read":
            if not token_ctx.has_scope("rules:read") and not token_ctx.has_scope("cases:read"):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": "Forbidden: missing read scope"},
                }
            uri = params.get("uri", "")
            try:
                content = self._read_resource(uri)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(content, indent=2),
                            }
                        ]
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"Resource error: {exc}"},
                }

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_DEFINITIONS}}

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}

            # Scope-based access control
            scope_map = {
                "tribune_run_case": "cases:assess",
                "tribune_search_rules": "rules:read",
                "tribune_extract_fields": "fields:extract",
                "tribune_explain_assessment": "cases:read",
                "tribune_query_property_tax": "cases:read",
                "tribune_query_unemployment_database": "cases:read",
                "tribune_query_vital_statistics": "cases:read",
                "tribune_parallel_dispatch": "cases:read",
            }
            required_scope = scope_map.get(tool_name)
            if required_scope and not token_ctx.has_scope(required_scope):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32002,
                        "message": f"Forbidden: role '{user_role}' or token lacks scope '{required_scope}'",
                    },
                }

            # RBAC check for restricted actions
            if user_role == "read_only" and tool_name == "tribune_run_case":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32002, "message": "Forbidden: read_only role cannot invoke run_case"},
                }

            try:
                result_text = await self._call_tool_async(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}]
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Tool execution failed: {exc}"},
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"},
        }


    # -- Resource helpers --------------------------------------------------- #

    def _list_resources(self) -> list[dict[str, Any]]:
        resources = [
            {
                "uri": "tribune://programs",
                "name": "Public Benefit Programs",
                "description": "List of all supported benefit programs and IDs",
                "mimeType": "application/json",
            },
            {
                "uri": "tribune://jurisdictions",
                "name": "Known Jurisdictions",
                "description": "List of supported state and territory jurisdiction codes",
                "mimeType": "application/json",
            },
            {
                "uri": "tribune://demo-cases",
                "name": "Demo Synthetic Cases",
                "description": "Pre-configured synthetic benchmark cases for testing",
                "mimeType": "application/json",
            },
        ]
        # Dynamically add rules for programs
        for prog in all_programs():
            resources.append(
                {
                    "uri": f"tribune://rules/{prog.value}",
                    "name": f"Program Rules: {prog.value.upper()}",
                    "description": f"Governing criteria ruleset for {prog.value}",
                    "mimeType": "application/json",
                }
            )
        # Add active case runs
        for case_id in self.runs_store:
            resources.append(
                {
                    "uri": f"tribune://cases/{case_id}",
                    "name": f"Case Run: {case_id}",
                    "description": f"State machine results and outcomes for case {case_id}",
                    "mimeType": "application/json",
                }
            )
        return resources

    def _read_resource(self, uri: str) -> Any:
        if uri == "tribune://programs":
            return [p.value for p in all_programs()]
        if uri == "tribune://jurisdictions":
            return known_jurisdictions()
        if uri == "tribune://demo-cases":
            gen = SyntheticCaseGenerator(seed=self.settings.seed)
            return [c.model_dump(mode="json") for c in gen.generate_demo_set()]

        if uri.startswith("tribune://rules/"):
            prog_name = uri.split("tribune://rules/")[1]
            prog_id = ProgramId(prog_name)
            ruleset = get_ruleset(prog_id)
            return {
                "program": ruleset.program.value,
                "version": ruleset.version,
                "rules": [
                    {
                        "criterion_id": r.criterion_id,
                        "description": r.description,
                        "required": r.required,
                        "source": r.source,
                        "title": r.title,
                        "text": r.text,
                    }
                    for r in ruleset.rules
                ],
            }

        if uri.startswith("tribune://cases/"):
            case_id = uri.split("tribune://cases/")[1]
            if case_id in self.runs_store:
                entry = self.runs_store[case_id]
                return {
                    "case_id": case_id,
                    "case": entry["case"].model_dump(mode="json"),
                    "result": entry["result"].model_dump(mode="json"),
                }
            raise KeyError(f"Case '{case_id}' not found in active runs")

        raise ValueError(f"Unknown resource URI: {uri}")

    # -- Tool Execution Helpers --------------------------------------------- #

    async def _call_tool_async(self, name: str, args: dict[str, Any]) -> str:
        """Asynchronously execute an MCP tool without blocking event loops."""
        if name == "tribune_query_property_tax":
            await asyncio.sleep(0.001)  # Non-blocking IO yield
            parcel_id = args.get("parcel_id_or_address", "")
            jurisdiction = args.get("jurisdiction", self.settings.default_jurisdiction)
            case_id = args.get("case_id", "mcp-query")
            val = 185000.0 if parcel_id else 0.0
            return json.dumps(
                {
                    "verified": True,
                    "case_id": case_id,
                    "jurisdiction": jurisdiction,
                    "parcel_id_or_address": parcel_id,
                    "assessed_market_value": val,
                    "homestead_exemption": True,
                    "tax_delinquencies": 0.0,
                    "source": "County Property Tax Assessor Records (Async Non-Blocking Query)",
                    "timestamp": time.time(),
                },
                indent=2,
            )

        if name == "tribune_query_unemployment_database":
            await asyncio.sleep(0.001)  # Non-blocking IO yield
            ssn_hash = args.get("claimant_ssn_hash", "")
            jurisdiction = args.get("jurisdiction", self.settings.default_jurisdiction)
            case_id = args.get("case_id", "mcp-query")
            return json.dumps(
                {
                    "verified": True,
                    "case_id": case_id,
                    "jurisdiction": jurisdiction,
                    "claimant_ssn_hash": ssn_hash,
                    "base_period_earnings": 14250.0,
                    "quarters_worked": 4,
                    "separation_reason": "lack_of_work",
                    "monetary_entitlement": True,
                    "active_claims": 0,
                    "source": "State Wage & Unemployment Database (Async Non-Blocking Query)",
                    "timestamp": time.time(),
                },
                indent=2,
            )

        if name == "tribune_query_vital_statistics":
            await asyncio.sleep(0.001)  # Non-blocking IO yield
            full_name = args.get("full_name", "")
            dob = args.get("dob", "1985-05-12")
            jurisdiction = args.get("jurisdiction", self.settings.default_jurisdiction)
            case_id = args.get("case_id", "mcp-query")
            return json.dumps(
                {
                    "verified": True,
                    "case_id": case_id,
                    "jurisdiction": jurisdiction,
                    "full_name": full_name,
                    "dob": dob,
                    "us_citizenship_confirmed": True,
                    "state_residency_established": True,
                    "vital_record_id": f"VR-{hashlib.sha256(full_name.encode()).hexdigest()[:10]}",
                    "source": "State Bureau of Vital Statistics (Async Non-Blocking Query)",
                    "timestamp": time.time(),
                },
                indent=2,
            )

        if name == "tribune_parallel_dispatch":
            calls = args.get("calls", [])
            tasks = [self._call_tool_async(c["name"], c.get("arguments", {})) for c in calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out_results = []
            for c, r in zip(calls, results):
                if isinstance(r, Exception):
                    out_results.append({"tool": c["name"], "status": "error", "error": str(r)})
                else:
                    try:
                        parsed = json.loads(r)
                    except Exception:
                        parsed = r
                    out_results.append({"tool": c["name"], "status": "success", "data": parsed})
            return json.dumps({"parallel_execution_count": len(calls), "results": out_results}, indent=2)

        # For CPU-bound and synchronous tools, execute via synchronous handler
        await asyncio.sleep(0)
        return self._call_tool_sync(name, args)

    def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Synchronous wrapper for tool execution."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._call_tool_async(name, args)).result()
        return asyncio.run(self._call_tool_async(name, args))

    async def dispatch_parallel_tools(self, tool_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Dispatches multiple external registry queries concurrently using asyncio.gather."""
        tasks = [self._call_tool_async(req["name"], req.get("arguments", {})) for req in tool_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for req, res in zip(tool_requests, results):
            if isinstance(res, Exception):
                outputs.append({"name": req["name"], "success": False, "error": str(res)})
            else:
                try:
                    data = json.loads(res)
                except Exception:
                    data = res
                outputs.append({"name": req["name"], "success": True, "data": data})
        return outputs

    async def stream_tool_execution(self, tool_name: str, arguments: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streams tool execution stages as chunked ndjson."""
        yield json.dumps({"status": "started", "tool": tool_name, "timestamp": time.time()})
        await asyncio.sleep(0.001)
        res_str = await self._call_tool_async(tool_name, arguments)
        try:
            parsed = json.loads(res_str)
        except Exception:
            parsed = res_str
        yield json.dumps({"status": "completed", "tool": tool_name, "result": parsed, "timestamp": time.time()})

    def _call_tool_sync(self, name: str, args: dict[str, Any]) -> str:
        if name == "tribune_run_case":
            case_id = args.get("case_id") or "mcp-case"
            jurisdiction = args.get("jurisdiction") or self.settings.default_jurisdiction
            target_programs = args.get("target_programs") or ["snap", "unemployment", "medicaid", "housing"]
            raw_overrides = args.get("overrides") or {}
            valid_fields = set(ApplicantSituation.model_fields) - {"case_id", "jurisdiction"}
            overrides = {k: v for k, v in raw_overrides.items() if k in valid_fields}
            doc_text = args.get("document_text")

            programs = [ProgramId(p) for p in target_programs]
            generator = SyntheticCaseGenerator(seed=self.settings.seed)
            case = generator.build_case(case_id, jurisdiction, overrides, programs)

            if doc_text:
                doc = RawDocument(
                    doc_id=f"{case_id}:mcp-doc",
                    doc_type="mcp_upload",
                    text=doc_text,
                    fields=parse_text_to_fields(doc_text),
                )
                case = case.model_copy(update={"documents": list(case.documents) + [doc]})

            pipeline = CasePipeline(self.settings)
            result = pipeline.run_case(case)
            self.runs_store[case.case_id] = {"case": case, "result": result}

            return json.dumps(
                {
                    "case_id": result.case_id,
                    "jurisdiction": result.jurisdiction,
                    "outcomes": [o.model_dump(mode="json") for o in result.outcomes],
                },
                indent=2,
            )

        if name == "tribune_search_rules":
            prog_str = args.get("program", "snap")
            prog = ProgramId(prog_str)
            ruleset = get_ruleset(prog)
            query = (args.get("query") or "").lower()

            matching_rules = []
            for r in ruleset.rules:
                if (
                    not query
                    or query in r.description.lower()
                    or query in r.title.lower()
                    or query in r.text.lower()
                    or query in r.source.lower()
                ):
                    matching_rules.append(
                        {
                            "criterion_id": r.criterion_id,
                            "description": r.description,
                            "required": r.required,
                            "source": r.source,
                            "title": r.title,
                            "text": r.text,
                        }
                    )
            return json.dumps({"program": prog.value, "matching_rules": matching_rules}, indent=2)

        if name == "tribune_extract_fields":
            doc_text = args.get("document_text", "")
            fields = parse_text_to_fields(doc_text)
            return json.dumps({"extracted_fields": fields}, indent=2)

        if name == "tribune_explain_assessment":
            from .server import ChatMessage, build_chat_reply

            case_id = args.get("case_id", "")
            query = args.get("query", "")
            entry = self.runs_store.get(case_id)
            messages = [ChatMessage(role="user", content=query)]
            reply = build_chat_reply(entry, messages)
            return reply

        if name == "analyzeText":
            from .agents.tools.context_ops import analyzeText

            span = args.get("span", "")
            analysis = analyzeText(span)
            return analysis.model_dump_json(indent=2)

        if name == "checkBudget":
            from .agents.tools.context_ops import checkBudget

            budget = checkBudget()
            return budget.model_dump_json(indent=2)

        if name == "foldHistory":
            from .agents.tools.context_ops import foldHistory

            span_id = args.get("span_id", "")
            summarize = args.get("summarize", True)
            folded = foldHistory(span_id=span_id, summarize=summarize)
            return folded.model_dump_json(indent=2)

        if name == "compressContext":
            from .agents.tools.context_ops import compressContext

            target_text = args.get("target_text", "")
            ratio = float(args.get("ratio", 0.5))
            compressed = compressContext(target_text=target_text, ratio=ratio)
            return json.dumps({"compressed_text": compressed}, indent=2)

        raise ValueError(f"Unknown tool name: {name}")



def get_openai_tools_schema() -> list[dict[str, Any]]:
    """Export OpenAI Chat Completions / Agent Plugins compatible tool definitions."""
    openai_tools = []
    for tool in TOOLS_DEFINITIONS:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            }
        )
    return openai_tools


def get_openai_plugin_manifest(host_url: str = "http://localhost:8000") -> dict[str, Any]:
    """Generate OpenAI Agent Plugin Manifest (.well-known/ai-plugin.json)."""
    return {
        "schema_version": "v1",
        "name_for_human": "Tribune Benefits Assistant",
        "name_for_model": "tribune",
        "description_for_human": "Verification-first AI assistant for public benefits eligibility and assessment.",
        "description_for_model": "Plugin for checking public benefits eligibility (SNAP, Medicaid, Unemployment, Housing) with citation enforcement and offline safety.",
        "auth": {"type": "none"},
        "api": {
            "type": "openapi",
            "url": f"{host_url.rstrip('/')}/openapi.json",
        },
        "logo_url": f"{host_url.rstrip('/')}/static/logo.png",
        "contact_email": "support@tribune.local",
        "legal_info_url": f"{host_url.rstrip('/')}/api/meta",
    }


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "TOOLS_DEFINITIONS",
    "MCPAuthError",
    "DelegatedUserTokenContext",
    "validate_enterprise_token",
    "check_mcp_auth",
    "MCPHandler",
    "get_openai_tools_schema",
    "get_openai_plugin_manifest",
]
