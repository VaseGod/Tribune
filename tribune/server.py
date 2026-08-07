"""FastAPI backend that wraps the TRIBUNE :class:`CasePipeline` for the web UI.

Everything the SPA needs sits behind ``/api/*``; the SPA itself is served as static
files from ``tribune/static/``. The same safety guarantees that hold in the library
hold here — assessments are cited, abstention is first-class, and a binding
submission only happens through the :class:`ActionGate` with an explicit human
sign-off.

Run it with ``tribune web`` (see :mod:`tribune.cli`).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .casegen.synthetic import SyntheticCaseGenerator
from .config import TribuneSettings, get_settings
from .corpus.programs import all_programs, benefit_programs, known_jurisdictions
from .governance.action_gate import ActionBlocked, ActionGate, HumanSignoff
from .ingestion.ocr import parse_text_to_fields
from .mcp import MCPHandler, get_openai_plugin_manifest, get_openai_tools_schema
from .orchestration.pipeline import CasePipeline
from .types import ApplicantSituation, CaseRunResult, ProgramId, RawDocument, SyntheticCase

PROGRAM_NAMES = {
    "snap": "SNAP (food assistance)",
    "unemployment": "Unemployment insurance",
    "medicaid": "Medicaid / health coverage",
    "housing": "Housing assistance",
    "appeals": "Appeal / fair hearing",
}

_DEFAULT_PROGRAMS = [p.value for p in benefit_programs()]
_SITUATION_FIELDS = set(ApplicantSituation.model_fields) - {"case_id", "jurisdiction"}

# In-memory record of the most recent run per case_id, so submit-action and chat
# can reference the prepared materials and reasoning. (A real deployment would use
# the access-controlled MemoryStore; this keeps the demo server self-contained.)
_RUNS: dict[str, dict] = {}


app = FastAPI(
    title="TRIBUNE",
    version=__version__,
    description="Verification-first benefits navigator — informational only, never a binding determination.",
)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class RunRequest(BaseModel):
    case_id: str | None = None
    jurisdiction: str | None = None
    overrides: dict | None = None
    target_programs: list[str] | None = None
    document_text: str | None = None
    # Settings overrides (from the Settings panel).
    provider: str | None = None
    abstention_threshold: float | None = None


class SubmitActionRequest(BaseModel):
    case_id: str
    program: str
    authorized_by: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    case_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ser(model) -> dict:
    return model.model_dump(mode="json")


def _settings_from(req: RunRequest) -> TribuneSettings:
    base = get_settings()
    updates: dict = {}
    if req.provider:
        updates["provider"] = req.provider
    if req.jurisdiction:
        updates["default_jurisdiction"] = req.jurisdiction
    if req.abstention_threshold is not None:
        updates["abstention_threshold"] = float(req.abstention_threshold)
    return base.model_copy(update=updates) if updates else base


def _demo_cases(settings: TribuneSettings) -> list[SyntheticCase]:
    return SyntheticCaseGenerator(seed=settings.seed).generate_demo_set()


def _find_demo_case(case_id: str, settings: TribuneSettings) -> SyntheticCase | None:
    for c in _demo_cases(settings):
        if c.case_id == case_id:
            return c
    return None


def _programs_from(req: RunRequest) -> list[ProgramId]:
    names = req.target_programs or _DEFAULT_PROGRAMS
    out: list[ProgramId] = []
    for n in names:
        try:
            out.append(ProgramId(n))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown program '{n}'") from exc
    return out


def _build_custom_case(req: RunRequest, settings: TribuneSettings) -> SyntheticCase:
    generator = SyntheticCaseGenerator(seed=settings.seed)
    jurisdiction = req.jurisdiction or settings.default_jurisdiction
    case_id = req.case_id or f"custom-{uuid4().hex[:8]}"
    overrides = {k: v for k, v in (req.overrides or {}).items() if k in _SITUATION_FIELDS}
    try:
        case = generator.build_case(case_id, jurisdiction, overrides, _programs_from(req))
    except Exception as exc:  # pydantic validation of the situation, bad jurisdiction, etc.
        raise HTTPException(status_code=422, detail=f"invalid case input: {exc}") from exc

    if req.document_text:
        extra = RawDocument(
            doc_id=f"{case_id}:user-doc",
            doc_type="user_document",
            text=req.document_text,
            fields=parse_text_to_fields(req.document_text),
        )
        case = case.model_copy(update={"documents": list(case.documents) + [extra]})
    return case


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/meta")
def meta() -> dict:
    s = get_settings()
    return {
        "jurisdictions": known_jurisdictions(),
        "programs": [p.value for p in all_programs()],
        "benefit_programs": _DEFAULT_PROGRAMS,
        "program_names": PROGRAM_NAMES,
        "providers": ["local_rules", "openai_compat", "router"],
        "defaults": {
            "jurisdiction": s.default_jurisdiction,
            "provider": s.provider,
            "abstention_threshold": s.abstention_threshold,
        },
    }


# --------------------------------------------------------------------------- #
# MCP & Agent Plugin routes
# --------------------------------------------------------------------------- #


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> dict:
    """Model Context Protocol (MCP) JSON-RPC 2.0 endpoint."""
    try:
        payload = await request.json()
    except Exception:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error: invalid JSON"},
        }
    headers = dict(request.headers)
    handler = MCPHandler(runs_store=_RUNS, settings=get_settings())
    return handler.handle_request(payload, headers)


@app.get("/mcp")
def mcp_info() -> dict:
    return {
        "status": "ok",
        "service": "tribune-mcp",
        "protocolVersion": "2024-11-05",
        "endpoint": "/mcp",
    }


@app.get("/.well-known/ai-plugin.json")
def openai_plugin_manifest(request: Request) -> dict:
    base_url = str(request.base_url)
    return get_openai_plugin_manifest(base_url)


@app.get("/api/openai-tools")
def openai_tools_schema() -> dict:
    return {"tools": get_openai_tools_schema()}


@app.get("/api/cases/demo")
def demo_cases() -> list[dict]:
    settings = get_settings()
    out = []
    for case in _demo_cases(settings):
        out.append(
            {
                "case_id": case.case_id,
                "jurisdiction": case.jurisdiction,
                "situation": _ser(case.situation),
                "target_programs": [p.value for p in case.target_programs],
                "program_names": [PROGRAM_NAMES.get(p.value, p.value) for p in case.target_programs],
                "documents": [_ser(d) for d in case.documents],
            }
        )
    return out


@app.post("/api/cases/run")
def run_case(req: RunRequest) -> dict:
    settings = _settings_from(req)
    is_demo = bool(req.case_id) and not req.overrides and not req.document_text
    case = _find_demo_case(req.case_id, settings) if is_demo else None
    if case is None:
        case = _build_custom_case(req, settings)

    pipeline = CasePipeline(settings)
    result: CaseRunResult = pipeline.run_case(case)
    chain_ok = pipeline.audit.verify_chain(case.case_id)
    _RUNS[case.case_id] = {"case": case, "result": result}

    return {
        "case": {
            "case_id": case.case_id,
            "jurisdiction": case.jurisdiction,
            "situation": _ser(case.situation),
            "target_programs": [p.value for p in case.target_programs],
            "documents": [_ser(d) for d in case.documents],
        },
        "result": _ser(result),
        "audit_chain_verified": chain_ok,
        "program_names": PROGRAM_NAMES,
        "settings": {
            "provider": settings.provider,
            "jurisdiction": case.jurisdiction,
            "abstention_threshold": settings.abstention_threshold,
        },
    }


@app.post("/api/cases/submit-action")
def submit_action(req: SubmitActionRequest) -> dict:
    entry = _RUNS.get(req.case_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="case not found; run the assessment first")
    try:
        program = ProgramId(req.program)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown program '{req.program}'") from exc

    outcome = entry["result"].outcome_for(program)
    if outcome is None or outcome.materials is None:
        raise HTTPException(
            status_code=409,
            detail="no prepared materials to submit for this program (it was not eligible / was abstained)",
        )

    signoff = HumanSignoff.issue(req.authorized_by, program.value, req.case_id)
    try:
        receipt = ActionGate().authorize_submission(outcome.materials, signoff)
    except ActionBlocked as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return {"receipt": _ser(receipt), "materials": _ser(outcome.materials)}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    entry = _RUNS.get(req.case_id) if req.case_id else None
    return {"reply": build_chat_reply(entry, req.messages)}


# --------------------------------------------------------------------------- #
# Grounded navigator chat (deterministic, offline-safe).
# --------------------------------------------------------------------------- #

_FOOTER = (
    "This is informational only — not legal advice and not an official determination. "
    "A navigator, caseworker, or the agency makes the final call, and you can always ask a person for help."
)


def build_chat_reply(entry: dict | None, messages: list[ChatMessage]) -> str:
    """A grounded assistant: it only states what the actual run found.

    It never invents eligibility facts; it explains the assessment, surfaces what was
    uncertain or missing, and points to the human path. If a served model is
    configured it could be swapped in here, but staying grounded is the on-brand
    default for a benefits tool.
    """
    user_msgs = [m for m in messages if m.role == "user"]
    last = (user_msgs[-1].content if user_msgs else "").lower()

    if entry is None:
        return (
            "I can help once you've run an assessment. Pick a demo case or fill in the "
            "New Case wizard, then ask me why TRIBUNE reached its conclusion or what "
            "documents would help. " + _FOOTER
        )

    result: CaseRunResult = entry["result"]
    abstained, prepared, ineligible, eligible_pending = [], [], [], []
    missing: dict[str, list[str]] = {}
    for o in result.outcomes:
        name = PROGRAM_NAMES.get(o.program.value, o.program.value)
        if o.abstained:
            reason = o.abstention.reason if o.abstention else "the case is genuinely uncertain"
            abstained.append((name, reason))
            if o.assessment is not None:
                unknown = [c.description for c in o.assessment.criteria if c.outcome.value == "unknown"]
                if unknown:
                    missing[name] = unknown
        elif o.materials is not None:
            prepared.append(name)
        elif o.assessment is not None and o.assessment.status.value == "likely_ineligible":
            ineligible.append(name)
        elif o.assessment is not None:
            eligible_pending.append(name)

    lines: list[str] = []

    # Tailor the opening to the user's intent.
    if any(k in last for k in ("document", "missing", "need", "provide", "upload", "paperwork", "proof")):
        lines.append("Here's what would help move your case forward:")
        if missing:
            for name, items in missing.items():
                lines.append(f"• {name}: TRIBUNE couldn't confirm — " + "; ".join(items) + ".")
            lines.append("Adding documentation for those and re-running should let TRIBUNE resolve them.")
        else:
            lines.append(
                "TRIBUNE had what it needed to check the listed criteria. Where it abstained, it was "
                "because the call is close or state-dependent — a navigator's judgment helps more than another document."
            )
    elif any(k in last for k in ("why", "explain", "reason", "how come")):
        lines.append("Here's why TRIBUNE reached these conclusions:")
        for o in result.outcomes:
            name = PROGRAM_NAMES.get(o.program.value, o.program.value)
            if o.abstained and o.abstention is not None:
                lines.append(f"• {name}: it abstained — {o.abstention.reason}.")
            elif o.assessment is not None:
                lines.append(f"• {name}: {o.assessment.status.value.replace('_', ' ')}.")
    elif any(k in last for k in ("human", "navigator", "help", "who", "talk", "person", "caseworker")):
        lines.append(
            "Yes — you can and should talk to a person. A benefits navigator, a community "
            "organization, legal aid, or the administering agency can review this with you and "
            "act on anything binding. TRIBUNE never replaces them; it prepares and explains."
        )
    elif "appeal" in last:
        lines.append(
            "If you were denied a benefit you believe you qualify for, you generally have a limited "
            "window to request a fair hearing / appeal. TRIBUNE can prepare an appeal packet (grounds, "
            "deadline, documents) — but you or your navigator must file it."
        )
    else:
        lines.append("Here's where your case stands:")

    # Always include the grounded status summary.
    if prepared:
        lines.append("✓ Prepared (not submitted) for: " + ", ".join(prepared) + ".")
    if eligible_pending:
        lines.append("• Assessed for: " + ", ".join(eligible_pending) + ".")
    if abstained:
        lines.append("⚠ Routed to a human (abstained) for: " + ", ".join(n for n, _ in abstained) + ".")
    if ineligible:
        lines.append("✗ Did not appear to meet the rules for: " + ", ".join(ineligible) + " — a human can confirm or help you contest this.")

    lines.append(_FOOTER)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Errors + static SPA (mounted last so /api/* takes precedence)
# --------------------------------------------------------------------------- #


@app.exception_handler(ActionBlocked)
def _action_blocked_handler(_request, exc: ActionBlocked):  # pragma: no cover - defensive
    return JSONResponse(status_code=403, content={"detail": str(exc)})


_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
