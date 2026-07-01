"""Integration tests for the FastAPI web backend.

Verify the server starts, routes return 200, case runs validate against TRIBUNE's
types, and the safety-critical web flows (cited assertions, action-gated submission)
behave correctly.
"""

from fastapi.testclient import TestClient

from tribune.server import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_meta_lists_programs_and_jurisdictions():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert "snap" in body["programs"]
    assert "EX" in body["jurisdictions"]
    assert 0.5 <= body["defaults"]["abstention_threshold"] <= 1.0


def test_demo_cases_returns_cases():
    r = client.get("/api/cases/demo")
    assert r.status_code == 200
    cases = r.json()
    assert len(cases) >= 1
    first = cases[0]
    for key in ("case_id", "jurisdiction", "situation", "target_programs", "documents"):
        assert key in first


def test_run_demo_case_validates_and_is_cited():
    cid = client.get("/api/cases/demo").json()[0]["case_id"]
    r = client.post("/api/cases/run", json={"case_id": cid})
    assert r.status_code == 200
    body = r.json()
    assert body["audit_chain_verified"] is True
    assert body["result"]["outcomes"], "expected at least one program outcome"
    # Every asserted (non-abstained) result must carry citations.
    for o in body["result"]["outcomes"]:
        a = o["assessment"]
        if a and not o["abstained"] and a["status"] != "indeterminate":
            assert a["citations"], "asserted result without citations"


def test_run_custom_case():
    r = client.post(
        "/api/cases/run",
        json={
            "overrides": {
                "household_size": 3,
                "monthly_income": 1400,
                "resident": True,
                "citizenship_status": "citizen",
            },
            "target_programs": ["snap", "medicaid"],
            "jurisdiction": "EX",
            "document_text": "monthly_income: 1400\nhousehold_size: 3\nresident: true",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["case"]["case_id"].startswith("custom-")
    assert {o["program"] for o in body["result"]["outcomes"]} == {"snap", "medicaid"}


def test_run_unknown_program_rejected():
    r = client.post("/api/cases/run", json={"overrides": {"household_size": 1}, "target_programs": ["nope"]})
    assert r.status_code == 422


def test_submit_action_flow_and_gate():
    # demo-parent-multi has a prepared SNAP packet and an abstained Housing outcome.
    run = client.post("/api/cases/run", json={"case_id": "demo-parent-multi"}).json()
    prepared = [o["program"] for o in run["result"]["outcomes"] if o["materials"]]
    abstained = [o["program"] for o in run["result"]["outcomes"] if o["abstained"]]
    assert prepared, "expected a prepared program in the demo case"

    ok = client.post(
        "/api/cases/submit-action",
        json={"case_id": "demo-parent-multi", "program": prepared[0], "authorized_by": "Navigator Jane"},
    )
    assert ok.status_code == 200
    receipt = ok.json()["receipt"]
    assert receipt["authorized_by"] == "Navigator Jane"
    assert receipt["signoff_token"]

    # Submitting a program with no prepared materials (abstained) is refused.
    if abstained:
        bad = client.post(
            "/api/cases/submit-action",
            json={"case_id": "demo-parent-multi", "program": abstained[0], "authorized_by": "Navigator Jane"},
        )
        assert bad.status_code == 409


def test_submit_action_unknown_case_404():
    r = client.post(
        "/api/cases/submit-action",
        json={"case_id": "does-not-exist", "program": "snap", "authorized_by": "X"},
    )
    assert r.status_code == 404


def test_chat_is_grounded():
    client.post("/api/cases/run", json={"case_id": "demo-coverage-gap"})
    r = client.post(
        "/api/chat",
        json={
            "case_id": "demo-coverage-gap",
            "messages": [{"role": "user", "content": "why did you abstain?"}],
        },
    )
    assert r.status_code == 200
    reply = r.json()["reply"]
    assert "not legal advice" in reply.lower()


def test_chat_without_case_is_safe():
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert r.json()["reply"]


def test_spa_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "TRIBUNE" in r.text
