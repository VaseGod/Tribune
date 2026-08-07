"""Tests for Model Context Protocol (MCP) server interface and plugin endpoints."""

import json
import pytest
from fastapi.testclient import TestClient

from tribune.config import TribuneSettings, get_settings, reset_settings_cache
from tribune.mcp import MCPHandler, MCPAuthError, check_mcp_auth
from tribune.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_mcp_initialize(client):
    req = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "initialize",
        "params": {},
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == "1"
    assert data["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in data["result"]["capabilities"]
    assert "resources" in data["result"]["capabilities"]
    assert data["result"]["serverInfo"]["name"] == "tribune-mcp"


def test_mcp_resources_list_and_read(client):
    # List resources
    req = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "resources/list",
        "params": {},
    }
    resp = client.post("/mcp", json=req)
    assert resp.status_code == 200
    data = resp.json()
    resources = data["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "tribune://programs" in uris
    assert "tribune://jurisdictions" in uris
    assert "tribune://demo-cases" in uris
    assert "tribune://rules/snap" in uris

    # Read resource: tribune://programs
    req_read = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "resources/read",
        "params": {"uri": "tribune://programs"},
    }
    resp_read = client.post("/mcp", json=req_read)
    assert resp_read.status_code == 200
    contents = resp_read.json()["result"]["contents"]
    assert len(contents) == 1
    prog_list = json.loads(contents[0]["text"])
    assert "snap" in prog_list
    assert "medicaid" in prog_list


def test_mcp_tools_list_and_call_search_rules(client):
    # List tools
    req_list = {
        "jsonrpc": "2.0",
        "id": "4",
        "method": "tools/list",
        "params": {},
    }
    resp_list = client.post("/mcp", json=req_list)
    assert resp_list.status_code == 200
    tools = resp_list.json()["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "tribune_run_case" in tool_names
    assert "tribune_search_rules" in tool_names
    assert "tribune_extract_fields" in tool_names
    assert "tribune_explain_assessment" in tool_names

    # Call tool: tribune_search_rules
    req_call = {
        "jsonrpc": "2.0",
        "id": "5",
        "method": "tools/call",
        "params": {
            "name": "tribune_search_rules",
            "arguments": {"program": "snap", "query": "income"},
        },
    }
    resp_call = client.post("/mcp", json=req_call)
    assert resp_call.status_code == 200
    res_content = resp_call.json()["result"]["content"]
    assert len(res_content) == 1
    rules_obj = json.loads(res_content[0]["text"])
    assert rules_obj["program"] == "snap"
    assert "matching_rules" in rules_obj


def test_mcp_tools_call_run_case(client):
    req_call = {
        "jsonrpc": "2.0",
        "id": "6",
        "method": "tools/call",
        "params": {
            "name": "tribune_run_case",
            "arguments": {
                "case_id": "mcp-test-1",
                "jurisdiction": "EX",
                "target_programs": ["snap"],
                "overrides": {"monthly_income": 800, "household_size": 2},
            },
        },
    }
    resp = client.post("/mcp", json=req_call)
    assert resp.status_code == 200
    res_text = resp.json()["result"]["content"][0]["text"]
    out_obj = json.loads(res_text)
    assert out_obj["case_id"] == "mcp-test-1"
    assert len(out_obj["outcomes"]) > 0


def test_mcp_extract_fields_and_explain(client):
    # Extract fields
    req_ext = {
        "jsonrpc": "2.0",
        "id": "7",
        "method": "tools/call",
        "params": {
            "name": "tribune_extract_fields",
            "arguments": {"document_text": "Paystub: gross pay $1,500 monthly. Household count: 3."},
        },
    }
    resp_ext = client.post("/mcp", json=req_ext)
    assert resp_ext.status_code == 200
    ext_text = resp_ext.json()["result"]["content"][0]["text"]
    ext_obj = json.loads(ext_text)
    assert "extracted_fields" in ext_obj

    # Explain assessment
    req_exp = {
        "jsonrpc": "2.0",
        "id": "8",
        "method": "tools/call",
        "params": {
            "name": "tribune_explain_assessment",
            "arguments": {"case_id": "nonexistent", "query": "Why was I denied?"},
        },
    }
    resp_exp = client.post("/mcp", json=req_exp)
    assert resp_exp.status_code == 200
    exp_text = resp_exp.json()["result"]["content"][0]["text"]
    assert "informational only" in exp_text.lower()


def test_mcp_auth_and_rbac(monkeypatch, client):
    monkeypatch.setenv("TRIBUNE_MCP_AUTH_TOKEN", "secret-mcp-token")
    reset_settings_cache()

    req_init = {
        "jsonrpc": "2.0",
        "id": "auth1",
        "method": "initialize",
        "params": {},
    }

    # Missing auth token -> 401/Unauthorized JSON-RPC error
    resp_unauth = client.post("/mcp", json=req_init)
    assert resp_unauth.status_code == 200
    assert resp_unauth.json()["error"]["code"] == -32001

    # Valid Bearer token
    headers = {"Authorization": "Bearer secret-mcp-token"}
    resp_auth = client.post("/mcp", json=req_init, headers=headers)
    assert resp_auth.status_code == 200
    assert "result" in resp_auth.json()

    # RBAC check: read_only role forbidden from run_case
    req_run = {
        "jsonrpc": "2.0",
        "id": "rbac1",
        "method": "tools/call",
        "params": {
            "name": "tribune_run_case",
            "arguments": {"case_id": "rbac-test"},
        },
    }
    rbac_headers = {"Authorization": "Bearer secret-mcp-token", "X-Tribune-Role": "read_only"}
    resp_rbac = client.post("/mcp", json=req_run, headers=rbac_headers)
    assert resp_rbac.status_code == 200
    assert resp_rbac.json()["error"]["code"] == -32002
    assert "Forbidden" in resp_rbac.json()["error"]["message"]


def test_openai_plugin_endpoints(client):
    # Manifest
    resp_manifest = client.get("/.well-known/ai-plugin.json")
    assert resp_manifest.status_code == 200
    manifest = resp_manifest.json()
    assert manifest["schema_version"] == "v1"
    assert manifest["name_for_model"] == "tribune"

    # OpenAI tools schema
    resp_tools = client.get("/api/openai-tools")
    assert resp_tools.status_code == 200
    tools_data = resp_tools.json()["tools"]
    assert len(tools_data) >= 4
    tool_names = [t["function"]["name"] for t in tools_data]
    assert "tribune_run_case" in tool_names


def test_mcp_get_info_endpoint(client):
    resp = client.get("/mcp")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "tribune-mcp"
    assert data["status"] == "ok"
