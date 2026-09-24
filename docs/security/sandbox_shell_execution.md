# Hardened Sandboxed Shell Execution & Sentinel Security Daemon

*Authored 2026-09-24 by Tribune AI Systems & Security Engineering.*

---

## 1. Executive Summary & Problem Statement

Static programmatic tool catalogs consume excessive prompt tokens and fail unpredictably when evaluated models attempt file manipulation and data pipeline operations. Conversely, naive monolithic shell execution allows degraded quantized models to leak credentials, run repetitive failure loops, probe system files, or attempt unauthorized network egress.

To resolve these tensions, Tribune implements a **hardened dual-zone execution architecture**:
1. **Trusted Host Tier:** Holds authentic credentials in-memory, runs the Sentinel security broker, hosts the Proactive Memory sidecar, and inspects all commands and egress requests out-of-band.
2. **Controlled Execution Tier:** Executes quantized models within an unprivileged, capability-dropped container (`ContainerShellRunner`) or isolated fallback (`LocalFallbackRunner`), viewing exclusively synthetic surrogate tokens (`MOCK_*`).

---

## 2. Threat Model & Trust Boundaries

```
┌────────────────────────────────────────────────────────────────────────┐
│ TRUSTED ZONE (Host Tier)                                               │
│ - Orchestration Ladder (`run_ladder()`)                                │
│ - Sentinel Daemon & In-Memory Secret Vault (holds sk-live-..., etc.)   │
│ - Proactive Memory Sidecar Engine & 5-Track State Bank                 │
│ - Unrestricted Host Network Interface (allowlist-mediated)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         UNIX DOMAIN SOCKET IPC
                     (JSON Requests, Allow/Deny Only)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ UNTRUSTED ZONE (Controlled Execution Tier)                             │
│ - Model-facing shell runner (`tribune-sandbox` UID 10001)              │
│ - Read-only root filesystem with ephemeral tmpfs scratch               │
│ - Dropped capabilities (`--cap-drop=ALL`)                              │
│ - Zero direct host network access (`--network=none`)                   │
│ - Injected environment has ONLY `MOCK_*` surrogate tokens              │
│ - Outbound command stdout/stderr auto-redacted before context return   │
└────────────────────────────────────────────────────────────────────────┘
```

### Threat Vectors Addressed
1. **Credential Exfiltration:** Quantized models dumping environment variables (`printenv`, `export -p`, `.env`) receive only deterministic synthetic handles (e.g. `MOCK_ACCESS_HANDLE_ALPHA`).
2. **Privilege Escalation:** Sandboxed processes run as non-root with dropped Linux capabilities, preventing `setuid`, `sudo`, or raw disk access.
3. **Container Escape & Docker Socket Abuse:** Protected system paths (`/var/run/docker.sock`, `/etc/shadow`, `/etc/sudoers`) are blocked by Sentinel AST/regex checks and container bind policies.
4. **Adversarial Network Tunneling:** Outbound network calls from the sandbox are denied by default. Any permitted API queries are routed through Sentinel, which injects authentic auth headers strictly on the host.

---

## 3. Sentinel Socket Protocol & IPC

The Sentinel daemon operates over a local Unix domain socket (`/tmp/tribune_sentinel.sock` by default, configurable via `TRIBUNE_SENTINEL_SOCKET_PATH`).

### 3.1 Check Command Request / Response
```json
// Runner -> Sentinel
{
  "action": "check_command",
  "command": "cat /workspace/data/input.csv | grep SNAP",
  "session_id": "session_102",
  "task_id": "task_snap_calc"
}

// Sentinel -> Runner
{
  "allowed": true,
  "reason": "Command authorized by Sentinel policy",
  "decision_code": "ALLOWED",
  "redacted_command": "cat /workspace/data/input.csv | grep SNAP"
}
```

### 3.2 Check Network Egress Request / Response
```json
// Runner -> Sentinel
{
  "action": "check_egress",
  "destination_host": "mock.internal.tribune",
  "port": 443,
  "scheme": "https",
  "method": "POST",
  "path": "/api/v1/verify",
  "task_id": "task_snap_calc"
}

// Sentinel -> Runner
{
  "allowed": true,
  "reason": "Allowed by rule 'allow_local_mock_api'",
  "auth_profile": "mock_internal_api",
  "matched_rule_id": "allow_local_mock_api"
}
```
*Note: Real secrets and authorization headers are never transmitted across the socket to the sandbox.*

---

## 4. Surrogate Token Brokering & Redaction

`TokenBroker` (`tribune/security/token_broker.py`) enforces strict separation between raw secrets and sandbox visibility:
- **Scan & Mask:** Environment variables such as `OPENAI_API_KEY`, `TRIBUNE_HMAC_SECRET`, or values matching secret regexes (`sk-...`, `Bearer ...`) are replaced with `MOCK_*` surrogates.
- **Host-Only Vault:** Mappings (`surrogate_token_id -> secret_ref -> injection_policy`) are maintained in host RAM only and never serialized into reports or logs.
- **Bidirectional Redaction:** Before any command output is returned to the model or persisted in evaluation traces, `TokenBroker.redact_text()` scans and replaces raw secret patterns with `[REDACTED_SECRET]`.

---

## 5. Machine-Readable Allowlist Format

Allowlists are stored in `config/sentinel_allowlist.yaml`:

```yaml
version: 1
default_policy: deny
rules:
  - id: allow_local_mock_api
    destination_host: mock.internal.tribune
    port: 443
    scheme: https
    methods: [GET, POST]
    auth_profile: mock_internal_api
    max_requests_per_task: 50

  - id: allow_static_dataset
    destination_host: datasets.internal.tribune
    port: 443
    scheme: https
    methods: [GET]
    auth_profile: none
    max_requests_per_task: 20
```

### Auth Profiles
- `none`: Egress allowed with no host auth injection.
- `bearer_host_secret_ref`: Host injects `Authorization: Bearer <real_secret>`.
- `api_key_header_host_secret_ref`: Host injects `X-API-Key: <real_secret>`.
- `custom_header_host_secret_ref`: Host injects custom headers configured in vault.

---

## 6. Operational Runbook & Configuration

### Configuration Flags

| Environment Variable | Allowed Values | Default | Purpose |
|---|---|---|---|
| `TRIBUNE_EXECUTION_MODE` | `container`, `local_fallback`, `legacy`, `auto` | `auto` | Selects execution runner backend. |
| `TRIBUNE_SENTINEL_ENABLED` | `true`, `false` | `true` | Toggles Sentinel daemon interception. |
| `TRIBUNE_SENTINEL_SOCKET_PATH` | Path | `/tmp/tribune_sentinel.sock` | UDS socket path. |
| `TRIBUNE_ALLOWLIST_PATH` | Path | `config/sentinel_allowlist.yaml` | Egress allowlist rules. |
| `TRIBUNE_CONTAINER_IMAGE` | Docker Image Tag | `tribune-sandbox:latest` | Sandboxed runtime container. |
| `TRIBUNE_CONTAINER_USER` | Username / UID | `tribune-sandbox` | Non-root sandbox user. |
| `TRIBUNE_CONTAINER_NETWORK_MODE` | `none`, `sentinel_proxy` | `none` | Network sandboxing mode. |
| `TRIBUNE_SURROGATE_TOKEN_PREFIX` | String | `MOCK_` | Prefix for synthetic handles. |
| `TRIBUNE_MAX_COMMAND_TIMEOUT_SECONDS` | Float | `60.0` | Maximum wall-clock command time. |

### How to Run Hardened Evaluations
```bash
# Run quantization ladder with full container isolation
export TRIBUNE_EXECUTION_MODE=container
export TRIBUNE_SENTINEL_ENABLED=true
pytest tests/test_quant_sensitivity.py
```

### How to Enable / Disable Legacy Mode
```bash
# Temporarily enable legacy unhardened mode for baseline comparison
export TRIBUNE_EXECUTION_MODE=legacy
export TRIBUNE_SENTINEL_ENABLED=false
```

### How to Run Tests
```bash
# Run Sentinel and runner unit tests
.venv/bin/pytest tests/test_sentinel_isolation.py tests/test_sandbox_runtime.py
```

---

## 7. Failure Modes & Known Limitations

1. **Docker Daemon Unavailable:** When running in environments lacking Docker (e.g. standard developer laptops or nested VMs), `auto` mode degrades gracefully to `LocalFallbackRunner`, which maintains full token surrogateing and allowlist checks, but issues `[SECURITY ALERT]` logs and flags `unsafe_for_production=True`.
2. **UDS Socket Permission Errors:** If the host process crashes unexpectedly without cleaning `/tmp/tribune_sentinel.sock`, `SentinelServer.start()` automatically unlinks stale socket files before binding.
3. **Denial Code 126:** Returned when commands violate Sentinel pattern rules or attempt access to system paths.
4. **Denial Code 127:** Returned when commands are quarantined by the Exploit Detection Engine for identical command looping.
