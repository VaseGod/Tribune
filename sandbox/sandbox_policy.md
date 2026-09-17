# Hardened Containerized Sandbox Policy & Architecture

## Overview

Tribune enforces strict isolation for code execution, document parsing, and file manipulation. The sandbox prevents:
1. Unauthorized network egress and data exfiltration.
2. Uncontrolled filesystem reads/writes outside the designated workspace.
3. Denial of service via unbounded runaway processes.
4. Privilege escalation and container breakouts.

---

## 1. Container Isolation Profile

When executed via `sandbox/sandbox_runtime.py` (`DockerSandboxRuntime`):

| Parameter | Configuration | Purpose |
| :--- | :--- | :--- |
| **Network** | `--network=none` | Absolute egress and ingress network denial at container boundary. |
| **User** | `uid 10001 (evaluser)` | Non-root execution preventing host daemon manipulation. |
| **Capabilities** | `--cap-drop=ALL` | Drops all Linux capabilities (e.g. `CAP_NET_RAW`, `CAP_SYS_ADMIN`). |
| **Memory Limit** | `--memory=2g` | Memory quota preventing out-of-memory host starvation. |
| **CPU Limit** | `--cpus=2.0` | Compute quota preventing CPU monopolization. |
| **Artifact Mount** | `-v /host/artifacts:/app/artifacts:rw` | Explicit directory boundary for deterministic artifact extraction. |

---

## 2. Execution Timeouts & Automated Kill Switch

- **Default Timeout**: 60.0 seconds (customizable per evaluation).
- **Graceful Signal**: At $t = \text{timeout}$, `SIGTERM` is dispatched to allow in-flight state checkpointing.
- **Hard Kill Switch**: If the container does not terminate within 3.0 seconds of `SIGTERM`, an unconditional `SIGKILL` (-9) is dispatched by the runtime supervisor.

---

## 3. Degraded Local Fallback Mode

If the Docker daemon is unreachable or disabled in local testing environments:
1. `get_sandbox_runtime()` automatically selects `LocalFallbackSandboxRuntime`.
2. A prominent security warning is emitted to `stderr` and the execution log.
3. In-process network egress interception (`TRIBUNE_NETGUARD=deny_all`) is enforced.
4. Wall-clock subprocess timeouts are maintained.
5. Production deployments must not permit degraded local fallback for untrusted code execution.
