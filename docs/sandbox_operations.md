# Sandboxed Execution Runtime Operations Guide

## Overview

Tribune executes appeals evaluation, code tools, document parsing, and file manipulation inside an isolated sandbox environment.

---

## 1. Production Docker Container Isolation

### Building the Sandbox Image

```bash
docker build -f sandbox/Dockerfile -t tribune-appeals-eval .
```

### Running with Network Denial and Artifact Extraction

```bash
docker run --rm \
  --network=none \
  --memory=2g \
  --cpus=2.0 \
  --cap-drop=ALL \
  -v $(pwd)/artifacts:/app/artifacts:rw \
  tribune-appeals-eval
```

---

## 2. Programmatic Python Sandbox Controller

Use `sandbox/sandbox_runtime.py`:

```python
from sandbox.sandbox_runtime import SandboxConfig, get_sandbox_runtime

config = SandboxConfig(
    image_name="tribune-appeals-eval:latest",
    timeout_s=60.0,
    network_disabled=True,
    memory_limit="2g",
    mode="auto",  # Uses Docker if available, else local fallback with warning
)

runtime = get_sandbox_runtime(config)
result = runtime.execute(["python", "-m", "tribune.eval.appeals_eval"])

print(f"Exit Code: {result.exit_code}")
print(f"Isolation Mode: {result.isolation_mode}")
print(f"Artifacts: {result.collected_artifacts}")
```

---

## 3. Degraded Local Fallback Mode

In developer environments where the Docker daemon is restricted:
- Set `TRIBUNE_SANDBOX_MODE=local_fallback`
- The runtime will emit a prominent security alert and enforce in-process netguard egress restrictions (`TRIBUNE_NETGUARD=deny_all`).
- Wall-clock timeouts and artifact directory separation remain enforced.
