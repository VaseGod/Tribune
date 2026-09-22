"""Secret resolution chain: env → file → external command.

Production deployments must not pass secrets only via process environment.
`resolve_secret` supports:

1. Direct env var: ``TRIBUNE_HMAC_SECRET``
2. File ref: ``TRIBUNE_HMAC_SECRET_FILE=/run/secrets/hmac`` (mode-checked;
   warns if group/other-readable). File content is stripped of trailing newline.
3. Command ref: ``TRIBUNE_HMAC_SECRET_CMD=vault kv get -field=secret …``
   (shlex-split, no shell, 10s timeout, stdout stripped).

Precedence: direct env > file > command. Values are never logged.
"""

from __future__ import annotations

import logging
import os
import shlex
import stat
import subprocess

logger = logging.getLogger(__name__)


def _file_secret(path: str, label: str) -> str | None:
    try:
        st = os.stat(path)
    except OSError as err:
        logger.warning("[SECRETS] %s file '%s' unreadable: %s", label, path, err)
        return None
    if st.st_mode & (stat.S_IRGRP | stat.S_IROTH):
        logger.warning(
            "[SECRETS] %s file '%s' is group/other-readable; restrict to 0600.",
            label,
            path,
        )
    try:
        with open(path, encoding="utf-8") as fh:
            value = fh.read().strip()
        return value or None
    except OSError as err:
        logger.warning("[SECRETS] %s file read failed: %s", label, err)
        return None


def _command_secret(argv: list[str], label: str, timeout_s: float = 10.0) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as err:
        logger.warning("[SECRETS] %s command failed: %s", label, err)
        return None
    if proc.returncode != 0:
        logger.warning(
            "[SECRETS] %s command exited %d.", label, proc.returncode
        )
        return None
    value = proc.stdout.strip().splitlines()
    return value[0].strip() if value and value[0].strip() else None


def resolve_secret(
    base_env: str,
    file_env: str | None = None,
    cmd_env: str | None = None,
) -> str | None:
    """Resolve a secret value without ever logging it.

    Args:
        base_env: direct env var name (e.g. ``TRIBUNE_HMAC_SECRET``).
        file_env: env var holding a file path (default ``<BASE>_FILE``).
        cmd_env: env var holding a command line (default ``<BASE>_CMD``).
    """
    direct = os.getenv(base_env, "")
    if direct:
        return direct
    file_var = file_env or f"{base_env}_FILE"
    path = os.getenv(file_var, "")
    if path:
        value = _file_secret(path, base_env)
        if value:
            return value
    cmd_var = cmd_env or f"{base_env}_CMD"
    cmdline = os.getenv(cmd_var, "")
    if cmdline:
        try:
            argv = shlex.split(cmdline)
        except ValueError as err:
            logger.warning("[SECRETS] %s command unparseable: %s", base_env, err)
            return None
        if argv:
            return _command_secret(argv, base_env)
    return None


__all__ = ["resolve_secret"]
