"""Hardened handling of user-supplied output paths."""

from __future__ import annotations

from pathlib import Path


class UnsafePath(ValueError):
    """Raised when an output path is not safe to write to."""


def safe_output_path(raw: str | Path, cwd: Path | None = None) -> Path:
    """Normalise a user-supplied output path and bound it to the workspace.

    The CLI accepts output locations from its own user, but ``..`` segments
    must never silently escape the working directory: relative paths are
    resolved strictly inside it, absolute paths are allowed as-is (explicit
    intent), and paths that resolve into an existing directory are rejected.
    """
    base = (cwd or Path.cwd()).resolve()
    given = Path(raw).expanduser()
    resolved = given.resolve() if given.is_absolute() else (base / given).resolve()
    if not given.is_absolute() and resolved != base and base not in resolved.parents:
        raise UnsafePath(f"relative output path escapes the working directory: {raw!r}")
    if resolved.is_dir():
        raise UnsafePath(f"output path is an existing directory: {raw!r}")
    return resolved
