"""Path utilities for the Grasp1 external IsaacLab project.

This module centralizes project-relative filesystem paths so task, asset, and
dataset code does not depend on the process working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_project_root() -> Path:
    """Return the Grasp1 project root.

    The optional ``GRASP1_ROOT`` environment variable can override the
    automatically derived root. This is useful when tools are launched from a
    different checkout or deployment layout.
    """
    override = os.environ.get("GRASP1_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_PROJECT_ROOT


PROJECT_ROOT = _resolve_project_root()
PACKAGE_ROOT = _PACKAGE_ROOT

SOURCE_DIR = PROJECT_ROOT / "source"
ASSETS_DIR = PROJECT_ROOT / "assets"
ROBOTS_ASSETS_DIR = ASSETS_DIR / "robots"
OBJECTS_ASSETS_DIR = ASSETS_DIR / "objects"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

PACKAGE_DATA_DIR = PACKAGE_ROOT / "data"
TASKS_DIR = PACKAGE_ROOT / "tasks"


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a path relative to the Grasp1 project root."""
    return PROJECT_ROOT.joinpath(*parts)


def robot_asset_dir(robot_name: str) -> Path:
    """Return the asset directory for a robot."""
    return ROBOTS_ASSETS_DIR / robot_name


def object_dataset_dir(dataset_name: str) -> Path:
    """Return the root directory for an object dataset."""
    return OBJECTS_ASSETS_DIR / dataset_name


def object_asset_dir(dataset_name: str, object_name: str) -> Path:
    """Return the directory for one object inside a dataset."""
    return object_dataset_dir(dataset_name) / object_name


def require_path(path: str | os.PathLike[str], *, kind: str = "any") -> Path:
    """Resolve and validate that a filesystem path exists."""
    if kind not in {"any", "file", "dir"}:
        raise ValueError(f"Unsupported path kind: {kind!r}")

    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    if kind == "file" and not resolved.is_file():
        raise FileNotFoundError(f"Expected a file: {resolved}")

    if kind == "dir" and not resolved.is_dir():
        raise FileNotFoundError(f"Expected a directory: {resolved}")

    return resolved


__all__ = [
    "PROJECT_ROOT",
    "PACKAGE_ROOT",
    "SOURCE_DIR",
    "ASSETS_DIR",
    "ROBOTS_ASSETS_DIR",
    "OBJECTS_ASSETS_DIR",
    "SCRIPTS_DIR",
    "LOGS_DIR",
    "PACKAGE_DATA_DIR",
    "TASKS_DIR",
    "project_path",
    "robot_asset_dir",
    "object_dataset_dir",
    "object_asset_dir",
    "require_path",
]
