"""JSON-based cache store for analyses and training runs."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_json(path: Path) -> dict | None:
    """Read a JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read {path}: {e}")
        return None


def write_json(path: Path, data: dict) -> None:
    """Write data to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {path}")


def append_jsonl(path: Path, data: dict) -> None:
    """Append a JSON line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all lines from a JSONL file."""
    if not path.exists():
        return []
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def list_dirs(parent: Path) -> list[Path]:
    """List all subdirectories in a parent directory."""
    if not parent.exists():
        return []
    return [d for d in parent.iterdir() if d.is_dir()]
