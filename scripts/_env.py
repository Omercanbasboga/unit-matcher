"""Tiny .env loader shared by the scripts (no python-dotenv needed)."""
import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding what's
    already set. Inline "value  # comment" is trimmed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        for sep in (" #", "\t#"):
            if sep in value:
                value = value.split(sep, 1)[0].rstrip()
                break
        if key and key not in os.environ:
            os.environ[key] = value
