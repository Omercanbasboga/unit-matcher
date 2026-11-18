"""Where the reference data lives.

The engine only cares about the file names and shapes in data/
(canonical_units.json, synonyms.json, abbreviations.json, kadro_kurallari.json),
not where they came from. Point UNIT_MATCHER_DATA_DIR at another folder to run
against a different organisation's data, or pass data_dir= directly.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def get_data_dir() -> Path:
    env_value = os.environ.get("UNIT_MATCHER_DATA_DIR")
    if env_value:
        return Path(env_value)
    return DEFAULT_DATA_DIR
