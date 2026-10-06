import hashlib
import json
from pathlib import Path

from app.config import CACHE_DIR


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def read_json(group: str, key: str):
    path = CACHE_DIR / group / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_json(group: str, key: str, value) -> None:
    folder = CACHE_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, default=str))
    temporary.replace(path)
