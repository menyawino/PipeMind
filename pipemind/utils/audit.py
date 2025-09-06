from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_invocation_log(out_dir: str, record: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"invocation_{ts}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return path
