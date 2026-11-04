"""Download the AIW response dataset at a pinned commit and verify its checksum.

The dataset comes from the "Alice in Wonderland" (AIW) repository by LAION-AI
(Apache License 2.0). We pin an exact commit so every run uses identical data.

Usage:
    python scripts/fetch_aiw_data.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

AIW_COMMIT = "4d5d079b773ae6ab7d2eb852971ea3fda0ea1ce8"
FILE_IN_REPO = "collected_responses/AIW_AIW_plus.json"
URL = f"https://raw.githubusercontent.com/LAION-AI/AIW/{AIW_COMMIT}/{FILE_IN_REPO}"
EXPECTED_SHA256 = "e84ce307d14657cb4650ae6f4cbfab995e7745e5ce5721b0cd6252080f3c2cc1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "data" / "raw" / "AIW_AIW_plus.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.exists() and sha256_of(TARGET) == EXPECTED_SHA256:
        print(f"Data already present and verified: {TARGET}")
        return 0

    print(f"Downloading {URL}")
    print("(about 45 MB, this can take a minute)")
    tmp = TARGET.with_suffix(".part")
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except OSError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Alternative: clone https://github.com/LAION-AI/AIW, check out commit "
            f"{AIW_COMMIT}, and copy {FILE_IN_REPO} to {TARGET}",
            file=sys.stderr,
        )
        return 1

    actual = sha256_of(tmp)
    if actual != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        print(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual}", file=sys.stderr)
        return 1

    tmp.replace(TARGET)
    print(f"Saved and verified: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
