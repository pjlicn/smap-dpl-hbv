#!/usr/bin/env python3
"""Generate payload manifest and SHA-256 checksums without self-reference."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "MANIFEST.txt"
    checksums_path = root / "checksums.sha256"

    payload = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path not in {manifest_path, checksums_path}
    )
    lines = [
        "# SMAP_dPL_WRR payload manifest",
        "# MANIFEST.txt and checksums.sha256 are control files and are not listed as payload.",
        "size_bytes\tpath",
    ]
    lines.extend(f"{path.stat().st_size}\t{path.relative_to(root).as_posix()}" for path in payload)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    checksum_targets = sorted(path for path in root.rglob("*") if path.is_file() and path != checksums_path)
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in checksum_targets
    ]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({len(payload)} payload files)")
    print(f"wrote {checksums_path} ({len(checksum_targets)} checksums)")


if __name__ == "__main__":
    main()
