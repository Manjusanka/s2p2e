from __future__ import annotations

from pathlib import Path
import gzip
import json
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.utils.io import ensure_dir, save_json


BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


def current_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    target_additional_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    target_additional_count = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    obj_dir = ensure_dir(ROOT / "objaverse" / "models")
    object_paths = json.load(gzip.open(ROOT / "objaverse" / "object-paths.json.gz", "rt", encoding="utf-8"))
    existing = {p.stem for p in obj_dir.glob("*.glb")}
    initial_size = current_size_bytes(obj_dir)
    budget_bytes = int(target_additional_gb * 1024 * 1024 * 1024)

    downloaded = 0
    downloaded_bytes = 0
    failed: list[str] = []
    manifest: list[dict[str, object]] = []

    session = requests.Session()
    for uid, rel_path in object_paths.items():
        if uid in existing:
            continue
        if downloaded >= target_additional_count or downloaded_bytes >= budget_bytes:
            break
        url = BASE_URL + rel_path + "?download=true"
        out_path = obj_dir / f"{uid}.glb"
        tmp_path = out_path.with_suffix(".glb.part")
        try:
            with session.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                size = int(response.headers.get("content-length", "0"))
                if size and downloaded_bytes + size > budget_bytes and downloaded > 0:
                    break
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_size = tmp_path.stat().st_size
            tmp_path.replace(out_path)
            downloaded += 1
            downloaded_bytes += tmp_size
            manifest.append({"uid": uid, "bytes": tmp_size, "path": str(out_path)})
            if downloaded % 50 == 0:
                save_json(
                    ROOT / "objaverse" / "download_manifest.json",
                    {
                        "downloaded": downloaded,
                        "downloaded_gb": round(downloaded_bytes / 1024 / 1024 / 1024, 3),
                        "failed": failed,
                        "manifest": manifest,
                    },
                )
        except Exception:
            failed.append(uid)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(0.2)

    save_json(
        ROOT / "objaverse" / "download_manifest.json",
        {
            "initial_gb": round(initial_size / 1024 / 1024 / 1024, 3),
            "downloaded": downloaded,
            "downloaded_gb": round(downloaded_bytes / 1024 / 1024 / 1024, 3),
            "failed": failed,
            "manifest": manifest,
        },
    )
    print(
        {
            "downloaded": downloaded,
            "downloaded_gb": round(downloaded_bytes / 1024 / 1024 / 1024, 3),
            "failed": len(failed),
        }
    )


if __name__ == "__main__":
    main()
