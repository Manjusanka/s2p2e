from __future__ import annotations

from pathlib import Path
import sys
import tarfile

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.utils.io import ensure_dir


URL = "https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed/resolve/main/modelnet40_normal_resampled.tar.gz?download=true"


def download_file(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
    mode = "ab" if downloaded > 0 else "wb"
    with requests.get(url, stream=True, timeout=120, headers=headers) as response:
        if response.status_code not in (200, 206):
            response.raise_for_status()
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)


def main() -> None:
    out_dir = ensure_dir(ROOT / "shapenet")
    archive = out_dir / "modelnet40_normal_resampled.tar.gz"
    extract_dir = out_dir / "modelnet40_normal_resampled"
    if not archive.exists():
        try:
            downloaded = hf_hub_download(
                repo_id="Pointcept/modelnet40_normal_resampled-compressed",
                repo_type="dataset",
                filename="modelnet40_normal_resampled.tar.gz",
                local_dir=str(out_dir),
                local_dir_use_symlinks=False,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != archive.resolve():
                downloaded_path.replace(archive)
        except Exception:
            download_file(URL, archive)
    if not extract_dir.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out_dir)
    txt_count = len(list(extract_dir.rglob("*.txt")))
    print({"archive": str(archive), "extract_dir": str(extract_dir), "txt_count": txt_count})


if __name__ == "__main__":
    main()
