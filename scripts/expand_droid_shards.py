from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    target_dir = ROOT / "public_data" / "droid_100" / "1.0.0"
    target_dir.mkdir(parents=True, exist_ok=True)
    gsutil = r"C:\Users\honey\AppData\Local\Programs\Python\Python312\Scripts\gsutil.exe"
    for idx in range(start, end + 1):
        shard = f"r2d2_faceblur-train.tfrecord-{idx:05d}-of-00031"
        cmd = f"& '{gsutil}' cp gs://gresearch/robotics/droid_100/1.0.0/{shard} {target_dir}/"
        print({"downloading": shard})
        code = __import__("subprocess").run(["powershell", "-Command", cmd], check=False).returncode
        print({"shard": shard, "code": code})


if __name__ == "__main__":
    main()
