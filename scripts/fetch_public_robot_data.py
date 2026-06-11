from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.utils.io import ensure_dir, save_json


def main() -> None:
    public_root = ensure_dir(ROOT / "public_data")
    manifest = {
        "droid": {
            "official_doc": "https://droid-dataset.github.io/droid/the-droid-dataset",
            "sample_download": "gsutil -m cp -r gs://gresearch/robotics/droid_100 <target_dir>",
            "notes": "RLDS debug subset is 2GB; full RLDS is 1.7TB.",
        },
        "rh20t": {
            "official_doc": "https://rh20t.github.io/",
            "api_repo": "https://github.com/rh20t/rh20t_api",
            "notes": "Contains joint angles and joint torques; 320x180 resized RGB subsets start around 4.4GB-30.3GB per config.",
        },
        "roboset": {
            "official_doc": "https://www.tensorflow.org/datasets/catalog/robo_set",
            "notes": "179.42 GiB dataset with state, state_velocity, images, and action.",
        },
    }
    save_json(public_root / "dataset_manifest.json", manifest)
    print({"public_root": str(public_root), "datasets": list(manifest.keys())})


if __name__ == "__main__":
    main()
