from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.assets import export_physics_assets, synthesize_physics_assets
from s2p2e.data.corpus import build_object_corpus
from s2p2e.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "experiment_joint.yaml")
    data_cfg = cfg["data"]
    records = build_object_corpus(
        ROOT / data_cfg["shapenet_root"],
        ROOT / data_cfg["objaverse_root"],
        data_cfg["shapenet_limit_per_category"],
        data_cfg["objaverse_limit"],
    )
    assets = synthesize_physics_assets(records, seed=cfg["seed"])
    export_physics_assets(ROOT / cfg["artifacts_dir"] / cfg["run_name"] / "assets", assets)
    print({"assets": len(assets)})


if __name__ == "__main__":
    main()
