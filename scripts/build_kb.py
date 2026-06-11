from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.kb import KnowledgeBaseBuilder
from s2p2e.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_small.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data_cfg = cfg["data"]
    shapenet_roots = data_cfg.get("shapenet_roots", data_cfg.get("shapenet_root"))
    if isinstance(shapenet_roots, list):
        shapenet_roots = [ROOT / path for path in shapenet_roots]
    else:
        shapenet_roots = ROOT / shapenet_roots
    records = build_object_corpus(
        shapenet_roots,
        ROOT / data_cfg["objaverse_root"],
        data_cfg["shapenet_limit_per_category"],
        data_cfg["objaverse_limit"],
    )
    builder = KnowledgeBaseBuilder(records, data_cfg["tasks"], data_cfg["num_points"])
    entries = builder.build()
    builder.export_json(ROOT / cfg["artifacts_dir"] / cfg["run_name"] / "kb", entries)
    print({"kb_entries": len(entries)})


if __name__ == "__main__":
    main()
