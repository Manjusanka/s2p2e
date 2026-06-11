from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.kb import KnowledgeBaseBuilder
from s2p2e.utils.config import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "experiment_small.yaml")
    data_cfg = cfg["data"]
    records = build_object_corpus(
        ROOT / data_cfg["shapenet_root"],
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
