from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.assets import synthesize_physics_assets
from s2p2e.data.corpus import build_object_corpus
from s2p2e.trainers.reporting import export_failure_analysis, export_paper_tables


def test_assets_and_reports(tmp_path: Path) -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    assets = synthesize_physics_assets(records[:2], seed=42)
    assert len(assets) == 2

    summary = {
        "joint": {"val": {"loss": 0.1}},
        "ablations": {
            "full": {"loss": 0.1},
            "w_o_rag": {"loss": 0.4},
            "w_o_ppe": {"loss": 0.3},
            "w_o_residual": {"loss": 0.2},
        },
    }
    export_paper_tables(tmp_path / "tables.json", summary)
    export_failure_analysis(tmp_path / "failures.json", summary)
    assert (tmp_path / "tables.json").exists()
    assert (tmp_path / "failures.json").exists()
