import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rag_dataset_v2_schema_files_exist():
    dataset_dir = ROOT / "public_data" / "s2p2e_rag_dataset"

    assert (dataset_dir / "manifest.json").exists()
    assert (dataset_dir / "SCHEMA.md").exists()
    assert (ROOT / "paper" / "source_data" / "rag_dataset_schema_summary.csv").exists()


def test_rag_dataset_v2_entry_has_enriched_fields():
    dataset_dir = ROOT / "public_data" / "s2p2e_rag_dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    first_entry = json.loads((dataset_dir / "entries.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert manifest["name"] == "S2P2E-RAG"
    assert manifest["num_entries"] > 0
    assert manifest["fields"] == [
        "entry_id",
        "schema_version",
        "object",
        "instruction",
        "geometric",
        "operational",
        "geometry_summary",
        "operational_summary",
        "physics_prior",
        "quality_flags",
        "provenance",
    ]
    assert first_entry["schema_version"] == "s2p2e-rag-v2"
    assert first_entry["instruction"]["phase_sequence"]
    assert "preferred_grasp_region" in first_entry["object"]
    assert "aspect_ratio" in first_entry["geometry_summary"]
    assert "contact_risk" in first_entry["operational_summary"]
    assert "mass_proxy_kg" in first_entry["physics_prior"]
    assert first_entry["quality_flags"]["finite_geometry"] is True
