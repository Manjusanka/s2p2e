from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


TASK_ONTOLOGY = {
    "pick": {
        "family": "pick_place",
        "phase_sequence": ["approach", "grasp", "lift"],
        "requires_upright": False,
        "contact_risk": "medium",
    },
    "place": {
        "family": "pick_place",
        "phase_sequence": ["transport", "align", "release"],
        "requires_upright": True,
        "contact_risk": "low",
    },
    "insert": {
        "family": "precision_alignment",
        "phase_sequence": ["approach", "align", "contact", "insert"],
        "requires_upright": True,
        "contact_risk": "high",
    },
    "pour": {
        "family": "fluid_transfer",
        "phase_sequence": ["grasp", "lift", "tilt", "recover"],
        "requires_upright": True,
        "contact_risk": "medium",
    },
    "tool_use": {
        "family": "tool_use",
        "phase_sequence": ["grasp", "orient", "contact", "actuate"],
        "requires_upright": False,
        "contact_risk": "high",
    },
}


CATEGORY_PRIORS = {
    "bottle": {"material_prior": "rigid_container", "grasp_region": "neck_or_body", "symmetry": "axial"},
    "cup": {"material_prior": "rigid_container", "grasp_region": "rim_or_handle", "symmetry": "axial"},
    "table": {"material_prior": "rigid_support", "grasp_region": "edge_or_leg", "symmetry": "planar"},
    "chair": {"material_prior": "rigid_support", "grasp_region": "back_or_leg", "symmetry": "partial"},
    "sofa": {"material_prior": "semi_rigid_support", "grasp_region": "side_or_base", "symmetry": "partial"},
    "lamp": {"material_prior": "fragile_rigid", "grasp_region": "stem_or_base", "symmetry": "axial"},
    "vase": {"material_prior": "fragile_container", "grasp_region": "body", "symmetry": "axial"},
    "plant": {"material_prior": "deformable_visual", "grasp_region": "pot", "symmetry": "axial"},
    "laptop": {"material_prior": "articulated_rigid", "grasp_region": "base_edge", "symmetry": "bilateral"},
    "keyboard": {"material_prior": "flat_rigid", "grasp_region": "long_edge", "symmetry": "bilateral"},
    "objaverse_asset": {"material_prior": "unknown_rigid", "grasp_region": "largest_stable_region", "symmetry": "unknown"},
}


def task_id(task: str) -> int:
    order = ["pick", "place", "insert", "pour", "tool_use"]
    return order.index(task)


def enrich_entry(entry: dict, index: int) -> dict:
    geometric = entry["geometric"]
    operational = entry["operational"]
    center = geometric[0:3]
    extents = geometric[3:6]
    curvature = geometric[6:9]
    approachability = geometric[9:12]
    pose = operational[0:8]
    grasp_quality = operational[8:12]
    volume_proxy = extents[0] * extents[1] * extents[2]
    longest = max(extents)
    shortest = max(min(extents), 1e-6)
    aspect_ratio = longest / shortest
    category_prior = CATEGORY_PRIORS.get(
        entry["category"],
        {"material_prior": "unknown_rigid", "grasp_region": "largest_stable_region", "symmetry": "unknown"},
    )
    ontology = TASK_ONTOLOGY[entry["task"]]
    quality_flags = {
        "finite_geometry": all(abs(float(x)) < 1e6 for x in geometric),
        "finite_operational": all(abs(float(x)) < 1e6 for x in operational),
        "nondegenerate_extent": shortest > 1e-5,
        "pose_quaternion_present": len(pose[3:7]) == 4,
        "task_id_matches": int(round(grasp_quality[3])) == task_id(entry["task"]),
    }
    enriched = {
        **entry,
        "entry_id": f"s2p2e-rag-{index:07d}",
        "schema_version": "s2p2e-rag-v2",
        "instruction": {
            "canonical": entry["semantic_text"],
            "task": entry["task"],
            "task_family": ontology["family"],
            "phase_sequence": ontology["phase_sequence"],
        },
        "object": {
            "object_id": entry["object_id"],
            "category": entry["category"],
            "source_dataset": entry["dataset"],
            "material_prior": category_prior["material_prior"],
            "symmetry_prior": category_prior["symmetry"],
            "preferred_grasp_region": category_prior["grasp_region"],
        },
        "geometry_summary": {
            "center_xyz": center,
            "extent_xyz": extents,
            "curvature_l123": curvature,
            "approachability": approachability,
            "volume_proxy": volume_proxy,
            "aspect_ratio": aspect_ratio,
            "slenderness": longest / (sum(extents) / 3.0 + 1e-6),
        },
        "operational_summary": {
            "target_pose_xyzwg": pose,
            "grasp_score_proxy": grasp_quality[0],
            "principal_curvature_score": grasp_quality[1],
            "size_score": grasp_quality[2],
            "task_id": int(round(grasp_quality[3])),
            "requires_upright": ontology["requires_upright"],
            "contact_risk": ontology["contact_risk"],
        },
        "physics_prior": {
            "mass_proxy_kg": max(0.2, min(4.0, volume_proxy * 8.0)),
            "rigidity_prior": category_prior["material_prior"],
            "torque_sensitivity": "high" if ontology["contact_risk"] == "high" or volume_proxy > 1.5 else "medium",
        },
        "quality_flags": quality_flags,
        "provenance": {
            "generated_by": "scripts/export_rag_dataset_release.py",
            "source_kb": "artifacts/s2p2e_rag_full/kb/kb_entries.json",
            "release_date": "2026-06-11",
        },
    }
    return enriched


def main() -> None:
    rag_root = ROOT / "artifacts" / "s2p2e_rag_full"
    physics_root = ROOT / "artifacts" / "s2p2e_physics_full"
    policy_root = ROOT / "artifacts" / "actor_critic_droid_full"
    out_dir = ROOT / "public_data" / "s2p2e_rag_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads((rag_root / "kb" / "kb_entries.json").read_text(encoding="utf-8"))["entries"]
    rag_summary = json.loads((rag_root / "metrics" / "rag_layer1_summary.json").read_text(encoding="utf-8"))
    physics_summary = json.loads((physics_root / "metrics" / "summary.json").read_text(encoding="utf-8"))
    policy_summary = json.loads((policy_root / "actor_critic_droid_history.json").read_text(encoding="utf-8"))

    enriched_entries = [enrich_entry(entry, idx) for idx, entry in enumerate(entries)]

    with (out_dir / "entries.jsonl").open("w", encoding="utf-8") as handle:
        for entry in enriched_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    train_end = int(len(entries) * 0.8)
    val_end = train_end + int(len(entries) * 0.1)
    splits = {
        "train": enriched_entries[:train_end],
        "val": enriched_entries[train_end:val_end],
        "test": enriched_entries[val_end:],
    }
    for split, split_entries in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for entry in split_entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    category_counts = Counter(entry["category"] for entry in entries)
    task_counts = Counter(entry["task"] for entry in entries)
    dataset_counts = Counter(entry["dataset"] for entry in entries)
    manifest = {
        "name": "S2P2E-RAG",
        "version": "2026-06-11",
        "num_entries": len(entries),
        "num_objects": rag_summary["records"],
        "num_categories": len(category_counts),
        "tasks": sorted(task_counts),
        "datasets": dict(sorted(dataset_counts.items())),
        "splits": {name: len(value) for name, value in splits.items()},
        "fields": [
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
        ],
        "source_artifacts": {
            "kb_entries": "artifacts/s2p2e_rag_full/kb/kb_entries.json",
            "rag_checkpoint": "artifacts/s2p2e_rag_full/checkpoints/layer1_best.pt",
            "full_stack_checkpoint": "artifacts/s2p2e_physics_full/checkpoints/joint_best.pt",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "category_counts.json", dict(sorted(category_counts.items())))
    write_json(out_dir / "task_counts.json", dict(sorted(task_counts.items())))

    card = f"""# S2P2E-RAG Dataset

S2P2E-RAG is the retrieval knowledge-base dataset generated for the S2P2E manipulation stack. It pairs object-level 3D assets with five tabletop manipulation task families and stores three retrieval tiers for each entry: semantic text, geometric descriptors, and operational pose/action descriptors.

## Scope

- Objects: {rag_summary['records']}
- Retrieval entries: {len(entries)}
- Categories: {len(category_counts)}
- Tasks: {', '.join(sorted(task_counts))}
- Splits: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}

## Intended Use

This dataset is intended for retrieval-augmented pose proposal, manipulation knowledge-base ablations, and reproducibility of the Layer I S2P2E experiments. It is not a full replacement for upstream Objaverse, ShapeNet/ModelNet, or DROID assets.

## Training Snapshot

- RAG best epoch: {rag_summary['best']['epoch']}
- RAG validation pose MAE: {rag_summary['best']['val']['pose_mae']:.6f}
- Full-stack validation loss: {physics_summary['joint']['val']['loss']:.6f}
- DROID policy samples: {policy_summary['num_samples']}

## Files

- `entries.jsonl`: all retrieval entries.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: deterministic split files.
- `manifest.json`: dataset metadata.
- `category_counts.json`, `task_counts.json`: dataset statistics.
- `SCHEMA.md`: field-level description for the enriched v2 entry format.
"""
    (out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    schema = """# S2P2E-RAG Schema

Each JSONL row represents one `(object, task)` retrieval entry.

## Core Fields

- `entry_id`: stable release identifier.
- `schema_version`: current entry schema.
- `object`: object identity, source dataset, category, and manipulation priors.
- `instruction`: canonical task text plus task family and phase sequence.
- `geometric`: 12-dimensional vector used by the model.
- `operational`: 12-dimensional vector containing target pose and grasp/action descriptors.

## Enriched Fields

- `geometry_summary`: interpretable center, extents, curvature, approachability, volume proxy, and aspect ratio.
- `operational_summary`: target pose, grasp score proxy, task id, upright requirement, and contact-risk class.
- `physics_prior`: mass proxy, rigidity/material prior, and torque-sensitivity label.
- `quality_flags`: finite-value, extent, quaternion, and task-consistency checks.
- `provenance`: script, source KB artifact, and release date.
"""
    (out_dir / "SCHEMA.md").write_text(schema, encoding="utf-8")

    csv_path = ROOT / "paper" / "source_data" / "training_expansion_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "training_data", "best_epoch", "primary_validation_metric", "value"])
        writer.writerow(["Layer I / S2P2E-RAG", f"{rag_summary['records']} objects; {len(entries)} KB entries", rag_summary["best"]["epoch"], "pose_mae", rag_summary["best"]["val"]["pose_mae"]])
        writer.writerow(["Layer II / PPE", "4096 synthetic inverse-dynamics trajectories", physics_summary["layer2"]["epoch"], "rmse", physics_summary["layer2"]["val"]["rmse"]])
        writer.writerow(["Layer III / residual control", "4096 residual-control trajectories", physics_summary["layer3"]["epoch"], "residual_loss", physics_summary["layer3"]["val"]["residual_loss"]])
        writer.writerow(["Joint S2P2E", f"{physics_summary['num_records']} 3D objects x 5 tasks", physics_summary["joint"]["epoch"], "joint_val_loss", physics_summary["joint"]["val"]["loss"]])
        writer.writerow(["Actor-Critic DROID bridge", f"{policy_summary['num_samples']} DROID transitions", policy_summary["history"][-1]["epoch"], "critic_loss", policy_summary["history"][-1]["critic_loss"]])

    schema_csv = ROOT / "paper" / "source_data" / "rag_dataset_schema_summary.csv"
    with schema_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field_group", "field", "description", "source"])
        writer.writerow(["identity", "entry_id", "Stable row identifier for release and citation.", "export script"])
        writer.writerow(["instruction", "task_family", "Task ontology grouping used for retrieval and ablation.", "S2P2E task vocabulary"])
        writer.writerow(["instruction", "phase_sequence", "Ordered manipulation phases for the requested task.", "S2P2E task vocabulary"])
        writer.writerow(["object", "material_prior", "Category-level material or rigidity prior.", "category prior table"])
        writer.writerow(["object", "preferred_grasp_region", "Category-level grasp-region prior.", "category prior table"])
        writer.writerow(["geometry_summary", "extent_xyz", "Object bounding-box dimensions used by pose proposal.", "computed descriptor"])
        writer.writerow(["geometry_summary", "curvature_l123", "Principal-shape descriptor from the local KB.", "computed descriptor"])
        writer.writerow(["geometry_summary", "aspect_ratio", "Scale-normalized elongation descriptor.", "computed descriptor"])
        writer.writerow(["operational_summary", "target_pose_xyzwg", "Pose and gripper target used by Layer I supervision.", "computed label"])
        writer.writerow(["operational_summary", "contact_risk", "Task-level contact-risk class for safety analysis.", "task ontology"])
        writer.writerow(["physics_prior", "mass_proxy_kg", "Object-scale proxy mass for downstream dynamics checks.", "computed prior"])
        writer.writerow(["quality_flags", "task_id_matches", "Consistency check between task label and operational vector.", "validation check"])

    print({"dataset_dir": str(out_dir), "entries": len(enriched_entries), "source_data": str(csv_path)})


if __name__ == "__main__":
    main()
