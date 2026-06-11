from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ObjectRecord:
    object_id: str
    category: str
    dataset: str
    path: str
    semantic_text: str
    affordance_text: str


def _iter_roots(root_or_roots: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(root_or_roots, (str, Path)):
        return [Path(root_or_roots)]
    return [Path(x) for x in root_or_roots]


def build_object_corpus(
    shapenet_root: str | Path | Iterable[str | Path],
    objaverse_root: str | Path,
    shapenet_limit_per_category: int = 0,
    objaverse_limit: int = 0,
) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []
    seen_paths: set[str] = set()
    for root in _iter_roots(shapenet_root):
        if not root.exists():
            continue
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(category_dir.glob("*.txt"))
            if shapenet_limit_per_category > 0:
                files = files[:shapenet_limit_per_category]
            for path in files:
                norm = str(path.resolve())
                if norm in seen_paths:
                    continue
                seen_paths.add(norm)
                category = category_dir.name
                records.append(
                    ObjectRecord(
                        object_id=path.stem,
                        category=category,
                        dataset="shapenet",
                        path=str(path),
                        semantic_text=f"{category} object for tabletop manipulation",
                        affordance_text=_affordance_text(category),
                    )
                )
    objaverse_root = Path(objaverse_root)
    objaverse_files = sorted(objaverse_root.glob("*.glb"))
    if objaverse_limit > 0:
        objaverse_files = objaverse_files[:objaverse_limit]
    for path in objaverse_files:
        records.append(
            ObjectRecord(
                object_id=path.stem,
                category="objaverse_asset",
                dataset="objaverse",
                path=str(path),
                semantic_text="generic objaverse asset for manipulation pretraining",
                affordance_text="graspable rigid object with unknown semantics",
            )
        )
    if not records:
        raise RuntimeError("No dataset files found. Check configured roots.")
    return records


def _affordance_text(category: str) -> str:
    tags = {
        "chair": "support, lift, align",
        "table": "support, place, align",
        "bottle": "grasp, pour, upright",
        "cup": "grasp, pour, place",
        "lamp": "grasp, rotate, place",
        "sofa": "support, align, place",
    }
    return tags.get(category, "grasp, place, align")
