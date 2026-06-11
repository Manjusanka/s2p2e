from __future__ import annotations

from copy import deepcopy


def build_curriculum_phases(config: dict) -> list[dict]:
    base = deepcopy(config)
    data_tasks = base["data"]["tasks"]
    phases = []

    phase1 = deepcopy(base)
    phase1["phase_name"] = "phase_1_pick_place"
    phase1["data"]["tasks"] = [task for task in data_tasks if task in {"pick", "place"}]
    phase1["data"]["objaverse_limit"] = min(base["data"].get("objaverse_limit", 0) or 48, 24)
    phase1["training"]["epochs_joint"] = max(1, base["training"].get("epochs_joint", 1))
    phases.append(phase1)

    phase2 = deepcopy(base)
    phase2["phase_name"] = "phase_2_alignment"
    phase2["data"]["tasks"] = [task for task in data_tasks if task in {"pick", "place", "insert"}]
    phase2["data"]["objaverse_limit"] = min(base["data"].get("objaverse_limit", 0) or 48, 36)
    phase2["training"]["epochs_joint"] = max(1, base["training"].get("epochs_joint", 1))
    phases.append(phase2)

    phase3 = deepcopy(base)
    phase3["phase_name"] = "phase_3_full_tasks"
    phase3["data"]["tasks"] = data_tasks
    phases.append(phase3)

    return phases
