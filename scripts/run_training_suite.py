from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print({"stage": name, "command": command}, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run_step("rag_layer1_full", [python, "scripts/train_rag_layer1.py", "--config", "configs/experiment_rag_full.yaml"])
    run_step("physics_stack_full", [python, "scripts/train_all.py", "--config", "configs/experiment_physics_full.yaml"])
    run_step(
        "actor_critic_droid",
        [
            python,
            "scripts/train_actor_critic_droid.py",
            "--config",
            "configs/public_robot_data.yaml",
            "--out-dir",
            "artifacts/actor_critic_droid_full",
        ],
    )


if __name__ == "__main__":
    main()
