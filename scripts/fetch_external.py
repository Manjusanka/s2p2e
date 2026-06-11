from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    external = ROOT / "external"
    external.mkdir(exist_ok=True)
    repos = {
        "sam2": "https://github.com/facebookresearch/sam2.git",
        "Depth-Anything-V2": "https://github.com/DepthAnything/Depth-Anything-V2.git",
        "PointNeXt": "https://github.com/guochengqian/PointNeXt.git",
    }
    print({"external_dir": str(external), "repos": repos})


if __name__ == "__main__":
    main()
