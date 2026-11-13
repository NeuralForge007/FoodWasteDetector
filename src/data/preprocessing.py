"""
FOODWISE AI: Data Preprocessing Pipeline
Prepares and validates dataset directories, config files, and image scaling for YOLOv11.
"""

from pathlib import Path
import yaml
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def setup_processed_dataset(
    raw_dir: Path = None,
    processed_dir: Path = None
) -> Path:
    """
    Validates the dataset directory and generates a clean dataset.yaml in data/processed/.
    """
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "roboflow" / "PROJ3---Food-waste-detection-6"
    if processed_dir is None:
        processed_dir = PROJECT_ROOT / "data" / "processed"

    raw_dir = Path(raw_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    src_yaml = raw_dir / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml in {raw_dir}")

    with open(src_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set absolute paths or relative paths pointing to train/valid/test
    cfg["path"] = str(raw_dir).replace("\\", "/")
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"

    dst_yaml = processed_dir / "dataset.yaml"
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Generated validated dataset configuration at: {dst_yaml}")
    return dst_yaml

if __name__ == "__main__":
    setup_processed_dataset()

