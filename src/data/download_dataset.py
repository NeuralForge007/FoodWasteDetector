"""
Roboflow Dataset Acquisition Script for FOODWISE AI
Downloads the PROJ3 - Food Waste Detection dataset (version 6) from Roboflow Universe
in an Ultralytics-compatible instance segmentation format.
"""

import os
import sys
from pathlib import Path
import json
import yaml
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

def get_roboflow_dataset(
    api_key: str = None,
    workspace: str = "project-eyfif",
    project_id: str = "proj3-food-waste-detection",
    version: int = 6,
    export_format: str = "yolov8", # YOLOv8 / YOLOv11 segmentation format
    target_dir: Path = None
):
    from roboflow import Roboflow

    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key or api_key.strip() == "your_roboflow_api_key_here":
        raise ValueError(
            "Missing ROBOFLOW_API_KEY! Please set the ROBOFLOW_API_KEY environment variable, "
            "add it to your .env file, or pass --api-key <YOUR_KEY>."
        )

    if target_dir is None:
        target_dir = PROJECT_ROOT / "data" / "raw" / "roboflow"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("   FOODWISE AI: ACQUIRING ROBOFLOW DATASET")
    print(f"   Workspace:  {workspace}")
    print(f"   Project:    {project_id}")
    print(f"   Version:    {version}")
    print(f"   Format:     {export_format}")
    print(f"   Target Dir: {target_dir}")
    print("=" * 70)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    dataset_version = project.version(version)

    print(f"Starting download for {project_id} version {version} ({export_format})...")
    # Change working directory temporarily to target_dir for roboflow export
    orig_cwd = os.getcwd()
    os.chdir(target_dir)
    try:
        dataset = dataset_version.download(export_format)
        dataset_location = Path(dataset.location).resolve()
        print(f"Successfully downloaded to: {dataset_location}")
    finally:
        os.chdir(orig_cwd)

    # Process into data/processed/
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find the data.yaml
    yaml_files = list(dataset_location.glob("*.yaml")) + list(dataset_location.glob("*.yml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML dataset config found in {dataset_location}")
    
    src_yaml = yaml_files[0]
    dst_yaml = processed_dir / "dataset.yaml"

    # Read and update yaml paths to absolute or relative to data/processed
    with open(src_yaml, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # Set root path in yaml
    yaml_data["path"] = str(dataset_location).replace("\\", "/")
    
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated processed dataset configuration at: {dst_yaml}")

    # Log metadata
    metadata = {
        "dataset_source": "Roboflow Universe",
        "workspace": workspace,
        "project_id": project_id,
        "version": version,
        "download_timestamp": datetime.now().isoformat(),
        "export_format": export_format,
        "location": str(dataset_location),
        "classes": yaml_data.get("names", []),
        "num_classes": len(yaml_data.get("names", [])),
        "license": "CC BY 4.0 / Open Source Research Attribution"
    }

    meta_path = PROJECT_ROOT / "data" / "raw" / "roboflow" / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved dataset acquisition metadata to: {meta_path}")
    return dataset_location, dst_yaml

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Roboflow Dataset for FoodWise AI")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API Key")
    parser.add_argument("--version", type=int, default=6, help="Dataset version (default: 6)")
    parser.add_argument("--format", type=str, default="yolov8", help="Export format")
    args = parser.parse_args()

    get_roboflow_dataset(api_key=args.api_key, version=args.version, export_format=args.format)

