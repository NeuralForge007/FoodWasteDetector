"""
FOODWISE AI: Streamlit Application Runner Script
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def launch_app():
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching FOODWISE AI Streamlit Application: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_app()

