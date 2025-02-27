import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add both src and project root to Python path
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.append(src_path)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Set the PYTHONPATH environment variable
os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"