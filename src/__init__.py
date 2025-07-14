import os
from pathlib import Path
from types import SimpleNamespace

# project root is determined as being parent directory of src
root = Path(__file__).resolve().parents[1]

PATHS = SimpleNamespace(
    project = root,
    data = os.path.join(root, "data"),
    raw_data = os.path.join(root, "data", "raw"),
    raw_images = os.path.join(root, "data", "raw", "RVL-CDIP", "images"),
    converted_images = os.path.join(root, "data", "converted", "images"),
    metadata = os.path.join(root, "data", "metadata"),
    processed_data = os.path.join(root, "data", "processed"),
    models = os.path.join(root, "models"),
    pipelines = os.path.join(root, "pipelines"),
)