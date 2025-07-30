import os
from pathlib import Path
from types import SimpleNamespace

# project root is determined as being parent directory of src
project_root = Path(__file__).resolve().parents[1]

PATHS = SimpleNamespace(
    project =           project_root,
    data =              project_root / "data",
    raw_data =          project_root / "data" / "raw",
    rvl_cdip =          project_root / "data" / "raw" / "RVL-CDIP",
    rvl_cdip_images =   project_root / "data" / "raw" / "RVL-CDIP" / "images",
    iit_cdip_images =   project_root / "data" / "raw" / "IIT-CDIP" / "images",
    iit_cdip_xmls =     project_root / "data" / "raw" / "IIT-CDIP" / "xmls",
    raw_images =        project_root / "data" / "raw" / "RVL-CDIP" / "images", # same as rvl_cdip
    labels =            project_root / "data" / "raw" / "labels",
    metadata =          project_root / "data" / "metadata",
    samples =           project_root / "data" / "metadata" / "samples",
    converted_images =  project_root / "data" / "converted" / "images",
    processed_data =    project_root / "data" / "processed",
    models =            project_root / "models",
    pipelines =         project_root / "pipelines",
)


def create_all_dirs(paths: SimpleNamespace) -> None:
    for path in paths.__dict__.values():
        os.makedirs(path, exist_ok=True)
# Allows to avoid any error linked to missing path
create_all_dirs(PATHS)