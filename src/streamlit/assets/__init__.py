import os
import sys
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

# project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if not project_root in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(project_root))

import src

PATHS = SimpleNamespace(
    **src.PATHS.__dict__,
    streamlit =         project_root / "src" / "streamlit",
    streamlit_images =  project_root / "src" / "streamlit" / "assets" / "images",
    stylesheet =        project_root / "src" / "streamlit" / "assets" / "style.css",
)


try:
    df = pd.read_parquet(
        os.path.join(PATHS.metadata, "df_labels_mapping.parquet"))
    LABELS = dict(zip(df.index, df.values[:,0]))
except:
    print("unable to load CLASSES. Imputing default value")
    LABELS = {
        0: 'letter',
        1: 'form',
        2: 'email',
        3: 'handwritten',
        4: 'advertisement',
        5: 'scientific report',
        6: 'scientific publication',
        7: 'specification',
        8: 'file folder',
        9: 'news article',
        10: 'budget',
        11: 'invoice',
        12: 'presentation',
        13: 'questionnaire',
        14: 'resume',
        15: 'memo'
        }

