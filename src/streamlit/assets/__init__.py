from pathlib import Path
from types import SimpleNamespace

# project root path
root = Path(__file__).resolve().parent.parent.parent.parent

PATHS = SimpleNamespace(
    streamlit =         root / "src" / "streamlit",
    streamlit_images =  root / "src" / "streamlit" / "assets" / "images",
    stylesheet =        root / "src" / "streamlit" / "assets" / "style.css",
)

