import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# config.py requires tokens.json at import time; tests only need the example tokens.
if not (ROOT / "tokens.json").exists():
    shutil.copyfile(ROOT / "tokens.example.json", ROOT / "tokens.json")
