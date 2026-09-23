"""Load scripts/helpers/collect_data.py as a module without executing its __main__."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_collector():
    if "aq_collect" in sys.modules:
        return sys.modules["aq_collect"]
    spec = importlib.util.spec_from_file_location(
        "aq_collect", Path(__file__).resolve().parent / "collect_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aq_collect"] = mod
    spec.loader.exec_module(mod)
    return mod
