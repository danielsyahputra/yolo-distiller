# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.4.19"

import os
from pathlib import Path as _Path

# Auto-sync latest official Ultralytics (YOLO26 / Segment26 / Proto26) + KD patches on first run when online
_sync_marker = _Path(__file__).resolve().parent / ".synced_v26_r2"
_sync_script = _Path(__file__).resolve().parent.parent / "sync_upstream.py"
if not _sync_marker.exists() and _sync_script.exists():
    try:
        import importlib.util as _ilu

        _spec = _ilu.spec_from_file_location("sync_upstream", _sync_script)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.sync_ultralytics()
    except Exception:
        pass

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
