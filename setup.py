#!/usr/bin/env python3
"""Setup script that ensures Ultralytics YOLO26 + Knowledge Distillation patches are synced on install."""

from pathlib import Path
from setuptools import setup

try:
    import sync_upstream

    sync_upstream.sync_ultralytics(force=False)
except Exception as e:
    print(f"Warning during pre-install sync: {e}")

setup()

