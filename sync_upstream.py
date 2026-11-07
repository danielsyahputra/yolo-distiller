#!/usr/bin/env python3
"""
Optional utility to sync the repository with the latest official Ultralytics YOLO release from GitHub
while preserving the cross-architecture Knowledge Distillation (CWD / MGD / SL2) implementation in:
- ultralytics/nn/distill_model.py
- ultralytics/engine/trainer.py
- ultralytics/cfg/default.yaml
"""

import io
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARBALL_URL = "https://codeload.github.com/ultralytics/ultralytics/tar.gz/refs/heads/main"


def _indent_block(block: str, indent: str) -> str:
    """Indents every non-empty line of a multi-line string by `indent`."""
    lines = block.strip("\n").splitlines()
    return "\n".join((indent + line if line.strip() else "") for line in lines)


def _inject_before(content: str, line_regex: str, snippet: str, count: int = 1) -> str:
    """Finds a line matching `^([ \\t]*)(line_regex)` and injects `snippet` (indented to match) before it."""
    pattern = re.compile(rf"(?m)^([ \t]*)({line_regex})")

    def repl(m: re.Match) -> str:
        indent, target = m.group(1), m.group(2)
        return f"{_indent_block(snippet, indent)}\n{indent}{target}"

    return pattern.sub(repl, content, count=count)


def patch_default_yaml(yaml_path: Path) -> None:
    content = yaml_path.read_text(encoding="utf-8")
    if "distillation_loss:" not in content:
        content = (
            content.rstrip()
            + "\n\n# Knowledge Distillation settings\n"
            + "teacher: # (str | nn.Module, optional) teacher model or checkpoint path for knowledge distillation\n"
            + "distillation_loss: cwd # (str) knowledge distillation loss method: 'cwd', 'mgd', or 'sl2'\n"
        )
        yaml_path.write_text(content, encoding="utf-8")


def patch_trainer_py(trainer_path: Path) -> None:
    content = trainer_path.read_text(encoding="utf-8")
    if "CWDLoss, DistillationModel" in content:
        return

    content = content.replace(
        "from ultralytics.nn.distill_model import DistillationModel",
        "from ultralytics.nn.distill_model import CWDLoss, DistillationModel, MGDAdapter, MGDLoss",
        1,
    )

    content = _inject_before(
        content,
        r"self\.args = get_cfg\(cfg, overrides\)",
        """\
self.teacher = None
if overrides:
    teacher_arg = overrides.pop("teacher", None)
    if teacher_arg not in (None, "None", "null"):
        if isinstance(teacher_arg, (str, Path)):
            overrides.setdefault("distill_model", str(teacher_arg))
        else:
            self.teacher = teacher_arg
            pt_path = (
                getattr(teacher_arg, "pt_path", None)
                or getattr(teacher_arg, "ckpt_path", None)
                or getattr(getattr(teacher_arg, "model", None), "pt_path", None)
            )
            if pt_path and not overrides.get("distill_model"):
                overrides["distill_model"] = str(pt_path)
    loss_arg = overrides.get("distillation_loss")
    if loss_arg in (None, "None", "null"):
        overrides["distillation_loss"] = "cwd\"""",
        count=1,
    )

    content = content.replace(
        "if self.args.distill_model is not None and self.args.compile:",
        "if (self.args.distill_model is not None or getattr(self, \"teacher\", None) is not None) and self.args.compile:",
        1,
    )

    content = content.replace(
        "        if isinstance(unwrap_model(self.model), DistillationModel):\n            freeze_layer_names.append(\"teacher_model.\")",
        """        if (
            isinstance(unwrap_model(self.model), DistillationModel)
            or self.args.distill_model is not None
            or getattr(self, "teacher", None) is not None
        ):
            freeze_layer_names.append("teacher_model.")""",
        1,
    )

    content = content.replace(
        "        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):\n            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)",
        """        teacher_source = self.teacher if getattr(self, "teacher", None) is not None else self.args.distill_model
        if teacher_source is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(
                student_model=self.model,
                teacher_model=teacher_source,
                loss_type=getattr(self.args, "distillation_loss", "cwd"),
            )""",
        1,
    )

    content = content.replace(
        "            teacher_model = weights.teacher_model if weights.teacher_model is not None else self.args.distill_model\n            model = DistillationModel(student_model=student_model, teacher_model=teacher_model)",
        """            teacher_model = (
                weights.teacher_model
                if weights.teacher_model is not None
                else (self.teacher if getattr(self, "teacher", None) is not None else self.args.distill_model)
            )
            loss_type = getattr(weights, "loss_type", getattr(self.args, "distillation_loss", "cwd"))
            model = DistillationModel(student_model=student_model, teacher_model=teacher_model, loss_type=loss_type)""",
        1,
    )

    content = content.replace(
        '                "distill_model",\n                "save_dir",',
        '                "distill_model",\n                "distillation_loss",\n                "save_dir",',
        1,
    )

    compile(content, str(trainer_path), "exec")
    trainer_path.write_text(content, encoding="utf-8")


def sync_ultralytics() -> bool:
    """Downloads latest official Ultralytics while preserving custom distill_model.py and applying trainer patches."""
    distill_model_backup = (ROOT / "ultralytics" / "nn" / "distill_model.py").read_text(encoding="utf-8")
    print("Syncing latest official Ultralytics from GitHub...")
    req = urllib.request.Request(TARBALL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(tmpdir)
        src_root = next(p for p in Path(tmpdir).iterdir() if p.is_dir())
        shutil.rmtree(ROOT / "ultralytics", ignore_errors=True)
        shutil.copytree(src_root / "ultralytics", ROOT / "ultralytics")
        if (src_root / "pyproject.toml").exists():
            shutil.copy2(src_root / "pyproject.toml", ROOT / "pyproject.toml")

    (ROOT / "ultralytics" / "nn" / "distill_model.py").write_text(distill_model_backup, encoding="utf-8")
    patch_default_yaml(ROOT / "ultralytics" / "cfg" / "default.yaml")
    patch_trainer_py(ROOT / "ultralytics" / "engine" / "trainer.py")
    print("Successfully synced latest Ultralytics codebase and applied Knowledge Distillation patches.")
    return True


if __name__ == "__main__":
    sync_ultralytics()
