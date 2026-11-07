#!/usr/bin/env python3
"""
Syncs the repository with the latest official Ultralytics YOLO release (including YOLO26 / Segment26 / Proto26)
and applies the Knowledge Distillation (CWD / MGD) enhancement to ultralytics/engine/trainer.py and default.yaml.

Can be executed directly:
    python3 sync_upstream.py
Or automatically invoked during `pip install -e .` via setup.py.
"""

import io
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARBALL_URL = "https://codeload.github.com/ultralytics/ultralytics/tar.gz/refs/heads/main"

KD_CLASSES_CODE = '''
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation` (https://arxiv.org/abs/2011.13256)."""

    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape, f"Shape mismatch at stage {idx}: student {s.shape} vs teacher {t.shape}"
            N, C, H, W = s.shape
            s_fp32 = s.float().view(-1, W * H) / self.tau
            t_fp32 = t.float().view(-1, W * H) / self.tau
            softmax_pred_T = F.softmax(t_fp32, dim=1)
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t_fp32) - softmax_pred_T * logsoftmax(s_fp32)
            ) * (self.tau ** 2)
            losses.append(cost / (C * N))
        return sum(losses)


class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation` (https://arxiv.org/abs/2205.01529)."""

    def __init__(self, student_channels, teacher_channels, alpha_mgd=0.00002, lambda_mgd=0.65, device=None):
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            ).to(device)
            for channel in teacher_channels
        ])

    def forward(self, y_s, y_t, layer=None):
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        return sum(losses)

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction="sum")
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W), device=device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(dtype=preds_S.dtype, device=device)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        return loss_mse(new_fea.float(), preds_T.float()) / N


class FeatureLoss(nn.Module):
    """Feature alignment and distillation loss wrapper supporting cross-scale student/teacher pairs."""

    def __init__(self, channels_s, channels_t, distiller="cwd", loss_weight=1.0, device=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.distiller = (distiller or "cwd").lower()[:3]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()

        for s_chan, t_chan in zip(channels_s, channels_t):
            self.align_module.append(
                nn.Sequential(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(t_chan, affine=False),
                ).to(device)
            )
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        if self.distiller == "mgd":
            self.feature_loss = MGDLoss(channels_s, channels_t, device=device)
        elif self.distiller == "cwd":
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError(f"Unsupported distillation_loss='{distiller}'. Expected 'cwd' or 'mgd'.")

    def forward(self, y_s, y_t):
        if len(y_s) != len(y_t):
            y_t = y_t[len(y_t) // 2:]
        tea_feats, stu_feats = [], []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            param_dtype = next(self.align_module[idx].parameters()).dtype
            s = self.align_module[idx](s.to(dtype=param_dtype))
            t = t.detach().to(dtype=param_dtype)
            if self.distiller != "cwd":
                t = self.norm[idx](t)
            stu_feats.append(s)
            tea_feats.append(t)
        return self.loss_weight * self.feature_loss(stu_feats, tea_feats)


class DistillationLoss:
    """Multi-scale forward-hook manager for YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8."""

    def __init__(self, models, modelt, distiller="cwd", device=None):
        self.distiller = (distiller or "cwd").lower()[:3]
        self.layers = ["6", "8", "13", "16", "19", "22"]
        self.models = de_parallel(models)
        self.modelt = de_parallel(modelt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        self.teacher_outputs = []
        self.student_outputs = []

        self._find_layers()
        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s,
            channels_t=self.channels_t,
            distiller=self.distiller,
            device=self.device,
        )

    @classmethod
    def _extract_model_layers(cls, model, default_layers):
        base = de_parallel(model)
        seq = getattr(base, "model", None)
        if isinstance(seq, nn.Sequential) and len(seq) > 0:
            head = seq[-1]
            head_from = getattr(head, "f", None)
            if isinstance(head_from, list) and all(isinstance(x, int) and x > 0 for x in head_from):
                neck_layers = [str(x) for x in head_from]
                if neck_layers == ["15", "18", "21"]:
                    target_layers = ["6", "8", "12", "15", "18", "21"]
                else:
                    target_layers = ["6", "8", "13", *neck_layers]
            else:
                target_layers = list(default_layers)

            found = []
            for l_str in target_layers:
                idx = int(l_str)
                if idx < len(seq):
                    block = seq[idx]
                    sub = getattr(block, "cv2", getattr(block, "cv4", getattr(block, "cv3", block)))
                    conv = getattr(sub, "conv", sub)
                    out_ch = getattr(conv, "out_channels", None)
                    if out_ch is not None:
                        found.append((sub, out_ch))
            if found:
                return found

        found = []
        for name, ml in base.named_modules():
            if name:
                parts = name.split(".")
                if parts[0] == "model" and len(parts) == 3 and parts[1] in default_layers and "cv2" in parts[2]:
                    if hasattr(ml, "conv"):
                        found.append((ml, ml.conv.out_channels))
        return found

    def _find_layers(self):
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []

        t_found = self._extract_model_layers(self.modelt, self.layers)
        s_found = self._extract_model_layers(self.models, self.layers)

        nl = min(len(s_found), len(t_found))
        for (s_mod, s_ch), (t_mod, t_ch) in zip(s_found[-nl:], t_found[-nl:]):
            self.student_module_pairs.append(s_mod)
            self.channels_s.append(s_ch)
            self.teacher_module_pairs.append(t_mod)
            self.channels_t.append(t_ch)

        LOGGER.info(
            f"{colorstr('Knowledge Distillation:')} mode={self.distiller.upper()}, "
            f"paired {nl} multi-scale layers (student={self.channels_s} -> teacher={self.channels_t})"
        )

    def clear_outputs(self):
        self.teacher_outputs.clear()
        self.student_outputs.clear()

    def register_hook(self):
        self.remove_handle_()
        self.clear_outputs()

        def make_student_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.clone())
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self):
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if len(self.teacher_outputs) != len(self.student_outputs):
            self.clear_outputs()
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs)
        if self.distiller != "cwd":
            quant_loss = quant_loss * 0.3
        self.clear_outputs()
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()
        self.clear_outputs()
'''


def patch_default_yaml(yaml_path: Path) -> None:
    content = yaml_path.read_text(encoding="utf-8")
    if "distillation_loss:" not in content:
        content = content.rstrip() + "\n\n# Knowledge Distillation settings\nteacher: None\ndistillation_loss: None\n"
        yaml_path.write_text(content, encoding="utf-8")


def patch_trainer_py(trainer_path: Path) -> None:
    content = trainer_path.read_text(encoding="utf-8")
    if "class DistillationLoss" in content:
        return

    # 1. Ensure torch.nn.functional as F and de_parallel are imported
    if "import torch.nn.functional as F" not in content:
        content = content.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
    if "de_parallel" not in content:
        content = content.replace(
            "from ultralytics.utils.torch_utils import (",
            "from ultralytics.utils.torch_utils import (\n    de_parallel,",
            1,
        )

    # 2. Insert KD classes before BaseTrainer
    content = content.replace("class BaseTrainer:", KD_CLASSES_CODE.strip() + "\n\n\nclass BaseTrainer:", 1)

    # 3. Extract teacher and distillation_loss in BaseTrainer.__init__ before get_cfg
    content = content.replace(
        "        self.args = get_cfg(cfg, overrides)",
        "        if overrides:\n"
        "            self.teacher = overrides.pop('teacher', None)\n"
        "            self.loss_type = overrides.pop('distillation_loss', None) or 'cwd'\n"
        "        else:\n"
        "            self.teacher = None\n"
        "            self.loss_type = None\n"
        "        self.distillation_loss = None\n"
        "        self.args = get_cfg(cfg, overrides)",
        1,
    )

    # 4. Initialize and freeze teacher in _setup_train
    content = content.replace(
        "        self.model = self.model.to(self.device)",
        "        self.model = self.model.to(self.device)\n"
        "        if self.teacher is not None:\n"
        "            if isinstance(self.teacher, (str, Path)):\n"
        "                self.teacher, _ = attempt_load_one_weight(str(self.teacher))\n"
        "            elif hasattr(self.teacher, 'model') and isinstance(self.teacher.model, nn.Module) and not hasattr(self.teacher, 'criterion'):\n"
        "                self.teacher = self.teacher.model\n"
        "            self.teacher = self.teacher.to(self.device).float().eval()\n"
        "            for _, v in self.teacher.named_parameters():\n"
        "                v.requires_grad = False\n"
        "            self.distillation_loss = DistillationLoss(\n"
        "                self.model, self.teacher, distiller=self.loss_type, device=self.device\n"
        "            )",
        1,
    )

    # 5. Register trainable KD adapter parameters in optimizer inside _setup_train
    content = content.replace(
        "        self._setup_scheduler()",
        "        if self.distillation_loss is not None:\n"
        "            kd_params = [p for p in self.distillation_loss.distill_loss_fn.parameters() if p.requires_grad]\n"
        "            if kd_params:\n"
        "                base_pg = self.optimizer.param_groups[0]\n"
        "                pg = {'params': kd_params, 'lr': base_pg.get('lr', self.args.lr0), 'weight_decay': 0.0}\n"
        "                for k in ('use_muon', 'momentum', 'betas'):\n"
        "                    if k in base_pg:\n"
        "                        pg[k] = False if k == 'use_muon' else base_pg[k]\n"
        "                self.optimizer.add_param_group(pg)\n"
        "        self._setup_scheduler()",
        1,
    )

    # 6. Hook registration & loss calculation in _do_train
    content = content.replace(
        "            self.tloss = None\n",
        "            self.tloss = None\n"
        "            if self.teacher is not None:\n"
        "                if self.distillation_loss is None:\n"
        "                    self.distillation_loss = DistillationLoss(\n"
        "                        self.model, self.teacher, distiller=self.loss_type, device=self.device\n"
        "                    )\n"
        "                self.distillation_loss.register_hook()\n",
        1,
    )

    # 7. Inject teacher forward and distillation loss inside batch loop before backward
    backward_target = "                # Backward\n                self.scaler.scale(self.loss).backward()"
    kd_forward_snippet = (
        "                if self.teacher is not None:\n"
        "                    distill_weight = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.1 - 1) + 1\n"
        "                    with autocast(self.amp):\n"
        "                        with torch.no_grad():\n"
        "                            _ = self.teacher(batch['img'])\n"
        "                        self.d_loss = self.distillation_loss.get_loss() * distill_weight\n"
        "                        if RANK != -1:\n"
        "                            self.d_loss *= world_size\n"
        "                        self.loss = self.loss + self.d_loss\n\n"
        + backward_target
    )
    content = content.replace(backward_target, kd_forward_snippet, 1)

    # 8. Remove hooks at end of epoch and end of training
    content = content.replace(
        "            self.lr = {f\"lr/pg{ir}\": x[\"lr\"] for ir, x in enumerate(self.optimizer.param_groups)}",
        "            if self.teacher is not None and self.distillation_loss is not None:\n"
        "                self.distillation_loss.remove_handle_()\n"
        "            self.lr = {f\"lr/pg{ir}\": x[\"lr\"] for ir, x in enumerate(self.optimizer.param_groups)}",
        1,
    )
    content = content.replace(
        "        self.run_callbacks(\"teardown\")",
        "        if self.teacher is not None and self.distillation_loss is not None:\n"
        "            self.distillation_loss.remove_handle_()\n"
        "        self.run_callbacks(\"teardown\")",
        1,
    )

    trainer_path.write_text(content, encoding="utf-8")


def sync_ultralytics(force: bool = False) -> bool:
    """Downloads latest official Ultralytics if not yet synced from upstream (or force=True) and applies KD patches."""
    init_py = ROOT / "ultralytics" / "__init__.py"
    sync_marker = ROOT / "ultralytics" / ".synced_v26"
    has_upstream_v26 = sync_marker.exists() or (
        init_py.exists() and "# UPSTREAM_SYNCED_V26" in init_py.read_text(encoding="utf-8")
    )

    if not has_upstream_v26 or force:
        try:
            print("Syncing latest official Ultralytics (with YOLO26 / Segment26) from GitHub...")
            req = urllib.request.Request(TARBALL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) > 100_000:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                        tar.extractall(tmpdir)
                    extracted_roots = [p for p in Path(tmpdir).iterdir() if p.is_dir()]
                    if extracted_roots:
                        src_root = extracted_roots[0]
                        src_ultralytics = src_root / "ultralytics"
                        if src_ultralytics.exists():
                            shutil.rmtree(ROOT / "ultralytics", ignore_errors=True)
                            shutil.copytree(src_ultralytics, ROOT / "ultralytics")
                            init_text = (ROOT / "ultralytics" / "__init__.py").read_text(encoding="utf-8")
                            (ROOT / "ultralytics" / "__init__.py").write_text(
                                init_text + "\n# UPSTREAM_SYNCED_V26\n", encoding="utf-8"
                            )
                            (ROOT / "ultralytics" / ".synced_v26").write_text("synced\n", encoding="utf-8")
                        src_pyproject = src_root / "pyproject.toml"
                        if src_pyproject.exists():
                            shutil.copy2(src_pyproject, ROOT / "pyproject.toml")
                        print("Successfully synced latest Ultralytics codebase.")
        except Exception as e:
            print(f"Note: Offline or unable to fetch upstream tarball ({e}); using local ultralytics tree.")

    patch_default_yaml(ROOT / "ultralytics" / "cfg" / "default.yaml")
    patch_trainer_py(ROOT / "ultralytics" / "engine" / "trainer.py")
    return True


if __name__ == "__main__":
    sync_ultralytics(force=True)
