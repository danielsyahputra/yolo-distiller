# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import copy_attr, unwrap_model

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the key (feature level) to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for backbone/neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation` (https://arxiv.org/abs/2011.13256)."""

    def __init__(
        self, channels_s: list[int] | None = None, channels_t: list[int] | None = None, tau: float = 1.0
    ) -> None:
        """Initialize Channel-wise Distillation loss with temperature tau."""
        super().__init__()
        self.tau = tau

    def forward(self, y_s: list[torch.Tensor], y_t: list[torch.Tensor]) -> torch.Tensor:
        """Compute channel-wise KL divergence between aligned student and teacher feature maps."""
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            assert s.shape == t.shape, f"Shape mismatch at stage {idx}: student {s.shape} vs teacher {t.shape}"
            N, C, H, W = s.shape
            s_fp32 = s.float().reshape(-1, W * H) / self.tau
            t_fp32 = t.float().reshape(-1, W * H) / self.tau
            softmax_pred_T = F.softmax(t_fp32, dim=1)
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t_fp32) - softmax_pred_T * logsoftmax(s_fp32)
            ) * (self.tau**2)
            losses.append(cost / (C * N))
        return sum(losses)


class MGDAdapter(nn.Module):
    """Single-stage feature alignment and masked generative block for MGD (https://arxiv.org/abs/2205.01529)."""

    def __init__(
        self, student_dim: int, teacher_dim: int, alpha_mgd: float = 0.00002, lambda_mgd: float = 0.65
    ) -> None:
        """Initialize 1x1 channel alignment, teacher normalization, and 3x3 generative block."""
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.align = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(teacher_dim, affine=False),
        )
        self.teacher_norm = nn.BatchNorm2d(teacher_dim, affine=False)
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dim, teacher_dim, kernel_size=3, padding=1),
        )

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Align student feature, apply random spatial mask, generate teacher feature, and compute scaled MSE loss."""
        param_dtype = next(self.align.parameters()).dtype
        s = self.align(s.to(dtype=param_dtype))
        t = t.detach().to(dtype=param_dtype)
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
        t = self.teacher_norm(t)
        N, C, H, W = t.shape
        device = s.device
        mat = torch.rand((N, 1, H, W), device=device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(dtype=s.dtype, device=device)
        masked_fea = torch.mul(s, mat)
        new_fea = self.generation(masked_fea)
        dis_loss = F.mse_loss(new_fea.float().contiguous(), t.float().contiguous(), reduction="sum") / N
        return dis_loss * self.alpha_mgd


class MGDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation` (https://arxiv.org/abs/2205.01529)."""

    def __init__(
        self,
        student_channels: list[int],
        teacher_channels: list[int],
        alpha_mgd: float = 0.00002,
        lambda_mgd: float = 0.65,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize multi-stage MGD loss."""
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                ).to(device)
                for channel in teacher_channels
            ]
        )

    def forward(self, y_s: list[torch.Tensor], y_t: list[torch.Tensor], layer: str | None = None) -> torch.Tensor:
        """Compute summed MGD loss across all feature pyramid stages."""
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        return sum(losses)

    def get_dis_loss(self, preds_S: torch.Tensor, preds_T: torch.Tensor, idx: int) -> torch.Tensor:
        """Compute masked generative MSE loss for a single stage."""
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            preds_S = F.interpolate(preds_S, size=preds_T.shape[-2:], mode="bilinear", align_corners=False)
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W), device=device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(dtype=preds_S.dtype, device=device)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        return F.mse_loss(new_fea.float().contiguous(), preds_T.float().contiguous(), reduction="sum") / N


class DistillationModel(nn.Module):
    """Cross-architecture YOLO knowledge distillation model supporting CWD, MGD, and SL2 losses.

    This class wraps any teacher-student YOLO pair (e.g., YOLO26 <-> YOLO12 <-> YOLO11 <-> YOLOv10 <-> YOLOv9 <-> YOLOv8)
    for knowledge distillation training. Multi-scale features are extracted from both models via topological graph tracing
    and forward hooks, then aligned via 1x1 projections and spatial interpolation if needed.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        loss_type (str): Distillation loss mode ('cwd', 'mgd', or 'sl2').
        feats_idx (list): Student layer indices for feature extraction.
        teacher_feats_idx (list): Teacher layer indices at the corresponding semantic feature levels.
        projector (nn.ModuleList): Stage adapters aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Examples:
        >>> from ultralytics import YOLO
        >>> student = YOLO("yolo26n-seg.pt")
        >>> teacher = YOLO("yolo11l-seg.pt")
        >>> student.train(data="coco8-seg.yaml", teacher=teacher.model, distillation_loss="cwd")
    """

    def __init__(
        self,
        teacher_model: str | Path | nn.Module,
        student_model: nn.Module,
        loss_type: str | None = None,
    ):
        """Initialize the distillation model with teacher, student, loss type, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path, YOLO wrapper, or nn.Module.
            student_model (nn.Module): Student model module to be trained.
            loss_type (str, optional): Distillation loss method ('cwd', 'mgd', or 'sl2'). Defaults to student_model
                config or 'cwd'.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        elif (
            hasattr(teacher_model, "model")
            and isinstance(teacher_model.model, nn.Module)
            and not isinstance(teacher_model.model, (nn.Sequential, nn.ModuleList))
        ):
            # Unwrap high-level YOLO(weights) object to its underlying BaseModel nn.Module
            teacher_model = teacher_model.model
        teacher_model = unwrap_model(teacher_model)

        resolved_loss = (
            loss_type
            or (getattr(student_model.args, "distillation_loss", None) if hasattr(student_model, "args") else None)
            or "cwd"
        )
        if str(resolved_loss) in {"None", "null", ""}:
            resolved_loss = "cwd"
        self.loss_type = str(resolved_loss).lower()[:3]
        if self.loss_type not in {"cwd", "mgd", "sl2"}:
            raise ValueError(f"Unsupported distillation_loss='{resolved_loss}'. Expected 'cwd', 'mgd', or 'sl2'.")

        student_param = next(student_model.parameters())
        device = student_param.device
        self.teacher_model = teacher_model.to(device).float()
        self._freeze_teacher()
        self.student_model = student_model

        self.feats_idx = self.get_distill_layers(student_model, loss_type=self.loss_type)
        self.teacher_feats_idx = self.get_distill_layers(self.teacher_model, loss_type=self.loss_type)
        if len(self.feats_idx) != len(self.teacher_feats_idx):
            if self.loss_type in {"cwd", "mgd"}:
                nl = min(len(self.feats_idx), len(self.teacher_feats_idx))
                self.feats_idx = self.feats_idx[-nl:]
                self.teacher_feats_idx = self.teacher_feats_idx[-nl:]
            else:
                raise ValueError("Teacher and student detection heads must have the same number of feature levels")

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs regardless of block type)
        imgsz = getattr(student_model.args, "imgsz", 640) if hasattr(student_model, "args") else 640
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        student_model.eval()
        with torch.no_grad():
            im_t = torch.zeros(2, ch, imgsz, imgsz, device=device, dtype=next(self.teacher_model.parameters()).dtype)
            im_s = torch.zeros(2, ch, imgsz, imgsz, device=device, dtype=student_param.dtype)
            self.teacher_model(im_t)
            student_model(im_s)
        student_model.train()

        n_feats = len(self.feats_idx) - 1 if self.loss_type == "sl2" else len(self.feats_idx)
        teacher_output = [self._teacher_feats[i] for i in range(n_feats)]
        student_output = [self._student_feats[i] for i in range(n_feats)]
        self._teacher_feats.clear()
        self._student_feats.clear()

        copy_attr(self, student_model)
        raw_dis = getattr(self.student_model.args, "dis", 6.0) if hasattr(self.student_model, "args") else 6.0
        if self.loss_type == "cwd":
            self.dis = 1.0 if raw_dis == 6.0 else float(raw_dis)
        elif self.loss_type == "mgd":
            self.dis = 0.3 if raw_dis == 6.0 else float(raw_dis)
        else:
            self.dis = float(raw_dis)

        self.cwd_loss = CWDLoss() if self.loss_type == "cwd" else None
        projectors = []
        s_channels, t_channels = [], []
        for student_out, teacher_out in zip(student_output, teacher_output):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            s_channels.append(student_dim)
            t_channels.append(teacher_dim)
            if self.loss_type == "cwd":
                projectors.append(
                    nn.Sequential(
                        nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(teacher_dim, affine=False),
                    )
                )
            elif self.loss_type == "mgd":
                projectors.append(MGDAdapter(student_dim, teacher_dim))
            else:  # sl2
                projectors.append(
                    nn.Sequential(
                        nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    )
                )
        self.projector = nn.ModuleList(projectors).to(device)

        if RANK in {-1, 0}:
            LOGGER.info(
                f"{colorstr('Knowledge Distillation:')} mode={self.loss_type.upper()}, "
                f"paired {n_feats} multi-scale layers "
                f"(student_layers={self.feats_idx[:n_feats]} ch={s_channels} -> "
                f"teacher_layers={self.teacher_feats_idx[:n_feats]} ch={t_channels})"
            )

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        state.setdefault("teacher_feats_idx", state["feats_idx"])  # checkpoints saved before teacher_feats_idx
        state.setdefault("loss_type", "cwd")
        state.setdefault("cwd_loss", CWDLoss() if state["loss_type"] == "cwd" else None)
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        for i, (s, t) in enumerate(zip(self.feats_idx, self.teacher_feats_idx)):  # features stored by level i
            self._clear_feature_hooks(self.student_model.model[s])
            self._student_hooks.append(
                self.student_model.model[s].register_forward_hook(FeatureHook(self._student_feats, i))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[t])
                self._teacher_hooks.append(
                    self.teacher_model.model[t].register_forward_hook(FeatureHook(self._teacher_feats, i))
                )

    @staticmethod
    def _get_skip_idx(seq: nn.Sequential, concat_idx: int, fallback: int) -> int:
        """Extract the positive skip-connection layer index from a Concat layer's .f attribute."""
        if 0 <= concat_idx < len(seq):
            f = getattr(seq[concat_idx], "f", None)
            if isinstance(f, (list, tuple)):
                pos = [x for x in f if isinstance(x, int) and x >= 0]
                if pos:
                    return pos[0]
        return fallback

    @staticmethod
    def get_distill_layers(model: nn.Module, loss_type: str = "cwd") -> list[int]:
        """Auto-detect multi-scale distillation feature layers from the model's topological graph.

        For 'cwd' and 'mgd', traces 6 Backbone + Neck stages [B_P4, B_P5, N_td_P4, N_P3, N_P4, N_P5]:
        - YOLO26 / YOLO11 / YOLOv10 -> [6, 8, 13, 16, 19, 22]
        - YOLO12                    -> [6, 8, 11, 14, 17, 20]
        - YOLOv8 / YOLOv9           -> [6, 8, 12, 15, 18, 21]

        For 'sl2', returns the 3 Neck output layers plus the Detect head index itself (e.g. [16, 19, 22, 23]).

        Args:
            model (nn.Module): Model whose `model` layers contain a Detect head.
            loss_type (str): Distillation loss mode ('cwd', 'mgd', or 'sl2').

        Returns:
            (list[int]): Layer indices to hook for feature distillation.

        Raises:
            ValueError: If the model has no Detect head.
        """
        mode = str(loss_type or "cwd").lower()[:3]
        seq = model.model
        for m in seq:
            if isinstance(m, Detect):
                head_f = list(m.f)
                if mode == "sl2":
                    return [*head_f, m.i]
                if len(head_f) >= 3 and all(isinstance(x, int) and x > 0 for x in head_f):
                    n_p3, n_p4 = head_f[0], head_f[1]
                    n_td_p4 = DistillationModel._get_skip_idx(seq, n_p4 - 1, n_p3 - 3)
                    b_p4 = DistillationModel._get_skip_idx(seq, n_td_p4 - 1, 6)
                    b_p5 = b_p4 + 2
                    candidate = [b_p4, b_p5, n_td_p4, *head_f]
                    if all(0 <= idx < len(seq) for idx in candidate):
                        return candidate
                return head_f
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed in eval mode for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Run the student model, or compute the combined loss when given a training batch.

        Args:
            x (torch.Tensor | dict): Input image tensor, or a batch dict with images and labels for loss computation.
            *args (Any): Additional positional arguments passed to `loss()` or the student `predict()`.
            **kwargs (Any): Additional keyword arguments passed to `loss()` or the student `predict()`.

        Returns:
            (Any): Loss tuple if x is a dict, otherwise student model predictions.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        preds = self.student_model.predict(x, *args, **kwargs)
        self._student_feats.clear()
        return preds

    def fuse(self, verbose: bool = True, imgsz: int | list[int] = 640):
        """Fuse and return the student model, dropping the training-only distillation wrapper.

        Args:
            verbose (bool): Whether to print model information after fusion.
            imgsz (int | list[int]): Input image size used for FLOPs calculation.

        Returns:
            (nn.Module): The fused student model.
        """
        self._remove_feature_hooks()
        return self.student_model.fuse(verbose=verbose, imgsz=imgsz)

    def _extract_teacher_scores(
        self, teacher_head_feat, neck_feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Extract spatial classification score maps from either end2end (YOLO26/v10) or standard (YOLO11/12/v8/v9) heads."""
        raw = teacher_head_feat[1] if isinstance(teacher_head_feat, tuple) else teacher_head_feat
        if isinstance(raw, dict):
            if "one2many" in raw and "one2one" in raw:
                scores = (raw["one2many"]["scores"] + raw["one2one"]["scores"]) / 2
                probs = scores.sigmoid()
            elif "one2one" in raw:
                probs = raw["one2one"]["scores"].sigmoid()
            elif "one2many" in raw:
                probs = raw["one2many"]["scores"].sigmoid()
            elif "scores" in raw:
                probs = raw["scores"].sigmoid()
            else:
                raise ValueError(f"Unsupported teacher head dict keys for SL2 distillation: {list(raw.keys())}")
        elif isinstance(teacher_head_feat, tuple) and isinstance(teacher_head_feat[0], torch.Tensor):
            nc = getattr(self.teacher_model.model[-1], "nc", self.student_model.model[-1].nc)
            probs = teacher_head_feat[0][:, 4 : 4 + nc]
        else:
            raise ValueError(f"Unsupported teacher head output format for SL2 distillation: {type(teacher_head_feat)}")

        parts = torch.split(probs, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        return tuple(p.max(dim=1, keepdim=True).values for p in parts)

    def loss(self, batch, preds=None):
        """Compute combined detection/segmentation and knowledge distillation loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Student predictions, used only in validation mode.

        Returns:
            loss (torch.Tensor): Student loss with the distillation loss appended.
            loss_items (dict[str, torch.Tensor]): Detached loss components, including `dis_loss`.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            self._student_feats.clear()
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features

        regular_loss, loss_items = self.student_model.loss(batch, preds)

        if self.loss_type == "cwd":
            n = len(self.feats_idx)
            stu_feats, tea_feats = [], []
            for i in range(n):
                t_feat = self.decouple_outputs(self._teacher_feats[i]).detach()
                s_feat = self.decouple_outputs(self._student_feats[i])
                param_dtype = next(self.projector[i].parameters()).dtype
                s_feat = self.projector[i](s_feat.to(dtype=param_dtype))
                t_feat = t_feat.to(dtype=param_dtype)
                if s_feat.shape[-2:] != t_feat.shape[-2:]:
                    s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)
                stu_feats.append(s_feat)
                tea_feats.append(t_feat)
            loss_distill = (self.cwd_loss(stu_feats, tea_feats) * self.dis).reshape(1)
            loss_items["dis_loss"] = loss_distill.detach()
        elif self.loss_type == "mgd":
            n = len(self.feats_idx)
            mgd_losses = []
            for i in range(n):
                t_feat = self.decouple_outputs(self._teacher_feats[i]).detach()
                s_feat = self.decouple_outputs(self._student_feats[i])
                mgd_losses.append(self.projector[i](s_feat, t_feat))
            loss_distill = (sum(mgd_losses) * self.dis).reshape(1)
            loss_items["dis_loss"] = loss_distill.detach()
        else:  # sl2
            n = len(self.feats_idx) - 1  # neck levels; level n is the Detect head
            teacher_head_feat = self._teacher_feats[n]
            neck_feats = [self.decouple_outputs(self._teacher_feats[i]) for i in range(n)]
            teacher_scores = self._extract_teacher_scores(teacher_head_feat, neck_feats)
            for i in range(n):
                teacher_feat = neck_feats[i]
                student_feat = self.projector[i](self.decouple_outputs(self._student_feats[i]))
                if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                    student_feat = F.interpolate(
                        student_feat, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss_distill += (
                    self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
                )
            loss_items["dis_loss"] = loss_distill.detach()
            loss_distill = loss_distill * batch["img"].shape[0]

        # Release feature references immediately so autograd can free intermediate activations during .backward()
        self._teacher_feats.clear()
        self._student_feats.clear()
        return torch.cat([regular_loss, loss_distill]), loss_items

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.reshape(n, c, -1)
        teacher_feat = teacher_feat.reshape(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
