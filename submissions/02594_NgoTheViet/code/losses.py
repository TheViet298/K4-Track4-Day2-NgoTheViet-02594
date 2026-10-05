"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
- Label smoothing (trang 56)
- Focal loss (trang 57)
- Mixup/CutMix (trang 48)

Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".
    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        smoothing = kw.get("smoothing", 0.1)
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = kw.get("gamma", 2.0)
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không nhận diện kind={kind}. Chọn: ce, ls, focal, ce_weighted")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56).
    Dùng torch.nn.functional.cross_entropy với tham số label_smoothing.
    Khi eps = 0.0, hàm tương đương 100% với standard CrossEntropyLoss.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).
    Khi gamma = 0 và alpha = None, bằng đúng standard CrossEntropyLoss.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], target: [B]
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_unsqueezed = target.unsqueeze(1)
        log_pt = log_p.gather(dim=-1, index=target_unsqueezed).squeeze(1)
        pt = p.gather(dim=-1, index=target_unsqueezed).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(dim=-1, index=target)
            loss = alpha_t * loss

        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (README.md mục 2.1).
    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về mean = 1
    - beta > 0: class-balanced theo số mẫu hiệu dụng (Cui et al., slide trang 57)
      w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng trọng số = số lớp C (9).
    """
    if hasattr(counts, "values"):
        if hasattr(counts, "sort_index"):
            counts = counts.sort_index().values
        else:
            counts = list(counts.values())

    counts_arr = np.array(counts, dtype=np.float64)
    num_classes = len(counts_arr)

    if beta == 0.0:
        weights = 1.0 / np.maximum(counts_arr, 1.0)
        weights = weights / np.mean(weights)
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * num_classes

    return torch.tensor(weights, dtype=torch.float32)


def _rand_bbox(size: tuple[int, int, int, int], lam: float) -> tuple[int, int, int, int]:
    """Tạo bounding box cắt cho CutMix và kẹp vào biên ảnh."""
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn (Mixup hoặc CutMix).
    - lam ~ Beta(alpha, alpha)
    - CutMix: tính lại lam theo DIỆN TÍCH THỰC của vùng cắt sau khi kẹp biên ảnh.
    - Trả về (x_mixed, (y_a, y_b, lam))
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[index]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # TÍNH LẠI LAM THEO DIỆN TÍCH THỰC TẾ
        H, W = x.size(2), x.size(3)
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_adj = 1.0 - float(actual_area) / float(H * W)

        return x_mixed, (y_a, y_b, lam_adj)
    else:
        raise ValueError(f"Không nhận diện mode={mode}. Chọn 'mixup' hoặc 'cutmix'")


def mixed_loss(criterion, logits: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
