"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2:
- TTA (trang 62-66, 75)
- Ensemble / Model soup (trang 67)
- Temperature scaling (trang 69)
- Gộp BatchNorm (trang 71)

Mọi hàm chạy ở chế độ eval, không gradient.
"""
from __future__ import annotations

import copy
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device | str, view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file.
    `view` là hàm biến đổi tensor ảnh trước khi đưa vào model (ví dụ view_hflip).
    Trả về (filenames, y_true [N], logits [N, 9]).
    """
    model.eval()
    dev = torch.device(device)
    all_filenames = []
    all_targets = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            if view is not None:
                images = view(images)

            images = images.to(dev, non_blocking=True)
            logits = model(images)

            all_filenames.extend(filenames)
            all_targets.append(targets.numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Giữ nguyên ảnh gốc (1 view mốc I00)."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W) qua chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> list[torch.Tensor]:
    """5-crop: 4 góc + góc giữa."""
    _, _, H, W = x.shape
    crops = []
    # 4 góc
    crops.append(x[:, :, :crop, :crop])
    crops.append(x[:, :, :crop, W - crop:])
    crops.append(x[:, :, H - crop:, :crop])
    crops.append(x[:, :, H - crop:, W - crop:])
    # Giữa
    start_h = (H - crop) // 2
    start_w = (W - crop) // 2
    crops.append(x[:, :, start_h:start_h + crop, start_w:start_w + crop])
    return crops


def views_multiscale(x: torch.Tensor, sizes: list[int] = [224, 256]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong sizes."""
    scaled = []
    for s in sizes:
        scaled.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
    return scaled


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một phân bố xác suất (slide trang 62).
      - space='prob' : trung bình softmax của từng view
      - space='logit': trung bình logit rồi mới softmax
    Trả về probs [N, 9].
    """
    if space == "prob":
        probs_list = [
            F.softmax(torch.from_numpy(l), dim=-1).numpy() for l in logits_per_view
        ]
        return np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return F.softmax(torch.from_numpy(avg_logits), dim=-1).numpy()
    else:
        raise ValueError(f"Không nhận diện space={space}. Chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu hoá NLL trên VAL: p = softmax(logits / T) (slide trang 69).
    Accuracy không đổi vì thứ tự argmax giữ nguyên.
    """
    logits_t = torch.from_numpy(val_logits).float()
    labels_t = torch.from_numpy(val_labels).long()

    class TemperatureWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_temp = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            temp = torch.exp(self.log_temp)
            return x / temp

    wrapper = TemperatureWrapper()
    optimizer = torch.optim.LBFGS(wrapper.parameters(), lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(wrapper(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    best_t = float(torch.exp(wrapper.log_temp).item())
    return max(0.01, best_t)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Áp dụng nhiệt độ T đã khớp và trả về softmax(logits / T)."""
    T = max(1e-4, float(T))
    scaled = logits / T
    return F.softmax(torch.from_numpy(scaled), dim=-1).numpy()


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm2d vào Conv2d liền kề lúc suy luận (slide trang 71, 75).
    Kiểm tra sai số đầu ra trước và sau khi gộp.
    """
    model_copy = copy.deepcopy(model).eval()

    # Dùng tiện ích chính thức của PyTorch nếu có
    try:
        fused = torch.nn.utils.fuse_conv_bn_eval(model_copy)
        return fused
    except Exception:
        pass

    # Tự gộp thủ công nếu là cấu hình tùy biến
    for m in model_copy.modules():
        for name, child in m.named_children():
            if isinstance(child, nn.Sequential):
                for idx in range(len(child) - 1):
                    if isinstance(child[idx], nn.Conv2d) and isinstance(child[idx + 1], nn.BatchNorm2d):
                        conv = child[idx]
                        bn = child[idx + 1]
                        
                        fused_conv = nn.Conv2d(
                            conv.in_channels,
                            conv.out_channels,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            bias=True
                        )
                        w = conv.weight
                        gamma = bn.weight
                        var = bn.running_var
                        eps = bn.eps
                        std = torch.sqrt(var + eps)
                        fused_conv.weight.data = w * (gamma / std).reshape(-1, 1, 1, 1)

                        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
                        beta = bn.bias
                        mean = bn.running_mean
                        fused_conv.bias.data = beta + gamma * (b - mean) / std

                        child[idx] = fused_conv
                        child[idx + 1] = nn.Identity()

    return model_copy
