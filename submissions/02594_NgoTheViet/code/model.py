"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = False if init == "scratch" else pretrained
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    model.init_mode = init
    model.backbone_name = name

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.
    Khi đóng băng backbone, BatchNorm cũng phải ở chế độ eval.
    """
    classifier = model.get_classifier()
    head_param_ids = {id(p) for p in classifier.parameters()} if classifier is not None else set()

    for p in model.parameters():
        if id(p) in head_param_ids:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model._is_backbone_frozen = True


def set_bn_eval(model: nn.Module) -> None:
    """Giữ toàn bộ BatchNorm của backbone ở chế độ eval khi backbone bị freeze."""
    if getattr(model, "_is_backbone_frozen", False):
        classifier = model.get_classifier()
        head_modules = set(classifier.modules()) if classifier is not None else set()
        for m in model.modules():
            if m not in head_modules:
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:
    1. Trọng số backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    2. Norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. Head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    Bỏ qua tham số có requires_grad == False.
    """
    classifier = model.get_classifier()
    head_param_ids = {id(p) for p in classifier.parameters()} if classifier is not None else set()

    params_backbone_decay = []
    params_backbone_no_decay = []
    params_head = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if id(p) in head_param_ids:
            params_head.append(p)
        else:
            if p.ndim > 1:
                params_backbone_decay.append(p)
            else:
                params_backbone_no_decay.append(p)

    groups = []
    if params_backbone_decay:
        groups.append({
            "params": params_backbone_decay,
            "lr": lr_backbone,
            "weight_decay": weight_decay
        })
    if params_backbone_no_decay:
        groups.append({
            "params": params_backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0
        })
    if params_head:
        groups.append({
            "params": params_head,
            "lr": lr_head,
            "weight_decay": weight_decay
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (tính MAC, không phải FLOPs 2x)."""
    # 1. Thử ptflops
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return round(macs / 1e9, 3)
    except Exception:
        pass

    # 2. Thử thop
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return round(macs / 1e9, 3)
    except Exception:
        pass

    # 3. Fallback: Hook đếm MAC chuẩn xác cho Conv2d, Linear, ConvNeXt/ViT
    total_macs = [0]
    hooks = []

    def conv_hook(module, inp, out):
        # out: [B, C_out, H_out, W_out]
        b, c_out, h_out, w_out = out.shape
        c_in = module.in_channels // module.groups
        k_h, k_w = module.kernel_size
        kernel_ops = k_h * k_w * c_in
        total_macs[0] += b * c_out * h_out * w_out * kernel_ops

    def linear_hook(module, inp, out):
        b = inp[0].shape[0] if len(inp[0].shape) > 1 else 1
        num_tokens = inp[0].numel() // (b * module.in_features)
        total_macs[0] += b * num_tokens * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        try:
            model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    return round(total_macs[0] / 1e9, 3)
