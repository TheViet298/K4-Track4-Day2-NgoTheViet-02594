"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo import được dataset, model, losses và eval
CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parents[2]  # submissions/02594_NgoTheViet/code -> repo root
for p in [str(CUR_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import dataset as ds_module
import model as model_module
import losses as loss_module
try:
    from eval import save_predictions, compute_metrics
except ImportError:
    compute_metrics = None
    save_predictions = None


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = model_module.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). Cập nhật theo iteration."""
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    total_steps = cfg.epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine decay từ 1.0 về 1e-4
        return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    if v.dtype.is_floating_point:
                        self.shadow[k].copy_(self.decay * self.shadow[k] + (1.0 - self.decay) * v)
                    else:
                        self.shadow[k].copy_(v)

    def apply_shadow(self, model: nn.Module) -> dict:
        """Lưu lại weights hiện tại và nạp shadow weights vào model."""
        original = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return original

    def restore(self, model: nn.Module, original: dict) -> None:
        """Khôi phục lại weights ban đầu."""
        model.load_state_dict(original)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict: {"train_loss": ..., "lr": ...}."""
    model.train()
    # Nếu đóng băng backbone, giữ BatchNorm của backbone ở chế độ eval
    if cfg.init == "frozen":
        model_module.set_bn_eval(model)

    running_loss = 0.0
    total_samples = 0
    use_amp = cfg.amp and (device.type == "cuda")

    for batch in loader:
        images, targets, _ = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixup hoặc CutMix
        if cfg.mix in ["mixup", "cutmix"]:
            mixed_images, mixed_targets = loss_module.mix_batch(
                images, targets, alpha=cfg.mix_alpha, mode=cfg.mix
            )
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(mixed_images)
                loss = loss_module.mixed_loss(criterion, logits, mixed_targets)
        else:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

        optimizer.zero_grad()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    current_lr = optimizer.param_groups[0]["lr"]
    return {
        "train_loss": running_loss / max(1, total_samples),
        "lr": current_lr
    }


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    """Chạy model trên loader ở chế độ eval.
    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    """
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets_dev)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_targets.append(targets.numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    avg_loss = total_loss / max(1, total_samples)

    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training: loss train/val và macro-F1 val theo epoch."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_f1s = [h["val_macro_f1"] for h in history]
    val_top1s = [h.get("val_top1", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Đồ thị 1: Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", color="#1f77b4")
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", color="#ff7f0e")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Đồ thị 2: Metrics
    axes[1].plot(epochs, val_f1s, label="Val Macro-F1", marker="^", color="#2ca02c")
    axes[1].plot(epochs, val_top1s, label="Val Top-1 Acc", marker="d", color="#d62728", linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{title} - Validation Metrics")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_folder = run_dir(cfg)
    out_folder.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)

    # Lưu config.json
    with open(out_folder / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # 1. Dữ liệu
    train_df, val_df, test_df = ds_module.load_split(cfg.labels_dir, fold=cfg.fold)
    split_info = ds_module.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = ds_module.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = ds_module.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds_module.make_loader(
        train_df, cfg.images_dir, transform=train_tf,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = ds_module.make_loader(
        val_df, cfg.images_dir, transform=val_tf,
        batch_size=cfg.batch_size, train=False, sampler=None, num_workers=cfg.num_workers
    )

    # 2. Model & Complexity
    model = model_module.build_model(
        name=cfg.backbone,
        pretrained=True,
        num_classes=ds_module.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    ).to(device)

    n_params = model_module.count_params(model)
    gmacs = model_module.count_gmacs(model, img_size=cfg.img_size)

    # 3. Criterion
    criterion_kwargs = {}
    if cfg.loss == "ls":
        criterion_kwargs["smoothing"] = cfg.label_smoothing
    elif cfg.loss == "focal":
        criterion_kwargs["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        train_counts = train_df["Label"].value_counts()
        weights = loss_module.class_weights(train_counts, beta=cfg.class_weight_beta or 0.0)
        criterion_kwargs["weight"] = weights.to(device)

    criterion = loss_module.build_criterion(cfg.loss, **criterion_kwargs)
    eval_criterion = nn.CrossEntropyLoss()

    # 4. Optimizer, Scheduler, AMP, EMA
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Training loop
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = out_folder / "best_checkpoint.pt"
    epoch_times = []

    print(f"\n[{cfg.exp_id}] Bắt đầu huấn luyện {cfg.backbone} ({cfg.init}) - {cfg.epochs} epochs:")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=ema
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # Đánh giá trên val (dùng EMA nếu có)
        original_weights = ema.apply_shadow(model) if ema is not None else None
        filenames, y_true, logits, val_loss = evaluate(model, val_loader, eval_criterion, device)
        if ema is not None:
            ema.restore(model, original_weights)

        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        y_pred = np.argmax(probs, axis=-1)

        if compute_metrics is not None:
            val_eval = compute_metrics(y_true, y_pred, probs)
            macro_f1 = val_eval["macro_f1"]
            top1 = val_eval["top1"]
        else:
            top1 = float((y_true == y_pred).mean())
            macro_f1 = top1

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "val_top1": top1,
            "lr": train_metrics["lr"],
            "time_sec": epoch_time
        }
        history.append(entry)

        # Lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn: > thay vì >=)
        is_best = macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": (ema.shadow if ema is not None else model.state_dict()),
                "val_macro_f1": macro_f1,
                "config": dataclasses.asdict(cfg)
            }, best_ckpt_path)

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
              f"Loss Train: {entry['train_loss']:.4f} | Val: {val_loss:.4f} | "
              f"Val Macro-F1: {macro_f1:.4f} {'[BEST]' if is_best else ''} | "
              f"Time: {epoch_time:.1f}s")

    # 6. Nạp checkpoint tốt nhất & lưu predictions trên VAL
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    val_filenames, val_y_true, val_logits, _ = evaluate(model, val_loader, eval_criterion, device)
    val_probs = F.softmax(torch.from_numpy(val_logits), dim=-1).numpy()
    np.save(out_folder / "val_logits.npy", val_logits)

    if save_predictions is not None:
        save_predictions(pred_path(cfg, "val"), val_filenames, val_y_true, val_probs)

    # 7. Đánh giá TEST nếu được bật (CHỈ BẬT Ở CHUNG KẾT BƯỚC 4)
    test_macro_f1 = None
    if cfg.save_test_predictions:
        test_tf = ds_module.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = ds_module.make_loader(
            test_df, cfg.images_dir, transform=test_tf,
            batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers
        )
        test_filenames, test_y_true, test_logits, _ = evaluate(model, test_loader, eval_criterion, device)
        test_probs = F.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
        np.save(out_folder / "test_logits.npy", test_logits)

        if save_predictions is not None:
            save_predictions(pred_path(cfg, "test"), test_filenames, test_y_true, test_probs)

        if compute_metrics is not None:
            test_pred = np.argmax(test_probs, axis=-1)
            test_eval = compute_metrics(test_y_true, test_pred, test_probs)
            test_macro_f1 = test_eval["macro_f1"]

    # 8. Lưu history.csv và vẽ đồ thị
    df_history = pd.DataFrame(history)
    df_history.to_csv(out_folder / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} - {cfg.backbone}")
    plot_curves(history, out_folder / "curves.png", title=f"{cfg.exp_id} - {cfg.backbone}")

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "init": cfg.init,
        "seed": cfg.seed,
        "n_params_M": n_params,
        "gmacs": gmacs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test_macro_f1": test_macro_f1,
        "avg_epoch_time_sec": float(np.mean(epoch_times)),
        "history_file": str(out_folder / "history.csv"),
        "best_checkpoint": str(best_ckpt_path)
    }

    with open(out_folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict ép kiểu chuẩn Config."""
    type_hints = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ (cần KEY=VALUE): '{pair}'")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key not in type_hints:
            valid_keys = ", ".join(type_hints.keys())
            raise KeyError(f"Key '{key}' không có trong Config. Các key hợp lệ: {valid_keys}")

        if value.lower() in ["none", "null"]:
            overrides[key] = None
        elif value.lower() in ["true", "yes"]:
            overrides[key] = True
        elif value.lower() in ["false", "no"]:
            overrides[key] = False
        else:
            # Ép kiểu int hoặc float nếu có thể
            try:
                overrides[key] = int(value)
            except ValueError:
                try:
                    overrides[key] = float(value)
                except ValueError:
                    overrides[key] = value

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0."""
    parser = argparse.ArgumentParser(description="DeepWeeds Training Runner")
    parser.add_argument("--set", nargs="*", default=[], help="Cấu hình ghi đè dạng KEY=VALUE")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    summary = run(cfg)
    print("\n=== HOÀN TẤT THÍ NGHIỆM ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
