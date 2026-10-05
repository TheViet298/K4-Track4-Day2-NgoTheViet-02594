"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo bắt buộc (RUBRIC mục 3):
  - Warmup: bỏ >= 10 lần chạy đầu
  - Đồng bộ GPU: torch.cuda.synchronize() trước và sau mỗi lượt đo
  - >= 50 lần đo thật, tính phân vị p50, p95, p99 (mili-giây)
  - Đo thật, không giả định (đặc biệt ở batch=1: AMP có thể chậm hơn FP32 do kernel launch overhead).
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 15, iters: int = 60, sync=None) -> dict:
    """Đo thời gian một hàm fn() (không tham số), trả về mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    # 2. Đo lường chính xác
    times_ms = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "std": float(np.std(times_arr, ddof=1)) if len(times_arr) > 1 else 0.0,
        "n": iters
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 15, iters: int = 60) -> dict:
    """Đo độ trễ forward của model với tensor ngẫu nhiên (batch_size, 3, img_size, img_size).
    Trả về dict để ghi thẳng vào sheet Latency của results.xlsx.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None
    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        dummy = dummy.half()

    use_amp = (dtype == "amp") and (dev.type == "cuda")

    @torch.inference_mode()
    def forward_fn():
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                return model(dummy)
        else:
            return model(dummy)

    timing = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    # Khôi phục model về float32 nếu đã half
    if dtype == "fp16":
        model.float()

    p50 = timing["p50"]
    throughput = (batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50_ms": round(timing["p50"], 3),
        "p95_ms": round(timing["p95"], 3),
        "p99_ms": round(timing["p99"], 3),
        "mean_ms": round(timing["mean"], 3),
        "images_per_s": round(throughput, 1),
        "torch_version": torch.__version__
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                device: str = "cuda", warmup: int = 15, iters: int = 60) -> dict:
    """Đo độ trễ thực tế khi chạy TTA K views so với K * p50 một view (slide trang 63)."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    @torch.inference_mode()
    def tta_fn():
        # Lượt 1: ảnh gốc
        out1 = model(dummy)
        # Lượt 2..K: các biến thể
        for _ in range(k_views - 1):
            flipped = torch.flip(dummy, dims=[-1])
            out2 = model(flipped)
        return out1

    timing = bench(tta_fn, warmup=warmup, iters=iters, sync=sync_fn)
    return {
        "k_views": k_views,
        "p50_ms": round(timing["p50"], 3),
        "p95_ms": round(timing["p95"], 3),
        "p99_ms": round(timing["p99"], 3),
        "mean_ms": round(timing["mean"], 3)
    }
