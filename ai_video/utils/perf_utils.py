from typing import Tuple
import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def setup_torch_cuda_hints(num_workers: int) -> None:
    """Enable safe CUDA and CPU hints for better performance."""
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # CPU threading hints
        cpu_threads = max(1, min(num_workers, (torch.get_num_threads() or 1)))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(cpu_threads)
    except Exception:
        pass


def enable_opencv_optimizations(num_threads: int) -> None:
    """Enable OpenCV threading and optimizations if available."""
    if cv2 is None:
        return
    try:
        cv2.setNumThreads(num_threads)
        cv2.useOptimized(True)
    except Exception:
        pass


def try_enable_low_latency_capture(cap: "cv2.VideoCapture") -> None:  # type: ignore
    """Attempt to set low-latency buffer and HW acceleration on capture."""
    if cv2 is None:
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    except Exception:
        pass
    try:
        if hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    except Exception:
        pass


def ensure_c_contiguous(frame: np.ndarray) -> np.ndarray:
    """Ensure numpy array uses contiguous memory layout."""
    return frame if frame.flags.get("C_CONTIGUOUS", False) else np.ascontiguousarray(frame)


def to_device_batch(
    batch: torch.Tensor,
    device: torch.device,
    non_blocking: bool = True,
    use_channels_last: bool = True,
) -> torch.Tensor:
    """Move a batch tensor to device with performance-friendly options."""
    if device.type == "cuda":
        try:
            batch = batch.pin_memory().to(device, non_blocking=non_blocking)
        except Exception:
            batch = batch.to(device)
        if use_channels_last and batch.ndim == 4:
            batch = batch.contiguous(memory_format=torch.channels_last)
        return batch
    return batch.to(device)


