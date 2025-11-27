"""
Configuration helpers for SDNQ integration
"""

import os
import torch
from typing import Optional
from pathlib import Path


def get_sdnq_models_dir() -> str:
    """
    Get the directory where SDNQ models should be stored.

    Uses ComfyUI's models/diffusers/ folder, creating an 'sdnq' subdirectory.
    Falls back to a default location if folder_paths is not available.

    Returns:
        Absolute path to SDNQ models directory
    """
    try:
        import folder_paths
        # Get ComfyUI's diffusers models folder
        diffusers_dirs = folder_paths.get_folder_paths("diffusers")
        if diffusers_dirs and len(diffusers_dirs) > 0:
            base_dir = diffusers_dirs[0]
            sdnq_dir = os.path.join(base_dir, "sdnq")
            os.makedirs(sdnq_dir, exist_ok=True)
            return sdnq_dir
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not access ComfyUI models folder: {e}")

    # Fallback to default location
    fallback_dir = os.path.expanduser("~/.cache/comfyui/models/diffusers/sdnq")
    os.makedirs(fallback_dir, exist_ok=True)
    return fallback_dir


def get_dtype_from_string(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation ("bfloat16", "float16", "float32")

    Returns:
        torch.dtype object

    Raises:
        ValueError: If dtype_str is not supported
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. "
            f"Supported values: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def get_device_map(device_str: str) -> Optional[str]:
    """
    Convert device string to device map for model loading.

    Args:
        device_str: Device selection ("auto", "cuda", "cpu")

    Returns:
        Device map string or None
    """
    device_map = {
        "auto": "auto",
        "cuda": "cuda",
        "cpu": "cpu",
    }

    return device_map.get(device_str, "auto")
