"""
SDNQ Core Package

Contains core functionality for SDNQ integration with ComfyUI:
- Model registry and catalog
- HuggingFace Hub downloader
- Configuration helpers

Performance Note: Heavy imports (torch, huggingface_hub) are lazy-loaded.
Only the lightweight registry module is imported at startup.
"""

# Only import lightweight registry module at startup
from .registry import (
    get_model_catalog,
    get_model_names,
    get_model_names_for_dropdown,
    get_model_info,
    get_repo_id_from_name,
    check_local_model_exists,
    recommend_models_by_vram,
    get_model_statistics
)


# Lazy imports for heavy modules (torch, huggingface_hub)
def get_dtype_from_string(dtype_str):
    """Lazy import wrapper for config.get_dtype_from_string."""
    from .config import get_dtype_from_string as _impl
    return _impl(dtype_str)


def get_device_map(device_str):
    """Lazy import wrapper for config.get_device_map."""
    from .config import get_device_map as _impl
    return _impl(device_str)


def download_model(*args, **kwargs):
    """Lazy import wrapper for downloader.download_model."""
    from .downloader import download_model as _impl
    return _impl(*args, **kwargs)


def check_model_cached(*args, **kwargs):
    """Lazy import wrapper for downloader.check_model_cached."""
    from .downloader import check_model_cached as _impl
    return _impl(*args, **kwargs)


def get_cached_model_path(*args, **kwargs):
    """Lazy import wrapper for downloader.get_cached_model_path."""
    from .downloader import get_cached_model_path as _impl
    return _impl(*args, **kwargs)


def get_model_size_estimate(*args, **kwargs):
    """Lazy import wrapper for downloader.get_model_size_estimate."""
    from .downloader import get_model_size_estimate as _impl
    return _impl(*args, **kwargs)


def download_model_with_status(*args, **kwargs):
    """Lazy import wrapper for downloader.download_model_with_status."""
    from .downloader import download_model_with_status as _impl
    return _impl(*args, **kwargs)


__all__ = [
    # Config (lazy)
    'get_dtype_from_string',
    'get_device_map',
    # Registry
    'get_model_catalog',
    'get_model_names',
    'get_model_names_for_dropdown',
    'get_model_info',
    'get_repo_id_from_name',
    'check_local_model_exists',
    'recommend_models_by_vram',
    'get_model_statistics',
    # Downloader (lazy)
    'download_model',
    'check_model_cached',
    'get_cached_model_path',
    'get_model_size_estimate',
    'download_model_with_status',
]
