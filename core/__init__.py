"""
SDNQ Core Package

Contains core functionality for SDNQ integration with ComfyUI:
- Model registry and catalog
- HuggingFace Hub downloader
- ComfyUI type wrappers (MODEL, CLIP, VAE)
- Configuration helpers
"""

from .wrapper import wrap_pipeline_components
from .config import get_dtype_from_string, get_device_map
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
from .downloader import (
    download_model,
    check_model_cached,
    get_cached_model_path,
    get_model_size_estimate,
    download_model_with_status
)

__all__ = [
    # Wrapper
    'wrap_pipeline_components',
    # Config
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
    # Downloader
    'download_model',
    'check_model_cached',
    'get_cached_model_path',
    'get_model_size_estimate',
    'download_model_with_status',
]
