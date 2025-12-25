"""
Model Registry and Catalog

Maintains a catalog of known SDNQ models from Disty0's collection with metadata
including VRAM requirements, quantization levels, and quality estimates.
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Catalog of pre-quantized SDNQ models from Disty0's HuggingFace collection
# All entries verified against https://huggingface.co/collections/Disty0/sdnq
# Last updated: 2025-12-23
SDNQ_MODEL_CATALOG = {
    # FLUX.1 Models - Original FLUX with older qint naming
    "FLUX.1-dev-qint8": {
        "repo_id": "Disty0/FLUX.1-dev-qint8",
        "type": "FLUX",
        "quant_level": "int8",
        "description": "FLUX.1-dev 8-bit - Best quality/VRAM balance",
        "priority": 1
    },
    "FLUX.1-dev-qint4": {
        "repo_id": "Disty0/FLUX.1-dev-qint4",
        "type": "FLUX",
        "quant_level": "uint4",
        "description": "FLUX.1-dev 4-bit - Extreme VRAM savings",
        "priority": 2
    },
    "FLUX.1-dev-SDNQ-uint4": {
        "repo_id": "Disty0/FLUX.1-dev-SDNQ-uint4-svd-r32",
        "type": "FLUX",
        "quant_level": "uint4",
        "description": "FLUX.1-dev 4-bit SVD - Text-to-Image",
        "priority": 3
    },
    "FLUX.1-schnell-SDNQ-uint4": {
        "repo_id": "Disty0/FLUX.1-schnell-SDNQ-uint4-svd-r32",
        "type": "FLUX",
        "quant_level": "uint4",
        "description": "FLUX.1-schnell 4-bit SVD - Fast generation",
        "priority": 4
    },
    "FLUX.1-Krea-dev-SDNQ-uint4": {
        "repo_id": "Disty0/FLUX.1-Krea-dev-SDNQ-uint4-svd-r32",
        "type": "FLUX",
        "quant_level": "uint4",
        "description": "FLUX.1-Krea-dev 4-bit SVD",
        "priority": 5
    },
    "FLUX.1-Kontext-dev-SDNQ-uint4": {
        "repo_id": "Disty0/FLUX.1-Kontext-dev-SDNQ-uint4-svd-r32",
        "type": "FLUX",
        "quant_level": "uint4",
        "description": "FLUX.1-Kontext-dev 4-bit SVD",
        "priority": 6
    },

    # FLUX.2 Models - Next generation (requires diffusers>=0.35)
    "FLUX.2-dev-SDNQ-uint4": {
        "repo_id": "Disty0/FLUX.2-dev-SDNQ-uint4-svd-r32",
        "type": "FLUX2",
        "quant_level": "uint4",
        "description": "FLUX.2-dev 4-bit SVD - Next-gen (needs diffusers>=0.35)",
        "priority": 7
    },

    # Qwen Image Models - Text-to-Image and Image-to-Image
    "Qwen-Image-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image 4-bit SVD - Text-to-Image",
        "priority": 8
    },
    "Qwen-Image-Lightning-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-Lightning-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image-Lightning 4-bit SVD - Fast T2I",
        "priority": 9
    },
    "Qwen-Image-Edit-2509-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-Edit-2509-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image-Edit-2509 4-bit SVD - Image-to-Image",
        "priority": 10
    },
    "Qwen-Image-Edit-2511-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image-Edit-2511 4-bit SVD - Latest Image-to-Image",
        "priority": 11
    },
    "Qwen-Image-Edit-Lightning-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-Edit-Lightning-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image-Edit-Lightning 4-bit SVD - Fast I2I",
        "priority": 12
    },
    "Qwen-Image-Edit-SDNQ-uint4": {
        "repo_id": "Disty0/Qwen-Image-Edit-SDNQ-uint4-svd-r32",
        "type": "Qwen",
        "quant_level": "uint4",
        "description": "Qwen-Image-Edit 4-bit SVD - Image-to-Image",
        "priority": 13
    },
    # NOTE: Qwen3-VL-32B-Instruct is a VLM (Vision-Language Model) that outputs TEXT, not images.
    # It requires transformers library, not diffusers, and needs a separate node implementation.
    # See docs/VLM_SUPPORT_NOTES.md for implementation details if VLM support is added later.

    # Z-Image Models - Latest addition (2025)
    "Z-Image-Turbo-SDNQ-int8": {
        "repo_id": "Disty0/Z-Image-Turbo-SDNQ-int8",
        "type": "Z-Image",
        "quant_level": "int8",
        "description": "Z-Image-Turbo 8-bit - Best quality, fast INT8 MatMul",
        "priority": 15
    },
    "Z-Image-Turbo-SDNQ-uint4": {
        "repo_id": "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
        "type": "Z-Image",
        "quant_level": "uint4",
        "description": "Z-Image-Turbo 4-bit SVD - Max VRAM savings",
        "priority": 16
    },

    # Other Models
    "Chroma1-HD-SDNQ-uint4": {
        "repo_id": "Disty0/Chroma1-HD-SDNQ-uint4-svd-r32",
        "type": "Chroma",
        "quant_level": "uint4",
        "description": "Chroma1-HD 4-bit SVD - Text-to-Image",
        "priority": 17
    },
    "ChronoEdit-14B-SDNQ-uint4": {
        "repo_id": "Disty0/ChronoEdit-14B-SDNQ-uint4-svd-r32",
        "type": "Chrono",
        "quant_level": "uint4",
        "description": "ChronoEdit-14B 4-bit SVD - Image editing",
        "priority": 18
    },
    "HunyuanImage3-SDNQ-uint4": {
        "repo_id": "Disty0/HunyuanImage3-SDNQ-uint4-svd-r32",
        "type": "Hunyuan",
        "quant_level": "uint4",
        "description": "HunyuanImage3 4-bit SVD - Text-to-Image (45B params)",
        "priority": 19
    },

    # NoobAI Models - Anime/Illustration focused
    "NoobAI-XL-Vpred-v1.0-SDNQ-uint4": {
        "repo_id": "Disty0/NoobAI-XL-Vpred-v1.0-SDNQ-uint4-svd-r128",
        "type": "SDXL",
        "quant_level": "uint4",
        "description": "NoobAI-XL v1.0 4-bit SVD - Anime/illustration",
        "priority": 20
    },
    "NoobAI-XL-v1.1-SDNQ-uint4": {
        "repo_id": "Disty0/NoobAI-XL-v1.1-SDNQ-uint4-svd-r128",
        "type": "SDXL",
        "quant_level": "uint4",
        "description": "NoobAI-XL v1.1 4-bit SVD - Anime/illustration",
        "priority": 21
    },

    # Video Models - Image-to-Video and Text-to-Video
    "Wan2.2-I2V-A14B-SDNQ-uint4": {
        "repo_id": "Disty0/Wan2.2-I2V-A14B-SDNQ-uint4-svd-r32",
        "type": "Wan",
        "quant_level": "uint4",
        "description": "Wan2.2-I2V-A14B 4-bit SVD - Image-to-Video",
        "priority": 22
    },
    "Wan2.2-T2V-A14B-SDNQ-uint4": {
        "repo_id": "Disty0/Wan2.2-T2V-A14B-SDNQ-uint4-svd-r32",
        "type": "Wan",
        "quant_level": "uint4",
        "description": "Wan2.2-T2V-A14B 4-bit SVD - Text-to-Video",
        "priority": 23
    },
}


def get_model_catalog() -> Dict[str, Dict]:
    """
    Get the catalog of available SDNQ models.

    Returns:
        Dictionary of model metadata
    """
    return SDNQ_MODEL_CATALOG


def get_model_names(sort_by_priority: bool = True) -> List[str]:
    """
    Get list of available model names.

    Args:
        sort_by_priority: Sort models by priority (recommended first)

    Returns:
        List of model names
    """
    if sort_by_priority:
        # Sort by priority field (lower number = higher priority)
        return sorted(
            SDNQ_MODEL_CATALOG.keys(),
            key=lambda x: SDNQ_MODEL_CATALOG[x].get("priority", 999)
        )
    return list(SDNQ_MODEL_CATALOG.keys())


def get_model_names_for_dropdown() -> List[str]:
    """
    Get model names formatted for ComfyUI dropdown.

    Returns:
        List of model names sorted by priority
    """
    return get_model_names(sort_by_priority=True)


def get_model_info(model_name: str) -> Optional[Dict]:
    """
    Get metadata for a specific model.

    Args:
        model_name: Name of the model (can include dropdown formatting)

    Returns:
        Model metadata dictionary or None if not found
    """
    # Strip dropdown formatting if present
    clean_name = model_name.split(" [")[0] if " [" in model_name else model_name
    return SDNQ_MODEL_CATALOG.get(clean_name)


def get_repo_id_from_name(model_name: str) -> Optional[str]:
    """
    Get HuggingFace repo ID from model name.

    Args:
        model_name: Name of the model (can include dropdown formatting)

    Returns:
        HuggingFace repo ID or None if not found
    """
    info = get_model_info(model_name)
    return info["repo_id"] if info else None


def check_local_model_exists(model_name: str, cache_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if a model is already downloaded locally.

    Args:
        model_name: Name of the model
        cache_dir: Optional custom cache directory

    Returns:
        Tuple of (exists: bool, path: Optional[str])
    """
    from huggingface_hub import try_to_load_from_cache

    info = get_model_info(model_name)
    if not info:
        return (False, None)

    repo_id = info["repo_id"]

    # Check HuggingFace cache
    # The model is downloaded if model_index.json exists in cache
    try:
        cached_path = try_to_load_from_cache(
            repo_id=repo_id,
            filename="model_index.json",
            cache_dir=cache_dir
        )
        if cached_path and cached_path != "_not_found_":
            # Get the model directory (parent of model_index.json)
            model_dir = os.path.dirname(cached_path)
            return (True, model_dir)
    except Exception:
        pass

    return (False, None)


def recommend_models_by_vram(vram_gb: int) -> List[str]:
    """
    Recommend models based on available VRAM.

    NOTE: VRAM estimates removed due to inaccuracy. This function now returns
    all models sorted by priority. Users should check model requirements on HuggingFace.

    Args:
        vram_gb: Available VRAM in GB (currently unused)

    Returns:
        List of all model names sorted by priority
    """
    # Return all models sorted by priority
    # Users can check actual VRAM requirements on the HuggingFace model pages
    return get_model_names(sort_by_priority=True)


def get_model_statistics() -> Dict[str, int]:
    """
    Get statistics about the model catalog.

    Returns:
        Dictionary with catalog statistics
    """
    stats = {
        "total_models": len(SDNQ_MODEL_CATALOG),
        "flux_models": len([m for m in SDNQ_MODEL_CATALOG.values() if m["type"] == "FLUX"]),
        "sd35_models": len([m for m in SDNQ_MODEL_CATALOG.values() if m["type"] == "SD3.5"]),
        "sdxl_models": len([m for m in SDNQ_MODEL_CATALOG.values() if m["type"] == "SDXL"]),
    }
    return stats
