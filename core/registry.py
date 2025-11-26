"""
Model Registry and Catalog

Maintains a catalog of known SDNQ models from Disty0's collection with metadata
including VRAM requirements, quantization levels, and quality estimates.
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Catalog of pre-quantized SDNQ models from Disty0's HuggingFace collection
# Organized by model family for easier navigation
SDNQ_MODEL_CATALOG = {
    # FLUX Models - Most popular for high-quality generation
    "FLUX.1-dev-qint8": {
        "repo_id": "Disty0/FLUX.1-dev-qint8",
        "type": "FLUX",
        "quant_level": "int8",
        "vram_required": "~12 GB",
        "vram_gb": 12,
        "quality": "~99%",
        "size_gb": "~15 GB",
        "description": "FLUX.1-dev 8-bit - Best quality/VRAM balance",
        "priority": 1  # High priority for display
    },
    "FLUX.1-dev-qint6": {
        "repo_id": "Disty0/FLUX.1-dev-qint6",
        "type": "FLUX",
        "quant_level": "int6",
        "vram_required": "~9 GB",
        "vram_gb": 9,
        "quality": "~97%",
        "size_gb": "~12 GB",
        "description": "FLUX.1-dev 6-bit - Great quality, lower VRAM",
        "priority": 2
    },
    "FLUX.1-dev-qint4": {
        "repo_id": "Disty0/FLUX.1-dev-qint4",
        "type": "FLUX",
        "quant_level": "uint4",
        "vram_required": "~6 GB",
        "vram_gb": 6,
        "quality": "~95%",
        "size_gb": "~9 GB",
        "description": "FLUX.1-dev 4-bit - Extreme VRAM savings",
        "priority": 3
    },
    "FLUX.1-schnell-qint8": {
        "repo_id": "Disty0/FLUX.1-schnell-qint8",
        "type": "FLUX",
        "quant_level": "int8",
        "vram_required": "~12 GB",
        "vram_gb": 12,
        "quality": "~99%",
        "size_gb": "~15 GB",
        "description": "FLUX.1-schnell 8-bit - Fast generation variant",
        "priority": 4
    },

    # FLUX.2 Models - Next generation FLUX
    "FLUX.2-dev-qint8": {
        "repo_id": "Disty0/FLUX.2-dev-qint8",
        "type": "FLUX",
        "quant_level": "int8",
        "vram_required": "~12 GB",
        "vram_gb": 12,
        "quality": "~99%",
        "size_gb": "~15 GB",
        "description": "FLUX.2-dev 8-bit - Next-gen FLUX model",
        "priority": 5
    },
    "FLUX.2-dev-qint4": {
        "repo_id": "Disty0/FLUX.2-dev-qint4",
        "type": "FLUX",
        "quant_level": "uint4",
        "vram_required": "~6 GB",
        "vram_gb": 6,
        "quality": "~95%",
        "size_gb": "~9 GB",
        "description": "FLUX.2-dev 4-bit - Next-gen with extreme savings",
        "priority": 6
    },

    # SD 3.5 Models - Latest Stable Diffusion
    "SD3.5-Large-qint8": {
        "repo_id": "Disty0/stable-diffusion-3.5-large-qint8",
        "type": "SD3.5",
        "quant_level": "int8",
        "vram_required": "~10 GB",
        "vram_gb": 10,
        "quality": "~99%",
        "size_gb": "~12 GB",
        "description": "SD 3.5 Large 8-bit - Latest flagship model",
        "priority": 7
    },
    "SD3.5-Large-Turbo-qint8": {
        "repo_id": "Disty0/stable-diffusion-3.5-large-turbo-qint8",
        "type": "SD3.5",
        "quant_level": "int8",
        "vram_required": "~10 GB",
        "vram_gb": 10,
        "quality": "~99%",
        "size_gb": "~12 GB",
        "description": "SD 3.5 Large Turbo 8-bit - Fast inference",
        "priority": 8
    },
    "SD3.5-Medium-qint8": {
        "repo_id": "Disty0/stable-diffusion-3.5-medium-qint8",
        "type": "SD3.5",
        "quant_level": "int8",
        "vram_required": "~6 GB",
        "vram_gb": 6,
        "quality": "~99%",
        "size_gb": "~8 GB",
        "description": "SD 3.5 Medium 8-bit - Smaller, faster",
        "priority": 9
    },

    # SDXL Models - Stable and widely compatible
    "SDXL-base-qint8": {
        "repo_id": "Disty0/stable-diffusion-xl-base-1.0-qint8",
        "type": "SDXL",
        "quant_level": "int8",
        "vram_required": "~6 GB",
        "vram_gb": 6,
        "quality": "~99%",
        "size_gb": "~7 GB",
        "description": "SDXL Base 1.0 8-bit - Classic high quality",
        "priority": 10
    },
    "SDXL-base-qint4": {
        "repo_id": "Disty0/stable-diffusion-xl-base-1.0-qint4",
        "type": "SDXL",
        "quant_level": "uint4",
        "vram_required": "~4 GB",
        "vram_gb": 4,
        "quality": "~95%",
        "size_gb": "~4 GB",
        "description": "SDXL Base 1.0 4-bit - Very low VRAM",
        "priority": 11
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
    Get model names formatted for ComfyUI dropdown with descriptions.

    Returns:
        List of formatted model names with metadata
    """
    names = []
    for model_name in get_model_names(sort_by_priority=True):
        info = SDNQ_MODEL_CATALOG[model_name]
        # Format: "Model Name [VRAM] - Description"
        formatted = f"{model_name} [{info['vram_required']}]"
        names.append(formatted)
    return names


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

    Args:
        vram_gb: Available VRAM in GB

    Returns:
        List of recommended model names
    """
    recommendations = []

    for model_name, info in SDNQ_MODEL_CATALOG.items():
        required_vram = info.get("vram_gb", 12)
        # Add 10% headroom for safety
        if required_vram * 1.1 <= vram_gb:
            recommendations.append(model_name)

    # Sort by priority
    recommendations.sort(key=lambda x: SDNQ_MODEL_CATALOG[x].get("priority", 999))

    return recommendations


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
