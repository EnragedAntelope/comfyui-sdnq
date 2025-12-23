"""
ComfyUI-SDNQ: Custom node pack for loading SDNQ quantized models in ComfyUI

SDNQ (SD.Next Quantization) is developed by Disty0
Repository: https://github.com/Disty0/sdnq
Pre-quantized models: https://huggingface.co/collections/Disty0/sdnq

This node pack enables loading SDNQ models with 50-75% VRAM savings while maintaining quality.
"""

from .nodes.sampler import SDNQSampler

# ============================================================================
# V1 API - Node Registration (ComfyUI Standard)
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SDNQSampler": SDNQSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDNQSampler": "SDNQ Sampler",
}

# ============================================================================
# V3 API - Modern Extension System (Future-Proofing)
# ============================================================================

def comfy_entrypoint():
    """
    V3 API entry point for ComfyUI.

    This provides forward compatibility with ComfyUI's V3 node system while
    maintaining V1 API support. V3 benefits:
    - Better type safety
    - Async execution support
    - Improved error handling
    - Schema validation

    Returns V1 mappings for now, but structure is ready for V3 enhancements.
    """
    return {
        "node_class_mappings": NODE_CLASS_MAPPINGS,
        "node_display_name_mappings": NODE_DISPLAY_NAME_MAPPINGS,
        "version": "0.3.0",
        "author": "ComfyUI-SDNQ Contributors",
        "description": "SDNQ quantized model support for ComfyUI",
        "license": "Apache-2.0",
    }


# Web directory for custom UI (optional, for future use)
WEB_DIRECTORY = "./web"

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'comfy_entrypoint',  # V3 API
]
