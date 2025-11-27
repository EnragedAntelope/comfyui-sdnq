"""
SDNQ Model Catalog Node

Phase 2: This node will display and manage available pre-quantized models.

Features to implement:
- Display available models from Disty0 collection
- Show model metadata (size, VRAM requirements, quality)
- Quick download/install functionality
- Check which models are already cached locally
"""

from typing import Tuple
from ..core.registry import get_model_catalog, get_model_names


class SDNQModelCatalog:
    """
    Display and manage available SDNQ models.

    This node shows available pre-quantized models, their metadata,
    and helps users discover and download models.

    TODO: Implement in Phase 2
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list_all", "show_cached", "recommend"], {
                    "default": "list_all"
                }),
            },
            "optional": {
                "available_vram_gb": ("INT", {
                    "default": 12,
                    "min": 4,
                    "max": 80,
                    "tooltip": "Available VRAM for model recommendations"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("catalog_info",)
    FUNCTION = "show_catalog"
    CATEGORY = "loaders/SDNQ"
    OUTPUT_NODE = True
    DESCRIPTION = "Display available SDNQ models and metadata (Phase 2)"

    def show_catalog(
        self,
        action: str = "list_all",
        available_vram_gb: int = 12
    ) -> Tuple[str]:
        """
        Display model catalog information.

        TODO: Full implementation in Phase 2
        """
        catalog = get_model_catalog()
        model_names = get_model_names()

        # Basic implementation - just list available models
        output = f"SDNQ Model Catalog ({len(model_names)} models)\n"
        output += "=" * 60 + "\n\n"

        for name in model_names:
            info = catalog[name]
            output += f"📦 {name}\n"
            output += f"   Repo: {info['repo_id']}\n"
            output += f"   Type: {info['type']}\n"
            output += f"   Quant: {info['quant_level']}\n"
            output += f"   VRAM: {info['vram_required']}\n"
            output += f"   Quality: {info['quality']}\n"
            output += f"   {info['description']}\n\n"

        output += "=" * 60 + "\n"
        output += "To use a model, copy the Repo ID into the SDNQ Model Loader node.\n"

        return (output,)


# Not exported yet - will be added in Phase 2
# NODE_CLASS_MAPPINGS = {"SDNQModelCatalog": SDNQModelCatalog}
# NODE_DISPLAY_NAME_MAPPINGS = {"SDNQModelCatalog": "SDNQ Model Catalog"}
