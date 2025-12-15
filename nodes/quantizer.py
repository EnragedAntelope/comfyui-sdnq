"""
SDNQ Model Quantizer Node

Quantizes existing ComfyUI models to SDNQ format for VRAM savings.
Uses ComfyUI's native MODEL input - works with any model loader!
"""

import os
import torch
from typing import Tuple, Optional
from pathlib import Path


class SDNQModelQuantizer:
    """
    Quantize loaded ComfyUI models to SDNQ format.

    Takes any MODEL from ComfyUI (checkpoints, diffusers, etc.) and quantizes it
    using SDNQ. Saves as a diffusers-format model that can be loaded via SDNQ Model Loader.

    This approach leverages ComfyUI's existing model loading infrastructure - no need
    to reinvent checkpoint loading!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Any loaded model from ComfyUI (from any checkpoint/diffusers loader)"
                }),
                "quant_type": (["int8", "int6", "uint4", "float8_e4m3fn"], {
                    "default": "int8",
                    "tooltip": "Quantization level (int8=best quality, uint4=most VRAM savings)"
                }),
                "output_name": ("STRING", {
                    "default": "my-quantized-model",
                    "multiline": False,
                    "tooltip": "Name for the quantized model (saved to ComfyUI/models/diffusers/sdnq/)"
                }),
            },
            "optional": {
                "use_svd": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable SVD compression for better quality (slower quantization)"
                }),
                "svd_rank": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": "SVD rank (higher = better quality, larger size)"
                }),
                "group_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "tooltip": "Quantization group size (0=auto, 128=common)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "quantize_model"
    CATEGORY = "loaders/SDNQ"
    OUTPUT_NODE = True  # Saves to disk
    DESCRIPTION = "Quantize any loaded model to SDNQ format. Takes MODEL from any ComfyUI loader."

    def quantize_model(
        self,
        model,
        quant_type: str,
        output_name: str,
        use_svd: bool = False,
        svd_rank: int = 32,
        group_size: int = 0
    ) -> Tuple[str]:
        """
        Quantize a loaded model to SDNQ format.

        Args:
            model: ComfyUI MODEL object (from any loader)
            quant_type: Quantization type (int8, int6, uint4, float8_e4m3fn)
            output_name: Name for output model
            use_svd: Enable SVD compression
            svd_rank: SVD rank for compression
            group_size: Quantization group size (0=auto)

        Returns:
            Path to the saved quantized model
        """
        try:
            from sdnq.loader import save_sdnq_model
            from sdnq import SDNQConfig
        except ImportError as e:
            raise RuntimeError(
                "SDNQ is not installed or not accessible. "
                "Please ensure sdnq is installed: pip install git+https://github.com/Disty0/sdnq.git"
            ) from e

        # Validate inputs
        if not output_name or not output_name.strip():
            raise ValueError("output_name cannot be empty")

        # CRITICAL: Sanitize the output name to prevent path traversal.
        # os.path.basename strips all directory information, leaving only the filename.
        sanitized_name = os.path.basename(output_name.strip())
        if not sanitized_name or sanitized_name in (".", ".."):
            raise ValueError("Invalid output_name. Please use a valid filename.")

        # Determine output directory (ComfyUI models folder)
        try:
            import folder_paths
            # Try to get ComfyUI models folder
            models_dir = folder_paths.get_folder_paths("diffusers")
            if models_dir and len(models_dir) > 0:
                base_dir = Path(models_dir[0])
            else:
                # Fallback to default ComfyUI structure
                base_dir = Path(folder_paths.models_dir) / "diffusers"
        except Exception:
            # Ultimate fallback
            base_dir = Path("./models/diffusers")

        # Create sdnq subdirectory
        sdnq_dir = base_dir / "sdnq"
        sdnq_dir.mkdir(parents=True, exist_ok=True)

        output_path = sdnq_dir / sanitized_name

        # Check if output already exists
        if output_path.exists():
            raise ValueError(
                f"Output model already exists: {output_path}\n"
                f"Please choose a different name or delete the existing model."
            )

        print(f"\n{'='*60}")
        print(f"SDNQ Model Quantizer")
        print(f"{'='*60}")
        print(f"Input: ComfyUI MODEL")
        print(f"Quantization: {quant_type}")
        print(f"SVD: {'Enabled' if use_svd else 'Disabled'}")
        if use_svd:
            print(f"SVD Rank: {svd_rank}")
        print(f"Group Size: {group_size if group_size > 0 else 'Auto'}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        try:
            # Extract the actual model from ComfyUI's wrapper
            # ComfyUI models are typically wrapped in ModelPatcher
            if hasattr(model, 'model'):
                actual_model = model.model
            else:
                actual_model = model

            print("Extracting model for quantization...")

            # Build SDNQ configuration
            config = SDNQConfig(
                quant_type=quant_type,
                use_svd=use_svd,
                svd_rank=svd_rank if use_svd else None,
                group_size=group_size if group_size > 0 else None,
            )

            print(f"Quantizing model (this may take several minutes)...")
            print(f"Progress will be shown below:\n")

            # Quantize and save
            # Note: save_sdnq_model handles the quantization and saving
            save_sdnq_model(
                model=actual_model,
                save_path=str(output_path),
                config=config,
            )

            print(f"\n{'='*60}")
            print(f"✓ Quantization complete!")
            print(f"{'='*60}")
            print(f"Saved to: {output_path}")
            print(f"\nTo use this model:")
            print(f"1. Add SDNQ Model Loader node")
            print(f"2. Select '--Custom Model--' from dropdown")
            print(f"3. Enter path: {output_path}")
            print(f"{'='*60}\n")

            return (str(output_path),)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Quantization failed!")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            print(f"\nTroubleshooting:")
            print(f"1. Ensure the model is a supported diffusion model (FLUX, SD3, SDXL)")
            print(f"2. Check you have enough disk space")
            print(f"3. Try a different quantization level")
            print(f"4. Check SDNQ installation: pip install git+https://github.com/Disty0/sdnq.git")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to quantize model: {str(e)}") from e


# Export for V1 API
NODE_CLASS_MAPPINGS = {
    "SDNQModelQuantizer": SDNQModelQuantizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDNQModelQuantizer": "SDNQ Model Quantizer",
}
