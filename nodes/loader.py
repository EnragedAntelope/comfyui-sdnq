"""
SDNQ Model Loader Node

Main node for loading pre-quantized SDNQ models in ComfyUI workflows.
Supports both dropdown selection from catalog and custom repo IDs.
"""

import os
import torch
from typing import Tuple

# Import SDNQ config to register quantization methods with diffusers
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available

import diffusers

from ..core.wrapper import wrap_pipeline_components
from ..core.config import get_dtype_from_string, get_device_map
from ..core.registry import get_model_names_for_dropdown, get_repo_id_from_name, get_model_info
from ..core.downloader import download_model, check_model_cached, get_cached_model_path


class SDNQModelLoader:
    """
    Load SDNQ (SD.Next Quantization) quantized models.

    SDNQ provides 50-75% VRAM savings while maintaining quality,
    enabling large models like FLUX and SD3.5 on consumer hardware.

    Features:
    - Dropdown selection from pre-configured models
    - Automatic download from HuggingFace Hub
    - Custom repo ID support
    - Multiple quantization levels (int8, int6, uint4, etc.)
    - Optional Triton acceleration
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get model names for dropdown
        model_options = ["--Custom Model--"] + get_model_names_for_dropdown()

        return {
            "required": {
                "model_selection": (model_options, {
                    "default": model_options[1] if len(model_options) > 1 else model_options[0],
                    "tooltip": "Select a pre-configured SDNQ model (auto-downloads from HuggingFace)"
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data type for model weights (bfloat16 recommended)"
                }),
                "use_quantized_matmul": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Triton quantized matmul for faster inference (Linux/WSL only)"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload model to CPU RAM to save VRAM (reduces speed, saves 60-70% VRAM)"
                }),
            },
            "optional": {
                "custom_repo_or_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Custom HuggingFace repo ID or local path (only used if 'Custom Model' selected)"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "loaders/SDNQ"
    DESCRIPTION = "Load SDNQ quantized models with automatic downloads. Models by Disty0."

    def load_model(
        self,
        model_selection: str,
        dtype: str,
        use_quantized_matmul: bool = True,
        cpu_offload: bool = True,
        custom_repo_or_path: str = "",
        device: str = "auto"
    ) -> Tuple:
        """
        Load an SDNQ quantized model and return ComfyUI-compatible components.

        Args:
            model_selection: Selected model from dropdown or "Custom Model"
            dtype: Data type for model weights
            use_quantized_matmul: Enable Triton quantized matmul optimization
            cpu_offload: Enable model CPU offloading
            custom_repo_or_path: Custom repo ID or path (when using Custom Model)
            device: Device placement strategy

        Returns:
            Tuple of (MODEL, CLIP, VAE) wrappers compatible with ComfyUI

        Raises:
            ValueError: If model selection is invalid
            RuntimeError: If model loading fails
        """
        # Determine which model to load
        if model_selection == "--Custom Model--":
            if not custom_repo_or_path or custom_repo_or_path.strip() == "":
                raise ValueError(
                    "Custom Model selected but no repo ID or path provided.\n"
                    "Please enter a HuggingFace repo ID (e.g., Disty0/FLUX.1-dev-qint8)\n"
                    "or local path in the 'custom_repo_or_path' field."
                )
            model_path = custom_repo_or_path.strip()
            model_info = None
            print("\n" + "="*60)
            print("SDNQ Model Loader - Custom Model")
            print("="*60)
        else:
            # Get repo ID from catalog
            repo_id = get_repo_id_from_name(model_selection)
            if not repo_id:
                raise ValueError(f"Invalid model selection: {model_selection}")

            model_info = get_model_info(model_selection)
            model_path = repo_id

            print("\n" + "="*60)
            if is_cached:
                print(f"✓ Model is already cached")
                cached_path = get_cached_model_path(repo_id)
                if cached_path:
                    model_path = cached_path
            else:
                print(f"📥 Model will be downloaded from HuggingFace Hub...")
                # Download the model (handles caching automatically)
                model_path = download_model(repo_id)

        model_path = model_path.strip()

        print(f"Model Path: {model_path}")
        print(f"Dtype: {dtype}")
        print(f"Quantized MatMul: {use_quantized_matmul}")
        print(f"CPU Offload: {cpu_offload}")
        print(f"Device: {device}")

        # Convert dtype string to torch dtype
        torch_dtype = get_dtype_from_string(dtype)

        # Determine if loading from local path or HuggingFace
        is_local = os.path.exists(model_path)

        print(f"Source: {'Local cached path' if is_local else 'HuggingFace Hub'}")

        try:
            # Load pipeline with SDNQ support
            # The SDNQConfig import above registers SDNQ into diffusers
            # SDNQ handles all quantization automatically through diffusers integration
            print("Loading model pipeline...")
            print("Note: If the progress bar appears stuck, it is likely verifying files or downloading large chunks. Please wait.")

            # Monkeypatch/Alias Flux2 classes to Flux classes if they don't exist
            # This fixes the "AttributeError: module diffusers has no attribute Flux2Transformer2DModel"
            if not hasattr(diffusers, "Flux2Pipeline") and hasattr(diffusers, "FluxPipeline"):
                print("Patching diffusers: Aliasing Flux2Pipeline -> FluxPipeline")
                diffusers.Flux2Pipeline = diffusers.FluxPipeline
            
            if not hasattr(diffusers, "Flux2Transformer2DModel") and hasattr(diffusers, "FluxTransformer2DModel"):
                print("Patching diffusers: Aliasing Flux2Transformer2DModel -> FluxTransformer2DModel")
                diffusers.Flux2Transformer2DModel = diffusers.FluxTransformer2DModel

            # Use DiffusionPipeline to support custom pipelines like Flux2Pipeline
            try:
                pipeline = diffusers.DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    local_files_only=is_local,
                    trust_remote_code=True, # Required for custom pipelines like Flux 2
                )
            except AttributeError as e:
                # Fallback for missing Flux2Pipeline in diffusers module
                # This happens if model_index.json specifies Flux2Pipeline but it's not in the library
                if "Flux2Pipeline" in str(e):
                    print("Warning: Flux2Pipeline not found in diffusers, falling back to FluxPipeline...")
                    from diffusers import FluxPipeline
                    pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        local_files_only=is_local,
                    )
                else:
                    raise


            print(f"Pipeline loaded: {type(pipeline).__name__}")

            # Apply quantized matmul optimization if requested and available
            if use_quantized_matmul:
                if triton_is_available:
                    print("Applying Triton quantized matmul optimization...")
                    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                        pipeline.transformer = apply_sdnq_options_to_model(
                            pipeline.transformer,
                            use_quantized_matmul=True
                        )
                    elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                        pipeline.unet = apply_sdnq_options_to_model(
                            pipeline.unet,
                            use_quantized_matmul=True
                        )
                else:
                    print("Warning: Triton not available, skipping quantized matmul optimization")

            # Apply CPU offloading if requested
            if cpu_offload:
                print("Enabling model CPU offload...")
                pipeline.enable_model_cpu_offload()

            # Wrap pipeline components for ComfyUI compatibility
            print("Wrapping pipeline components for ComfyUI...")
            model_wrapper, clip_wrapper, vae_wrapper = wrap_pipeline_components(pipeline)

            print(f"{'='*60}")
            print("✓ Model loaded successfully!")
            print(f"{'='*60}\n")

            return (model_wrapper, clip_wrapper, vae_wrapper)

        except Exception as e:
            print(f"\n{'='*60}")
            print("✗ Model loading failed!")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            print(f"\nTroubleshooting:")
            print(f"1. Verify the model path is correct")
            print(f"2. For HuggingFace models, check internet connection")
            print(f"3. Ensure the model is SDNQ-quantized")
            print(f"4. Check that required dependencies are installed")
            print(f"{'='*60}\n")
            raise RuntimeError(f"Failed to load SDNQ model: {str(e)}") from e
