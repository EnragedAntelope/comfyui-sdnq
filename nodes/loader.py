"""
SDNQ Model Loader Node

Main node for loading pre-quantized SDNQ models in ComfyUI workflows.
Supports both dropdown selection from catalog and custom repo IDs.
"""

import os
import torch
import gc
from typing import Tuple, Dict, Any, Optional

# Import SDNQ config to register quantization methods with diffusers
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available

import diffusers
from diffusers import DiffusionPipeline

# Import ComfyUI modules for native model loading
import comfy.sd
import folder_paths

from ..core.config import get_dtype_from_string
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

    @staticmethod
    def cleanup_resources(pipeline=None, model_component=None, model_state_dict=None,
                          clip_data=None, vae_state_dict=None, force=True):
        """
        Comprehensive cleanup of resources to prevent torch compile state pollution.

        This is critical to prevent the "black image" bug where failed loads break
        subsequent ComfyUI workflows. Based on KJNodes approach to torch compile cleanup.

        Args:
            pipeline: Diffusers pipeline to delete
            model_component: Transformer/UNet component to delete
            model_state_dict: Model state dict to delete
            clip_data: CLIP state dicts to delete
            vae_state_dict: VAE state dict to delete
            force: Force aggressive cleanup (torch compile reset, gc)
        """
        try:
            # Delete references in reverse order of creation
            if vae_state_dict is not None:
                del vae_state_dict
            if clip_data is not None:
                del clip_data
            if model_state_dict is not None:
                del model_state_dict
            if model_component is not None:
                del model_component
            if pipeline is not None:
                # Ensure pipeline components are moved to CPU before deletion
                try:
                    if hasattr(pipeline, 'to'):
                        pipeline.to('cpu')
                except:
                    pass
                del pipeline

            if force:
                # Force garbage collection
                gc.collect()

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Synchronize to ensure cleanup completes
                    torch.cuda.synchronize()

                # Reset torch dynamo (torch.compile) cache to prevent state pollution
                # This prevents the "black image" bug from torch compile failures
                try:
                    torch._dynamo.reset()
                except:
                    pass  # Not available in all torch versions

        except Exception as cleanup_error:
            print(f"Warning: Error during resource cleanup: {cleanup_error}")
            # Don't raise - cleanup errors shouldn't break execution

    def load_model(
        self,
        model_selection: str,
        dtype: str,
        use_quantized_matmul: bool = True,
        custom_repo_or_path: str = "",
        device: str = "auto"
    ) -> Tuple:
        """
        Load an SDNQ quantized model and return ComfyUI-compatible components.

        Args:
            model_selection: Selected model from dropdown or "Custom Model"
            dtype: Data type for model weights
            use_quantized_matmul: Enable Triton quantized matmul optimization
            custom_repo_or_path: Custom repo ID or path (when using Custom Model)
            device: Device placement strategy

        Returns:
            Tuple of (MODEL, CLIP, VAE) objects compatible with ComfyUI

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
            print("SDNQ Model Loader")
            print("="*60)
            print(f"Selected: {model_selection}")
            if model_info:
                print(f"Type: {model_info['type']}")
                print(f"Quantization: {model_info['quant_level']}")

            # Check if already cached
            is_cached = check_model_cached(repo_id)
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

        # Convert dtype string to torch dtype
        torch_dtype = get_dtype_from_string(dtype)

        # Determine if loading from local path or HuggingFace
        is_local = os.path.exists(model_path)

        print(f"Source: {'Local cached path' if is_local else 'HuggingFace Hub'}")

        # Track resources for cleanup
        pipeline = None
        model_component = None
        model_state_dict = None
        clip_data = None
        vae_state_dict = None

        try:
            # Load pipeline with SDNQ support
            # The SDNQConfig import above registers SDNQ into diffusers
            # SDNQ pre-quantized models will be loaded with quantization preserved
            # DiffusionPipeline auto-detects the correct pipeline type from model_index.json
            # (T2I, I2I, I2V, T2V, multimodal, etc.)
            print("Loading SDNQ model pipeline...")

            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=is_local,
            )

            print(f"Pipeline loaded: {type(pipeline).__name__}")

            # Extract state dictionaries from pipeline components
            print("Extracting model components for ComfyUI integration...")

            # Get transformer/unet component
            # Note: Different pipeline types (T2I, I2I, I2V, T2V, multimodal) all use
            # transformer or unet architecture. DiffusionPipeline loads the correct type:
            # - FLUX.1/FLUX.2: FluxPipeline (text-to-image with optional image guidance)
            # - Qwen-Image-Edit: QwenImageEditPipeline (image editing, requires input image)
            # - Wan2.2: Video pipelines (I2V, T2V with temporal components)
            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                model_component = pipeline.transformer
                model_type = "transformer"
            elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                model_component = pipeline.unet
                model_type = "unet"
            else:
                raise RuntimeError("Pipeline missing transformer or unet component")

            print(f"Model component: {model_type}")

            # Extract state dictionary from model component
            # The quantized weights are preserved in the state_dict
            print("Extracting model state dictionary...")
            model_state_dict = model_component.state_dict()

            # DEBUG: Log state dict keys for troubleshooting
            print(f"\n{'─'*60}")
            print(f"DEBUG: State Dict Analysis")
            print(f"{'─'*60}")
            print(f"Total keys in state_dict: {len(model_state_dict.keys())}")
            sample_keys = list(model_state_dict.keys())[:10]
            print(f"Sample keys (first 10):")
            for i, key in enumerate(sample_keys, 1):
                tensor_shape = tuple(model_state_dict[key].shape) if hasattr(model_state_dict[key], 'shape') else "N/A"
                tensor_dtype = model_state_dict[key].dtype if hasattr(model_state_dict[key], 'dtype') else "N/A"
                print(f"  {i}. {key}")
                print(f"     Shape: {tensor_shape}, Dtype: {tensor_dtype}")
            print(f"{'─'*60}\n")

            # Convert state_dict keys if needed (diffusers → ComfyUI format)
            # For transformers, the keys should already be compatible
            # For UNet, may need remapping
            model_options = {"dtype": torch_dtype}

            # Load via ComfyUI's native diffusion model loader
            # This creates a proper ModelPatcher with latent_format attribute
            print("Loading via ComfyUI native model loader...")
            print("This step detects model architecture from state_dict keys...")
            model = comfy.sd.load_diffusion_model_state_dict(
                model_state_dict,
                model_options=model_options
            )
            print(f"✓ Model architecture detected: {type(model).__name__ if model else 'Unknown'}")

            # Apply SDNQ Triton optimizations to the model inside the ModelPatcher
            # This adds quantized matmul operations for faster inference
            # WARNING: This uses torch.compile which can pollute torch state on failure
            if use_quantized_matmul and triton_is_available:
                print("Applying SDNQ Triton quantized matmul optimization...")
                print("(This uses torch.compile - if errors occur, torch state will be reset)")
                try:
                    # ModelPatcher has a model attribute containing the actual BaseModel
                    # Save original reference in case we need to restore it
                    original_model = model.model
                    model.model = apply_sdnq_options_to_model(
                        model.model,
                        use_quantized_matmul=True
                    )
                    print("✓ Triton optimizations applied successfully")
                except Exception as opt_error:
                    print(f"Warning: Could not apply Triton optimizations: {opt_error}")
                    print("Model will still work with quantized weights, just without Triton acceleration")
                    # Restore original model if optimization failed
                    try:
                        model.model = original_model
                    except:
                        pass
                    # Reset torch compile state to prevent pollution
                    try:
                        torch._dynamo.reset()
                        print("✓ Torch compile state reset after optimization failure")
                    except:
                        pass
            elif use_quantized_matmul and not triton_is_available:
                print("Note: Triton not available - model uses quantized weights but not Triton kernels")

            # Extract CLIP components
            print("Extracting CLIP components...")
            clip_data = self._extract_clip_state_dicts(pipeline)

            if clip_data:
                embedding_dir = folder_paths.get_folder_paths("embeddings")[0] if folder_paths.get_folder_paths("embeddings") else None
                clip = comfy.sd.load_text_encoder_state_dicts(
                    clip_data,
                    embedding_directory=embedding_dir
                )
            else:
                print("Warning: No CLIP components found in pipeline")
                clip = None

            # Extract VAE
            print("Extracting VAE component...")
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                vae_state_dict = pipeline.vae.state_dict()
                vae = comfy.sd.VAE(sd=vae_state_dict)
            else:
                print("Warning: No VAE component found in pipeline")
                vae = None

            # Clean up pipeline and intermediate objects
            # Use non-forced cleanup since we succeeded
            print("Cleaning up intermediate resources...")
            self.cleanup_resources(
                pipeline=pipeline,
                model_component=model_component,
                model_state_dict=model_state_dict,
                clip_data=clip_data,
                vae_state_dict=vae_state_dict,
                force=False  # Normal cleanup, no torch dynamo reset needed
            )

            print(f"{'='*60}")
            print("✓ Model loaded successfully via ComfyUI native loaders!")
            print(f"{'='*60}\n")

            return (model, clip, vae)

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
            print(f"5. Check ComfyUI logs for detailed error messages")
            print(f"6. Review DEBUG state dict keys above for compatibility issues")
            print(f"{'='*60}\n")

            # CRITICAL: Aggressive cleanup to prevent torch compile state pollution
            # This prevents the "black image" bug where failed loads break other workflows
            print("Performing aggressive cleanup to prevent session corruption...")
            self.cleanup_resources(
                pipeline=pipeline,
                model_component=model_component,
                model_state_dict=model_state_dict,
                clip_data=clip_data,
                vae_state_dict=vae_state_dict,
                force=True  # Force torch dynamo reset and full cleanup
            )
            print("✓ Cleanup complete - ComfyUI session should remain stable")

            raise RuntimeError(f"Failed to load SDNQ model: {str(e)}") from e

    def _extract_clip_state_dicts(self, pipeline) -> Dict[str, Any]:
        """
        Extract CLIP text encoder state dictionaries from a diffusers pipeline.

        Args:
            pipeline: Diffusers pipeline object

        Returns:
            Dictionary with clip_l and/or clip_g state dicts, or None if no text encoders
        """
        clip_data = {}

        # Check for text_encoder (CLIP-L or similar)
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            clip_data['clip_l'] = pipeline.text_encoder.state_dict()

        # Check for text_encoder_2 (CLIP-G or T5)
        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            clip_data['clip_g'] = pipeline.text_encoder_2.state_dict()

        # Check for text_encoder_3 (some models have 3 encoders)
        if hasattr(pipeline, 'text_encoder_3') and pipeline.text_encoder_3 is not None:
            clip_data['t5xxl'] = pipeline.text_encoder_3.state_dict()

        return clip_data if clip_data else None
