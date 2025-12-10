"""
SDNQ Model Loader Node

Main node for loading pre-quantized SDNQ models in ComfyUI workflows.
Supports both dropdown selection from catalog and custom repo IDs.
"""

import os
import sys
import torch
import gc
import subprocess
from typing import Tuple, Dict, Any, Optional

# Import SDNQ config to register quantization methods with diffusers
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available

import diffusers
from diffusers import DiffusionPipeline

# Import ComfyUI modules
import folder_paths

from ..core.config import get_dtype_from_string
from ..core.registry import get_model_names_for_dropdown, get_repo_id_from_name, get_model_info
from ..core.downloader import download_model, check_model_cached, get_cached_model_path
from ..core.wrapper import wrap_pipeline_components


def check_cpp_compiler_available() -> bool:
    """
    Check if C++ compiler is available (required for torch.compile on Windows).

    Returns:
        True if compiler is available, False otherwise
    """
    if sys.platform != "win32":
        # On Linux/Mac, gcc/clang usually available
        return True

    try:
        # Check if cl.exe (MSVC compiler) is available on Windows
        result = subprocess.run(
            ["cl"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
    def cleanup_resources(pipeline=None, force=True):
        """
        Cleanup pipeline resources to prevent torch compile state pollution.

        This is critical to prevent the "black image" bug where failed loads break
        subsequent ComfyUI workflows.

        Args:
            pipeline: Diffusers pipeline to delete
            force: Force aggressive cleanup (torch compile reset, gc)
        """
        try:
            if pipeline is not None:
                # Move pipeline to CPU before deletion to free VRAM
                try:
                    if hasattr(pipeline, 'to'):
                        pipeline.to('cpu')
                except:
                    pass
                del pipeline

            if force:
                # Force garbage collection multiple times to ensure cleanup
                gc.collect()
                gc.collect()

                # Clear CUDA cache and synchronize
                if torch.cuda.is_available():
                    try:
                        # Synchronize all CUDA operations first
                        torch.cuda.synchronize()
                        # Empty cache
                        torch.cuda.empty_cache()
                        # Reset peak memory stats
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as cuda_error:
                        print(f"Warning: CUDA cleanup error: {cuda_error}")

                # Reset torch dynamo (torch.compile) cache to prevent state pollution
                # This prevents the "black image" bug from torch compile failures
                try:
                    torch._dynamo.reset()
                except:
                    pass  # Not available in all torch versions

                # Clear any remaining references
                gc.collect()

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

        # Check if C++ compiler is available (needed for torch.compile)
        compiler_available = check_cpp_compiler_available()

        # Configure torch.compile error handling
        # SDNQ quantized models use torch.compile for weight dequantization optimization
        if not compiler_available:
            print("⚠ C++ compiler not detected - using fallback mode for SDNQ optimizations")
            print("  ✓ Model attention: SDPA (GPU-accelerated, no compiler needed)")
            print("  ✓ Weight dequantization: Eager mode (GPU, slightly slower)")
            print("  ✓ VRAM usage: Same as with compiler")
            print("  ⚠ Performance: ~10-20% slower than with compiler optimizations")

            # Suppress compilation errors for SDNQ's dequantization code
            # This allows graceful fallback to eager mode for weight dequantization
            # while still using SDPA for model attention (set via attn_implementation="sdpa")
            torch._dynamo.config.suppress_errors = True

            # Reduce error verbosity
            torch._dynamo.config.verbose = False

        # Pre-load cleanup to clear any leftover state from previous failed loads
        # This prevents hanging if a previous load left resources in a bad state
        print("Pre-load cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        try:
            torch._dynamo.reset()
        except:
            pass

        # Track resources for cleanup
        pipeline = None

        try:
            # Load pipeline with SDNQ support
            # The SDNQConfig import above registers SDNQ into diffusers
            # SDNQ pre-quantized models will be loaded with quantization preserved
            # DiffusionPipeline auto-detects the correct pipeline type from model_index.json
            # (T2I, I2I, I2V, T2V, multimodal, etc.)
            print("Loading SDNQ model pipeline...", flush=True)
            print("This may take a moment for large models...", flush=True)

            # Use SDPA (Scaled Dot Product Attention) for model attention layers
            # SDPA is GPU-accelerated and widely supported (PyTorch 2.0+)
            # Does NOT require C++ compiler (unlike torch.compile)
            # Much faster than eager mode for attention operations
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=is_local,
                attn_implementation="sdpa",  # Use SDPA for fast GPU attention (no compiler needed)
            )

            print(f"✓ Pipeline loading complete", flush=True)
            print(f"Pipeline loaded: {type(pipeline).__name__}")

            # Apply SDNQ optimizations to pipeline components (optional)
            # Note: Optimizations require Triton and C++ compiler (already checked above)
            if use_quantized_matmul and triton_is_available and compiler_available:
                print("Applying SDNQ optimizations to all quantized components...")
                try:
                    # Apply to transformer or unet component
                    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                        pipeline.transformer = apply_sdnq_options_to_model(
                            pipeline.transformer,
                            use_quantized_matmul=True
                        )
                        print("✓ Optimizations applied to transformer")

                    elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
                        pipeline.unet = apply_sdnq_options_to_model(
                            pipeline.unet,
                            use_quantized_matmul=True
                        )
                        print("✓ Optimizations applied to unet")

                    # Apply to text encoders (FLUX.2 has quantized text encoder)
                    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                        pipeline.text_encoder = apply_sdnq_options_to_model(
                            pipeline.text_encoder,
                            use_quantized_matmul=True
                        )
                        print("✓ Optimizations applied to text_encoder")

                    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                        pipeline.text_encoder_2 = apply_sdnq_options_to_model(
                            pipeline.text_encoder_2,
                            use_quantized_matmul=True
                        )
                        print("✓ Optimizations applied to text_encoder_2")

                except Exception as opt_error:
                    print(f"Warning: Could not apply optimizations: {opt_error}")
                    print("Model will still work with quantized weights, just without optimizations")
                    # Reset torch compile state to prevent pollution
                    try:
                        torch._dynamo.reset()
                        print("✓ Torch compile state reset after optimization failure")
                    except:
                        pass

            elif use_quantized_matmul and not compiler_available:
                print("⚠ C++ compiler not found - SDNQ optimizations disabled")
                print("  torch.compile requires Visual Studio Build Tools with C++ compiler on Windows")
                print("  Model will still use quantized weights (same memory savings)")
                print("  To enable optimizations:")
                print("    1. Install Visual Studio Build Tools 2019 or later")
                print("    2. Include 'Desktop development with C++' workload")
                print("    3. Add cl.exe to PATH (usually in: C:\\Program Files\\Microsoft Visual Studio\\...\\VC\\Tools\\MSVC\\...\\bin\\Hostx64\\x64)")

            elif use_quantized_matmul and not triton_is_available:
                print("Note: Triton not available - using standard SDNQ dequantization")

            # Get model type from registry if available
            model_type = None
            if model_info:
                model_type = model_info.get('type', None)

            # Wrap pipeline components for ComfyUI compatibility
            # This creates MODEL, CLIP, VAE objects that work directly with ComfyUI nodes
            print("Wrapping pipeline components for ComfyUI integration...")
            model, clip, vae = wrap_pipeline_components(pipeline, model_type=model_type)

            print(f"{'='*60}")
            print("✓ Model loaded successfully!")
            print(f"  MODEL type: {type(model).__name__}")
            print(f"  CLIP type: {type(clip).__name__}")
            print(f"  VAE type: {type(vae).__name__}")
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
                force=True  # Force torch dynamo reset and full cleanup
            )
            print("✓ Cleanup complete - ComfyUI session should remain stable")

            raise RuntimeError(f"Failed to load SDNQ model: {str(e)}") from e
