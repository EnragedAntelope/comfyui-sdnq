"""
SDNQ Standalone Sampler Node - V3 API Compatible

This node loads SDNQ quantized models and generates images in one step.
No MODEL/CLIP/VAE outputs - just IMAGE output for ComfyUI.

Architecture: User Input → Load/Download SDNQ Model → Generate → Output IMAGE

Based on verified APIs from:
- diffusers documentation (https://huggingface.co/docs/diffusers)
- SDNQ repository (https://github.com/Disty0/sdnq)
- ComfyUI nodes.py (IMAGE format specification)

Performance Note: Heavy imports (sdnq, diffusers, huggingface_hub) are lazy-loaded
to minimize ComfyUI startup time. They are only imported when actually needed.
"""

import torch
import numpy as np
from PIL import Image
import traceback
import sys
import os
import warnings
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

# ComfyUI imports for LoRA folder access and interrupt handling
try:
    import folder_paths
    import comfy.model_management
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    comfy = None
    print("[SDNQ Sampler] Warning: ComfyUI modules not available - LoRA dropdown and interrupt handling will be disabled")

# ============================================================================
# LAZY IMPORT HELPERS
# ============================================================================
# Heavy imports (sdnq, diffusers, huggingface_hub) are loaded on first use
# to reduce ComfyUI startup time from ~15s to ~2-3s
# ============================================================================

# Module-level cache for lazy imports
_sdnq_initialized = False
_diffusers_pipeline_class = None
_scheduler_classes = None
_sdnq_apply_options = None
_triton_available = None

def _ensure_sdnq_initialized():
    """Initialize SDNQ (registers into diffusers). Called before loading models."""
    global _sdnq_initialized
    if not _sdnq_initialized:
        from sdnq import SDNQConfig  # noqa: F401 - import side effect registers SDNQ
        _sdnq_initialized = True

def _get_diffusers_pipeline_class():
    """Lazy load DiffusionPipeline class."""
    global _diffusers_pipeline_class
    if _diffusers_pipeline_class is None:
        from diffusers import DiffusionPipeline
        _diffusers_pipeline_class = DiffusionPipeline
    return _diffusers_pipeline_class

def _get_scheduler_classes():
    """Lazy load all scheduler classes."""
    global _scheduler_classes
    if _scheduler_classes is None:
        from diffusers.schedulers import (
            FlowMatchEulerDiscreteScheduler,
            DDIMScheduler,
            DDPMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            KDPM2DiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            DEISMultistepScheduler,
            UniPCMultistepScheduler,
        )
        _scheduler_classes = {
            "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "UniPCMultistepScheduler": UniPCMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "HeunDiscreteScheduler": HeunDiscreteScheduler,
            "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler,
            "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler,
            "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
            "DEISMultistepScheduler": DEISMultistepScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "DDPMScheduler": DDPMScheduler,
            "PNDMScheduler": PNDMScheduler,
        }
    return _scheduler_classes

def _get_sdnq_optimization_tools():
    """Lazy load SDNQ optimization tools."""
    global _sdnq_apply_options, _triton_available
    if _sdnq_apply_options is None:
        try:
            from sdnq.loader import apply_sdnq_options_to_model
            from sdnq.common import use_torch_compile as triton_is_available
            _sdnq_apply_options = apply_sdnq_options_to_model
            _triton_available = triton_is_available
        except ImportError:
            print("[SDNQ Sampler] Warning: Could not import SDNQ optimization tools. Quantized MatMul will be disabled.")
            _sdnq_apply_options = lambda model, **kwargs: model
            _triton_available = False
    return _sdnq_apply_options, _triton_available

# Local imports for model catalog (these are lightweight, no heavy dependencies)
from ..core.registry import (
    get_model_names_for_dropdown,
    get_repo_id_from_name,
    get_model_info,
)

def _lazy_import_downloader():
    """Lazy import downloader functions to avoid huggingface_hub import at startup."""
    from ..core.downloader import (
        download_model,
        get_cached_model_path,
        check_model_cached,
    )
    return download_model, get_cached_model_path, check_model_cached


class SDNQSampler:
    """
    Standalone SDNQ sampler that loads quantized models and generates images.

    All-in-one node that handles:
    - Model selection from pre-configured catalog OR custom paths
    - Auto-download from HuggingFace Hub
    - Loading SDNQ models from local paths
    - Setting up generation parameters
    - Generating images with proper seeding
    - Converting output to ComfyUI IMAGE format
    - Graceful error handling and interruption support

    ComfyUI V3 API Compatible with V1 backward compatibility.
    """

    def __init__(self):
        """Initialize sampler with empty pipeline cache."""
        self.pipeline = None
        self.current_model_path = None
        self.current_dtype = None
        self.current_memory_mode = None
        self.current_scheduler = None
        self.current_lora_path = None
        self.current_lora_strength = None
        # Performance optimization settings cache
        self.current_use_xformers = None
        self.current_enable_vae_tiling = None
        self.current_use_quantized_matmul = None
        self.current_use_torch_compile = None
        self.interrupted = False

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs following ComfyUI V3 conventions with V1 compatibility.

        Parameters organized logically:
        1. Model Selection (what to use)
        2. Generation Prompts (what to create)
        3. Generation Settings (how to create)
        4. Model Configuration (technical settings)
        5. Enhancements (optional improvements)

        All parameters verified from diffusers documentation.
        """
        # Get model names from catalog
        model_names = get_model_names_for_dropdown()

        # Get available LoRAs from ComfyUI loras folder
        lora_list = ["[None]", "[Custom Path]"]
        if COMFYUI_AVAILABLE:
            try:
                available_loras = folder_paths.get_filename_list("loras")
                lora_list.extend(available_loras)
            except Exception as e:
                print(f"[SDNQ Sampler] Warning: Could not load LoRA list: {e}")

        # Scheduler options
        # Flow-based schedulers ONLY work with FLUX/SD3/Qwen/Z-Image models
        # Traditional schedulers ONLY work with SDXL/SD1.5 models
        scheduler_list = [
            # Flow-based (for FLUX, SD3, Qwen, Z-Image)
            "FlowMatchEulerDiscreteScheduler",
            # Traditional diffusion (for SDXL, SD1.5) - Top 10 most popular
            "DPMSolverMultistepScheduler",
            "UniPCMultistepScheduler",
            "EulerDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "DDIMScheduler",
            "HeunDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "DPMSolverSinglestepScheduler",
            "DEISMultistepScheduler",
            "LMSDiscreteScheduler",
            "DDPMScheduler",
            "PNDMScheduler",
        ]

        return {
            "required": {
                # ============================================================
                # GROUP 1: MODEL SELECTION (What to use)
                # ============================================================

                "model_selection": (["[Custom Path]"] + model_names, {
                    "default": model_names[0] if model_names else "[Custom Path]",
                    "tooltip": "Select a pre-configured SDNQ model (auto-downloads from HuggingFace) or choose [Custom Path] to specify a local model directory"
                }),

                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local path to SDNQ model directory (only used when [Custom Path] is selected). Example: /path/to/model or C:\\path\\to\\model"
                }),

                # ============================================================
                # GROUP 2: GENERATION PROMPTS (What to create)
                # ============================================================

                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text description of the image to generate. Be descriptive for best results."
                }),

                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature",
                    "multiline": True,
                    "tooltip": "What to avoid in the image. Default includes common quality issues. Clear this for no negative prompt. Note: FLUX-schnell (cfg=0) ignores negative prompts."
                }),

                # ============================================================
                # GROUP 3: GENERATION SETTINGS (How to create)
                # ============================================================

                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 150,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More steps = better quality but slower. 20-30 is typical for most models."
                }),

                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Guidance scale - how closely to follow the prompt. Higher = more literal. FLUX-schnell uses 0.0, FLUX-dev uses 3.5-7.0, SDXL uses 7.0-9.0."
                }),

                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Image width in pixels. Must be multiple of 8. Larger = more VRAM usage. 1024 is standard for FLUX/SDXL."
                }),

                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Image height in pixels. Must be multiple of 8. Larger = more VRAM usage. 1024 is standard for FLUX/SDXL."
                }),

                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation. Same seed + settings = same image."
                }),

                "scheduler": (scheduler_list, {
                    "default": "DPMSolverMultistepScheduler",
                    "tooltip": "⚠️ IMPORTANT: Use FlowMatchEulerDiscreteScheduler for FLUX/SD3/Qwen/Z-Image. Use DPMSolver/Euler/UniPC for SDXL/SD1.5. Wrong scheduler = broken images!"
                }),

                # ============================================================
                # GROUP 4: MODEL CONFIGURATION (Technical settings)
                # ============================================================

                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended for FLUX (best quality/speed). float16 for older GPUs. float32 for CPU."
                }),

                "memory_mode": (["gpu", "balanced", "lowvram"], {
                    "default": "balanced",
                    "tooltip": "Memory management: 'gpu' = All on GPU (fastest, needs 24GB+ VRAM). 'balanced' = Model offloading (12-16GB VRAM). 'lowvram' = Sequential offloading (8GB VRAM, slowest)."
                }),

                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download model from HuggingFace if not found locally. Disable to only use local models."
                }),
            },
            "optional": {
                # ============================================================
                # GROUP 5: ENHANCEMENTS (Optional improvements)
                # ============================================================

                "lora_selection": (lora_list, {
                    "default": "[None]",
                    "tooltip": "Select LoRA from ComfyUI loras folder ([None] = disabled, [Custom Path] = use custom path below). LoRAs add styles or concepts to generation."
                }),

                "lora_custom_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom LoRA path or HuggingFace repo ID (only used when [Custom Path] is selected). Example: /path/to/lora.safetensors or username/lora-repo"
                }),

                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "LoRA influence strength. 1.0 = full strength, 0.5 = half strength, 0.0 = disabled. Negative values invert the LoRA effect. Range: -5.0 to +5.0."
                }),

                # ============================================================
                # PERFORMANCE OPTIMIZATIONS (Optional speedups)
                # ============================================================

                "use_quantized_matmul": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Triton quantized matmul for 30-80% speedup (varies by GPU). Linux: install 'triton'. Windows: install 'triton-windows' (pip install triton-windows). Auto-disabled if Triton unavailable."
                }),

                "use_xformers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable xFormers memory-efficient attention for 10-45% speedup. Works with all memory modes (gpu/balanced/lowvram). Auto-fallback to SDPA if xformers not installed or incompatible. Requires: pip install xformers"
                }),

                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VAE tiling for very large images (>1536px). Prevents out-of-memory errors on high resolutions. Minimal performance impact. Recommended for images >1536x1536."
                }),

                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Compile transformer with torch.compile for ~30% speedup after first run. First run has 30-60s overhead (cached after). Requires PyTorch 2.0+ and Triton. CONFLICTS with xFormers - but torch.compile + SDPA (default) is actually faster than xFormers!"
                }),

                # ============================================================
                # IMAGE INPUTS (For image editing models like Qwen-Image-Edit)
                # ============================================================

                "image1": ("IMAGE", {
                    "tooltip": "Optional source image for image editing models (Qwen-Image-Edit, ChronoEdit, etc.). Leave unconnected for text-to-image generation."
                }),

                "image2": ("IMAGE", {
                    "tooltip": "Optional second image for multi-image editing models (Qwen-Image-Edit-2509/2511). Not all models support multiple images."
                }),

                "image3": ("IMAGE", {
                    "tooltip": "Optional third image for multi-image editing models."
                }),

                "image4": ("IMAGE", {
                    "tooltip": "Optional fourth image for multi-image editing models."
                }),

                "image_resize": (["No Resize", "Small (512px)", "Medium (768px)", "Large (1024px)", "XL (1536px)"], {
                    "default": "No Resize",
                    "tooltip": "Resize input images before processing. Smaller = faster inference, less VRAM. 'No Resize' keeps original dimensions."
                }),
            }
        }

    # V3 API: Return type hints
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    # V1 API: Function name
    FUNCTION = "generate"

    # Category for node menu
    CATEGORY = "sampling/SDNQ"

    # V3 API: Output node (can save/display results)
    OUTPUT_NODE = False

    # V3 API: Node description
    DESCRIPTION = "Load SDNQ quantized models and generate images with 50-75% VRAM savings. Supports FLUX, SD3, SDXL, video models, and more."

    def check_interrupted(self):
        """Check if generation should be interrupted (ComfyUI interrupt support)."""
        if self.interrupted:
            return True
        # Check ComfyUI's global interrupt flag
        if COMFYUI_AVAILABLE:
            try:
                return comfy.model_management.processing_interrupted()
            except Exception:
                pass
        return False

    def _create_interrupt_callback(self):
        """Create a callback for diffusers pipeline that checks for interrupts each step."""
        def interrupt_callback(pipeline, step, timestep, callback_kwargs):
            if self.check_interrupted():
                # Set the pipeline's internal interrupt flag
                pipeline._interrupt = True
                print(f"[SDNQ Sampler] Interrupt detected at step {step}, stopping generation...")
            return callback_kwargs
        return interrupt_callback

    def _convert_comfyui_image_to_pil(self, image_tensor, resize_option: str = "No Resize") -> Optional[Image.Image]:
        """
        Convert ComfyUI IMAGE tensor to PIL Image with optional resizing.

        ComfyUI images are [B, H, W, C] float tensors in 0-1 range.

        Args:
            image_tensor: ComfyUI IMAGE tensor
            resize_option: Resize option string

        Returns:
            PIL Image or None if conversion fails
        """
        if image_tensor is None:
            return None

        try:
            # ComfyUI images are [B, H, W, C] - take first image from batch
            img_array = image_tensor[0].cpu().numpy()
            # Convert from 0-1 float to 0-255 uint8
            img_array = (img_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            # Resize if requested
            resize_dimensions = {
                "Small (512px)": 512,
                "Medium (768px)": 768,
                "Large (1024px)": 1024,
                "XL (1536px)": 1536,
            }

            max_dim = resize_dimensions.get(resize_option)
            if max_dim is not None:
                width, height = pil_image.size
                max_current = max(width, height)
                if max_current > max_dim:
                    scale = max_dim / max_current
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                    print(f"[SDNQ Sampler] Resized input image: {width}x{height} → {new_width}x{new_height}")

            return pil_image.convert("RGB")

        except Exception as e:
            print(f"[SDNQ Sampler] Warning: Failed to convert input image: {e}")
            return None

    def load_or_download_model(self, model_selection: str, custom_path: str, auto_download: bool) -> Tuple[str, bool]:
        """
        Load model from catalog or custom path, downloading if needed.

        Args:
            model_selection: Selected model from dropdown
            custom_path: Custom model path (if [Custom Path] selected)
            auto_download: Whether to auto-download from HuggingFace

        Returns:
            Tuple of (model_path: str, was_downloaded: bool)

        Raises:
            ValueError: If model not found and auto_download is False
            Exception: If download fails
        """
        # Check if using custom path
        if model_selection == "[Custom Path]":
            if not custom_path or custom_path.strip() == "":
                raise ValueError(
                    "Custom model path is empty. Please provide a valid path to a local SDNQ model directory, "
                    "or select a pre-configured model from the dropdown."
                )

            model_path = custom_path.strip()

            # Verify path exists
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Custom model path does not exist: {model_path}\n"
                    f"Please verify the path and try again."
                )

            # Verify it's a valid model directory (has model_index.json)
            if not os.path.exists(os.path.join(model_path, "model_index.json")):
                raise ValueError(
                    f"Invalid model directory: {model_path}\n"
                    f"Directory must contain model_index.json file. "
                    f"This should be a diffusers model directory."
                )

            print(f"[SDNQ Sampler] Using custom model path: {model_path}")
            return (model_path, False)

        # Using catalog model
        model_info = get_model_info(model_selection)
        if not model_info:
            raise ValueError(
                f"Model not found in catalog: {model_selection}\n"
                f"This may indicate an issue with the model registry. "
                f"Try selecting a different model or using [Custom Path]."
            )

        repo_id = model_info["repo_id"]
        print(f"[SDNQ Sampler] Selected model: {model_selection}")
        print(f"[SDNQ Sampler] Repository: {repo_id}")

        # Lazy import downloader to avoid huggingface_hub at startup
        download_model, get_cached_model_path, check_model_cached = _lazy_import_downloader()

        # Check if model already cached
        cached_path = get_cached_model_path(repo_id)
        if cached_path:
            print(f"[SDNQ Sampler] Found cached model at: {cached_path}")
            return (cached_path, False)

        # Model not cached - download if auto_download enabled
        if not auto_download:
            raise ValueError(
                f"Model not found locally: {model_selection} ({repo_id})\n\n"
                f"Options:\n"
                f"1. Enable 'auto_download' to download automatically from HuggingFace\n"
                f"2. Download manually using: huggingface-cli download {repo_id}\n"
                f"3. Select a different model that's already downloaded"
            )

        print(f"[SDNQ Sampler] Model not cached - downloading from HuggingFace...")
        print(f"[SDNQ Sampler] This may take a while (models are 5-20+ GB)")

        try:
            downloaded_path = download_model(repo_id)
            print(f"[SDNQ Sampler] Download complete: {downloaded_path}")
            return (downloaded_path, True)
        except Exception as e:
            raise Exception(
                f"Failed to download model {model_selection} ({repo_id})\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Check your internet connection\n"
                f"2. Verify HuggingFace Hub is accessible\n"
                f"3. Try downloading manually: huggingface-cli download {repo_id}\n"
                f"4. Check disk space (models are large)\n"
                f"5. If download was interrupted, try again - it will resume"
            )

    def load_pipeline(self, model_path: str, dtype_str: str, memory_mode: str = "gpu",
                     use_xformers: bool = False, enable_vae_tiling: bool = False,
                     use_quantized_matmul: bool = True, use_torch_compile: bool = False):
        """
        Load SDNQ model using diffusers pipeline.

        Uses DiffusionPipeline which auto-detects the correct pipeline class
        from the model's model_index.json file. This works with:
        - FLUX.1, FLUX.2
        - SD3, SD3.5, SDXL
        - Video models (CogVideoX, Wan, etc.)
        - Multimodal models (Z-Image, Qwen-Image, etc.)

        Args:
            model_path: Local path to SDNQ model directory
            dtype_str: String dtype ("bfloat16", "float16", "float32")
            memory_mode: Memory management strategy ("gpu", "balanced", "lowvram")
            use_xformers: Enable xFormers memory-efficient attention
            enable_vae_tiling: Enable VAE tiling for large images
            use_quantized_matmul: Enable Triton quantized matmul optimization
            use_torch_compile: Enable torch.compile for transformer (experimental)

        Returns:
            Loaded diffusers pipeline

        Raises:
            Exception: If pipeline loading fails

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/loading
        https://huggingface.co/docs/diffusers/main/optimization/memory
        """
        # Lazy import SDNQ and diffusers (heavy dependencies)
        _ensure_sdnq_initialized()
        DiffusionPipeline = _get_diffusers_pipeline_class()
        apply_sdnq_options_to_model, triton_is_available = _get_sdnq_optimization_tools()

        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype_str]

        print(f"[SDNQ Sampler] Loading model from: {model_path}")
        print(f"[SDNQ Sampler] Using dtype: {dtype_str} ({torch_dtype})")
        print(f"[SDNQ Sampler] Memory mode: {memory_mode}")

        try:
            # Load pipeline - DiffusionPipeline auto-detects model type
            # SDNQ quantization is automatically detected from model config
            # Note: Pipeline loads to CPU by default - we move to GPU below

            # Suppress torch_dtype deprecation warning from transformers components
            # The warning comes from transformers library (used for CLIP/T5 text encoders)
            # diffusers still uses torch_dtype as the official parameter in 0.36.x
            # See: https://github.com/huggingface/peft/issues/2835
            with warnings.catch_warnings():
                # Filter both message pattern and FutureWarning category for comprehensive suppression
                warnings.filterwarnings("ignore", message=".*torch_dtype.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    local_files_only=True,  # Only load from local path
                )

            print(f"[SDNQ Sampler] Model loaded successfully!")
            print(f"[SDNQ Sampler] Pipeline type: {type(pipeline).__name__}")

            # Apply SDNQ optimizations (Quantized MatMul)
            # This must be done BEFORE memory management moves things around
            if use_quantized_matmul:
                if triton_is_available and torch.cuda.is_available():
                    print(f"[SDNQ Sampler] Applying Triton Quantized MatMul optimizations...")
                    try:
                        # Apply to transformer (FLUX, SD3)
                        if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                            pipeline.transformer = apply_sdnq_options_to_model(
                                pipeline.transformer,
                                use_quantized_matmul=True
                            )
                            print("[SDNQ Sampler] ✓ Optimization applied to transformer")

                        # Apply to UNet (SDXL, SD1.5)
                        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                            pipeline.unet = apply_sdnq_options_to_model(
                                pipeline.unet,
                                use_quantized_matmul=True
                            )
                            print("[SDNQ Sampler] ✓ Optimization applied to UNet")

                        # NOTE: We intentionally DO NOT apply quantized_matmul to text_encoders
                        # VL models (Qwen, etc.) have text encoder dimensions that aren't multiples
                        # of 8 (e.g., 3420), which causes SDNQ int8 matmul to fail with:
                        # "mat2.size(1) must be a multiple of 8 for INT8 GEMM"
                        # Only transformer/unet components are safe for this optimization.

                    except Exception as e:
                        print(f"[SDNQ Sampler] ⚠️  Failed to apply optimizations: {e}")
                        print("[SDNQ Sampler] Continuing without optimizations...")
                else:
                    if not torch.cuda.is_available():
                        print("[SDNQ Sampler] ℹ️  Quantized MatMul requires CUDA. Optimization disabled.")
                    elif not triton_is_available:
                        print("[SDNQ Sampler] ℹ️  Triton not available. Install: 'pip install triton' (Linux) or 'pip install triton-windows' (Windows).")
            else:
                print("[SDNQ Sampler] Quantized MatMul optimization disabled.")

            # NOTE: torch.compile is applied AFTER model is on GPU (see below)
            # This avoids crashes on Windows with triton-windows during .to("cuda")

            # CRITICAL: Apply xFormers BEFORE memory management
            # xFormers must be enabled before CPU offloading is set up
            if use_xformers:
                try:
                    print(f"[SDNQ Sampler] Enabling xFormers memory-efficient attention...")
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("[SDNQ Sampler] ✓ xFormers memory-efficient attention enabled")

                except ModuleNotFoundError as e:
                    # xFormers package not installed
                    print(f"[SDNQ Sampler] ⚠️  xFormers not installed: {e}")
                    print("[SDNQ Sampler] Install with: pip install xformers")
                    print("[SDNQ Sampler] Falling back to SDPA (PyTorch 2.0+ default attention)")

                except ValueError as e:
                    # CUDA not available - xFormers requires GPU
                    print(f"[SDNQ Sampler] ⚠️  xFormers requires CUDA: {e}")
                    print("[SDNQ Sampler] Falling back to SDPA")

                except NotImplementedError as e:
                    # Model architecture doesn't support xFormers
                    print(f"[SDNQ Sampler] ℹ️  xFormers not supported for this model architecture")
                    print(f"[SDNQ Sampler] Details: {e}")
                    print("[SDNQ Sampler] Using SDPA instead (this is normal for some models)")

                except (RuntimeError, AttributeError) as e:
                    # Version incompatibility, dimension mismatch, or API changes
                    print(f"[SDNQ Sampler] ⚠️  xFormers compatibility issue: {type(e).__name__}")
                    print(f"[SDNQ Sampler] Error: {e}")
                    print("[SDNQ Sampler] This may indicate:")
                    print("[SDNQ Sampler]   - xFormers version mismatch with PyTorch/CUDA")
                    print("[SDNQ Sampler]   - GPU architecture incompatibility")
                    print("[SDNQ Sampler]   - Tensor dimension issues with this model")
                    print("[SDNQ Sampler] Try: pip install -U xformers --force-reinstall")
                    print("[SDNQ Sampler] Falling back to SDPA")

                except Exception as e:
                    # Unexpected error - log full details for debugging
                    print(f"[SDNQ Sampler] ⚠️  Unexpected xFormers error: {type(e).__name__}")
                    print(f"[SDNQ Sampler] Error message: {e}")
                    print("[SDNQ Sampler] Full traceback:")
                    traceback.print_exc()
                    print("[SDNQ Sampler] Falling back to SDPA")
            else:
                print("[SDNQ Sampler] Using SDPA (scaled dot product attention, default in PyTorch 2.0+)")

            # Apply memory management strategy
            # Based on: https://huggingface.co/docs/diffusers/main/optimization/memory
            if memory_mode == "gpu":
                # Full GPU mode: Fastest performance, needs 24GB+ VRAM
                # Load entire pipeline to GPU
                print(f"[SDNQ Sampler] Moving model to GPU (full GPU mode)...")
                pipeline.to("cuda")
                print(f"[SDNQ Sampler] ✓ Model loaded to GPU (all components on VRAM)")

            elif memory_mode == "balanced":
                # Model CPU offload: Good balance for 12-16GB VRAM
                # Moves whole models between CPU and GPU as needed
                print(f"[SDNQ Sampler] Enabling model CPU offload (balanced mode)...")
                pipeline.enable_model_cpu_offload()
                print(f"[SDNQ Sampler] ✓ Model offloading enabled (efficient VRAM usage)")

            elif memory_mode == "lowvram":
                # Sequential CPU offload: Maximum memory savings for 8GB VRAM
                # Slowest but works on limited VRAM
                print(f"[SDNQ Sampler] Enabling sequential CPU offload (low VRAM mode)...")
                pipeline.enable_sequential_cpu_offload()
                print(f"[SDNQ Sampler] ✓ Sequential offloading enabled (minimal VRAM usage)")

            # VAE tiling (works with all memory modes)
            if enable_vae_tiling:
                try:
                    pipeline.enable_vae_tiling()
                    print("[SDNQ Sampler] ✓ VAE tiling enabled")
                except Exception as e:
                    print(f"[SDNQ Sampler] ⚠️  VAE tiling failed: {e}")

            # Apply torch.compile optimization (EXPERIMENTAL)
            # IMPORTANT: Applied AFTER model is on GPU to avoid crashes on Windows
            # with triton-windows during .to("cuda") operations
            # Compiles transformer/unet for ~30% speedup after first run
            # NOTE: torch.compile + SDPA (default) is faster than xFormers per HuggingFace docs
            if use_torch_compile:
                # Check for xFormers conflict first
                if use_xformers:
                    print("[SDNQ Sampler] ⚠️  torch.compile conflicts with xFormers - skipping")
                    print("[SDNQ Sampler] ℹ️  Tip: torch.compile + SDPA is faster than xFormers")
                elif not torch.cuda.is_available():
                    print("[SDNQ Sampler] ℹ️  torch.compile requires CUDA. Skipping.")
                elif memory_mode != "gpu":
                    # torch.compile works best with full GPU mode
                    print("[SDNQ Sampler] ℹ️  torch.compile works best with memory_mode='gpu'")
                    print("[SDNQ Sampler] ℹ️  Skipping compilation for balanced/lowvram modes")
                else:
                    print("[SDNQ Sampler] Applying torch.compile (EXPERIMENTAL)...")
                    print("[SDNQ Sampler] ⚠️  First run will be slow (30-60s compilation)")

                    # Platform warnings
                    if sys.platform == "win32":
                        py_version = sys.version_info
                        if py_version >= (3, 13):
                            print(f"[SDNQ Sampler] ⚠️  Python {py_version.major}.{py_version.minor} + Windows + torch.compile is experimental")
                            print("[SDNQ Sampler] ⚠️  If crashes occur, disable torch.compile")

                    try:
                        # Check PyTorch version
                        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                        if torch_version < (2, 0):
                            print(f"[SDNQ Sampler] ⚠️  torch.compile requires PyTorch 2.0+, got {torch.__version__}")
                        else:
                            # Use 'reduce-overhead' mode for better stability
                            # 'max-autotune' can cause issues on some configurations
                            compile_mode = "reduce-overhead"

                            # Compile transformer (FLUX, SD3)
                            if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
                                pipeline.transformer = torch.compile(
                                    pipeline.transformer,
                                    backend="inductor",
                                    mode=compile_mode
                                )
                                print("[SDNQ Sampler] ✓ torch.compile applied to transformer")

                            # Compile UNet (SDXL, SD1.5)
                            if hasattr(pipeline, 'unet') and pipeline.unet is not None:
                                pipeline.unet = torch.compile(
                                    pipeline.unet,
                                    backend="inductor",
                                    mode=compile_mode
                                )
                                print("[SDNQ Sampler] ✓ torch.compile applied to UNet")

                    except Exception as e:
                        print(f"[SDNQ Sampler] ⚠️  torch.compile failed: {e}")
                        print("[SDNQ Sampler] Continuing without compilation")

            return pipeline

        except AttributeError as e:
            error_str = str(e)
            # Check for Qwen/VL config attribute errors (version compatibility issues)
            # Common patterns: 'Qwen2_5_VLConfig' has no attribute 'vision_start_token_id'
            # These typically indicate transformers version is too old for the model
            if "Config" in error_str and ("has no attribute" in error_str or "object has no attribute" in error_str):
                # Extract the config class name and missing attribute for clearer messaging
                config_match = error_str.split("'")[1] if "'" in error_str else "Config"
                attr_match = error_str.split("'")[-2] if error_str.count("'") >= 4 else "unknown"
                raise Exception(
                    f"Failed to load SDNQ model from: {model_path}\n\n"
                    f"Error: {error_str}\n\n"
                    f"⚠️  This is a transformers/diffusers version compatibility issue!\n\n"
                    f"The model requires a config attribute ('{attr_match}') that your installed\n"
                    f"transformers version doesn't support for '{config_match}'.\n\n"
                    f"FIX: Update to the latest transformers and diffusers:\n"
                    f"  pip install --upgrade transformers diffusers\n\n"
                    f"If that doesn't work, install from source:\n"
                    f"  pip install git+https://github.com/huggingface/transformers.git\n"
                    f"  pip install git+https://github.com/huggingface/diffusers.git\n\n"
                    f"Note: Qwen-Image and vision-language models often require the latest\n"
                    f"transformers versions for full support."
                )
            else:
                # Other AttributeError - provide general guidance
                raise Exception(
                    f"Failed to load SDNQ model from: {model_path}\n\n"
                    f"Error: {error_str}\n\n"
                    f"This appears to be a library compatibility issue.\n\n"
                    f"Troubleshooting:\n"
                    f"1. Update transformers: pip install --upgrade transformers\n"
                    f"2. Update diffusers: pip install --upgrade diffusers\n"
                    f"3. For Qwen/VL models, you may need the latest from source:\n"
                    f"   pip install git+https://github.com/huggingface/transformers.git\n"
                    f"4. Check the model's HuggingFace page for version requirements"
                )

        except Exception as e:
            error_str = str(e)
            # Check if this looks like a version/compatibility error
            version_indicators = [
                "does not recognize this architecture",
                "Unrecognized model type",
                "quantization method is gonna be supported",
                "not supported",
            ]
            is_version_error = any(indicator in error_str for indicator in version_indicators)

            if is_version_error:
                raise Exception(
                    f"Failed to load SDNQ model from: {model_path}\n\n"
                    f"Error: {error_str}\n\n"
                    f"⚠️  This appears to be a version compatibility issue!\n\n"
                    f"The model requires a newer version of transformers or diffusers.\n\n"
                    f"FIX: Update to the latest versions:\n"
                    f"  pip install --upgrade transformers diffusers\n\n"
                    f"For cutting-edge models, install from source:\n"
                    f"  pip install git+https://github.com/huggingface/transformers.git\n"
                    f"  pip install git+https://github.com/huggingface/diffusers.git"
                )

            raise Exception(
                f"Failed to load SDNQ model from: {model_path}\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify the path contains a valid diffusers model (should have model_index.json)\n"
                f"2. Check if model download completed successfully\n"
                f"3. Try a different dtype (bfloat16 requires modern GPUs)\n"
                f"4. Check VRAM availability (use smaller model if needed)\n"
                f"5. For Qwen/VL models, try: pip install --upgrade transformers diffusers\n"
                f"6. Look at the error message above for specific details"
            )

    def load_lora(self, pipeline, lora_path: str, lora_strength: float = 1.0):
        """
        Load LoRA weights into pipeline.

        Supports both local .safetensors files and HuggingFace repo IDs.

        Args:
            pipeline: Loaded diffusers pipeline
            lora_path: Path to LoRA file or HuggingFace repo ID
            lora_strength: LoRA influence strength (0.0 to 2.0)

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/api/loaders/lora
        https://huggingface.co/blog/lora-fast
        """
        import os

        if not lora_path or lora_path.strip() == "":
            print(f"[SDNQ Sampler] No LoRA specified, skipping...")
            return

        print(f"[SDNQ Sampler] Loading LoRA...")
        print(f"[SDNQ Sampler]   Path: {lora_path}")
        print(f"[SDNQ Sampler]   Strength: {lora_strength}")

        try:
            # Check if it's a local file or HuggingFace repo
            is_local_file = os.path.exists(lora_path) and os.path.isfile(lora_path)

            if is_local_file:
                # Local .safetensors file
                # Extract directory and filename
                lora_dir = os.path.dirname(lora_path)
                lora_file = os.path.basename(lora_path)

                pipeline.load_lora_weights(
                    lora_dir,
                    weight_name=lora_file,
                    adapter_name="lora"
                )
            else:
                # Assume it's a HuggingFace repo ID
                pipeline.load_lora_weights(
                    lora_path,
                    adapter_name="lora"
                )

            # Set LoRA strength
            if lora_strength != 1.0:
                pipeline.set_adapters(["lora"], adapter_weights=[lora_strength])
            else:
                pipeline.set_adapters(["lora"])

            print(f"[SDNQ Sampler] ✓ LoRA loaded successfully")

        except Exception as e:
            raise Exception(
                f"Failed to load LoRA\n\n"
                f"Error: {str(e)}\n\n"
                f"LoRA path: {lora_path}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify LoRA file exists (.safetensors format)\n"
                f"2. For HuggingFace repos, verify repo ID is correct\n"
                f"3. Ensure LoRA is compatible with the model architecture\n"
                f"4. Check if LoRA is for the correct model type (FLUX, SDXL, etc.)\n"
                f"5. Try with lora_strength=1.0 first"
            )

    def unload_lora(self, pipeline):
        """
        Unload LoRA weights from pipeline.

        Args:
            pipeline: Pipeline with loaded LoRA
        """
        try:
            if hasattr(pipeline, 'unload_lora_weights'):
                print(f"[SDNQ Sampler] Unloading previous LoRA...")
                pipeline.unload_lora_weights()
        except Exception as e:
            # Non-critical error, just log it
            print(f"[SDNQ Sampler] Warning: Failed to unload LoRA: {e}")

    def swap_scheduler(self, pipeline, scheduler_name: str):
        """
        Swap the pipeline's scheduler.

        Uses the from_config() pattern to create a new scheduler with the same
        configuration as the current scheduler but with different algorithm.

        Args:
            pipeline: Loaded diffusers pipeline
            scheduler_name: Name of scheduler to use

        Raises:
            Exception: If scheduler swap fails

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers
        """
        print(f"[SDNQ Sampler] Swapping scheduler to: {scheduler_name}")

        try:
            # Lazy load scheduler classes
            scheduler_map = _get_scheduler_classes()

            if scheduler_name not in scheduler_map:
                raise ValueError(
                    f"Unknown scheduler: {scheduler_name}\n"
                    f"Available schedulers: {list(scheduler_map.keys())}"
                )

            scheduler_class = scheduler_map[scheduler_name]

            # Swap scheduler using from_config pattern
            # This preserves scheduler configuration while changing the algorithm
            pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

            print(f"[SDNQ Sampler] ✓ Scheduler swapped successfully")

        except Exception as e:
            raise Exception(
                f"Failed to swap scheduler\n\n"
                f"Error: {str(e)}\n\n"
                f"Requested scheduler: {scheduler_name}\n\n"
                f"Troubleshooting:\n"
                f"1. Ensure scheduler is compatible with the model type\n"
                f"2. FLUX/SD3/Qwen/Z-Image: Use FlowMatchEulerDiscreteScheduler\n"
                f"3. SDXL/SD1.5: Use DPMSolver, Euler, UniPC, or DDIM\n"
                f"4. Wrong scheduler type will produce broken/corrupted images\n"
                f"5. Check diffusers version (requires >=0.36.0)"
            )

    def generate_image(self, pipeline, prompt: str, negative_prompt: str,
                      steps: int, cfg: float, width: int, height: int, seed: int,
                      source_images: Optional[list] = None) -> Image.Image:
        """
        Generate image using the loaded pipeline.

        Args:
            pipeline: Loaded diffusers pipeline
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (optional)
            steps: Number of inference steps
            cfg: Guidance scale (classifier-free guidance strength)
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            seed: Random seed for reproducibility
            source_images: Optional list of PIL Images for image editing (Qwen-Image-Edit, etc.)

        Returns:
            PIL Image object

        Raises:
            Exception: If generation fails or is interrupted

        Based on verified API from FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        is_img2img = source_images and len(source_images) > 0
        pipeline_name = type(pipeline).__name__
        is_qwen_pipeline = "Qwen" in pipeline_name or "Edit" in pipeline_name

        if is_img2img:
            mode = "image-to-image"
        elif is_qwen_pipeline:
            mode = "text-to-image (blank canvas)"
        else:
            mode = "text-to-image"

        print(f"[SDNQ Sampler] Generating image ({mode})...")
        print(f"[SDNQ Sampler]   Pipeline: {pipeline_name}")
        print(f"[SDNQ Sampler]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[SDNQ Sampler]   Steps: {steps}, CFG: {cfg}")
        print(f"[SDNQ Sampler]   Size: {width}x{height}")
        print(f"[SDNQ Sampler]   Seed: {seed}")
        if is_img2img:
            print(f"[SDNQ Sampler]   Source images: {len(source_images)}")

        # Check for interruption before starting
        if self.check_interrupted():
            raise InterruptedError("Generation interrupted by user")

        try:
            # Create generator for reproducible generation
            # Generator handles random sampling during denoising
            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Build pipeline call kwargs
            # Only include parameters that are supported by the specific pipeline
            # Different pipelines have different signatures (FLUX.2 doesn't accept negative_prompt)
            pipeline_kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
                "generator": generator,
                "callback_on_step_end": self._create_interrupt_callback(),
            }

            # Add image input for image editing pipelines (Qwen-Image-Edit, ChronoEdit, etc.)
            # If source_images provided, this is img2img - don't set width/height (use source size)
            # If no source_images, this is txt2img - set width/height
            # Note: pipeline_name and is_qwen_pipeline are computed earlier in this method

            if is_img2img:
                # For single image, pass directly; for multiple, pass as list
                if len(source_images) == 1:
                    pipeline_kwargs["image"] = source_images[0]
                else:
                    pipeline_kwargs["image"] = source_images
            elif is_qwen_pipeline:
                # Qwen/Edit pipelines require an image even for "T2I" mode
                # Create a blank white image of the requested size as a starting point
                # This allows the model to generate from scratch while satisfying the image requirement
                print(f"[SDNQ Sampler] ℹ️  {pipeline_name} requires an image input.")
                print(f"[SDNQ Sampler] Creating blank {width}x{height} image for T2I mode...")
                blank_image = Image.new("RGB", (width, height), color=(255, 255, 255))
                pipeline_kwargs["image"] = blank_image
                # Don't set width/height - let the pipeline use the image dimensions
            else:
                # Text-to-image: specify output dimensions
                pipeline_kwargs["width"] = width
                pipeline_kwargs["height"] = height

            # Only add negative_prompt if it's not empty
            # Will be automatically removed if pipeline doesn't support it
            if negative_prompt and negative_prompt.strip():
                pipeline_kwargs["negative_prompt"] = negative_prompt

            # Try calling pipeline with all parameters
            # If certain parameters are unsupported, retry without them
            try:
                result = pipeline(**pipeline_kwargs)
            except TypeError as e:
                error_str = str(e)
                import re
                match = re.search(r"unexpected keyword argument '(\w+)'", error_str)
                param_name = match.group(1) if match else None

                # Handle 'callback_on_step_end' not supported (older pipelines)
                if param_name == "callback_on_step_end":
                    print(f"[SDNQ Sampler] ⚠️  Pipeline {type(pipeline).__name__} doesn't support step callbacks - interrupt may be delayed")
                    if "callback_on_step_end" in pipeline_kwargs:
                        del pipeline_kwargs["callback_on_step_end"]
                    result = pipeline(**pipeline_kwargs)

                # Handle 'negative_prompt' not supported (e.g., FLUX.2, FLUX-schnell)
                elif param_name == "negative_prompt":
                    print(f"[SDNQ Sampler] ⚠️  Pipeline {type(pipeline).__name__} doesn't support negative_prompt - skipping it")
                    if "negative_prompt" in pipeline_kwargs:
                        del pipeline_kwargs["negative_prompt"]
                    result = pipeline(**pipeline_kwargs)

                # Handle 'image' not supported - fallback to text-to-image
                elif param_name == "image":
                    print(f"[SDNQ Sampler] ⚠️  Pipeline {type(pipeline).__name__} doesn't support image input - falling back to text-to-image mode")
                    if "image" in pipeline_kwargs:
                        del pipeline_kwargs["image"]
                    # Add width/height back since we're doing text-to-image now
                    pipeline_kwargs["width"] = width
                    pipeline_kwargs["height"] = height
                    result = pipeline(**pipeline_kwargs)

                else:
                    # Different TypeError - re-raise with helpful message
                    raise Exception(
                        f"Pipeline doesn't support parameter: '{param_name or 'unknown'}'\n\n"
                        f"Error: {error_str}\n\n"
                        f"Pipeline type: {type(pipeline).__name__}\n"
                        f"This pipeline has a different signature than expected.\n\n"
                        f"Please report this issue on GitHub with the pipeline type above."
                    )
            except AttributeError as e:
                # Handle various AttributeError scenarios during generation
                error_str = str(e)

                # Check for Qwen/VL config attribute errors (version compatibility)
                # e.g., 'Qwen2_5_VLConfig' has no attribute 'vision_start_token_id'
                if "Config" in error_str and ("has no attribute" in error_str or "object has no attribute" in error_str):
                    config_match = error_str.split("'")[1] if "'" in error_str else "Config"
                    attr_match = error_str.split("'")[-2] if error_str.count("'") >= 4 else "unknown"
                    raise Exception(
                        f"Failed to generate image\n\n"
                        f"Error: {error_str}\n\n"
                        f"⚠️  This is a transformers/diffusers version compatibility issue!\n\n"
                        f"The model requires config attribute '{attr_match}' that your installed\n"
                        f"transformers version doesn't support for '{config_match}'.\n\n"
                        f"FIX: Update to the latest transformers and diffusers:\n"
                        f"  pip install --upgrade transformers diffusers\n\n"
                        f"If that doesn't work, install from source:\n"
                        f"  pip install git+https://github.com/huggingface/transformers.git\n"
                        f"  pip install git+https://github.com/huggingface/diffusers.git\n\n"
                        f"Note: Qwen-Image and vision-language models often require the latest\n"
                        f"transformers versions for full support."
                    )

                # Check if this looks like a missing image error
                if "'NoneType' object" in error_str and ("size" in error_str or "shape" in error_str):
                    raise ValueError(
                        f"This model requires an image input!\n\n"
                        f"Pipeline: {pipeline_name}\n"
                        f"Error: {error_str}\n\n"
                        f"This appears to be an image-editing model.\n"
                        f"Please connect a LoadImage node to the 'image1' input.\n\n"
                        f"If you want text-to-image generation, try a different model\n"
                        f"or report this as a bug if the model should support T2I."
                    )

                # Other AttributeError - re-raise with helpful context
                raise Exception(
                    f"Failed to generate image\n\n"
                    f"Error: {error_str}\n\n"
                    f"This may be a library compatibility issue.\n\n"
                    f"Troubleshooting:\n"
                    f"1. Update transformers: pip install --upgrade transformers\n"
                    f"2. Update diffusers: pip install --upgrade diffusers\n"
                    f"3. Check the model's HuggingFace page for version requirements"
                )

            # Check for interruption after generation (catches both callback and manual interrupts)
            if self.check_interrupted():
                raise InterruptedError("Generation interrupted by user")

            # Check if result has images (may be empty if interrupted)
            if not hasattr(result, 'images') or not result.images:
                if self.check_interrupted():
                    raise InterruptedError("Generation interrupted by user")
                raise RuntimeError("Pipeline returned no images - generation may have failed silently")

            # Extract first image from results
            # result.images[0] is a PIL.Image.Image object
            image = result.images[0]

            print(f"[SDNQ Sampler] Image generated! Size: {image.size}")

            return image

        except InterruptedError:
            raise
        except Exception as e:
            error_str = str(e)
            # Don't double-wrap exceptions we already formatted
            if "Pipeline doesn't support parameter" in error_str:
                raise
            if "transformers/diffusers version compatibility" in error_str:
                raise
            if "library compatibility issue" in error_str:
                raise

            # Check for version-related errors
            version_indicators = [
                "does not recognize this architecture",
                "Unrecognized model type",
                "quantization method is gonna be supported",
                "not supported",
                "Config" if "has no attribute" in error_str else "",
            ]
            is_version_error = any(indicator and indicator in error_str for indicator in version_indicators)

            if is_version_error:
                raise Exception(
                    f"Failed to generate image\n\n"
                    f"Error: {error_str}\n\n"
                    f"⚠️  This appears to be a version compatibility issue!\n\n"
                    f"FIX: Update to the latest versions:\n"
                    f"  pip install --upgrade transformers diffusers\n\n"
                    f"For cutting-edge models (Qwen, VL), install from source:\n"
                    f"  pip install git+https://github.com/huggingface/transformers.git\n"
                    f"  pip install git+https://github.com/huggingface/diffusers.git"
                )

            # Other errors - provide troubleshooting
            raise Exception(
                f"Failed to generate image\n\n"
                f"Error: {error_str}\n\n"
                f"Troubleshooting:\n"
                f"1. Check VRAM usage (reduce size or use smaller model)\n"
                f"2. Verify parameters are valid (size multiple of 8, CFG reasonable)\n"
                f"3. Try reducing steps if running out of memory\n"
                f"4. Some models have specific parameter requirements (check HuggingFace page)\n"
                f"5. For Qwen/VL models, try: pip install --upgrade transformers diffusers\n"
                f"6. Look at the error message above for specific details"
            )

    def pil_to_comfy_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to ComfyUI IMAGE tensor format.

        ComfyUI IMAGE format (verified from nodes.py LoadImage node):
        - Shape: [N, H, W, C] (batch, height, width, channels)
        - Dtype: torch.float32
        - Range: 0.0 to 1.0 (normalized)
        - Color: RGB

        Args:
            pil_image: PIL.Image.Image object

        Returns:
            torch.Tensor in ComfyUI format [1, H, W, 3]

        Based on verified conversion from ComfyUI nodes.py:
        https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py
        """
        # Ensure image is RGB (no alpha channel)
        pil_image = pil_image.convert("RGB")

        # Convert to numpy array and normalize to 0-1 range
        # PIL images are uint8 (0-255), ComfyUI uses float32 (0.0-1.0)
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        # [H, W, C] -> [1, H, W, C]
        tensor = torch.from_numpy(numpy_image)[None, :]

        print(f"[SDNQ Sampler] Converted to ComfyUI tensor: shape={tensor.shape}, dtype={tensor.dtype}")

        return tensor

    def generate(self, model_selection: str, custom_model_path: str, prompt: str,
                negative_prompt: str, steps: int, cfg: float, width: int, height: int,
                seed: int, scheduler: str, dtype: str, memory_mode: str, auto_download: bool,
                lora_selection: str = "[None]", lora_custom_path: str = "", lora_strength: float = 1.0,
                use_xformers: bool = False, enable_vae_tiling: bool = False,
                use_quantized_matmul: bool = True, use_torch_compile: bool = False,
                image1=None, image2=None, image3=None, image4=None,
                image_resize: str = "No Resize") -> Tuple[torch.Tensor]:
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model_selection: Selected model from dropdown
            custom_model_path: Custom model path (if [Custom Path] selected)
            prompt: Text prompt
            negative_prompt: Negative prompt
            steps: Inference steps
            cfg: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            scheduler: Scheduler algorithm name
            dtype: Data type string
            memory_mode: Memory management mode ("gpu", "balanced", "lowvram")
            auto_download: Whether to auto-download models
            lora_selection: Selected LoRA from dropdown ([None], [Custom Path], or filename)
            lora_custom_path: Custom LoRA path (used when [Custom Path] selected)
            lora_strength: LoRA influence strength (-5.0 to +5.0)
            use_xformers: Enable xFormers memory-efficient attention (10-45% speedup)
            enable_vae_tiling: Enable VAE tiling for large images
            use_quantized_matmul: Enable Triton quantized matmul optimization
            use_torch_compile: Enable torch.compile for transformer (experimental, ~30% speedup)
            image1: Optional source image for image editing (ComfyUI IMAGE tensor)
            image2: Optional second image for multi-image editing
            image3: Optional third image for multi-image editing
            image4: Optional fourth image for multi-image editing
            image_resize: Resize option for input images

        Returns:
            Tuple containing (IMAGE,) in ComfyUI format

        Raises:
            ValueError: For invalid inputs
            FileNotFoundError: For missing models/paths
            Exception: For other errors during loading/generation
        """
        print(f"\n{'='*60}")
        print(f"[SDNQ Sampler] Starting generation")
        print(f"{'='*60}\n")

        self.interrupted = False

        try:
            # Step 1: Load or download model
            model_path, was_downloaded = self.load_or_download_model(
                model_selection,
                custom_model_path,
                auto_download
            )

            # Step 2: Load pipeline (with caching)
            # Check if we need to reload the pipeline
            # Cache is invalidated if any of these change:
            # - Model path, dtype, memory mode
            # - Performance optimization settings (xformers, vae_tiling, quantized_matmul, torch_compile)
            if (self.pipeline is None or
                self.current_model_path != model_path or
                self.current_dtype != dtype or
                self.current_memory_mode != memory_mode or
                self.current_use_xformers != use_xformers or
                self.current_enable_vae_tiling != enable_vae_tiling or
                self.current_use_quantized_matmul != use_quantized_matmul or
                self.current_use_torch_compile != use_torch_compile):

                print(f"[SDNQ Sampler] Pipeline cache miss - loading model...")
                self.pipeline = self.load_pipeline(
                    model_path, dtype, memory_mode,
                    use_xformers=use_xformers,
                    enable_vae_tiling=enable_vae_tiling,
                    use_quantized_matmul=use_quantized_matmul,
                    use_torch_compile=use_torch_compile
                )
                self.current_model_path = model_path
                self.current_dtype = dtype
                self.current_memory_mode = memory_mode
                self.current_use_xformers = use_xformers
                self.current_enable_vae_tiling = enable_vae_tiling
                self.current_use_quantized_matmul = use_quantized_matmul
                self.current_use_torch_compile = use_torch_compile
                # Clear LoRA and scheduler cache when pipeline changes
                self.current_lora_path = None
                self.current_lora_strength = None
                self.current_scheduler = None
            else:
                print(f"[SDNQ Sampler] Using cached pipeline")

            # Step 2.5: Handle LoRA loading/unloading
            # Resolve actual LoRA path from lora_selection and lora_custom_path
            lora_path = None
            if lora_selection == "[None]":
                lora_path = None
            elif lora_selection == "[Custom Path]":
                lora_path = lora_custom_path if lora_custom_path and lora_custom_path.strip() else None
            else:
                # User selected a LoRA from the dropdown
                # Build full path from ComfyUI loras folder
                if COMFYUI_AVAILABLE:
                    try:
                        lora_folders = folder_paths.get_folder_paths("loras")
                        if lora_folders:
                            # Try to find the file in lora folders
                            for lora_folder in lora_folders:
                                potential_path = os.path.join(lora_folder, lora_selection)
                                if os.path.exists(potential_path):
                                    lora_path = potential_path
                                    break
                            if not lora_path:
                                # Fallback: use first folder + filename
                                lora_path = os.path.join(lora_folders[0], lora_selection)
                    except Exception as e:
                        print(f"[SDNQ Sampler] Warning: Could not resolve LoRA path: {e}")
                        lora_path = lora_selection  # Try using it as-is

            # Check if LoRA configuration has changed
            lora_changed = (lora_path != self.current_lora_path or
                           lora_strength != self.current_lora_strength)

            if lora_path and lora_path.strip():
                # User wants to use LoRA
                if lora_changed:
                    print(f"[SDNQ Sampler] LoRA configuration changed - updating...")

                    # Unload previous LoRA if it exists
                    if self.current_lora_path:
                        self.unload_lora(self.pipeline)

                    # Load new LoRA
                    self.load_lora(self.pipeline, lora_path, lora_strength)
                    self.current_lora_path = lora_path
                    self.current_lora_strength = lora_strength
                else:
                    print(f"[SDNQ Sampler] Using cached LoRA: {lora_path}")
            else:
                # User doesn't want LoRA, but we have one loaded
                if self.current_lora_path:
                    print(f"[SDNQ Sampler] Unloading LoRA...")
                    self.unload_lora(self.pipeline)
                    self.current_lora_path = None
                    self.current_lora_strength = None

            # Step 2.6: Handle scheduler swap
            # Check if scheduler changed from cached value
            if scheduler != self.current_scheduler:
                print(f"[SDNQ Sampler] Scheduler changed - swapping to {scheduler}...")
                self.swap_scheduler(self.pipeline, scheduler)
                self.current_scheduler = scheduler
            else:
                if self.current_scheduler:
                    print(f"[SDNQ Sampler] Using cached scheduler: {scheduler}")

            # Step 3: Process optional source images for image editing
            source_images = []
            for idx, img_tensor in enumerate([image1, image2, image3, image4], start=1):
                if img_tensor is not None:
                    pil_img = self._convert_comfyui_image_to_pil(img_tensor, image_resize)
                    if pil_img:
                        source_images.append(pil_img)
                        print(f"[SDNQ Sampler] Added source image {idx}: {pil_img.size}")

            # Step 4: Generate image
            pil_image = self.generate_image(
                self.pipeline,
                prompt,
                negative_prompt,
                steps,
                cfg,
                width,
                height,
                seed,
                source_images=source_images if source_images else None
            )

            # Step 5: Convert to ComfyUI format
            comfy_tensor = self.pil_to_comfy_tensor(pil_image)

            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Generation complete!")
            print(f"{'='*60}\n")

            # Return as tuple (ComfyUI expects tuple of outputs)
            return (comfy_tensor,)

        except InterruptedError as e:
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Generation interrupted")
            print(f"{'='*60}\n")
            raise

        except (ValueError, FileNotFoundError) as e:
            # User errors - display message clearly
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Error: {str(e)}")
            print(f"{'='*60}\n")
            raise

        except Exception as e:
            # Unexpected errors - provide full traceback
            print(f"\n{'='*60}")
            print(f"[SDNQ Sampler] Unexpected error occurred")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            raise
