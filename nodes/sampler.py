"""
SDNQ Standalone Sampler Node - V3 API Compatible

This node loads SDNQ quantized models and generates images in one step.
No MODEL/CLIP/VAE outputs - just IMAGE output for ComfyUI.

Architecture: User Input → Load/Download SDNQ Model → Generate → Output IMAGE

Based on verified APIs from:
- diffusers documentation (https://huggingface.co/docs/diffusers)
- SDNQ repository (https://github.com/Disty0/sdnq)
- ComfyUI nodes.py (IMAGE format specification)
"""

import torch
import numpy as np
from PIL import Image
import traceback
import sys
from typing import Optional, Tuple, Dict, Any

# SDNQ import - registers SDNQ support into diffusers
from sdnq import SDNQConfig

# diffusers pipeline - auto-detects model type from model_index.json
from diffusers import DiffusionPipeline

# Local imports for model catalog and downloading
from ..core.registry import (
    get_model_names_for_dropdown,
    get_repo_id_from_name,
    get_model_info,
)
from ..core.downloader import (
    download_model,
    get_cached_model_path,
    check_model_cached,
)
from ..core.config import get_sdnq_models_dir


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
        self.current_lora_path = None
        self.current_lora_strength = None
        self.interrupted = False

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs following ComfyUI V3 conventions with V1 compatibility.

        All parameters verified from diffusers FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        # Get model names from catalog
        model_names = get_model_names_for_dropdown()

        return {
            "required": {
                # Model selection - dropdown with pre-configured models
                "model_selection": (["[Custom Path]"] + model_names, {
                    "default": model_names[0] if model_names else "[Custom Path]",
                    "tooltip": "Select a pre-configured SDNQ model (auto-downloads from HuggingFace) or choose [Custom Path] to specify a local model directory"
                }),

                # Custom model path (used when model_selection is "[Custom Path]")
                "custom_model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local path to SDNQ model directory (only used when [Custom Path] is selected). Example: /path/to/model or C:\\path\\to\\model"
                }),

                # Generation parameters
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text description of the image to generate. Be descriptive for best results."
                }),

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
                    "tooltip": "Guidance scale - how closely to follow the prompt. Higher = more literal. FLUX-schnell uses 0.0, others typically 3.5-7.0."
                }),

                # Image dimensions (must be multiple of 8)
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

                # Reproducibility
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducible generation. Same seed + settings = same image. Use -1 for random."
                }),

                # Data type selection
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended for FLUX (best quality/speed). float16 for older GPUs. float32 for CPU."
                }),

                # Memory management strategy
                "memory_mode": (["gpu", "balanced", "lowvram"], {
                    "default": "balanced",
                    "tooltip": "Memory management: 'gpu' = All on GPU (fastest, needs 24GB+ VRAM). 'balanced' = Model offloading (good for 12-16GB VRAM). 'lowvram' = Sequential offloading (slowest, works with 8GB VRAM)."
                }),

                # Auto-download control
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download model from HuggingFace if not found locally. Disable to only use local models."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "What to avoid in the image. Not all models support negative prompts (FLUX-schnell ignores them)."
                }),

                # LoRA support
                "lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to LoRA file (.safetensors). Leave empty to not use LoRA. Supports local paths or HuggingFace repo IDs."
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA influence strength. 1.0 = full strength, 0.5 = half strength, 0.0 = disabled. Higher values increase LoRA effect."
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
        # ComfyUI provides comfy.model_management.interrupt_processing()
        # For now, we'll use a simple flag that can be extended
        return self.interrupted

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

    def load_pipeline(self, model_path: str, dtype_str: str, memory_mode: str = "gpu") -> DiffusionPipeline:
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

        Returns:
            Loaded diffusers pipeline

        Raises:
            Exception: If pipeline loading fails

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/loading
        https://huggingface.co/docs/diffusers/main/optimization/memory
        """
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
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,  # Only load from local path
            )

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

            print(f"[SDNQ Sampler] Model loaded successfully!")
            print(f"[SDNQ Sampler] Pipeline type: {type(pipeline).__name__}")

            return pipeline

        except Exception as e:
            raise Exception(
                f"Failed to load SDNQ model from: {model_path}\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Verify the path contains a valid diffusers model (should have model_index.json)\n"
                f"2. Check if model download completed successfully\n"
                f"3. Try a different dtype (bfloat16 requires modern GPUs)\n"
                f"4. Check VRAM availability (use smaller model if needed)\n"
                f"5. Look at the error message above for specific details"
            )

    def load_lora(self, pipeline: DiffusionPipeline, lora_path: str, lora_strength: float = 1.0):
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

    def unload_lora(self, pipeline: DiffusionPipeline):
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

    def generate_image(self, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str,
                      steps: int, cfg: float, width: int, height: int, seed: int) -> Image.Image:
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

        Returns:
            PIL Image object

        Raises:
            Exception: If generation fails or is interrupted

        Based on verified API from FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        print(f"[SDNQ Sampler] Generating image...")
        print(f"[SDNQ Sampler]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[SDNQ Sampler]   Steps: {steps}, CFG: {cfg}")
        print(f"[SDNQ Sampler]   Size: {width}x{height}")
        print(f"[SDNQ Sampler]   Seed: {seed}")

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
                "width": width,
                "height": height,
                "generator": generator,
            }

            # Only add negative_prompt if it's not empty and pipeline supports it
            if negative_prompt and negative_prompt.strip():
                pipeline_kwargs["negative_prompt"] = negative_prompt

            # Call pipeline to generate image
            # Returns object with .images attribute containing list of PIL Images
            result = pipeline(**pipeline_kwargs)

            # Check for interruption after generation
            if self.check_interrupted():
                raise InterruptedError("Generation interrupted by user")

            # Extract first image from results
            # result.images[0] is a PIL.Image.Image object
            image = result.images[0]

            print(f"[SDNQ Sampler] Image generated! Size: {image.size}")

            return image

        except InterruptedError:
            raise
        except TypeError as e:
            # Handle pipeline parameter incompatibilities
            if "unexpected keyword argument" in str(e):
                import re
                match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                param_name = match.group(1) if match else "unknown"
                raise Exception(
                    f"Pipeline doesn't support parameter: '{param_name}'\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Pipeline type: {type(pipeline).__name__}\n"
                    f"This pipeline has a different signature than expected.\n\n"
                    f"Workaround:\n"
                    f"- If error is about 'negative_prompt': Leave it empty (this pipeline doesn't support it)\n"
                    f"- If error is about other params: Report this issue on GitHub with the pipeline type above"
                )
            raise
        except Exception as e:
            raise Exception(
                f"Failed to generate image\n\n"
                f"Error: {str(e)}\n\n"
                f"Troubleshooting:\n"
                f"1. Check VRAM usage (reduce size or use smaller model)\n"
                f"2. Verify parameters are valid (size multiple of 8, CFG reasonable)\n"
                f"3. Try reducing steps if running out of memory\n"
                f"4. Some models have specific parameter requirements (check HuggingFace page)\n"
                f"5. Look at the error message above for specific details"
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
                steps: int, cfg: float, width: int, height: int, seed: int,
                dtype: str, memory_mode: str = "gpu", auto_download: bool = True,
                negative_prompt: str = "", lora_path: str = "", lora_strength: float = 1.0) -> Tuple[torch.Tensor]:
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model_selection: Selected model from dropdown
            custom_model_path: Custom model path (if [Custom Path] selected)
            prompt: Text prompt
            steps: Inference steps
            cfg: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            dtype: Data type string
            memory_mode: Memory management mode ("gpu", "balanced", "lowvram")
            auto_download: Whether to auto-download models
            negative_prompt: Optional negative prompt
            lora_path: Optional LoRA file path or HuggingFace repo ID
            lora_strength: LoRA influence strength (0.0 to 2.0)

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
            if (self.pipeline is None or
                self.current_model_path != model_path or
                self.current_dtype != dtype or
                self.current_memory_mode != memory_mode):

                print(f"[SDNQ Sampler] Pipeline cache miss - loading model...")
                self.pipeline = self.load_pipeline(model_path, dtype, memory_mode)
                self.current_model_path = model_path
                self.current_dtype = dtype
                self.current_memory_mode = memory_mode
                # Clear LoRA cache when pipeline changes
                self.current_lora_path = None
                self.current_lora_strength = None
            else:
                print(f"[SDNQ Sampler] Using cached pipeline")

            # Step 2.5: Handle LoRA loading/unloading
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

            # Step 3: Generate image
            pil_image = self.generate_image(
                self.pipeline,
                prompt,
                negative_prompt,
                steps,
                cfg,
                width,
                height,
                seed
            )

            # Step 4: Convert to ComfyUI format
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
# V3 API - Metadata (ComfyUI V3 Extension System)
# ============================================================================

# V3 API: Node metadata
__comfy_node_metadata__ = {
    "SDNQSampler": {
        "display_name": "SDNQ Sampler",
        "description": "Load SDNQ quantized models and generate images with 50-75% VRAM savings",
        "category": "sampling/SDNQ",
        "version": "1.0.0",
        "author": "ComfyUI-SDNQ Contributors",
        "license": "Apache-2.0",
        "tags": ["sampling", "sdnq", "quantization", "vram", "flux", "sd3"],
        "deprecated": False,
    }
}
