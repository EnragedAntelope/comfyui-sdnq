"""
SDNQ Standalone Sampler Node

This node loads SDNQ quantized models and generates images in one step.
No MODEL/CLIP/VAE outputs - just IMAGE output for ComfyUI.

Architecture: User Input → Load SDNQ Model → Generate → Output IMAGE

Based on verified APIs from:
- diffusers documentation (https://huggingface.co/docs/diffusers)
- SDNQ repository (https://github.com/Disty0/sdnq)
- ComfyUI nodes.py (IMAGE format specification)
"""

import torch
import numpy as np
from PIL import Image

# SDNQ import - registers SDNQ support into diffusers
from sdnq import SDNQConfig

# diffusers pipeline - auto-detects model type from model_index.json
from diffusers import DiffusionPipeline


class SDNQSampler:
    """
    Standalone SDNQ sampler that loads quantized models and generates images.

    All-in-one node that handles:
    - Loading SDNQ models from local paths
    - Setting up generation parameters
    - Generating images with proper seeding
    - Converting output to ComfyUI IMAGE format
    """

    def __init__(self):
        """Initialize sampler with empty pipeline cache."""
        self.pipeline = None
        self.current_model_path = None
        self.current_dtype = None

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define node inputs following ComfyUI conventions.

        All parameters verified from diffusers FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        return {
            "required": {
                # Model loading
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),

                # Generation parameters
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 150,
                    "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                }),

                # Image dimensions (must be multiple of 8)
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                }),

                # Reproducibility
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),

                # Data type selection
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "sampling/SDNQ"

    def load_pipeline(self, model_path, dtype_str):
        """
        Load SDNQ model using diffusers pipeline.

        Uses DiffusionPipeline which auto-detects the correct pipeline class
        from the model's model_index.json file. This works with:
        - FLUX.1, FLUX.2
        - SD3, SD3.5, SDXL
        - Video models (CogVideoX, etc.)
        - Multimodal models (Z-Image, Qwen-Image, etc.)

        Args:
            model_path: Local path to SDNQ model directory
            dtype_str: String dtype ("bfloat16", "float16", "float32")

        Returns:
            Loaded diffusers pipeline

        Based on verified API from:
        https://huggingface.co/docs/diffusers/en/using-diffusers/loading
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

        # Load pipeline - DiffusionPipeline auto-detects model type
        # SDNQ quantization is automatically detected from model config
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            local_files_only=True,  # Only load from local path
        )

        # Enable CPU offload for memory efficiency
        # This automatically manages device placement (model components on GPU when needed)
        # Verified from FLUX examples: https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        pipeline.enable_model_cpu_offload()

        print(f"[SDNQ Sampler] Model loaded successfully!")
        print(f"[SDNQ Sampler] Pipeline type: {type(pipeline).__name__}")

        return pipeline

    def generate_image(self, pipeline, prompt, negative_prompt, steps, cfg, width, height, seed):
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

        Based on verified API from FLUX examples:
        https://huggingface.co/docs/diffusers/main/api/pipelines/flux
        """
        print(f"[SDNQ Sampler] Generating image...")
        print(f"[SDNQ Sampler]   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"[SDNQ Sampler]   Steps: {steps}, CFG: {cfg}")
        print(f"[SDNQ Sampler]   Size: {width}x{height}")
        print(f"[SDNQ Sampler]   Seed: {seed}")

        # Create generator for reproducible generation
        # Generator handles random sampling during denoising
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Call pipeline to generate image
        # Returns object with .images attribute containing list of PIL Images
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            generator=generator,
        )

        # Extract first image from results
        # result.images[0] is a PIL.Image.Image object
        image = result.images[0]

        print(f"[SDNQ Sampler] Image generated! Size: {image.size}")

        return image

    def pil_to_comfy_tensor(self, pil_image):
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

    def generate(self, model_path, prompt, steps, cfg, width, height, seed, dtype, negative_prompt=""):
        """
        Main generation function called by ComfyUI.

        This is the entry point when the node executes in a workflow.

        Args:
            model_path: Local path to SDNQ model
            prompt: Text prompt
            steps: Inference steps
            cfg: Guidance scale
            width: Image width
            height: Image height
            seed: Random seed
            dtype: Data type string
            negative_prompt: Optional negative prompt

        Returns:
            Tuple containing (IMAGE,) in ComfyUI format
        """
        print(f"\n{'='*60}")
        print(f"[SDNQ Sampler] Starting generation")
        print(f"{'='*60}\n")

        # Check if we need to reload the pipeline
        # Cache pipeline to avoid reloading on every generation
        if (self.pipeline is None or
            self.current_model_path != model_path or
            self.current_dtype != dtype):

            print(f"[SDNQ Sampler] Pipeline cache miss - loading model...")
            self.pipeline = self.load_pipeline(model_path, dtype)
            self.current_model_path = model_path
            self.current_dtype = dtype
        else:
            print(f"[SDNQ Sampler] Using cached pipeline")

        # Generate image using the pipeline
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

        # Convert PIL image to ComfyUI tensor format
        comfy_tensor = self.pil_to_comfy_tensor(pil_image)

        print(f"\n{'='*60}")
        print(f"[SDNQ Sampler] Generation complete!")
        print(f"{'='*60}\n")

        # Return as tuple (ComfyUI expects tuple of outputs)
        return (comfy_tensor,)


# Export for ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SDNQSampler": SDNQSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDNQSampler": "SDNQ Sampler",
}
