# Research Notes - Standalone Sampler Implementation

**Date**: 2025-12-09
**Purpose**: Document verified API information before implementation

---

## diffusers Pipeline API (VERIFIED)

### Loading Pipelines

**Method 1: Specific Pipeline Class** (Recommended for known model types)
```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "model-path",
    torch_dtype=torch.bfloat16  # or torch.float16, torch.float32
)
pipe.to("cuda")  # Move to device
```

**Method 2: Using enable_model_cpu_offload** (Memory efficient)
```python
pipe = FluxPipeline.from_pretrained("model-path", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Automatically manages device placement
```

**Method 3: DiffusionPipeline (Auto-detects model type)**
```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "model-path",
    torch_dtype=torch.bfloat16,
    device_map="cuda"  # Alternative to .to()
)
```

### Generating Images

**Standard Pipeline Call**:
```python
result = pipe(
    prompt="text prompt",
    negative_prompt="negative text",  # Optional
    num_inference_steps=25,
    guidance_scale=7.0,
    height=1024,
    width=1024,
    generator=torch.Generator(device="cuda").manual_seed(seed)
)

# Result has .images attribute containing list of PIL Images
image = result.images[0]  # PIL.Image.Image
```

**Return Type**: `result.images[0]` returns a PIL Image object

**Generator for Seed**:
```python
# Can specify device or not
generator = torch.Generator().manual_seed(seed)
# OR
generator = torch.Generator(device="cuda").manual_seed(seed)
```

### Available Parameters (from FLUX examples)

- `prompt`: str - Text description
- `negative_prompt`: str (optional) - What to avoid
- `num_inference_steps`: int - Denoising steps (more = better quality, slower)
- `guidance_scale`: float - How closely to follow prompt (0.0 = none for schnell, 3.5-7.0 typical)
- `height`: int - Image height (must be multiple of 8)
- `width`: int - Image width (must be multiple of 8)
- `generator`: torch.Generator - For reproducible generation

---

## ComfyUI IMAGE Format (VERIFIED)

### Tensor Specifications

**Shape**: `[N, H, W, C]` (NHWC format)
- N: Batch dimension (usually 1 for single image)
- H: Height in pixels
- W: Width in pixels
- C: Channels (3 for RGB, 4 with alpha)

**Dtype**: `torch.float32`

**Value Range**: `0.0 to 1.0` (normalized)
- 0.0 = black/minimum intensity
- 1.0 = white/maximum intensity

### Conversion from PIL Image

**From ComfyUI's LoadImage node**:
```python
import numpy as np
import torch

# Convert PIL to RGB
pil_image = pil_image.convert("RGB")

# Convert to numpy, normalize to 0-1
numpy_image = np.array(pil_image).astype(np.float32) / 255.0

# Convert to tensor and add batch dimension
tensor = torch.from_numpy(numpy_image)[None, :]  # [None, :] adds batch dim at front

# Result shape: [1, H, W, 3]
```

### Conversion back to PIL (for reference)

**From ComfyUI's SaveImage node**:
```python
# Clip and denormalize
numpy_image = 255.0 * tensor.cpu().numpy()
numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)

# Convert to PIL
from PIL import Image
pil_image = Image.fromarray(numpy_image[0])  # Remove batch dimension
```

---

## SDNQ Integration (VERIFIED)

### Import Pattern

```python
from sdnq import SDNQConfig
```

**Effect**: Importing `sdnq` registers SDNQ model support into diffusers and transformers. After this import, diffusers pipelines can automatically load SDNQ-quantized models.

### Loading SDNQ Models

**No special code needed** - just load normally:
```python
from sdnq import SDNQConfig  # Register SDNQ
from diffusers import FluxPipeline
import torch

# Load SDNQ model exactly like unquantized model
pipe = FluxPipeline.from_pretrained(
    "Disty0/FLUX.1-dev-qint8",  # SDNQ quantized model
    torch_dtype=torch.bfloat16
)
```

**SDNQ is auto-detected** from the model's config files. No explicit configuration needed for pre-quantized models.

### Optional: Quantized MatMul Optimization

```python
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available

# Apply quantized matmul if Triton available
if triton_is_available and torch.cuda.is_available():
    pipe.transformer = apply_sdnq_options_to_model(
        pipe.transformer,
        use_quantized_matmul=True
    )
```

**Note**: This is optional optimization. Model works without it, just slightly slower.

---

## Implementation Checklist

Based on verified information:

- [x] diffusers pipeline loading API
- [x] diffusers generation call signature
- [x] Return type from pipeline call (PIL Image)
- [x] ComfyUI IMAGE tensor format (NHWC, float32, 0-1)
- [x] PIL to ComfyUI tensor conversion
- [x] SDNQ import pattern
- [x] Generator for seed handling

---

## Sources

All information verified from official sources:

**diffusers API**:
- [FLUX Pipeline Documentation](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)
- [DiffusionPipeline Loading Guide](https://huggingface.co/docs/diffusers/en/using-diffusers/loading)
- [diffusers GitHub Examples](https://github.com/huggingface/diffusers)

**ComfyUI IMAGE Format**:
- [ComfyUI nodes.py](https://github.com/comfyanonymous/ComfyUI/blob/master/nodes.py) - LoadImage and SaveImage nodes

**SDNQ**:
- [SDNQ Repository](https://github.com/Disty0/sdnq)
- [Pre-quantized Models](https://huggingface.co/collections/Disty0/sdnq)

---

## Next Steps

With this verified information, I can now implement:
1. `nodes/sampler.py` - Using exact API patterns documented above
2. No assumptions or guesses - everything based on real code
3. Test incrementally with real model
