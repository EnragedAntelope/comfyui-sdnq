# Standalone SDNQ Sampler Implementation Plan

**Date**: 2025-12-09
**Goal**: Create a working, self-contained SDNQ sampler node that generates images without relying on ComfyUI's MODEL/CLIP/VAE system.

---

## Design Overview

### Architecture

```
User Input → SDNQSampler Node → Image Output
```

**No intermediate MODEL/CLIP/VAE outputs** - everything happens inside the node.

### Node Specification

```python
class SDNQSampler:
    """
    Standalone SDNQ sampler that loads models and generates images in one step.

    Inputs:
    - model_selection: Dropdown of pre-configured SDNQ models
    - model_path: Text input for custom model paths
    - prompt: Text (multiline)
    - negative_prompt: Text (multiline)
    - width, height: Int
    - steps: Int (num_inference_steps)
    - cfg: Float (guidance_scale)
    - seed: Int
    - dtype: Dropdown (bfloat16, float16, float32)

    Output:
    - IMAGE: ComfyUI image tensor format
    """
```

---

## Implementation Steps

### Step 1: Archive Old Code ✅

Move broken wrapper-based code to `archive/` directory:
- `nodes/loader.py` → `archive/nodes/loader.py`
- `core/wrapper.py` → `archive/core/wrapper.py`
- Keep `core/config.py` (model catalog is still useful)
- Keep `core/downloader.py` (HF integration still useful)

**Why**: Keep old code for reference, but make clear it's not active.

### Step 2: Create Minimal Sampler Node ✅

File: `nodes/sampler.py`

**Key Requirements**:
1. NO assumptions about diffusers API - verify everything
2. NO complex error handling yet - get basic case working first
3. Clear comments explaining each step
4. Start with local model loading only (no HF downloads yet)

**Minimal Implementation**:
```python
import torch
from PIL import Image
import numpy as np

class SDNQSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "sampling/SDNQ"

    def generate(self, model_path, prompt, steps, cfg, width, height, seed, negative_prompt=""):
        # Implementation here
        pass
```

### Step 3: Implement Pipeline Loading

**Research First**: Verify the correct API by reading diffusers docs and SDNQ examples.

**Questions to Answer**:
1. What's the correct import for DiffusionPipeline?
2. What parameters does from_pretrained() accept?
3. How to set device properly?
4. How to enable CPU offload?
5. What dtype options are available?

**Implementation Pattern** (verify before using):
```python
import torch
from diffusers import DiffusionPipeline
from sdnq import SDNQConfig  # This import enables SDNQ support

def load_pipeline(model_path, dtype_str):
    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]

    # Load pipeline - VERIFY PARAMETERS
    pipeline = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,  # Start with local only
    )

    # Enable CPU offload - VERIFY THIS METHOD EXISTS
    pipeline.enable_model_cpu_offload()

    return pipeline
```

**Validation Steps**:
- [ ] Read diffusers DiffusionPipeline.from_pretrained() docs
- [ ] Verify enable_model_cpu_offload() exists and works
- [ ] Test with a real model path

### Step 4: Implement Image Generation

**Research First**: Verify the pipeline call API.

**Questions to Answer**:
1. What's the correct method signature for pipeline()?
2. What's returned? (images list, or .images attribute?)
3. How to set generator for seed?
4. What format is the image? (PIL, tensor, numpy?)

**Implementation Pattern** (verify before using):
```python
def generate_image(pipeline, prompt, negative_prompt, steps, cfg, width, height, seed):
    # Create generator for reproducibility
    # VERIFY: Does generator accept device parameter?
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate image
    # VERIFY: What parameters are accepted?
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
        generator=generator,
    )

    # Extract image
    # VERIFY: Is it result.images[0] or result[0] or something else?
    image = result.images[0]

    return image
```

**Validation Steps**:
- [ ] Read diffusers pipeline call documentation
- [ ] Verify return type (check FLUX.2 example from Disty's repo)
- [ ] Test with a real model

### Step 5: Implement PIL to ComfyUI Conversion

**Research First**: Find out ComfyUI's expected image format.

**Questions to Answer**:
1. What's ComfyUI's image tensor format? (shape, dtype, range)
2. Is it NHWC or NCHW?
3. What's the value range? (0-1, 0-255?)
4. Does it expect batch dimension?

**Research Method**:
1. Look at existing ComfyUI nodes that output IMAGE
2. Check ComfyUI documentation
3. Look at image_utils or similar modules

**Implementation Pattern** (verify before using):
```python
def pil_to_comfy_tensor(pil_image):
    """
    Convert PIL Image to ComfyUI tensor format.

    Expected format (VERIFY THIS):
    - Shape: [B, H, W, C] (batch, height, width, channels)
    - Dtype: torch.float32
    - Range: 0.0 to 1.0
    - Color: RGB
    """
    # Convert to numpy
    np_image = np.array(pil_image).astype(np.float32) / 255.0

    # Add batch dimension
    np_image = np_image[np.newaxis, :]  # [1, H, W, C]

    # Convert to tensor
    tensor = torch.from_numpy(np_image)

    return tensor
```

**Validation Steps**:
- [ ] Research ComfyUI image format
- [ ] Test by connecting output to PreviewImage node
- [ ] Verify image displays correctly

### Step 6: Put It All Together

Combine all pieces into the `generate()` method:

```python
def generate(self, model_path, prompt, steps, cfg, width, height, seed, negative_prompt=""):
    # Cache pipeline to avoid reloading every time
    # TODO: Add pipeline caching in later step

    # Load pipeline
    pipeline = self.load_pipeline(model_path, "bfloat16")

    # Generate image
    pil_image = self.generate_image(
        pipeline, prompt, negative_prompt,
        steps, cfg, width, height, seed
    )

    # Convert to ComfyUI format
    comfy_tensor = self.pil_to_comfy_tensor(pil_image)

    return (comfy_tensor,)
```

### Step 7: Update __init__.py

Replace old loader with new sampler:

```python
from .nodes.sampler import SDNQSampler

NODE_CLASS_MAPPINGS = {
    "SDNQSampler": SDNQSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDNQSampler": "SDNQ Sampler",
}
```

---

## Testing Strategy

### Test 1: Node Appears in ComfyUI

**Steps**:
1. Copy node pack to ComfyUI/custom_nodes/
2. Restart ComfyUI server
3. Check if "SDNQ Sampler" appears under "sampling/SDNQ" category

**Success**: Node visible in menu

### Test 2: Node Can Be Added to Workflow

**Steps**:
1. Add node to canvas
2. Verify all inputs appear correctly
3. Check default values

**Success**: Node adds without errors, inputs look correct

### Test 3: Load Local SDNQ Model

**Setup**:
1. Download a small SDNQ model manually (e.g., FLUX.1-schnell-qint8)
2. Place in a test directory
3. Note the full path

**Steps**:
1. Set model_path to full path
2. Set a simple prompt: "a cat"
3. Use default parameters
4. Connect to PreviewImage node
5. Run workflow

**Success**: Image generates without errors

### Test 4: Verify Image Quality

**Steps**:
1. Generate same prompt with different seeds
2. Check images are different (seed working)
3. Check images match prompt
4. Visual quality check

**Success**: Images look reasonable, seed works

### Test 5: Test Different Parameters

**Steps**:
1. Try different step counts (10, 25, 50)
2. Try different CFG values (3.5, 7.0, 12.0)
3. Try different sizes (512x512, 1024x1024)

**Success**: All parameters affect output as expected

---

## Error Handling Plan (Phase 2)

**NOT implementing yet** - get basic case working first. But plan for:

1. Model not found
2. Invalid model path
3. Out of memory
4. CUDA not available
5. Invalid dimensions (not multiple of 8)
6. Model type not supported

---

## Optimization Plan (Phase 3)

**NOT implementing yet** - but plan for:

1. Pipeline caching (don't reload same model)
2. Device selection (auto, cuda, cpu)
3. Dtype selection
4. use_quantized_matmul option
5. Batch generation
6. Progress callbacks

---

## Success Criteria

The standalone sampler is "done" when:

1. ✅ Node appears in ComfyUI
2. ✅ Can load local SDNQ model
3. ✅ Generates images successfully
4. ✅ Images display correctly in PreviewImage
5. ✅ All basic parameters work (prompt, steps, cfg, seed, size)
6. ✅ No errors or warnings during normal operation
7. ✅ context.md updated with lessons learned

---

## Non-Goals (Out of Scope for Now)

- ❌ HuggingFace auto-download (manual download for now)
- ❌ Model catalog dropdown (text input only)
- ❌ Advanced parameters (LoRA, ControlNet, etc.)
- ❌ Batch generation
- ❌ Video models
- ❌ Error recovery
- ❌ Pipeline caching

These can be added later once basic functionality is proven.

---

## Research Checklist

Before implementing each step, research and verify:

### For Pipeline Loading:
- [ ] Read diffusers.DiffusionPipeline documentation
- [ ] Check Disty's SDNQ examples for correct usage
- [ ] Verify parameter names and types

### For Image Generation:
- [ ] Read pipeline.__call__() documentation
- [ ] Verify return type structure
- [ ] Check FLUX.2 specific examples

### For ComfyUI Integration:
- [ ] Find ComfyUI IMAGE format specification
- [ ] Look at existing nodes that output IMAGE
- [ ] Test conversion with PreviewImage node

---

## Notes for Implementation

1. **Start Simple**: Get the absolute minimum working first
2. **Research First**: Never assume API behavior
3. **Test Incrementally**: Test each piece as you build it
4. **Document Everything**: Update context.md with findings
5. **No Optimization**: Don't add caching, batching, etc. yet
6. **Clear Errors**: If something fails, stop and investigate - don't work around it

---

## Update Log

- **2025-12-09 Initial**: Created plan based on ASSESSMENT.md Option A
