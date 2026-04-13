# CLAUDE.md - ComfyUI SDNQ Node Pack Development Guide

## Project Overview

You are developing **ComfyUI-SDNQ**, a custom node pack that enables loading and using SDNQ (SD.Next Quantization) models in ComfyUI. This provides significant VRAM savings (50-75%) while maintaining quality, enabling users to run large models like FLUX and SD3.5 on consumer hardware.

**Key Principle**: This integration requires NO monkeypatching of ComfyUI. SDNQ integrates cleanly through the `diffusers` library API.

---

## Critical First Steps

1. **Create context.md** immediately after starting. Update it continuously with:
   - What you've completed
   - Current blockers or issues
   - Lessons learned
   - Next steps / TODOs

2. **Read this entire file** before writing any code

3. **Test incrementally** - get basic loading working before adding features

---

## Architecture Decisions (DO NOT DEVIATE)

### Integration Approach
```python
# SDNQ registers into diffusers via import side-effect
from sdnq import SDNQConfig  # This single import enables SDNQ loading
import diffusers

# Pre-quantized models load transparently
pipe = diffusers.FluxPipeline.from_pretrained(
    "Disty0/FLUX.1-dev-qint8",
    torch_dtype=torch.bfloat16
)
```

### Model Storage
- Store SDNQ models in: `ComfyUI/models/diffusers/sdnq/`
- Support `extra_model_paths.yaml` for custom paths
- Use huggingface_hub for downloads to standard HF cache

### Node API Compatibility
- **Primary**: V1 API (NODE_CLASS_MAPPINGS) for broad compatibility
- **Secondary**: V3 API schema for future-proofing
- Both APIs should be supported from the same node classes

---

## Repository Structure

```
ComfyUI-SDNQ/
├── __init__.py                    # Entry point with dual V1/V3 support
├── nodes/
│   ├── __init__.py                # Node exports
│   ├── loader.py                  # SDNQModelLoader node
│   ├── quantizer.py               # SDNQQuantizer node (optional, Phase 2)
│   └── catalog.py                 # SDNQModelCatalog node (optional)
├── core/
│   ├── __init__.py
│   ├── registry.py                # Model registry & catalog
│   ├── downloader.py              # HuggingFace Hub integration
│   ├── wrapper.py                 # ComfyUI type wrappers (MODEL, CLIP, VAE)
│   └── config.py                  # Configuration helpers
├── requirements.txt               # Python dependencies
├── install.py                     # ComfyUI Manager install hook
├── pyproject.toml                 # Modern packaging
├── LICENSE                        # Apache 2.0
├── README.md                      # User documentation
├── CREDITS.md                     # Attribution to Disty0
└── context.md                     # YOUR RUNNING NOTES (create this!)
```

---

## Implementation Order

### Phase 1: Minimum Viable Product

**Goal**: Load a pre-quantized SDNQ model and output ComfyUI-compatible types

1. **Create project structure**
   - All directories and placeholder files
   - requirements.txt with dependencies
   - Basic __init__.py with empty NODE_CLASS_MAPPINGS

2. **Implement core/wrapper.py**
   - Create wrappers to convert diffusers pipeline components to ComfyUI types
   - Focus on MODEL, CLIP, VAE outputs
   - Study existing ComfyUI diffusers loaders for patterns

3. **Implement nodes/loader.py - SDNQModelLoader**
   - Input: model path (local file selection)
   - Input: dtype selection (bfloat16, float16)
   - Input: device options
   - Output: MODEL, CLIP, VAE
   - Start with local loading only

4. **Test with real model**
   - Download Disty0/FLUX.1-dev-qint8 manually to test folder
   - Verify node appears in ComfyUI
   - Verify outputs connect to standard ComfyUI nodes

### Phase 2: HuggingFace Integration

5. **Implement core/registry.py**
   - Hardcoded catalog of known SDNQ models from Disty0 collection
   - Model metadata (type, quant level, size, etc.)
   - Detection of locally installed models

6. **Implement core/downloader.py**
   - Use `huggingface_hub` for downloads
   - Progress callbacks for UI
   - Caching and resume support

7. **Update nodes/loader.py**
   - Add dropdown for model catalog
   - Add HuggingFace repo ID input for custom models
   - Auto-download on first use

### Phase 3: Advanced Features (Optional)

8. **Quantization node**
   - Convert existing checkpoints to SDNQ format
   - Use sdnq.loader.save_sdnq_model()

9. **V3 API schemas**
   - Add comfy_entrypoint() function
   - Define IO schemas with type hints

---

## Key Code Patterns

### Basic Node Structure (V1 API)
```python
class SDNQModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "dtype": (["bfloat16", "float16", "float32"],),
            },
            "optional": {
                "use_quantized_matmul": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "loaders/SDNQ"

    def load_model(self, model_path, dtype, use_quantized_matmul=True):
        # Implementation here
        return (model, clip, vae)

NODE_CLASS_MAPPINGS = {"SDNQModelLoader": SDNQModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"SDNQModelLoader": "SDNQ Model Loader"}
```

### SDNQ Loading Pattern
```python
import torch
from sdnq import SDNQConfig
from sdnq.loader import apply_sdnq_options_to_model
from sdnq.common import use_torch_compile as triton_is_available
import diffusers

def load_sdnq_model(model_path, dtype_str, use_quantized_matmul):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]
    
    # Load pipeline - SDNQ is auto-detected from model config
    pipe = diffusers.AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,  # For local models
    )
    
    # Apply quantized matmul optimization if requested
    if use_quantized_matmul and triton_is_available:
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, 
            use_quantized_matmul=True
        )
    
    pipe.enable_model_cpu_offload()
    
    return pipe
```

### ComfyUI Type Wrapping
```python
# Research how ComfyUI expects MODEL, CLIP, VAE types
# Look at:
# - comfy/model_management.py
# - comfy/sd.py
# - Existing diffusers loader nodes

# The wrapper needs to provide whatever interface ComfyUI's
# sampler nodes expect. This is the most research-intensive part.
```

---

## Dependencies

```txt
# requirements.txt
sdnq>=0.1.7
diffusers>=0.36.0
huggingface-hub>=0.20.0
safetensors>=0.4.0
accelerate>=0.25.0
hf-xet>=1.2.0
```

---

## Testing Strategy

1. **Unit Tests** (if time permits)
   - Test registry functions
   - Test wrapper compatibility

2. **Integration Tests** (REQUIRED)
   - Install node pack in ComfyUI
   - Verify node appears in menu
   - Load a real SDNQ model
   - Connect to KSampler
   - Generate an image

3. **Models to Test**
   - Disty0/FLUX.1-dev-qint8 (most common)
   - Any SDXL SDNQ model
   - One video model if time permits

---

## Common Pitfalls to Avoid

1. **Don't try to bypass diffusers** - SDNQ is designed to work through it
2. **Don't modify ComfyUI core files** - This should be a drop-in custom node
3. **Don't hardcode paths** - Use ComfyUI's folder_paths module
4. **Don't skip error handling** - Model loading has many failure modes
5. **Don't forget progress reporting** - Large model downloads need feedback
6. **Don't ignore dtype** - SDNQ models typically need bfloat16

---

## Resources

### SDNQ Documentation
- Repository: https://github.com/Disty0/sdnq
- Wiki: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
- Pre-quantized models: https://huggingface.co/collections/Disty0/sdnq

### ComfyUI Node Development
- Official docs: https://docs.comfy.org/custom-nodes/
- V3 Migration: https://docs.comfy.org/custom-nodes/v3_migration
- Example nodes: https://github.com/comfyanonymous/ComfyUI/tree/master/comfy_extras

### Existing Diffusers Loaders (study these!)
- ComfyUI-DiffusersLoader: https://github.com/Scorpinaus/ComfyUI-DiffusersLoader
- ComfyUI-Diffusers: (search in custom node list)

---

## Attribution Requirements

**IMPORTANT**: Credit Disty0 prominently in:
- README.md header
- CREDITS.md file
- Node category description
- Any published documentation

Sample credit text:
```
SDNQ (SD.Next Quantization) is developed by Disty0
Repository: https://github.com/Disty0/sdnq
Pre-quantized models: https://huggingface.co/collections/Disty0/sdnq
```

---

## Success Criteria

Your implementation is complete when:

1. ✅ Node appears in ComfyUI under "loaders/SDNQ" category
2. ✅ Can load local SDNQ models from models/diffusers/sdnq/
3. ✅ Outputs (MODEL, CLIP, VAE) connect to standard ComfyUI nodes
4. ✅ Can generate images using loaded model
5. ✅ Error messages are helpful when things fail
6. ✅ README.md has clear installation and usage instructions
7. ✅ context.md documents your journey and lessons learned

---

## When You Get Stuck

1. **Check context.md** - Have you documented the issue?
2. **Search existing loaders** - ComfyUI-DiffusersLoader solved similar problems
3. **Read SDNQ source** - The sdnq package is well-documented
4. **Simplify** - Get the minimum working before adding features
5. **Document** - Even failures are valuable information for context.md

Good luck! This is a valuable contribution to the ComfyUI ecosystem.
