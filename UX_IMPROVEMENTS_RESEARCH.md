# UX Improvements Research (December 2025)

## User Requests

1. Reorder node more logically
2. Include a default basic negative prompt
3. Include samplers that diffusers natively/easily supports
4. Make LoRAs selectable from existing LoRAs in ComfyUI's standard lora folder

---

## 1. Scheduler Support for Different Model Types

### Research Question
What schedulers does diffusers support for SDXL and other traditional diffusion models (beyond FLUX)?

### Key Finding
**SDXL and traditional diffusion models support MANY schedulers**, unlike FLUX which only supports FlowMatchEulerDiscreteScheduler.

### Available Schedulers (Traditional Diffusion Models)

According to diffusers documentation and KarrasDiffusionSchedulers class:

1. **DDIMScheduler** - Denoising Diffusion Implicit Models
2. **DDPMScheduler** - Denoising Diffusion Probabilistic Models
3. **PNDMScheduler** - Pseudo Numerical Methods for Diffusion
4. **LMSDiscreteScheduler** - Linear Multistep Method
5. **EulerDiscreteScheduler** - Euler method (fast, stable)
6. **HeunDiscreteScheduler** - Second-order method
7. **EulerAncestralDiscreteScheduler** - Euler ancestral (adds noise)
8. **DPMSolverMultistepScheduler** - Fast high-order solver (very popular!)
9. **DPMSolverSinglestepScheduler** - Single-step variant
10. **KDPM2DiscreteScheduler** - Karras DPM2
11. **KDPM2AncestralDiscreteScheduler** - Karras DPM2 ancestral
12. **DEISMultistepScheduler** - Diffusion Exponential Integrator Sampler
13. **UniPCMultistepScheduler** - Unified Predictor-Corrector (fast, high quality)
14. **DPMSolverSDEScheduler** - Stochastic sampler from EDM paper
15. **EDMEulerScheduler** - EDM formulation of Euler

### Most Popular for SDXL (Based on Community Usage)

Top 5 recommended:
1. **DPMSolverMultistepScheduler** - Best balance of speed/quality (20-25 steps)
2. **UniPCMultistepScheduler** - Very fast, high quality
3. **EulerDiscreteScheduler** - Simple, reliable
4. **EulerAncestralDiscreteScheduler** - Good for creative results
5. **DDIMScheduler** - Classic, deterministic

### FLUX Models

**ONLY FlowMatchEulerDiscreteScheduler** works with FLUX/flow-based models.

### Compatibility Check

All these schedulers work with SDXL pipelines. You can verify compatibility programmatically:
```python
compatible = pipeline.scheduler.compatibles
```

### Sources

- [Load schedulers and models](https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers)
- [SDXL Scheduler Testing](https://github.com/tillo13/sample_schedulers)
- [Stable Diffusion Samplers Guide](https://stable-diffusion-art.com/samplers/)
- [ML Guide to Schedulers](https://blog.segmind.com/what-are-schedulers-in-stable-diffusion/)
- [diffusers scheduling_utils.py](https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/schedulers/scheduling_utils.py)

---

## 2. ComfyUI LoRA Folder Integration

### Research Question
How to access LoRAs from ComfyUI's standard lora folder?

### ComfyUI folder_paths API

ComfyUI provides a `folder_paths` module for accessing model directories.

**Function**: `folder_paths.get_filename_list(folder_name: str) -> list[str]`

**Usage for LoRAs**:
```python
import folder_paths

# Get list of available LoRA files
lora_files = folder_paths.get_filename_list("loras")
# Returns: ["lora1.safetensors", "lora2.safetensors", ...]

# Get folder paths
lora_paths = folder_paths.get_folder_paths("loras")
# Returns: ["/path/to/ComfyUI/models/loras", ...]
```

### Default LoRA Location

- **Primary**: `ComfyUI/models/loras/`
- **Extensions**: `.safetensors`, `.ckpt`, `.pt`, `.bin`
- **Subdirectories**: Supported (e.g., `loras/SDXL/`, `loras/FLUX/`)

### Extra Model Paths

Users can configure additional paths via `extra_model_paths.yaml`:
```yaml
comfyui:
    loras: /custom/path/to/loras
```

### Implementation Pattern

From existing ComfyUI nodes:
```python
import folder_paths

class MyLoraNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {})
            }
        }
```

### Sources

- [ComfyUI folder_paths.py](https://github.com/comfyanonymous/ComfyUI/blob/master/folder_paths.py)
- [ComfyUI Folder Structure](https://comfyui-wiki.com/en/interface/files)
- [LoRA Installation Guide](https://comfyui-wiki.com/en/install/install-models/install-lora)
- [folder_paths extra_model_paths issue](https://github.com/comfyanonymous/ComfyUI/issues/6039)

---

## 3. Default Negative Prompt

### Research Question
What's a good universal negative prompt for quality improvement?

### Recommended Default

Based on common practices and SDXL/FLUX usage patterns:

```
"blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature"
```

### Rationale

- **blurry, low quality**: Improves overall sharpness
- **distorted, deformed**: Prevents structural artifacts
- **ugly, bad anatomy**: Helps with realistic generation
- **bad hands**: Common problem area in image generation
- **text, watermark, signature**: Prevents unwanted text artifacts

### Model-Specific Notes

- **FLUX-schnell**: Ignores negative prompts (cfg=0.0)
- **FLUX-dev**: Uses negative prompts (cfg=3.5-7.0)
- **SDXL**: Benefits significantly from negative prompts

---

## 4. Logical Parameter Ordering

### Current Organization Issues

Parameters are scattered across required/optional without clear grouping.

### Proposed Logical Grouping

**Group 1: Model Selection** (What to use)
1. model_selection
2. custom_model_path

**Group 2: Generation Prompts** (What to create)
3. prompt
4. negative_prompt

**Group 3: Generation Settings** (How to create)
5. steps
6. cfg
7. width
8. height
9. seed
10. scheduler

**Group 4: Model Configuration** (Technical settings)
11. dtype
12. memory_mode
13. auto_download

**Group 5: Enhancements** (Optional improvements)
14. lora_selection
15. lora_strength

### Rationale

1. **Top to bottom workflow**: Select model → Write prompts → Configure generation → Tweak technical settings → Add enhancements
2. **Frequency of use**: Most-changed parameters at top (prompts, settings)
3. **Grouped by purpose**: Related parameters together
4. **Optional last**: Enhancements are truly optional and less frequently changed

---

## Implementation Plan

### 1. Import folder_paths
```python
import folder_paths
```

### 2. Add LoRA Dropdown
```python
# Get available LoRAs
lora_list = ["[None]"] + folder_paths.get_filename_list("loras")

"lora_selection": (lora_list, {
    "default": "[None]",
    "tooltip": "Select LoRA from ComfyUI loras folder, or choose [Custom Path] for manual path"
})
```

### 3. Expand Scheduler List
```python
# Separate by model type
scheduler_options = [
    # Flow-based (FLUX, SD3)
    "FlowMatchEulerDiscreteScheduler",
    # Traditional diffusion (SDXL, SD1.5)
    "DPMSolverMultistepScheduler",
    "UniPCMultistepScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "DDIMScheduler",
    # ... more options
]
```

### 4. Add Default Negative Prompt
```python
"negative_prompt": ("STRING", {
    "default": "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, text, watermark, signature",
    "multiline": True,
    "tooltip": "..."
})
```

### 5. Reorder INPUT_TYPES
Reorganize following the logical grouping above.

---

## Testing Plan

1. **LoRA dropdown**: Verify it populates with files from `ComfyUI/models/loras/`
2. **Scheduler with SDXL model**: Test DPMSolverMultistepScheduler with NoobAI-XL
3. **Scheduler with FLUX model**: Verify FlowMatchEulerDiscreteScheduler still works
4. **Default negative prompt**: Test generation with default value
5. **Parameter order**: Verify logical flow in ComfyUI UI

---

## Compatibility Notes

### Scheduler Compatibility Matrix

| Model Type | Works With | Doesn't Work With |
|------------|------------|-------------------|
| FLUX.1/FLUX.2 | FlowMatchEulerDiscreteScheduler | All traditional schedulers |
| SDXL | All traditional schedulers | Flow-match schedulers |
| SD3/SD3.5 | FlowMatchEulerDiscreteScheduler | All traditional schedulers |
| Qwen | FlowMatchEulerDiscreteScheduler | All traditional schedulers |

**Warning**: Using wrong scheduler type produces incorrect/corrupted images!

### Auto-Detection Strategy

Since this node supports multiple model types, we should:
1. Provide all compatible schedulers in dropdown
2. Add tooltip warning about compatibility
3. Default to model-appropriate scheduler:
   - FLUX → FlowMatchEulerDiscreteScheduler
   - SDXL → DPMSolverMultistepScheduler
4. Let user experiment if they want

**Note**: Cannot auto-detect model type from path/repo, so user must know which scheduler to use.
