# Scheduler Research for FLUX Models (December 2025)

## Research Question
What schedulers does diffusers support for FLUX models as of diffusers>=0.36.0?

---

## Key Findings

### 1. FLUX Default Scheduler
**FluxPipeline** officially uses **FlowMatchEulerDiscreteScheduler** as its default and primary scheduler.

**Source**: [FluxPipeline source code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)

```python
def __init__(
    self,
    scheduler: FlowMatchEulerDiscreteScheduler,  # ← Required type
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    # ...
)
```

### 2. Available Flow-Match Schedulers in Diffusers

According to [GitHub Issue #9924](https://github.com/huggingface/diffusers/issues/9924):

The diffusers library has only **two** flow-based schedulers:
1. **FlowMatchEulerDiscreteScheduler** ✅ Works with FLUX
2. **FlowMatchHeunDiscreteScheduler** ❌ Has compatibility issues

**Quote from issue**: "There is an older implementation of FlowMatchHeunDiscreteScheduler, but its not updated to support required set_timesteps or mu inputs, so it cannot be used with newer DiT models like Flux.1"

### 3. Why Other Schedulers Don't Work

From [GitHub Issue #9924](https://github.com/huggingface/diffusers/issues/9924):

> "Advanced schedulers such as DDIM and DPM++ 2M do work with flow based models, but diffusers only includes FlowMatchEulerDiscreteScheduler and FlowMatchHeunDiscreteScheduler. **DPMSolverMultistepScheduler does not generate correct images with flow-based models.**"

**Reason**: Standard schedulers like these are "not designed for flow match derived sampling" in their standard form.

### 4. List of Standard Diffusers Schedulers (NOT compatible with FLUX)

From [diffusers documentation](https://github.com/ai-forever/diffusers-new/blob/main/docs/source/en/using-diffusers/schedulers.md):

Available for traditional diffusion models (SDXL, SD1.5, etc.):
1. PNDMScheduler
2. DPMSolverSDEScheduler
3. EulerDiscreteScheduler
4. LMSDiscreteScheduler
5. DDIMScheduler
6. DDPMScheduler
7. HeunDiscreteScheduler
8. DPMSolverMultistepScheduler
9. DEISMultistepScheduler
10. EulerAncestralDiscreteScheduler
11. UniPCMultistepScheduler
12. KDPM2DiscreteScheduler
13. DPMSolverSinglestepScheduler
14. KDPM2AncestralDiscreteScheduler

**These do NOT work with FLUX** - they are for traditional diffusion models.

### 5. How to Swap Schedulers (General Pattern)

From [diffusers documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers):

```python
# Method 1: Using from_config()
from diffusers import DiffusionPipeline, LMSDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained("model_id")
pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)

# Method 2: Check compatible schedulers
compatible_schedulers = pipeline.scheduler.compatibles
```

**BUT**: FlowMatchEulerDiscreteScheduler has `_compatibles = []` (empty list) - no other schedulers are officially compatible.

### 6. Community Efforts (NOT in Official Diffusers)

From [GitHub Discussion #9781](https://github.com/huggingface/diffusers/discussions/9781):

Some community members have created:
- Custom FlowMatchDPMSolverMultistepScheduler (experimental)
- Third-party wrappers to bypass scheduler checks
- Multi-step schedulers with 6 different samplers

**These are NOT available in official diffusers 0.36.0** and would require external dependencies.

---

## Conclusion for Implementation

### What to Expose in ComfyUI-SDNQ Node

**Option 1: No Scheduler Parameter** (Current State)
- Keep using default FlowMatchEulerDiscreteScheduler
- No configuration needed
- **Simplest and most reliable**

**Option 2: Expose FlowMatchEulerDiscreteScheduler Only**
- Add scheduler dropdown with single option: "FlowMatchEulerDiscreteScheduler (default)"
- Provides transparency but no actual choice
- Future-proof: can add more options when diffusers supports them

**Option 3: Advanced - Expose Both Flow-Match Schedulers**
- Include both FlowMatchEulerDiscreteScheduler and FlowMatchHeunDiscreteScheduler
- Add warning tooltip that Heun may not work with all models
- Risk: Heun scheduler may fail or produce bad results

### Recommendation: Option 2

Expose scheduler as a dropdown with **only FlowMatchEulerDiscreteScheduler** for now:
- Shows transparency (user knows what scheduler is being used)
- Future-proof (easy to add more when diffusers supports them)
- No risk of broken generations
- Matches user request: "whatever samplers and schedulers that diffusers easily or natively supports"

**Verdict**: FlowMatchEulerDiscreteScheduler is the ONLY scheduler that diffusers natively and easily supports for FLUX as of December 2025.

---

## Evidence-Based Decision

### User Request
> "whatever samplers and schedulers that diffusers easily or natively supports, i want those"

### Factual Answer
Diffusers **natively and easily supports** only **FlowMatchEulerDiscreteScheduler** for FLUX models in version 0.36.0 (December 2025).

- FlowMatchHeunDiscreteScheduler: NOT updated for FLUX compatibility
- All other schedulers: NOT designed for flow-based models, produce incorrect images

### Implementation Plan
1. Add `scheduler` parameter to node with dropdown
2. Default: "FlowMatchEulerDiscreteScheduler"
3. Only option: "FlowMatchEulerDiscreteScheduler" (for now)
4. Add helpful tooltip explaining this is FLUX's native scheduler
5. Code structure allows easy addition of more schedulers when diffusers adds support

---

## Sources

1. [FLUX Pipeline Source Code](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py) - Shows FlowMatchEulerDiscreteScheduler as required type
2. [GitHub Issue #9924: Can we get more schedulers for flow based models](https://github.com/huggingface/diffusers/issues/9924) - Explains why only 2 flow-match schedulers exist
3. [GitHub Discussion #9781: New Schedulers for Black Forest Flux pipelines](https://github.com/huggingface/diffusers/discussions/9781) - Community efforts for custom schedulers
4. [Diffusers Schedulers Documentation](https://github.com/ai-forever/diffusers-new/blob/main/docs/source/en/using-diffusers/schedulers.md) - Lists all available schedulers
5. [Load schedulers and models](https://huggingface.co/docs/diffusers/en/using-diffusers/schedulers) - How to swap schedulers
6. [FlowMatchEulerDiscreteScheduler Docs](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete) - API documentation

---

## Note About "Samplers"

The user asked about "samplers and schedulers". In diffusers terminology:
- **Scheduler** = The denoising algorithm (noise schedule and step calculation)
- **Sampler** = Often used interchangeably with scheduler in ComfyUI context

There is no separate "sampler" concept in diffusers - the scheduler IS the sampler. ComfyUI's KSampler is specific to ComfyUI's architecture and not applicable to standalone diffusers pipelines.
