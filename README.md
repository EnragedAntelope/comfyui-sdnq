# ComfyUI-SDNQ

**Load and run SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

Run large models like FLUX.2, FLUX.1, SD3.5, Qwen-Image, and more on consumer hardware with significantly reduced VRAM requirements.

<img width="2315" height="1121" alt="image" src="https://github.com/user-attachments/assets/4011ab36-e197-4543-8d7f-cf90ee4dee82" />

## Features

- **All-in-one node** - Select model, enter prompt, generate
- **20+ pre-configured models** with auto-download from HuggingFace
- **50-75% VRAM savings** with SDNQ quantization
- **Memory modes**: GPU (fastest), balanced (12-16GB), lowvram (8GB)
- **LoRA support**, image editing, 14 schedulers
- **Performance options**: Triton acceleration, xFormers, VAE tiling

## Installation

### ComfyUI Manager (Recommended)
Search for "comfyui-sdnq" → Install → Restart ComfyUI

### Manual
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/EnragedAntelope/comfyui-sdnq.git
cd comfyui-sdnq && pip install -r requirements.txt
```

## Quick Start

1. Add **SDNQ Sampler** node (under `sampling/SDNQ`)
2. Select a model from dropdown (auto-downloads on first use)
3. Enter your prompt → Queue Prompt → Done!

**Hover over inputs for tooltips** - all parameters are documented in the UI.

## Models

30+ pre-quantized models available: FLUX.1, FLUX.2, Qwen-Image (including 2512 Dec update), Z-Image, GLM-Image, LTX-2 video, and more.

Browse all models: [Disty0's SDNQ Collection](https://huggingface.co/collections/Disty0/sdnq)

### Video Models (Experimental)

LTX-2 video models are now supported. Set `num_frames` > 1 for video generation. Output is a batch of images (frames) that can be connected to video export nodes.

## Performance

For best speed (30-80% faster), install Triton:
- **Linux**: `pip install triton`
- **Windows**: `pip install triton-windows`

Triton enables optimized quantized matmul operations. Enabled by default when available.

**Scheduler tip**: Use `FlowMatchEulerDiscreteScheduler` for FLUX/SD3/Qwen. Use `DPMSolverMultistepScheduler` for SDXL/SD1.5.

## Troubleshooting

**Model loading errors** → Update libraries:
```bash
pip install --upgrade transformers diffusers
```

**Newest models (FLUX.2-klein, GLM-Image, Qwen-Image-2512, LTX-2)** → Build diffusers from source:
```bash
pip install git+https://github.com/huggingface/diffusers.git
```
This ensures you have the latest pipeline support for cutting-edge models.

**Out of memory** → Try `balanced` or `lowvram` memory mode, or use uint4 models.

**Slow performance** → Install Triton (see above), or try `use_xformers=True`.

## Credits

**SDNQ by [Disty0](https://github.com/Disty0/sdnq)** - All quantization technology is developed and maintained by Disty0.

- [SDNQ Repository](https://github.com/Disty0/sdnq)
- [Pre-quantized Models](https://huggingface.co/collections/Disty0/sdnq)
