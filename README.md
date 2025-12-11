# ComfyUI-SDNQ

**Load and run SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables running [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models directly in ComfyUI. Run large models like FLUX.2, FLUX.1, SD3.5, and more on consumer hardware with significantly reduced VRAM requirements while maintaining quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.

---

## Features

- **🎨 Standalone Sampler**: All-in-one node - load model, generate images, done
- **📦 Model Catalog**: 30+ pre-configured SDNQ models with auto-download
- **💾 Smart Caching**: Download once, use forever
- **🚀 VRAM Savings**: 50-75% memory reduction with quantization
- **⚡ Performance Optimizations**: Optional xFormers, VAE tiling, SDPA (automatic)
- **🎯 LoRA Support**: Load LoRAs from ComfyUI loras folder
- **📅 Multi-Scheduler**: 14 schedulers (FLUX/SD3 flow-match + traditional diffusion)
- **🔧 Memory Modes**: GPU (fastest), balanced (12-16GB VRAM), lowvram (8GB VRAM)

---

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "SDNQ" in the manager
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/EnragedAntelope/comfyui-sdnq.git
cd comfyui-sdnq
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## Quick Start

1. Add **SDNQ Sampler** node (under `sampling/SDNQ`)
2. Select a model from dropdown (auto-downloads on first use)
3. Enter your prompt
4. Click Queue Prompt
5. Done! Image output connects directly to SaveImage

**Defaults are optimized** - select model, enter prompt, generate!

---

## Node Reference

### SDNQ Sampler

**Category**: `sampling/SDNQ`

**Main Parameters**:
- `model_selection`: Dropdown with 30+ pre-configured models
- `custom_model_path`: For local models or custom HuggingFace repos
- `prompt` / `negative_prompt`: What to create / what to avoid
- `steps`, `cfg`, `width`, `height`, `seed`: Standard generation controls
- `scheduler`: FlowMatchEulerDiscreteScheduler (FLUX/SD3) or traditional samplers

**Memory Management**:
- `memory_mode`:
  - `gpu` = Full GPU (fastest, 24GB+ VRAM required)
  - `balanced` = CPU offloading (12-16GB VRAM)
  - `lowvram` = Sequential offloading (8GB VRAM, slowest)
- `dtype`: bfloat16 (recommended), float16, or float32

**Performance Optimizations** (optional):
- `use_xformers`: 10-45% speedup (safe to try, auto-fallback to SDPA)
- `enable_vae_tiling`: For large images >1536px (prevents OOM)
- SDPA (Scaled Dot Product Attention): Always active - automatic PyTorch 2.0+ optimization

**LoRA Support**:
- `lora_selection`: Dropdown from ComfyUI loras folder
- `lora_custom_path`: Custom LoRA path or HuggingFace repo
- `lora_strength`: -5.0 to +5.0 (1.0 = full strength)

**Outputs**: `IMAGE` (connects to SaveImage, Preview, etc.)

---

## Available Models

30+ pre-configured models including:
- **FLUX**: FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, FLUX.1-Krea, FLUX.1-Kontext
- **Qwen**: Qwen-Image variants (Edit, Lightning, Turbo)
- **SD3/SDXL**: SD3-Medium, SD3.5-Large, NoobAI-XL variants
- **Others**: Z-Image-Turbo, Chroma1-HD, HunyuanImage3, Video models

Most available in uint4 (max VRAM savings) or int8 (best quality). Browse: https://huggingface.co/collections/Disty0/sdnq

---

## Performance Tips

**For All Memory Modes**:
- SDPA (Scaled Dot Product Attention) is always active - automatic PyTorch 2.0+ optimization
- Enable `use_xformers=True` for 10-45% additional speedup (safe to try)
- Use `enable_vae_tiling=True` for large images (>1536px) to prevent OOM

**Scheduler Selection**:
- FLUX/SD3/Qwen/Z-Image: Use `FlowMatchEulerDiscreteScheduler`
- SDXL/SD1.5: Use `DPMSolverMultistepScheduler`, `EulerDiscreteScheduler`, or `UniPCMultistepScheduler`
- Wrong scheduler = broken images!

---

## Model Storage

Downloaded models are stored in:
- **Location**: `ComfyUI/models/diffusers/sdnq/`
- **Format**: Standard diffusers format

Models are cached automatically - download once, use forever!

---

## Troubleshooting

### xFormers Not Working

If you see "xFormers not available" but have it installed:
- This is usually fine - the node automatically falls back to SDPA (PyTorch 2.0+ default)
- SDPA provides good performance without xFormers
- If xFormers is incompatible with your GPU/model, fallback is automatic

### Performance is Slow

**Balanced/lowvram modes**: Inherently slower due to CPU↔GPU data movement. Options:
- Enable `use_xformers=True` (10-45% speedup if compatible)
- SDPA is always active for automatic optimization
- Upgrade to more VRAM for full GPU mode
- Use smaller model (uint4 vs int8)

### Out of Memory

1. Use lower memory mode (gpu → balanced → lowvram)
2. Use more aggressive quantization (uint4 instead of int8)
3. Reduce resolution or batch size
4. Enable `enable_vae_tiling=True` for large images

### Model Loading Fails

1. Check internet connection (for auto-download)
2. Verify repo ID is correct for custom models
3. For local models, ensure path points to directory (not a file)
4. Check model is actually SDNQ-quantized (from Disty0's collection)

---

## Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Test with multiple model types
3. Update documentation for new features

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

This project integrates with [SDNQ by Disty0](https://github.com/Disty0/sdnq).

---

## Credits

### SDNQ - SD.Next Quantization Engine
- **Author**: Disty0
- **Repository**: https://github.com/Disty0/sdnq
- **Pre-quantized models**: https://huggingface.co/collections/Disty0/sdnq

This node pack provides ComfyUI integration for SDNQ. All quantization technology is developed and maintained by Disty0.
