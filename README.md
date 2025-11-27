# ComfyUI-SDNQ

**Load SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables loading [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models in ComfyUI workflows. Run large models like FLUX.2, FLUX.1, Qwen-Image, Z-Image, HunyuanImage3, and more on consumer hardware with significantly reduced VRAM requirements while maintaining image quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.

---

## Features

- **📦 Model Dropdown**: Select from pre-configured SDNQ models from Disty0's collection
- **⚡ Auto-Download**: Models download automatically from HuggingFace on first use
- **💾 Smart Caching**: Download once, use forever
- **🚀 VRAM Savings**: 50-75% memory reduction with quantization
- **🎨 Quality Maintained**: Minimal quality loss with high-quality quantization
- **🔌 Compatible**: Works with standard ComfyUI nodes (KSampler, VAE Decode, etc.)
- **🏃 Optional Optimizations**: Triton acceleration for faster inference
- **🛠️ Model Quantization**: Convert your own models to SDNQ format

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

### ⚠️ Critical: diffusers 0.36.0+ Required

This node pack requires **diffusers 0.36.0 or higher** for FLUX.2, Z-Image, video models, and multimodal support.

**If you see `ImportError: cannot import name 'DiffusionPipeline'` or similar errors:**

```bash
# Install latest diffusers from GitHub (0.36.0 not yet on PyPI)
pip install git+https://github.com/huggingface/diffusers.git

# Or upgrade when 0.36.0 is released:
pip install --upgrade diffusers>=0.36.0
```

**Important Changes in diffusers 0.36.0:**
- `AutoPipeline` class was removed
- This node pack now uses `DiffusionPipeline` which auto-detects all pipeline types
- Supports all model architectures: T2I, I2I, I2V, T2V, multimodal

---

## Quick Start

### 1. Basic Usage

1. Add the **SDNQ Model Loader** node (under `loaders/SDNQ`)
2. **Select a model** from the dropdown
3. **First use**: Model auto-downloads from HuggingFace (cached for future use)
4. Connect outputs:
   - `MODEL` → KSampler
   - `CLIP` → CLIP Text Encode
   - `VAE` → VAE Decode

**Defaults are optimized** - just select a model and go!

### 2. Custom Models

Select `--Custom Model--` from dropdown, then enter:
- **HuggingFace repo ID**: `Disty0/your-model-qint8`
- **Local path**: `/path/to/model`

### 3. Available Models (21+ Pre-Configured)

The dropdown includes:
- **FLUX Models**: FLUX.1-dev, FLUX.1-schnell, FLUX.2, FLUX.1-Krea, FLUX.1-Kontext
- **Qwen Models**: Qwen-Image, Qwen-Image-Lightning, Qwen-Image-Edit variants, Qwen3-VL-32B
- **Other Models**: Z-Image-Turbo, Chroma1-HD, ChronoEdit-14B, HunyuanImage3
- **Anime/Illustration**: NoobAI-XL variants
- **Video**: Wan2.2-I2V, Wan2.2-T2V

Most models available in uint4 quantization. Browse full collection: https://huggingface.co/collections/Disty0/sdnq

---

## Node Reference

### SDNQ Model Loader

**Category**: `loaders/SDNQ`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_selection | DROPDOWN | First model | Select pre-configured model (auto-downloads) |
| custom_repo_or_path | STRING | "" | For custom models: repo ID or local path |
| dtype | CHOICE | bfloat16 | Weight data type |
| use_quantized_matmul | BOOLEAN | True | Triton optimization (Linux/WSL only) |
| device | CHOICE | auto | Device placement |

**Outputs**: `MODEL`, `CLIP`, `VAE` (compatible with standard ComfyUI nodes)

---

## Performance Notes

SDNQ quantization provides significant VRAM savings while maintaining quality:
- **int8**: Best quality/VRAM balance
- **uint4**: Maximum VRAM savings

Actual VRAM usage varies by:
- Model architecture (FLUX, Qwen, etc.)
- Image resolution
- Batch size
- System configuration

Check model pages on HuggingFace for specific requirements: https://huggingface.co/collections/Disty0/sdnq

---

## Model Storage

Downloaded models are stored in:
- **Location**: `ComfyUI/models/diffusers/sdnq/`
- **Format**: Standard diffusers format (works with other tools)

Models are cached automatically - download once, use forever!

---

## Troubleshooting

### "Triton not available" Warning

Triton is an optional optimization. The model will work without it, just slightly slower.

To enable Triton (Linux/WSL only):
```bash
pip install triton
```

Triton is not available on native Windows. Use WSL2 for Triton support.

### Out of Memory Errors

1. Use a more aggressive quantization (uint4 instead of int8)
2. Reduce batch size or resolution
3. Close other GPU applications
4. Use ComfyUI's built-in VRAM management settings

### Model Loading Fails

1. Check internet connection (for HuggingFace models)
2. Verify the repo ID is correct
3. For local models, ensure the path points to the model directory (not a file)
4. Check that the model is actually SDNQ-quantized (from Disty0's collection)

### "Pipeline missing transformer/unet" Error

The model may not be in the expected diffusers format. SDNQ models should have a standard diffusers directory structure with `model_index.json`.

### Integration with ComfyUI

SDNQ models are loaded via ComfyUI's native model loading system and work seamlessly with:
- KSampler and all sampling nodes
- VAE Encode/Decode
- CLIP Text Encode
- ComfyUI's built-in VRAM management
- Other compatible custom nodes

The quantized weights are preserved and the models integrate fully with ComfyUI's workflows.

---

## Quantizing Your Own Models

### SDNQ Model Quantizer Node

Convert any loaded ComfyUI model to SDNQ format:

1. Load a model with any ComfyUI loader (CheckpointLoaderSimple, etc.)
2. Add **SDNQ Model Quantizer** node (under `loaders/SDNQ`)
3. Connect the MODEL output to the quantizer
4. Configure:
   - **quant_type**: int8, int6, uint4, or float8_e4m3fn
   - **output_name**: Name for your quantized model
   - **use_svd**: Enable for better quality (optional)
5. Execute to quantize and save

Quantized models are saved to `ComfyUI/models/diffusers/sdnq/` and can be loaded with the SDNQ Model Loader.

---

## Development Status

### Phase 1: ✅ Complete
- [x] Basic SDNQ model loading
- [x] Local and HuggingFace Hub support
- [x] ComfyUI native integration (proper MODEL, CLIP, VAE objects)
- [x] Triton optimization support

### Phase 2: ✅ Complete
- [x] Model catalog with dropdown selection (21+ models)
- [x] Automatic model downloading with progress tracking
- [x] Smart caching in ComfyUI models folder
- [x] Custom model support

### Phase 3: ✅ Complete
- [x] Model quantization node (convert your own models)
- [x] All latest models (FLUX.2, Qwen, Z-Image, HunyuanImage3, Video models)
- [x] ComfyUI native model loading integration

### Future Enhancements:
- [ ] LoRA support with SDNQ models
- [ ] Memory usage reporting node
- [ ] Additional optimization options

---

## Contributing

Contributions welcome! Please:
1. Follow the existing code style
2. Test with multiple model types (FLUX, SD3, SDXL)
3. Update documentation for new features

---

## License

Apache License 2.0 - See [LICENSE](LICENSE)

This project integrates with [SDNQ by Disty0](https://github.com/Disty0/sdnq). Please respect the upstream project's license.

---

## Links

- **This Repository**: https://github.com/EnragedAntelope/comfyui-sdnq
- **SDNQ Engine**: https://github.com/Disty0/sdnq
- **Pre-quantized Models**: https://huggingface.co/collections/Disty0/sdnq
- **SDNQ Documentation**: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI

---

**Made possible by [Disty0's SDNQ](https://github.com/Disty0/sdnq)** - bringing large models to consumer hardware!
