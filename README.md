# ComfyUI-SDNQ

**Load SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables loading [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models in ComfyUI workflows. Run large models like FLUX.1 and SD3.5 on consumer hardware with significantly reduced VRAM requirements while maintaining image quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.
> See [CREDITS.md](CREDITS.md) for full attribution.

---

## Features

- **📦 Model Dropdown**: Select from pre-configured SDNQ models from Disty0's collection
- **⚡ Auto-Download**: Models download automatically from HuggingFace on first use
- **💾 Smart Caching**: Download once, use forever
- **🚀 VRAM Savings**: 50-75% memory reduction with quantization
- **🎨 Quality Maintained**: Minimal quality loss (int8: ~99%, int4: ~95%)
- **🔌 Compatible**: Works with standard ComfyUI nodes
- **🏃 Optional Optimizations**: Triton acceleration, CPU offloading

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

### 1. Basic Usage

1. Add the **SDNQ Model Loader** node (under `loaders/SDNQ`)
2. **Select a model** from the dropdown (shows VRAM requirements)
3. **First use**: Model auto-downloads from HuggingFace (cached for future use)
4. Connect outputs:
   - `MODEL` → KSampler
   - `CLIP` → CLIP Text Encode
   - `VAE` → VAE Decode

**Defaults are optimized** - just select a model and go!

**Need more VRAM savings?** Enable `cpu_offload` (reduces speed, saves 60-70% VRAM)

### 2. Custom Models

Select `--Custom Model--` from dropdown, then enter:
- **HuggingFace repo ID**: `Disty0/your-model-qint8`
- **Local path**: `/path/to/model`

### 3. Available Models

Includes FLUX, FLUX.2, SD3.5, and SDXL models with int8/int6/int4 quantization levels.

Browse full collection: https://huggingface.co/collections/Disty0/sdnq

---

## Node Reference

### SDNQ Model Loader

**Category**: `loaders/SDNQ`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_selection | DROPDOWN | First model | Select pre-configured model (auto-downloads) |
| custom_repo_or_path | STRING | "" | For custom models: repo ID or local path |
| dtype | CHOICE | bfloat16 | Weight data type |
| use_quantized_matmul | BOOLEAN | True | Triton optimization (Linux/WSL) |
| cpu_offload | BOOLEAN | False | Offload to CPU RAM (slower, saves VRAM) |
| device | CHOICE | auto | Device placement |

**Outputs**: `MODEL`, `CLIP`, `VAE` (compatible with standard ComfyUI nodes)

---

## Performance Comparison

Typical VRAM usage for FLUX.1-dev on RTX 4090:

| Version | VRAM Usage | Quality |
|---------|------------|---------|
| Full fp16 | ~24 GB | 100% (reference) |
| SDNQ int8 | ~12 GB | ~99% |
| SDNQ int6 | ~9 GB | ~97% |
| SDNQ uint4 | ~6 GB | ~95% |

*Measurements may vary based on resolution, batch size, and system configuration.*

---

## Model Storage

Models are cached following Diffusers/HuggingFace Hub conventions:
- **HuggingFace downloads**: `~/.cache/huggingface/hub/`
- **Recommended local path**: `ComfyUI/models/diffusers/sdnq/`

You can move models from the cache to your ComfyUI models folder and reference them by path to avoid re-downloading.

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

1. Enable **cpu_offload** in the node settings
2. Use a more aggressive quantization (int6 or uint4)
3. Reduce batch size or resolution
4. Close other GPU applications

### Model Loading Fails

1. Check internet connection (for HuggingFace models)
2. Verify the repo ID is correct
3. For local models, ensure the path points to the model directory (not a file)
4. Check that the model is actually SDNQ-quantized (from Disty0's collection)

### "Pipeline missing transformer/unet" Error

The model may not be in the expected diffusers format. SDNQ models should have a standard diffusers directory structure with `model_index.json`.

### ComfyUI Weight Streaming Compatibility

**Important**: Our `cpu_offload` option uses diffusers/Accelerate offloading, which operates independently from ComfyUI's weight streaming system.

**What this means**:
- ✅ `cpu_offload=True` works great for VRAM savings (uses diffusers' system)
- ❓ ComfyUI's native weight streaming likely won't work with our wrappers (different architecture)
- 🔧 `cpu_offload=False` keeps the model fully in VRAM (fastest, but uses more VRAM)

**Recommendation**: Use the built-in `cpu_offload` option for VRAM management. It's well-tested with diffusers models and provides 60-70% VRAM reduction.

See [WEIGHT_STREAMING.md](WEIGHT_STREAMING.md) for technical details.

---

## Quantizing Your Own Models

**Coming in Phase 3**: Support for quantizing existing checkpoints to SDNQ format directly in ComfyUI.

For now, use the [sdnq](https://github.com/Disty0/sdnq) package directly or use pre-quantized models from the [Disty0 collection](https://huggingface.co/collections/Disty0/sdnq).

---

## Development Status

### Phase 1: ✅ Complete
- [x] Basic SDNQ model loading
- [x] Local and HuggingFace Hub support
- [x] ComfyUI type compatibility (MODEL, CLIP, VAE)
- [x] Triton optimization support
- [x] CPU offloading

### Phase 2: ✅ Complete
- [x] Model catalog with dropdown selection
- [x] Automatic model downloading with progress tracking
- [x] Smart caching
- [x] Model metadata display
- [x] Custom model support

### Phase 3 (Planned):
- [ ] Checkpoint quantization node (convert your own models)
- [ ] LoRA support with SDNQ models
- [ ] Memory usage reporting
- [ ] Advanced optimization options
- [ ] Video model support (Wan2.2, etc.)

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
