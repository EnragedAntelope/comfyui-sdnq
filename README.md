# ComfyUI-SDNQ

**Load SDNQ quantized models in ComfyUI with 50-75% VRAM savings!**

This custom node pack enables loading [SDNQ (SD.Next Quantization)](https://github.com/Disty0/sdnq) models in ComfyUI workflows. Run large models like FLUX.1 and SD3.5 on consumer hardware with significantly reduced VRAM requirements while maintaining image quality.

> **SDNQ is developed by [Disty0](https://github.com/Disty0)** - this node pack provides ComfyUI integration.
> See [CREDITS.md](CREDITS.md) for full attribution.

---

## Features

- **📦 Model Dropdown**: Select from 11+ pre-configured SDNQ models - no typing required!
- **⚡ Auto-Download**: Models download automatically on first use from HuggingFace
- **💾 Smart Caching**: Downloads cached and reused - load instantly after first download
- **🚀 Massive VRAM Savings**: 50-75% reduction in memory usage
- **🎨 Quality Maintained**: Minimal to no degradation in output quality
- **⚛️ Multiple Quant Levels**: Support for int8, int6, uint4, and more
- **🔌 Drop-in Compatibility**: Works with standard ComfyUI nodes (KSampler, etc.)
- **🏃 Triton Acceleration**: Optional quantized matrix multiplication speedup

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

### 1. Using the Node (Easy Mode - Dropdown Selection)

**🆕 NEW: Just select and go!** No need to manually enter model names.

1. Add the **SDNQ Model Loader** node (under `loaders/SDNQ`)
2. **Select a model from the dropdown**:
   - `FLUX.1-dev-qint8 [~12 GB]` - Best quality/VRAM balance ⭐ **Recommended**
   - `FLUX.1-dev-qint6 [~9 GB]` - Great quality, lower VRAM
   - `FLUX.1-dev-qint4 [~6 GB]` - Extreme VRAM savings
   - `SD3.5-Large-qint8 [~10 GB]` - Latest Stable Diffusion
   - `SDXL-base-qint8 [~6 GB]` - Classic high quality
   - And more...
3. Configure settings (defaults work great):
   - **dtype**: `bfloat16` ✅ (recommended)
   - **use_quantized_matmul**: `True` ✅ (faster inference with Triton)
   - **cpu_offload**: `True` ✅ (saves VRAM)
4. **First use**: The model will auto-download from HuggingFace Hub
   - Progress shown in console
   - Models are cached - only download once!
   - Future loads are instant
5. Connect outputs to standard ComfyUI nodes:
   - `MODEL` → KSampler
   - `CLIP` → CLIP Text Encode
   - `VAE` → VAE Decode

### 2. Using the Node (Advanced - Custom Models)

For custom models or local paths:

1. In the **model_selection** dropdown, choose `--Custom Model--`
2. Enter in **custom_repo_or_path**:
   - **HuggingFace repo ID**: `Disty0/your-model-qint8`
   - **Local path**: `/path/to/downloaded/model`
3. Configure other settings as above

### 3. Example Workflow

```
SDNQ Model Loader
├─ model_selection: FLUX.1-dev-qint8 [~12 GB] ← Just select from dropdown!
├─ dtype: bfloat16
├─ use_quantized_matmul: ✓
└─ cpu_offload: ✓

     ↓ (MODEL, CLIP, VAE)

  KSampler ← (your prompts, seeds, etc.)

     ↓ (LATENT)

  VAE Decode ← (VAE from SDNQ loader)

     ↓ (IMAGE)

 Save Image
```

### 4. Available Models

**All models auto-download on first use:**

**FLUX Models** (Highest quality):
- FLUX.1-dev-qint8 (~12 GB VRAM, ~15 GB download) ⭐ **Recommended**
- FLUX.1-dev-qint6 (~9 GB VRAM, ~12 GB download)
- FLUX.1-dev-qint4 (~6 GB VRAM, ~9 GB download)
- FLUX.1-schnell-qint8 (~12 GB VRAM) - Fast variant
- FLUX.2-dev-qint8 (~12 GB VRAM, ~15 GB download) - Next-gen FLUX
- FLUX.2-dev-qint4 (~6 GB VRAM, ~9 GB download) - Next-gen low VRAM

**SD 3.5 Models** (Latest Stable Diffusion):
- SD3.5-Large-qint8 (~10 GB VRAM, ~12 GB download)
- SD3.5-Large-Turbo-qint8 (~10 GB VRAM) - Fast inference
- SD3.5-Medium-qint8 (~6 GB VRAM, ~8 GB download)

**SDXL Models** (Widely compatible):
- SDXL-base-qint8 (~6 GB VRAM, ~7 GB download)
- SDXL-base-qint4 (~4 GB VRAM, ~4 GB download)

Browse all models: https://huggingface.co/collections/Disty0/sdnq

---

## Node Reference

### SDNQ Model Loader

**Category**: `loaders/SDNQ`

**Inputs**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_selection | DROPDOWN | FLUX.1-dev-qint8 | Select from 11+ pre-configured models with auto-download |
| custom_repo_or_path | STRING | "" | Custom HuggingFace repo ID or local path (only when "Custom Model" selected) |
| dtype | CHOICE | bfloat16 | Weight data type (bfloat16/float16/float32) |
| use_quantized_matmul | BOOLEAN | True | Enable Triton quantized matmul (faster inference) |
| cpu_offload | BOOLEAN | True | Offload model to CPU when not in use |
| device | CHOICE | auto | Device placement (auto/cuda/cpu) |

**Outputs**:
- `MODEL`: Quantized diffusion model (compatible with KSampler)
- `CLIP`: Text encoder (compatible with CLIP Text Encode)
- `VAE`: Variational autoencoder (compatible with VAE Decode/Encode)

**How It Works**:
1. Select a model from the dropdown (or choose "Custom Model")
2. Model automatically downloads on first use (if not cached)
3. Subsequent loads use the cached model (instant loading)
4. Progress and status shown in ComfyUI console

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

### Phase 2 (Current): ✅ Complete
- [x] Model catalog with dropdown selection (11+ models including FLUX.2)
- [x] Automatic model downloading with progress tracking
- [x] Smart caching (download once, use forever)
- [x] Model metadata display (VRAM, size, quality)
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
