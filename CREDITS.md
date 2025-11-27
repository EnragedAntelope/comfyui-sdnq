# Credits

## SDNQ - SD.Next Quantization Engine

**SDNQ (SD.Next Quantization)** is developed and maintained by **Disty0**.

- **Author**: Disty0
- **Repository**: https://github.com/Disty0/sdnq
- **License**: Check upstream repository for details
- **Pre-quantized Models**: https://huggingface.co/collections/Disty0/sdnq

### About SDNQ

SDNQ is a powerful quantization engine that enables running large diffusion models (FLUX, SD3.5, SDXL) with 50-75% VRAM savings while maintaining image quality. It supports multiple quantization formats (int8, int6, uint4) and optional SVD compression.

Key features:
- Cross-platform support (CUDA, ROCm, Intel Arc, CPU)
- Transparent integration with Hugging Face Diffusers
- Optional Triton acceleration via torch.compile
- Pre-quantized model collection on HuggingFace

### ComfyUI Integration

This node pack (ComfyUI-SDNQ) provides ComfyUI integration for SDNQ technology. All quantization algorithms, model formats, and core functionality are developed and maintained by Disty0.

**The ComfyUI-SDNQ integration is simply a wrapper** to make these incredible models accessible in ComfyUI workflows.

---

## Additional Resources

- **SDNQ Wiki**: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Diffusers Library**: https://github.com/huggingface/diffusers

---

**Thank you to Disty0 for developing SDNQ and making high-quality AI image generation accessible to more users!**
