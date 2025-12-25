# SDNQ VLM (Vision-Language Model) Support Notes

This document captures research findings for potential future support of SDNQ-quantized VLM models that generate text from images.

## Background

SDNQ supports quantizing both:
1. **Diffusers models** (image generation) - Currently supported by SDNQSampler
2. **Transformers models** (VLM text generation) - NOT currently supported

## Example VLM Model

`Disty0/Qwen3-VL-32B-Instruct-SDNQ-uint4-svd-r32`

This is a Vision-Language Model that:
- Takes images as input
- Generates text descriptions/analysis as output
- Uses `transformers` library (NOT diffusers)

## Architecture Differences

| Aspect | Image Generation (SDNQSampler) | VLM Text Generation |
|--------|--------------------------------|---------------------|
| Library | `diffusers` | `transformers` |
| Model Class | `DiffusionPipeline.from_pretrained()` | `Qwen3VLForConditionalGeneration.from_pretrained()` |
| Additional | None | `AutoProcessor.from_pretrained()` |
| Input | Text prompt (+ optional images) | Image + Text question/instruction |
| Output | IMAGE, LATENT | STRING (text tokens) |

## VLM Loading Pattern

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from sdnq import SDNQConfig  # Register SDNQ into transformers

# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Disty0/Qwen3-VL-32B-Instruct-SDNQ-uint4-svd-r32",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load processor for tokenization
processor = AutoProcessor.from_pretrained(
    "Disty0/Qwen3-VL-32B-Instruct-SDNQ-uint4-svd-r32"
)
```

## VLM Inference Pattern

```python
# Prepare messages with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},  # PIL Image
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Tokenize
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=512)

# Decode output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
```

## Proposed Implementation (Future)

### New Node: SDNQCaptioner

**Inputs:**
- `model_selection`: Dropdown filtered to VLM models only
- `custom_model_path`: For custom VLM paths
- `image`: ComfyUI IMAGE input (required)
- `prompt`: Text prompt/question (e.g., "Describe this image")
- `max_tokens`: Maximum tokens to generate (default: 512)
- `dtype`: bfloat16, float16, float32
- `memory_mode`: gpu, balanced, lowvram
- `auto_download`: Boolean

**Outputs:**
- `STRING`: Generated text description

### Registry Changes

Add `output_type` field to distinguish models:
```python
"Qwen3-VL-32B-Instruct-SDNQ-uint4": {
    "repo_id": "Disty0/Qwen3-VL-32B-Instruct-SDNQ-uint4-svd-r32",
    "type": "Qwen-VL",
    "output_type": "text",  # NEW FIELD: "image" or "text"
    "quant_level": "uint4",
    "description": "Qwen3-VL-32B 4-bit SVD - Image-to-Text captioning",
    "priority": 50  # Lower priority since separate node
}
```

### New Files Required

1. `nodes/captioner.py` - SDNQCaptioner node implementation
2. Update `nodes/__init__.py` - Export new node
3. Update `__init__.py` - Register new node

## Requirements

- `transformers >= 4.57.0` (for Qwen3-VL support)
- May need `qwen_vl_utils` package for some utilities

## References

- Qwen3-VL GitHub: https://github.com/QwenLM/Qwen3-VL
- Transformers docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl
- SDNQ GitHub: https://github.com/Disty0/sdnq

---
*Last updated: 2024-12-25*
*Status: Research complete, implementation deferred*
