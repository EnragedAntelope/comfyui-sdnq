"""
ComfyUI Weight Streaming and Model Management Compatibility Notes

This document explains how SDNQ models interact with ComfyUI's weight streaming
and model management features.
"""

# ComfyUI Weight Streaming Overview

ComfyUI has a sophisticated model management system that can:
1. **Stream weights from disk/RAM to VRAM** as needed
2. **Automatically offload models** when not in use
3. **Manage multiple models** in limited VRAM

# Our Implementation

## Current Approach
We use diffusers pipelines with optional CPU offloading via:
- `pipeline.enable_model_cpu_offload()` (from Accelerate library)
- This is a diffusers/Accelerate-level feature, NOT ComfyUI-level

## Compatibility Status

### ❓ ComfyUI Weight Streaming
**Status**: Likely NOT compatible with standard ComfyUI weight streaming

**Why**:
- ComfyUI's weight streaming expects models that follow ComfyUI's ModelPatcher interface
- Our wrappers expose diffusers pipelines, which use a different architecture
- ComfyUI won't be able to directly manage the weights of our wrapped models

### ✅ Our CPU Offload
**Status**: Works independently

**How it works**:
- `cpu_offload=True` uses diffusers' `enable_model_cpu_offload()`
- This operates at the Accelerate library level
- Moves model layers to CPU RAM when not actively computing
- Streams layers back to VRAM as needed during inference
- This is SEPARATE from and INDEPENDENT of ComfyUI's system

## User Options

### Option 1: Use Our CPU Offload (Recommended for now)
```python
# In the node settings:
cpu_offload = True  # Uses diffusers/Accelerate offloading
```

**Pros**:
- Works reliably with diffusers models
- Managed by Accelerate library (well-tested)
- Automatically handles layer-by-layer streaming

**Cons**:
- Doesn't integrate with ComfyUI's model management
- Can't share VRAM budget with other ComfyUI models

### Option 2: Disable Our CPU Offload
```python
# In the node settings:
cpu_offload = False  # Keep model fully in VRAM
```

**Use when**:
- You have enough VRAM for the full model
- You want fastest inference (no CPU<->GPU transfers)
- You're not running multiple models

## Technical Details

### What happens with cpu_offload=True?
1. Model loads into system RAM first
2. During inference:
   - Current layer moves to VRAM
   - Previous layer moves back to RAM
   - Next layer pre-loads to VRAM
3. Only ~1-2 layers in VRAM at a time
4. Total VRAM usage reduced by 60-70%

### What happens with cpu_offload=False?
1. Entire model loads into VRAM
2. Stays in VRAM during and after inference
3. Faster inference (no transfers)
4. Requires full model VRAM

## Future Integration Possibilities

### Potential Phase 3 Enhancement: ComfyUI Model Management Integration
To make our models work with ComfyUI's weight streaming, we would need to:

1. **Implement ModelPatcher interface**:
   ```python
   class SDNQModelPatcher:
       def patch_model(self, model):
           # Integrate with ComfyUI's patching system
       def unpatch_model(self, model):
           # Remove patches
       # ... other ModelPatcher methods
   ```

2. **Extract and expose weights**:
   - Decompose diffusers pipeline into individual weight tensors
   - Provide ComfyUI with direct access to weights
   - Allow ComfyUI to manage loading/unloading

3. **Coordinate offloading**:
   - Disable diffusers' offloading when ComfyUI is managing
   - Use ComfyUI's offloading system instead
   - Share VRAM budget with other ComfyUI models

**Complexity**: HIGH - would require significant refactoring

**Benefit**: Better integration with ComfyUI ecosystem, shared VRAM management

## Recommendations

### For Users
1. **Start with defaults**: `cpu_offload=True` for best VRAM savings
2. **Disable if you have VRAM**: Set `cpu_offload=False` if you have enough VRAM
3. **Monitor performance**: Try both settings and see what works best

### For Developers
1. **Current implementation is good for Phase 1 & 2**
2. **Consider ComfyUI integration in Phase 3** if demand exists
3. **Document limitations clearly** for users

## Summary

**TLDR**:
- ✅ Our `cpu_offload` works independently using diffusers/Accelerate
- ❌ ComfyUI's weight streaming likely WON'T work with our current wrappers
- 🔧 `cpu_offload=False` disables our offloading, model stays in VRAM
- 🚀 Future: Could integrate with ComfyUI's system (Phase 3)

**Current Status**: Our CPU offloading works great as-is for saving VRAM. ComfyUI's
weight streaming is a separate system that we don't currently integrate with.
