# Performance Optimization Research (December 2025)

## User Request

> "Now that it works - what can be done to speed it up? That's a long processing time. Can we offer different attn mechanisms or torch compile if that would be supported within the diffusers node?"

**Current Performance**: ~2-3 seconds per step with FLUX.2-dev (40 steps = ~120 seconds total)

**Requirements**:
- Evidence-based optimizations only
- Good error handling
- Clean interrupt handling
- Good user experience

---

## Research Findings

### 1. Attention Mechanisms

#### Scaled Dot Product Attention (SDPA) - Default in PyTorch 2.0+

**Status**: Automatically enabled in PyTorch 2.0+ and diffusers
**Performance**: Moderate speedups (2% on datacenter GPUs, up to 20% on consumer GPUs)
**Backend**: Automatically selects best (FlashAttention, xFormers, or C++)

**Source**: [SDPA vs. xformers Discussion #3793](https://github.com/huggingface/diffusers/issues/3793)

**Key Finding**: "SDPA is enabled by default if you're using PyTorch 2.0 and the latest version of Diffusers, so no code changes are required"

#### xFormers - Explicit Memory-Efficient Attention

**Status**: Optional, requires separate installation (`pip install xformers`)
**Performance**: Up to 45% speedup on RTX 4090, 10-20% on RTX 3090
**Memory**: Reduced memory consumption vs standard attention

**Source**: [xFormers Documentation](https://huggingface.co/docs/diffusers/en/optimization/xformers)

**API**: `pipeline.enable_xformers_memory_efficient_attention()`

**Caveat**: "Don't enable attention slicing if you're already using SDPA or xFormers" - these are already memory efficient

### 2. torch.compile - Massive Speedups

#### Performance Benchmarks (Official)

**H100 (Datacenter)**:
- FLUX.1-Dev baseline: 6.431 seconds
- With torch.compile: 3.483 seconds
- **Speedup: 1.8x**

**RTX 4090 (Consumer)**:
- FLUX.1-Dev baseline: 32.27 seconds
- With torch.compile + quantized: 9.668 seconds
- **Speedup: 3.3x**

**H100 (FLUX Schnell)**:
- With torch.compile: ~700ms per image
- **Result: Sub-second generation**

**Sources**:
- [torch.compile and Diffusers Guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [Run Flux fast on H100s with torch.compile](https://modal.com/docs/examples/flux)
- [Fast LoRA inference for Flux](https://huggingface.co/blog/lora-fast)

#### Implementation Pattern

```python
# Set memory format for optimal performance
pipeline.transformer.to(memory_format=torch.channels_last)

# Compile transformer (main bottleneck)
pipeline.transformer = torch.compile(
    pipeline.transformer,
    mode="max-autotune",  # Best for latency
    fullgraph=True        # Compile entire graph
)

# Optional: Compile VAE decoder
pipeline.vae.decode = torch.compile(
    pipeline.vae.decode,
    mode="max-autotune",
    fullgraph=True
)
```

**Modes**:
- `"default"`: Balanced speed/compile time
- `"reduce-overhead"`: Good for small models
- `"max-autotune"`: **Best for latency** (uses inductor + CUDA graphs)

**Source**: [Accelerated PyTorch 2.0 support](https://huggingface.co/docs/diffusers/v0.20.0/optimization/torch2.0)

#### Compilation Overhead

**First Run**: Compile time can be significant (30-60 seconds)
- "Cold start" - compiles the computational graph
- Subsequent runs use cached compiled version

**Regional Compilation**: Recommended approach
- Only compiles frequently-repeated blocks
- **8-10x faster compilation** with same runtime speedup

**Recommendation**: Warm up on first run, cache benefits all subsequent runs

### 3. VAE Optimizations

#### VAE Tiling

**Purpose**: Decode large images without OOM
**Performance Impact**: Minimal (tiling overhead)
**When to use**: Images >1024x1024 on lower VRAM

**API**: `pipeline.enable_vae_tiling()`

**Source**: [Reduce memory usage](https://huggingface.co/docs/diffusers/en/optimization/memory)

#### VAE Slicing

**Purpose**: Decode large batches
**Performance Impact**: Small overhead for large batches
**When to use**: Batch size >1

**API**: `pipeline.enable_vae_slicing()`

### 4. Attention Slicing - NOT RECOMMENDED

**Purpose**: Reduce memory during attention computation
**Performance Impact**: **SLOWER** (trades speed for memory)

**CRITICAL**: "Don't enable attention slicing if you're already using SDPA from PyTorch > 2.0 or xFormers"

**Verdict**: Skip this - we're using PyTorch 2.0+ (SDPA enabled by default)

### 5. Model Offloading - Already Implemented

We already use `enable_model_cpu_offload()` in "balanced" mode.

**Alternative**: `enable_sequential_cpu_offload()` for even lower VRAM (but slower)

---

## Recommended Optimizations

### Priority 1: torch.compile (Highest Impact)

**Expected Gain**: 1.8-3.3x speedup
**Trade-off**: First-run compilation overhead (~30-60s)
**Risk**: Low - well-tested in diffusers

**Implementation**:
- Add `use_torch_compile` parameter (default: False for backward compat)
- Compile transformer + VAE decoder
- Warmup message: "Compiling model (first run only, ~60s)..."
- Graceful error handling if compilation fails

### Priority 2: xFormers (Moderate Impact)

**Expected Gain**: 10-45% speedup (especially RTX 4090)
**Trade-off**: Requires xformers package installation
**Risk**: Low - optional, graceful fallback

**Implementation**:
- Try to enable xformers if available
- Fallback to SDPA if not installed
- Clear console message about which is active

### Priority 3: VAE Tiling (Conditional)

**Expected Gain**: Enables larger images without OOM
**Trade-off**: Minimal overhead
**Risk**: None

**Implementation**:
- Auto-enable for images >1536x1536
- Or add user parameter

### Not Recommended: Attention Slicing

Conflicts with SDPA/xFormers. Skip entirely.

---

## Implementation Plan

### 1. Add Performance Parameters (Optional Section)

```python
"optional": {
    # ... existing parameters ...

    # Performance optimizations
    "use_torch_compile": ("BOOLEAN", {
        "default": False,
        "tooltip": "Use torch.compile for 1.8-3.3x speedup. First run adds ~60s compilation, then all runs are faster. Requires PyTorch 2.0+."
    }),

    "use_xformers": ("BOOLEAN", {
        "default": True,
        "tooltip": "Use xFormers memory-efficient attention (10-45% speedup). Requires: pip install xformers. Falls back to SDPA if not available."
    }),

    "enable_vae_tiling": ("BOOLEAN", {
        "default": False,
        "tooltip": "Enable VAE tiling for very large images (>1536px). Minimal performance impact, prevents OOM on large resolutions."
    }),
}
```

### 2. Apply Optimizations in load_pipeline()

```python
def load_pipeline(self, model_path, dtype, memory_mode,
                  use_torch_compile=False, use_xformers=True,
                  enable_vae_tiling=False):
    # ... existing loading code ...

    # Apply performance optimizations
    self.apply_performance_optimizations(
        pipeline, use_torch_compile, use_xformers, enable_vae_tiling
    )

    return pipeline

def apply_performance_optimizations(self, pipeline,
                                   use_torch_compile, use_xformers,
                                   enable_vae_tiling):
    """Apply performance optimizations with error handling"""

    # 1. xFormers (try first, fallback to SDPA)
    if use_xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("[SDNQ Sampler] ✓ xFormers enabled")
        except Exception as e:
            print(f"[SDNQ Sampler] ⚠️  xFormers not available, using SDPA: {e}")

    # 2. torch.compile (high impact, optional)
    if use_torch_compile:
        try:
            import torch
            print("[SDNQ Sampler] Compiling model (first run only, ~60s)...")

            # Set memory format
            pipeline.transformer.to(memory_format=torch.channels_last)

            # Compile transformer
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode="max-autotune",
                fullgraph=True
            )

            # Compile VAE decoder
            pipeline.vae.decode = torch.compile(
                pipeline.vae.decode,
                mode="max-autotune",
                fullgraph=True
            )

            print("[SDNQ Sampler] ✓ torch.compile enabled (warmup on first generation)")
        except Exception as e:
            print(f"[SDNQ Sampler] ⚠️  torch.compile failed, continuing without: {e}")

    # 3. VAE tiling (optional, low overhead)
    if enable_vae_tiling:
        try:
            pipeline.enable_vae_tiling()
            print("[SDNQ Sampler] ✓ VAE tiling enabled")
        except Exception as e:
            print(f"[SDNQ Sampler] ⚠️  VAE tiling failed: {e}")
```

### 3. Cache Invalidation

Add to cache tracking:
```python
self.current_use_torch_compile = None
self.current_use_xformers = None
# ... etc
```

Invalidate cache if optimization settings change.

### 4. Error Handling Requirements

- **All optimizations must be optional and non-blocking**
- **Graceful fallback if not supported**
- **Clear console messages about what's enabled**
- **Preserve interrupt handling** (already works)
- **Don't crash on compilation errors**

---

## Performance Expectations

### Conservative Estimates (RTX 3090/4090)

**Baseline** (your current): ~2.5s/step, 40 steps = 100s total

**With xFormers only**: ~2.0s/step, 40 steps = 80s total (~20% faster)

**With torch.compile only**: ~1.4s/step, 40 steps = 56s total (~1.8x faster)

**With both**: ~1.1s/step, 40 steps = 44s total (~2.3x faster)

**First run with torch.compile**: +60s compilation overhead

### Realistic User Experience

**First generation** (with torch.compile enabled):
- Compilation: ~60 seconds (one-time)
- Generation: ~56 seconds
- **Total: ~116 seconds**

**Second+ generations** (compiled cache used):
- Generation: ~44 seconds
- **Total: ~44 seconds** (2.3x faster than baseline)

**Trade-off**: Worth it for users generating multiple images per session!

---

## Testing Plan

1. **Baseline**: Measure current performance
2. **xFormers only**: Enable xformers, measure
3. **torch.compile only**: Enable compile, measure (first run + second run)
4. **Both**: Enable both, measure
5. **Error cases**: Test without xformers installed, test compilation failures
6. **Interrupt handling**: Verify Ctrl+C still works during compilation

---

## Sources

1. [SDPA vs. xformers Discussion](https://github.com/huggingface/diffusers/issues/3793)
2. [xFormers Documentation](https://huggingface.co/docs/diffusers/en/optimization/xformers)
3. [torch.compile and Diffusers Guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
4. [Run Flux fast on H100s with torch.compile](https://modal.com/docs/examples/flux)
5. [Fast LoRA inference for Flux](https://huggingface.co/blog/lora-fast)
6. [Accelerated PyTorch 2.0 support](https://huggingface.co/docs/diffusers/v0.20.0/optimization/torch2.0)
7. [Reduce memory usage](https://huggingface.co/docs/diffusers/en/optimization/memory)
8. [diffusers-torchao repository](https://github.com/sayakpaul/diffusers-torchao)

---

## Key Decision: Default Settings

**use_torch_compile**: Default `False`
- Reason: First-run overhead might confuse users
- Users can opt-in after seeing baseline performance

**use_xformers**: Default `True`
- Reason: No overhead, graceful fallback
- Clear benefit if available

**enable_vae_tiling**: Default `False`
- Reason: Only needed for very large images
- Auto-enable threshold: >1536x1536?

---

## Risk Assessment

**Low Risk**:
- xFormers: Optional, graceful fallback, well-tested
- VAE tiling: Minimal impact, well-tested

**Medium Risk**:
- torch.compile: Compilation can fail on some systems
- Mitigation: Try/except, clear error messages, continue without

**No Risk to Core Functionality**:
- All optimizations are additive
- Failures fall back to current working state
- Interrupt handling preserved
