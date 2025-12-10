# Archive - Broken Wrapper Implementation

**Date Archived**: 2025-12-09

This directory contains the original wrapper-based implementation that was found to be fundamentally incompatible with ComfyUI's architecture.

## Why Archived

After grounding research in actual source code (see ASSESSMENT.md), discovered that:

1. ComfyUI expects `ModelPatcher` objects for MODEL output
2. Our custom wrappers don't inherit from ComfyUI base classes
3. Missing required infrastructure: `clone()`, `latent_format`, patching system
4. GitHub Issue #14 confirmed no one successfully generated images with this approach

## Contents

- `nodes/loader.py` - Original SDNQModelLoader node (outputs MODEL/CLIP/VAE)
- `core/wrapper.py` - Custom wrapper classes (SDNQModelWrapper, SDNQCLIPWrapper, SDNQVAEWrapper)

## Current Implementation

The new approach uses a standalone sampler node (see `nodes/sampler.py`) that:
- Loads SDNQ models directly via diffusers
- Generates images in one step
- Outputs IMAGE only (no MODEL/CLIP/VAE)
- Actually works!

See `STANDALONE_SAMPLER_PLAN.md` for implementation details.

## Future

If ModelPatcher integration is needed later, this code provides a starting point for understanding what NOT to do. The proper approach would require inheriting from ComfyUI's base classes, not creating custom wrappers.
