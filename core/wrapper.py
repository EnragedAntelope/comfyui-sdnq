"""
ComfyUI Type Wrappers for SDNQ Models

This module wraps diffusers pipeline components (transformer/unet, text_encoder, vae)
into ComfyUI-compatible types (MODEL, CLIP, VAE) that can be used with standard
ComfyUI nodes like KSampler.
"""

import torch
from typing import Any, Tuple, Optional


class SDNQModelWrapper:
    """
    Wraps a diffusers transformer/unet model for ComfyUI MODEL compatibility.

    This wrapper provides the interface expected by ComfyUI's sampling nodes.
    """

    def __init__(self, pipeline, model_component):
        """
        Args:
            pipeline: The full diffusers pipeline
            model_component: The transformer or unet component from the pipeline
        """
        self.pipeline = pipeline
        self.model = model_component
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect the type of model (FLUX, SD3, SDXL, etc.)"""
        if hasattr(self.pipeline, 'transformer'):
            # FLUX or SD3 style
            return "flux"
        elif hasattr(self.pipeline, 'unet'):
            # SDXL/SD1.5 style
            return "sdxl"
        else:
            return "unknown"

    def get_model(self):
        """Return the underlying model for direct access"""
        return self.model

    def get_pipeline(self):
        """Return the full pipeline for generation"""
        return self.pipeline


class SDNQCLIPWrapper:
    """
    Wraps a diffusers text encoder for ComfyUI CLIP compatibility.

    This wrapper provides text encoding functionality compatible with ComfyUI workflows.
    """

    def __init__(self, pipeline, text_encoder, tokenizer):
        """
        Args:
            pipeline: The full diffusers pipeline
            text_encoder: The text encoder component(s) from the pipeline
            tokenizer: The tokenizer component(s) from the pipeline
        """
        self.pipeline = pipeline
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def tokenize(self, text, images=None, **kwargs):
        """
        Tokenize text (and optionally images) for encoding.
        
        This matches ComfyUI's CLIP.tokenize() interface.
        ComfyUI's standard CLIPTextEncode only passes text, but custom nodes
        may pass images for multimodal models.
        
        Args:
            text: Input text prompt
            images: Optional images for multimodal processors (e.g., Flux 2 with vision)
            **kwargs: Additional tokenizer options
        
        Returns:
            Tokenized output (format depends on tokenizer type)
        """
        # Handle multimodal processors (like PixtralProcessor for Flux 2)
        # These CAN accept both text and images, but images are optional
        if hasattr(self.tokenizer, 'tokenizer'):
            # This is a composite processor (has .tokenizer attribute for text-only)
            # Check if images were provided
            if images is not None:
                # Use full multimodal processor
                return self.tokenizer(text=text, images=images, return_tensors="pt", padding=True, **kwargs)
            else:
                # Use text-only tokenizer component
                text_tokenizer = self.tokenizer.tokenizer
                return text_tokenizer(text, return_tensors="pt", padding=True, **kwargs)
        elif hasattr(self.tokenizer, '__call__'):
            # Standard diffusers tokenizer (text-only)
            return self.tokenizer(text, return_tensors="pt", padding=True, **kwargs)
        else:
            raise NotImplementedError(f"Tokenizer type {type(self.tokenizer)} not supported")

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False, **kwargs):
        """
        Encode tokens to embeddings.
        
        This matches ComfyUI's CLIP.encode_from_tokens() interface.
        
        Args:
            tokens: Tokenized input (BatchEncoding or dict with tensors)
            return_pooled: Whether to return pooled output (bool or "unprojected")
            return_dict: Whether to return a dictionary instead of tuple
            **kwargs: Additional encoding options
        
        Returns:
            Encoded embeddings (and optionally pooled output)
        """
        # Extract tensors from BatchEncoding or dict
        # BatchEncoding is dict-like but we need to extract only the tensor keys
        if hasattr(tokens, 'data'):
            # BatchEncoding object - extract the underlying dict
            token_dict = tokens.data
        elif isinstance(tokens, dict):
            token_dict = tokens
        else:
            # Assume it's raw input_ids tensor
            token_dict = {"input_ids": tokens}
        
        # Move tensors to text encoder's device and prepare inputs
        # Only pass keys that the text encoder expects (input_ids, attention_mask, etc.)
        inputs = {}
        for key in ['input_ids', 'attention_mask', 'pixel_values']:
            if key in token_dict:
                value = token_dict[key]
                if hasattr(value, 'to'):
                    inputs[key] = value.to(self.text_encoder.device)
                else:
                    inputs[key] = value
        
        # Encode using text encoder
        outputs = self.text_encoder(**inputs)
        
        # Extract embeddings and pooled output
        cond = outputs.last_hidden_state
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, 'pooled_output') and outputs.pooled_output is not None:
            pooled = outputs.pooled_output
        else:
            # No pooled output, use first token as fallback
            pooled = cond[:, 0]
        
        # Return based on requested format
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            return out
        elif return_pooled:
            return cond, pooled
        else:
            return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict=None, show_pbar=True):
        """
        Encode tokens with optional scheduling support.
        
        This matches ComfyUI's CLIP.encode_from_tokens_scheduled() interface.
        For diffusers models, we don't have the scheduling/hooks infrastructure,
        so we just call encode_from_tokens and format the output appropriately.
        
        Args:
            tokens: Tokenized input
            unprojected: Whether to use unprojected pooled output
            add_dict: Additional dictionary to merge into output
            show_pbar: Whether to show progress bar (ignored for simple encoding)
        
        Returns:
            List of [cond, pooled_dict] pairs (ComfyUI format)
        """
        if add_dict is None:
            add_dict = {}
        
        # Encode tokens
        return_pooled = "unprojected" if unprojected else True
        pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)
        
        # Extract cond and create output format
        cond = pooled_dict.pop("cond")
        
        # Merge in any additional dict items
        pooled_dict.update(add_dict)
        
        # Return in ComfyUI's expected format: list of [cond, pooled_dict] pairs
        return [[cond, pooled_dict]]

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to embeddings in one step (convenience method).

        Args:
            text: Input text prompt

        Returns:
            Text embeddings tensor
        """
        tokens = self.tokenize(text, return_tensors="pt", padding=True)
        return self.encode_from_tokens(tokens)

    def get_text_encoder(self):
        """Return the underlying text encoder"""
        return self.text_encoder

    def get_tokenizer(self):
        """Return the underlying tokenizer"""
        return self.tokenizer


class SDNQVAEWrapper:
    """
    Wraps a diffusers VAE for ComfyUI VAE compatibility.

    This wrapper provides encoding/decoding functionality for latent space operations.
    """

    def __init__(self, vae):
        """
        Args:
            vae: The VAE component from the pipeline
        """
        self.vae = vae

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: Input images tensor [B, C, H, W]

        Returns:
            Latent representations
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: Latent representations

        Returns:
            Decoded images tensor [B, C, H, W]
        """
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
        return images

    def get_vae(self):
        """Return the underlying VAE"""
        return self.vae


def wrap_pipeline_components(pipeline) -> Tuple[SDNQModelWrapper, SDNQCLIPWrapper, SDNQVAEWrapper]:
    """
    Wrap a diffusers pipeline into ComfyUI-compatible components.

    Args:
        pipeline: A diffusers pipeline (FluxPipeline, StableDiffusion3Pipeline, etc.)

    Returns:
        Tuple of (model_wrapper, clip_wrapper, vae_wrapper) compatible with ComfyUI types

    Raises:
        ValueError: If pipeline components are missing or invalid
    """
    # Extract model component (transformer or unet)
    if hasattr(pipeline, 'transformer') and pipeline.transformer is not None:
        model_component = pipeline.transformer
    elif hasattr(pipeline, 'unet') and pipeline.unet is not None:
        model_component = pipeline.unet
    else:
        raise ValueError("Pipeline missing transformer or unet component")

    # Extract text encoder and tokenizer
    text_encoder = getattr(pipeline, 'text_encoder', None)
    tokenizer = getattr(pipeline, 'tokenizer', None)

    if text_encoder is None or tokenizer is None:
        raise ValueError("Pipeline missing text_encoder or tokenizer component")

    # Extract VAE
    vae = getattr(pipeline, 'vae', None)
    if vae is None:
        raise ValueError("Pipeline missing vae component")

    # Create wrappers
    model_wrapper = SDNQModelWrapper(pipeline, model_component)
    clip_wrapper = SDNQCLIPWrapper(pipeline, text_encoder, tokenizer)
    vae_wrapper = SDNQVAEWrapper(vae)

    return (model_wrapper, clip_wrapper, vae_wrapper)
