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

    def tokenize(self, text, **kwargs):
        """
        Tokenize text for encoding.
        
        This matches ComfyUI's CLIP.tokenize() interface.
        
        Args:
            text: Input text prompt
            **kwargs: Additional tokenizer options
        
        Returns:
            Tokenized output (format depends on tokenizer type)
        """
        # For diffusers tokenizers, return the tokenized output
        if hasattr(self.tokenizer, '__call__'):
            return self.tokenizer(text, **kwargs)
        else:
            raise NotImplementedError(f"Tokenizer type {type(self.tokenizer)} not supported")

    def encode_from_tokens(self, tokens, return_pooled=False, **kwargs):
        """
        Encode tokens to embeddings.
        
        This matches ComfyUI's CLIP.encode_from_tokens() interface.
        
        Args:
            tokens: Tokenized input
            return_pooled: Whether to return pooled output
            **kwargs: Additional encoding options
        
        Returns:
            Encoded embeddings (and optionally pooled output)
        """
        # Handle different token formats
        if isinstance(tokens, dict):
            # Already tokenized dict from diffusers tokenizer
            inputs = {k: v.to(self.text_encoder.device) if hasattr(v, 'to') else v 
                     for k, v in tokens.items()}
        else:
            # Assume it's raw tokens
            inputs = {"input_ids": tokens}
        
        # Encode using text encoder
        outputs = self.text_encoder(**inputs)
        
        if return_pooled:
            # Return both sequence output and pooled output
            if hasattr(outputs, 'pooler_output'):
                return outputs.last_hidden_state, outputs.pooler_output
            elif hasattr(outputs, 'pooled_output'):
                return outputs.last_hidden_state, outputs.pooled_output
            else:
                # No pooled output, return hidden state twice
                return outputs.last_hidden_state, outputs.last_hidden_state[:, 0]
        else:
            return outputs.last_hidden_state

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
