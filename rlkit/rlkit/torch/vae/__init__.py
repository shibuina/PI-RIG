"""
VAE (Variational Auto-Encoder) implementations for RL-Kit.

This module contains various VAE implementations including:
- Standard VAE trainer
- Physics-informed VAE trainers
- Enhanced P³-VAE trainer
- Configuration modules for different environments
"""

from .vae_trainer import ConvVAETrainer
from .conv_vae import ConvVAE
from .enhanced_p3_vae_trainer import EnhancedP3VAETrainer

__all__ = [
    'ConvVAE', 
    'ConvVAETrainer',
    'EnhancedP3VAETrainer',
]
