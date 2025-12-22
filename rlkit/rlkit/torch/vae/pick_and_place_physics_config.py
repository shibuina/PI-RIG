"""
Physics configuration for Pick-and-Place task based on methodology requirements.

This module provides configuration functions for physics-informed training
as described in the methodology paper, implementing the Enhanced P³-VAE
with physics constraints and ODE-based temporal consistency.
"""

import torch
from rlkit.torch.vae.enhanced_p3_vae_trainer import EnhancedP3VAETrainer


def get_pick_and_place_physics_config():
    """
    Get physics configuration for pick-and-place task based on methodology.
    
    Implements the physics constraints from methodology Section 2.3:
    - Kinematic consistency (Equation 5)
    - Newton's First Law (Equation 6) 
    - Conservation Laws (Equations 7-8)
    - Contact Dynamics (Equation 9)
    - ODE temporal consistency (Section 2.4)
    
    Returns:
        dict: Physics configuration parameters
    """
    return {
        # Physics constraint weights (methodology Section 2.4)
        'physics_loss_weights': {
            'kinematic': 0.1,      # λ_kinematic for Equation 5
            'newton_first': 0.05,  # λ_newton for Equation 6
            'momentum': 0.1,       # λ_momentum for Equation 7
            'energy': 0.05,        # λ_energy for Equation 8
            'contact': 0.2,        # λ_contact for Equation 9
            'temporal_ode': 0.15,  # λ_temporal for ODE consistency
        },
        
        # Physics parameters for pick-and-place (methodology values)
        'physics_params': {
            'dt': 0.02,                    # Time step (methodology: 0.02s)
            'hand_mass': 1.0,              # kg (methodology assumption)
            'obj_mass': 0.1,               # kg (typical object mass)
            'contact_threshold': 0.05,     # m (contact detection)
            'contact_stiffness': 1000.0,   # N/m (methodology: k_c = 1000)
            'friction_coeff': 0.3,         # Friction coefficient (methodology: μ = 0.3)
            'workspace_bounds': 0.5,       # m (workspace limits)
            'gripper_range': 0.05,         # m (gripper opening range)
            'velocity_limit': 2.0,         # m/s (reasonable velocity limit)
        },
        
        # Latent space configuration (methodology Section 2.2)
        'latent_config': {
            'z_I_dim': 7,     # Physics variables: [gripper_opening, hand_xyz, obj_xyz]
            'z_E_dim': 3,     # Environmental factors
            'physics_type': 'pick_and_place',
        },
        
        # Loss function configuration (methodology Equation 10-11)
        'loss_config': {
            'physics_weight': 0.2,         # λ_physics overall weight
            'conservation_weight': 0.05,   # Conservation law emphasis
            'regularization_weight': 0.01, # L2 regularization (methodology)
            'supervision_ratio': 0.5,      # α in Equation 3
            'beta_kl': 0.5,               # β for KL divergence
            'grad_clip_value': 5.0,       # Gradient clipping for stability
        },
        
        # ODE integration configuration (methodology Section 2.4)
        'ode_config': {
            'use_torchdiffeq': True,       # Prefer torchdiffeq when available
            'ode_method': 'euler',         # Integration method
            'ode_rtol': 1e-3,             # Relative tolerance
            'ode_atol': 1e-5,             # Absolute tolerance
        }
    }


def create_pick_and_place_vae_trainer_kwargs(physics_config):
    """
    Create VAE trainer kwargs for physics-informed pick-and-place training.
    
    Args:
        physics_config (dict): Physics configuration from get_pick_and_place_physics_config()
        
    Returns:
        dict: Trainer configuration for EnhancedP3VAETrainer
    """
    latent_config = physics_config['latent_config']
    loss_config = physics_config['loss_config'] 
    physics_params = physics_config['physics_params']
    ode_config = physics_config['ode_config']
    
    return {
        # Trainer class specification
        'vae_trainer_class': EnhancedP3VAETrainer,
        
        # Physics parameters from methodology
        'physics_type': latent_config['physics_type'],
        'z_I_dim': latent_config['z_I_dim'],
        'z_E_dim': latent_config['z_E_dim'],
        'dt': physics_params['dt'],
        
        # Loss configuration (methodology Equations 10-11)
        'use_physics_constraints': True,
        'physics_weight': loss_config['physics_weight'],
        'conservation_weight': loss_config['conservation_weight'],
        'regularization_weight': loss_config['regularization_weight'],
        'grad_clip_value': loss_config['grad_clip_value'],
        'beta': loss_config['beta_kl'],
        
        # Additional training parameters for robustness
        'lr': 1e-3,                    # Learning rate
        'batch_size': 256,             # Batch size
        'normalize': False,            # Don't normalize images
        'mse_weight': 0.1,            # Reconstruction weight
        'is_auto_encoder': False,      # Use VAE, not AE
        'background_subtract': False,  # No background subtraction
        
        # Enhanced P³-VAE specific parameters
        'supervision_ratio': loss_config['supervision_ratio'],
    }


def get_physics_informed_goal_sampling_config():
    """
    Get configuration for physics-informed goal sampling (methodology Algorithm 1).
    
    Returns:
        dict: Goal sampling configuration
    """
    return {
        'num_candidates': 100,         # N in Algorithm 1
        'physics_weight': 0.7,         # Weight for physics validation
        'reachability_weight': 0.3,    # Weight for reachability
        'min_distance_threshold': 0.05, # Minimum goal distance
        'max_distance_threshold': 0.3,  # Maximum goal distance
        'workspace_bounds': {
            'hand_low': [-0.2, 0.55, 0.05],
            'hand_high': [0.2, 0.75, 0.3],
            'obj_low': [-0.15, 0.6, 0.02],
            'obj_high': [0.15, 0.7, 0.1],
        },
        'physics_constraints': {
            'max_velocity': 2.0,        # m/s
            'gripper_range': [0.0, 0.05], # m
            'collision_threshold': 0.03, # m
        }
    }


def create_physics_informed_rl_config(base_rl_config):
    """
    Enhance RL configuration with physics-informed goal sampling.
    
    Args:
        base_rl_config (dict): Base RL configuration
        
    Returns:
        dict: Enhanced RL configuration with physics-informed goal sampling
    """
    enhanced_config = base_rl_config.copy()
    
    # Add physics-informed goal sampling configuration
    goal_sampling_config = get_physics_informed_goal_sampling_config()
    
    # Update environment wrapper kwargs to use physics-informed sampling
    if 'vae_wrapped_env_kwargs' not in enhanced_config:
        enhanced_config['vae_wrapped_env_kwargs'] = {}
    
    enhanced_config['vae_wrapped_env_kwargs'].update({
        'use_physics_informed_goals': True,
        'goal_sampling_config': goal_sampling_config,
        'sample_from_true_prior': False,  # Use physics-informed sampling instead
    })
    
    # Update replay buffer to support physics-informed goals
    if 'replay_buffer_kwargs' in enhanced_config:
        enhanced_config['replay_buffer_kwargs'].update({
            'use_physics_informed_relabeling': True,
        })
    
    return enhanced_config
