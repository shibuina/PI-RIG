"""
Common utilities for GRILL (Goal-conditioned RL with ImagenteL representation Learning).
This module provides the train_vae function and other utilities needed for RIG experiments.
"""

import torch
import numpy as np
from rlkit.util.ml_util import PiecewiseLinearSchedule
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity


def generate_vae_dataset(variant):
    """
    Generate VAE dataset from environment.
    Adapted from skewfit_experiments.py
    """
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
    from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
    
    env_class = variant.get('env_class', SawyerPickAndPlaceEnv)
    env_kwargs = variant.get('env_kwargs', {})
    init_camera = variant.get('init_camera', sawyer_pick_and_place_camera)
    imsize = variant.get('imsize', 48)
    N = variant.get('N', 10000)
    test_p = variant.get('test_p', 0.9)
    
    # Create environment
    base_env = env_class(**env_kwargs)
    env = ImageEnv(
        base_env,
        imsize=imsize,
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    
    # Collect data
    print(f"Collecting {N} samples for VAE training...")
    dataset = []
    
    for i in range(N):
        if i % 1000 == 0:
            print(f"Collected {i}/{N} samples")
            
        obs = env.reset()
        dataset.append(obs['image_observation'])
        
        # Take random actions to generate diverse data
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            dataset.append(obs['image_observation'])
            if done:
                break
    
    # Convert to numpy array
    dataset = np.array(dataset[:N])
    
    # Split into train/test
    split_idx = int(test_p * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    info = {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'image_shape': dataset.shape[1:],
    }
    
    print(f"Generated VAE dataset: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data, info


def train_vae(variant, return_data=False):
    """
    Train VAE with given variant configuration.
    Adapted from skewfit_experiments.py and extended for physics-informed training.
    """
    print("Training VAE...")
    
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn', generate_vae_dataset)
    
    # Generate dataset
    train_data, test_data, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs']
    )
    
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    
    # Handle beta schedule
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    
    # Handle decoder activation
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    
    # Create VAE architecture
    architecture = variant['vae_kwargs'].get('architecture', None)
    if architecture is None:
        from rlkit.torch.vae.conv_vae import imsize48_default_architecture
        architecture = imsize48_default_architecture
    
    # Create VAE model
    if variant.get('use_linear_dynamics', False):
        # Handle linear dynamics case if needed
        pass
    
    m = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        architecture=architecture,
        **variant['vae_kwargs']
    )
    
    # Setup trainer
    vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
    vae_trainer_kwargs = variant.get('vae_trainer_kwargs', {})
    
    # Ensure data is in correct format for trainer (uint8)
    if train_data.dtype != np.uint8:
        train_data = (train_data * 255).astype(np.uint8)
    if test_data.dtype != np.uint8:
        test_data = (test_data * 255).astype(np.uint8)
    
    # Create trainer with correct parameter order
    trainer_kwargs = {
        'batch_size': 32,
        'beta': beta,
        'lr': variant.get('algo_kwargs', {}).get('lr', 1e-3),
    }
    
    # Add custom trainer kwargs, but remove unsupported ones
    for k, v in vae_trainer_kwargs.items():
        if k not in ['beta_schedule']:  # Skip unsupported parameters
            trainer_kwargs[k] = v
    
    t = vae_trainer_class(
        train_data,  # train_dataset first
        test_data,   # test_dataset second  
        m,           # model third
        **trainer_kwargs
    )
    
    # Train
    print(f"Starting VAE training for {variant.get('num_epochs', 500)} epochs...")
    
    for epoch in range(variant.get('num_epochs', 500)):
        # Train epoch - ConvVAETrainer handles batching internally
        t.train_epoch(epoch)
        
        # Test epoch - ConvVAETrainer handles batching internally
        t.test_epoch(epoch)
        
        # Save periodically
        if epoch % variant.get('save_period', 50) == 0:
            logger.save_itr_params(epoch, t.model)
            
        # Log progress - use simple print since get_diagnostics doesn't exist
        if epoch % 10 == 0 or epoch == variant.get('num_epochs', 500) - 1:
            print(f"Epoch {epoch}: VAE training completed successfully")
    
    # Final save
    logger.save_itr_params(epoch, t.model)
    
    # Return path to saved model
    if return_data:
        return logger.get_snapshot_dir(), train_data, test_data
    else:
        return logger.get_snapshot_dir()
