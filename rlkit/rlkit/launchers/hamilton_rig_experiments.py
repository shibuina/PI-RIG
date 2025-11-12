"""
Hamiltonian RIG experiments (drop-in)
- Trains the VAE with HamiltonVAETrainer
- Updates variant with VAE path
- Runs the standard RIG (HER+TD3) experiment
"""

import os
import torch
from rlkit.core import logger
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.conv_vae import ConvVAE

# Baseline helpers (dataset, preprocess, RL stage)
from rlkit.launchers.rig_experiments import (
    full_experiment_variant_preprocess,
    generate_vae_dataset,
    grill_her_td3_experiment,   # RL stage after VAE training
)

# Your Hamilton trainer (make sure import path matches your repo)
from rlkit.torch.vae.hamilton_vae_trainer import HamiltonVAETrainer


def _train_hamilton_vae_and_update_variant(variant):
    """
    1) Build dataset (exactly like baseline)
    2) Build ConvVAE
    3) Train with HamiltonVAETrainer
    4) Save VAE and write back into variant for RL stage
    """
    tvv = variant['train_vae_variant']
    algo = tvv.setdefault('algo_kwargs', {})

    # 1) dataset (baseline helper adds env_id, camera, imsize, etc.)
    train_data, test_data = generate_vae_dataset(variant)

    # 2) VAE model (mirror baseline args)
    rep_size = tvv['representation_size']
    imsize = variant['imsize']
    vae_kwargs = dict(tvv.get('vae_kwargs', {}))
    vae = ConvVAE(representation_size=rep_size, imsize=imsize, **vae_kwargs).to(ptu.device)

    # 3) Hamilton trainer (only Hamilton-specific args here)
    trainer = HamiltonVAETrainer(
        train_data, test_data, vae,
        batch_size=algo.get('batch_size', 128),
        beta=tvv.get('beta', 0.5),
        lr=algo.get('lr', 1e-3),
        q_dim=algo.get('q_dim', 1),
        p_dim=algo.get('p_dim', 1),
        hidden=algo.get('hidden', 128),
        dt=algo.get('dt', 0.05),
        lambda_dyn=algo.get('lambda_dyn', 1.0),
        lambda_energy=algo.get('lambda_energy', 0.1),
        energy_mode=algo.get('energy_mode', 'const'),  # 'const' | 'decay' | 'none'
        rollout_K=algo.get('rollout_K', 0),
    )

    num_epochs = int(tvv.get('num_epochs', 300))
    save_period = int(tvv.get('save_period', 10))

    for epoch in range(num_epochs):
        trainer.train_epoch(epoch)
        # keep test path consistent with your trainers
        save_now = (epoch % save_period == 0) or (epoch == num_epochs - 1)
        trainer.test_epoch(epoch, save_reconstruction=save_now, save_vae=save_now)

    # 4) Save VAE and update variant for the RL stage
    snapshot_dir = logger.get_snapshot_dir()
    os.makedirs(snapshot_dir, exist_ok=True)
    vae_path = os.path.join(snapshot_dir, 'hamilton_vae.pt')
    # Save whole module (same style many rlkit forks expect)
    torch.save(vae, vae_path)

    # Write back into grill variant for the RL stage
    gv = variant.setdefault('grill_variant', {})
    gv['vae_path'] = vae_path
    gv['representation_size'] = rep_size
    gv.setdefault('vae_wrapped_env_kwargs', {})
    # (leave other RL config untouched; baseline uses gv for the RL stage)


def hamilton_grill_her_td3_full_experiment(variant):
    """
    Full Hamiltonian RIG:
      - preprocess (mutates variant in place)
      - train Hamilton VAE and update variant (vae_path, etc.)
      - run the standard RIG RL stage
    """
    # DO NOT reassign; this mutates in place and returns None in many versions
    full_experiment_variant_preprocess(variant)

    # Train VAE with the Hamilton trainer and update variant
    _train_hamilton_vae_and_update_variant(variant)

    # Run the standard RL, identical to baseline
    return grill_her_td3_experiment(variant['grill_variant'])