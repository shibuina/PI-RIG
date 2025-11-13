"""
Hamiltonian RIG experiments (drop-in)
- Trains the VAE with HamiltonVAETrainer
- Updates variant with VAE path
- Runs the standard RIG (HER+TD3) experiment
"""

import torch
from rlkit.launchers.rig_experiments import (
    full_experiment_variant_preprocess,
    generate_vae_dataset,
    get_envs, get_exploration_strategy,  # giữ nguyên import như bản gốc
)
from rlkit.torch.vae.hamilton_vae_trainer import HamiltonVAETrainer
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch.vae.conv_vae import ConvVAE
import rlkit.torch.vae.conv_vae as conv_vae
from rlkit.torch import pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.core import logger


def train_hamilton_vae(variant, return_data=False):
    """
    Train a Hamiltonian VAE (giống train_physics_informed_vae)
    """
    beta = variant["beta"]
    representation_size = variant["representation_size"]

    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn', generate_vae_dataset)
    train_data, test_data, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()

    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity

    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    m = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **variant['vae_kwargs']
    )
    m.to(ptu.device)

    trainer_class = variant.get('trainer_class', ConvVAETrainer)

    if trainer_class == HamiltonVAETrainer:
        print("Using Hamilton VAE Trainer")
        t = HamiltonVAETrainer(
            train_data, test_data, m, beta=beta,
            **variant['algo_kwargs']
        )
    else:
        print("Using Standard VAE Trainer")
        t = ConvVAETrainer(
            train_data, test_data, m, beta=beta,
            **variant['algo_kwargs']
        )

    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
        )
        if should_save_imgs:
            t.dump_samples(epoch)

    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def train_vae_and_update_variant_hamilton(variant):

    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']

    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output('progress.csv', relative_to_snapshot_dir=True)
        logger.add_tabular_output('vae_progress.csv', relative_to_snapshot_dir=True)

        vae, vae_train_data, vae_test_data = train_hamilton_vae(
            train_vae_variant,
            return_data=True,
        )

        if grill_variant.get('save_vae_data', False):
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')

        logger.remove_tabular_output('vae_progress.csv', relative_to_snapshot_dir=True)
        logger.add_tabular_output('progress.csv', relative_to_snapshot_dir=True)

        grill_variant['vae_path'] = vae
    else:
        grill_variant['vae_path'] = variant['train_vae_variant']['vae_path']

    if grill_variant.get('vae_path', None) is None:

        grill_variant['vae_path'] = grill_variant['train_vae_variant']['generate_vae_dataset_kwargs']


def hamilton_grill_her_td3_full_experiment(variant):
    """
    Full RIG + Hamilton VAE (giống physics_informed_grill_her_td3_full_experiment)
    """
    # preprocess (mutate in place)
    full_experiment_variant_preprocess(variant)

    # Train VAE (Hamilton) và cập nhật variant
    train_vae_and_update_variant_hamilton(variant)

    from rlkit.launchers.rig_experiments import grill_her_td3_experiment
    return grill_her_td3_experiment(variant['grill_variant'])