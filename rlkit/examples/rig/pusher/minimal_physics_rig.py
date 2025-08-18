"""
Minimal working example of physics-informed RIG
"""
import sys
import os

# Add required paths
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../multiworld'))

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.torch.vae.physics_informed_vae_trainer import PhysicsInformedConvVAETrainer

# Modified train_vae function to use physics-informed trainer
def train_physics_vae(variant, return_data=False):
    from rlkit.launchers.rig_experiments import generate_vae_dataset
    from rlkit.torch.vae.conv_vae import ConvVAE
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch import pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    from rlkit.core import logger
    import torch
    
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    
    print("Generating VAE dataset...")
    train_data, test_data, info = generate_vae_dataset(
        variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    
    # VAE model setup
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
        
    architecture = conv_vae.imsize84_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')
    
    print("Creating VAE model...")
    m = ConvVAE(
        representation_size,
        decoder_output_activation=decoder_activation,
        **variant['vae_kwargs']
    )
    m.to(ptu.device)
    
    # Use physics-informed trainer
    print("Creating Physics-Informed VAE Trainer...")
    t = PhysicsInformedConvVAETrainer(
        train_data, test_data, m, beta=beta,
        **variant['algo_kwargs']
    )
    
    print("Training VAE...")
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(epoch, save_reconstruction=should_save_imgs)
        if should_save_imgs:
            t.dump_samples(epoch)
        print(f"Completed VAE epoch {epoch}/{variant['num_epochs']}")
    
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m

# Monkey patch the train_vae function
import rlkit.launchers.rig_experiments
rlkit.launchers.rig_experiments.train_vae = train_physics_vae

if __name__ == "__main__":
    print("Starting Minimal Physics-Informed RIG Test...")
    
    variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        grill_variant=dict(
            save_video=True,
            save_video_period=25,
            qf_kwargs=dict(hidden_sizes=[400, 300]),
            policy_kwargs=dict(hidden_sizes=[400, 300]),
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=50,  # Very short for testing
                    num_steps_per_epoch=500,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=1000,
                    batch_size=64,
                    max_path_length=50,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    reward_scale=1,
                ),
                her_kwargs=dict(),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e5),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(type='latent_distance'),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(sample_from_true_prior=True)
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0 / 128,
            num_epochs=25,  # Short VAE training
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=2000,  # Small dataset for speed
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(),
                show=False,
            ),
            vae_kwargs=dict(input_channels=3),
            algo_kwargs=dict(
                batch_size=64,
                lr=1e-3,
                # Physics parameters
                physics_weight=0.1,
                contact_weight=0.05,
                momentum_weight=0.05,
                temporal_consistency_weight=0.02,
            ),
            save_period=5,
        ),
        algorithm='Minimal-Physics-RIG',
    )

    run_experiment(
        grill_her_td3_full_experiment,
        exp_prefix='minimal-physics-rig-test',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,
    )
