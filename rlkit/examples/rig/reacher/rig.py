import sys
import os
import argparse

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.launchers.physics_informed_rig_experiments import physics_informed_grill_her_td3_full_experiment
from rlkit.torch.vae.physics_informed_vae_trainer import PhysicsInformedConvVAETrainer

def get_physics_informed_variant():
    """Get configuration for physics-informed RIG variant"""
    return dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYZEnv-v1',  # Using 3D reacher environment
        
        # Physics-informed parameters tuned for reacher dynamics
        physics_weight=0.15,        # Slightly higher for arm dynamics
        contact_weight=0.03,        # Lower since no contact with objects
        momentum_weight=0.08,       # Higher for arm momentum conservation
        temporal_consistency_weight=0.04,  # Higher for smooth arm movements
        
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=300,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=50,  # Shorter episodes for reacher
                    discount=0.99,
                    num_updates_per_env_step=4,
                    reward_scale=1,
                ),
                her_kwargs=dict(),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=4,  # 3D position + orientation
            beta=10.0 / 128,
            num_epochs=300,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            trainer_class=PhysicsInformedConvVAETrainer,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                batch_size=128,
                lr=1e-3,
                # Physics-informed parameters for reacher
                physics_weight=0.15,
                contact_weight=0.03,
                momentum_weight=0.08,
                temporal_consistency_weight=0.04,
            ),
            save_period=10,
        ),
        algorithm='Physics-Informed-RIG-Reacher',
    )

def get_standard_variant():
    """Get configuration for standard RIG variant"""
    return dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYZEnv-v1',
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=300,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=50,  # Shorter episodes for reacher
                    discount=0.99,
                    num_updates_per_env_step=4,
                    reward_scale=1,
                ),
                her_kwargs=dict(),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=4,  # 3D position + orientation
            beta=10.0 / 128,
            num_epochs=300,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                batch_size=128,
                lr=1e-3,
            ),
            save_period=10,
        ),
        algorithm='Standard-RIG-Reacher',
    )

def main():
    parser = argparse.ArgumentParser(description='Train Reacher RIG variants')
    parser.add_argument('--variant', choices=['physics', 'standard', 'both'], 
                       default='both', help='Which variant to train')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU for training')
    
    args = parser.parse_args()
    
    if args.variant in ['physics', 'both']:
        print("Running Physics-Informed RIG experiment for Reacher...")
        physics_variant = get_physics_informed_variant()
        run_experiment(
            physics_informed_grill_her_td3_full_experiment,
            exp_prefix='rlkit-reacher-physics-rig',
            mode='here_no_doodad',
            variant=physics_variant,
            use_gpu=args.gpu,
        )
    
    if args.variant in ['standard', 'both']:
        print("Running Standard RIG experiment for Reacher...")
        standard_variant = get_standard_variant()
        run_experiment(
            grill_her_td3_full_experiment,
            exp_prefix='rlkit-reacher-standard-rig',
            mode='here_no_doodad',
            variant=standard_variant,
            use_gpu=args.gpu,
        )

if __name__ == "__main__":
    main()
