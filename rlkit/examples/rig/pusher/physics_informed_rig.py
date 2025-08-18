import sys
import os

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.launchers.physics_informed_rig_experiments import physics_informed_grill_her_td3_full_experiment
from rlkit.torch.vae.physics_informed_vae_trainer import PhysicsInformedConvVAETrainer

if __name__ == "__main__":
    # Physics-informed RIG variant
    physics_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        
        # Physics-informed parameters
        physics_weight=0.1,        # Overall physics loss weight
        contact_weight=0.05,       # Contact dynamics weight
        momentum_weight=0.05,      # Momentum conservation weight
        temporal_consistency_weight=0.02,  # Temporal smoothness weight
        
        grill_variant=dict(
            save_video=True,
            save_video_period=50,  # More frequent videos for comparison
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=300,  # Reduced for faster comparison
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
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
            representation_size=4,
            beta=10.0 / 128,
            num_epochs=300,  # Reduced for faster training
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            trainer_class=PhysicsInformedConvVAETrainer,  # Use physics-informed trainer
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                batch_size=128,
                lr=1e-3,
                # Physics-informed parameters
                physics_weight=0.1,
                contact_weight=0.05,
                momentum_weight=0.05,
                temporal_consistency_weight=0.02,
            ),
            save_period=10,  # More frequent VAE saves for monitoring
        ),
        algorithm='Physics-Informed-RIG',
    )

    # Standard RIG variant for comparison
    standard_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
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
                    num_epochs=300,  # Same as physics-informed for fair comparison
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
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
            representation_size=4,
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
                vae_dataset_specific_kwargs=dict(
                ),
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
        algorithm='Standard-RIG',
    )

    # Run physics-informed RIG
    print("Running Physics-Informed RIG experiment...")
    run_experiment(
        physics_informed_grill_her_td3_full_experiment,
        exp_prefix='rlkit-pusher-physics-rig',
        mode='here_no_doodad',
        variant=physics_variant,
        use_gpu=True,
    )

    # Run standard RIG for comparison
    print("Running Standard RIG experiment for comparison...")
    run_experiment(
        grill_her_td3_full_experiment,
        exp_prefix='rlkit-pusher-standard-rig',
        mode='here_no_doodad',
        variant=standard_variant,
        use_gpu=True,
    )
