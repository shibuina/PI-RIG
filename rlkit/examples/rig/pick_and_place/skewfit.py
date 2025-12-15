#!/usr/bin/env python3
"""
Skew-Fit training script for Sawyer Pick and Place Environment

This script trains a Skew-Fit agent using vision-based observations on the 
SawyerPickupEnv-v0 environment, which involves:
- Picking up an object from the table
- Moving it to a target location
- Releasing it at the goal position

Skew-Fit uses a Variational Autoencoder (VAE) to learn visual representations
and performs goal-conditioned reinforcement learning in the latent space.
"""
import sys
import os

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit_original')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')

# Register multiworld environments before importing other modules
import multiworld
multiworld.register_all_envs()

import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture


if __name__ == "__main__":
    variant = dict(
        algorithm='Skew-Fit',
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_pick_and_place_camera,
        env_class=SawyerPickAndPlaceEnv,  # Use direct environment class like oracle
        env_kwargs=dict(
            # Same configuration as oracle example
            obj_init_positions=((0.0, 0.6, 0.02),),  # Single object start position
            random_init=True,                   # Randomize initial positions
            fix_goal=False,                     # Dynamic goal positioning
            hide_goal_markers=False,            # Show goal markers for debugging
            reset_free=False,                   # Use full environment resets
            oracle_reset_prob=0.0,              # No oracle resets
            reward_type='hand_and_obj_distance', # Distance-based reward
            indicator_threshold=0.06,           # Success threshold (6cm)
            
            # Hand workspace bounds (pick and place requires larger workspace)
            hand_low=(-0.2, 0.55, 0.05),       # Hand position lower bounds
            hand_high=(0.2, 0.75, 0.3),        # Hand position upper bounds
            
            # Object workspace bounds
            obj_low=(-0.15, 0.55, 0.02),       # Object position lower bounds  
            obj_high=(0.15, 0.7, 0.02),        # Object position upper bounds
        ),
        skewfit_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=150,  # Longer episodes for pick and place (more complex task)
            algo_kwargs=dict(
                batch_size=256,                # Batch size
                num_epochs=300,                # More epochs for complex task
                num_eval_steps_per_epoch=300,  # More evaluation steps
                num_expl_steps_per_train_loop=300,  # More exploration steps
                num_trains_per_train_loop=300,      # More training steps
                min_num_steps_before_training=4000, # More warm-up for complex task
                vae_training_schedule=vae_schedules.custom_schedule_2,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),          # Larger buffer for complex task
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='vae_prior',
            ),
            exploration_goal_sampling_mode='vae_prior',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
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
            ),
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=50,  # More training data for complex task
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                oracle_dataset_using_set_to_goal=True,
                n_random_steps=100,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=-1,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),
            save_period=25,
        ),
    )
    
    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # Configure for local execution
    n_seeds = 1
    mode = 'here_no_doodad'  # Use local execution without doodad
    exp_prefix = '08-26-skewfit-pickup-local-test'

    print("🤖 Starting Skew-Fit Pick and Place Experiment")
    print("=" * 60)
    print(f"Algorithm: {variant['algorithm']}")
    print(f"Environment: {variant['env_class'].__name__}")
    print(f"Image size: {variant['imsize']}")
    print(f"Max path length: {variant['skewfit_variant']['max_path_length']}")
    print(f"Representation size: {variant['train_vae_variant']['representation_size']}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Execution mode: {mode}")
    print(f"VAE dataset size: {variant['train_vae_variant']['generate_vae_dataset_kwargs']['N']}")
    print(f"Training epochs: {variant['skewfit_variant']['algo_kwargs']['num_epochs']}")
    print(f"Success threshold: {variant['env_kwargs']['indicator_threshold']}m")
    print("=" * 60)
    print("Task: Pick up object and place at goal location using vision")
    print("Method: VAE representation learning + goal-conditioned RL")
    print("=" * 60)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                skewfit_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )

    print("\n🎉 Skew-Fit Pick and Place training completed!")
    print("\nResults will be saved in:")
    print("  - Logs: rlkit/data/")
    print("  - Videos: rlkit/data/*/video_*.mp4")  
    print("  - VAE models: rlkit/data/*/vae_*.pkl")
    print("  - Policy models: rlkit/data/*/params.pkl")
    print("  - Progress: rlkit/data/*/progress.csv")
