"""
Oracle training script for Sawyer Reacher Environment
Based on the pusher oracle example, adapted for reacher-specific configuration.

This script trains a TD3+HER agent using state-based observations (Oracle),
providing an upper-bound baseline for comparison with vision-based methods.
"""
import sys
import os

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_td3_experiment

if __name__ == "__main__":
    # Oracle configuration for Reacher environment
    # This provides state-based observations as an upper bound baseline
    variant = dict(
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=300,                    # Fewer epochs than pusher (reaching is simpler)
                num_steps_per_epoch=1000,         # Standard epoch length
                num_steps_per_eval=1000,          # Evaluation length
                max_path_length=50,               # Shorter episodes for reaching
                num_updates_per_env_step=4,       # Update frequency
                batch_size=128,                   # Batch size
                discount=0.99,                    # Discount factor
                min_num_steps_before_training=2000,  # Less warm-up needed for simpler task
                reward_scale=1.0,                 # Reward scaling
                render=False,                     # No rendering during training
                tau=1e-2,                         # Soft update rate
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),                    # Replay buffer size
            fraction_goals_rollout_goals=0.1,     # Goal relabeling ratio
            fraction_goals_env_goals=0.5,         # Environment goal ratio
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],              # Q-function network size
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],              # Policy network size
        ),
        version='normal',
        es_kwargs=dict(
            max_sigma=.2,                         # Exploration noise
        ),
        exploration_type='ou',                    # Ornstein-Uhlenbeck noise
        observation_key='state_observation',     # Use state observations (Oracle)
        desired_goal_key='state_desired_goal',   # Use state goals
        init_camera=sawyer_init_camera_zoomed_in, # Camera configuration
        do_state_exp=True,                       # Enable state-based experiments

        # Video and logging configuration
        save_video=True,                         # Save videos for visualization
        imsize=84,                              # Image size (for video only)

        # Snapshotting configuration
        snapshot_mode='gap_and_last',           # Save checkpoints
        snapshot_gap=50,                        # Save every 50 epochs

        # Environment configuration
        env_class=SawyerReachXYZEnv,            # 3D reacher environment
        env_kwargs=dict(
            reward_type='hand_distance',         # Distance-based reward
            norm_order=2,                       # L2 norm for distance
            indicator_threshold=0.06,           # Success threshold (6cm)
            fix_goal=False,                     # Dynamic goal positioning
            hide_goal_markers=False,            # Show goal markers for debugging
        ),

        algorithm='Oracle-Reacher',             # Algorithm identifier
    )

    # Experiment configuration
    n_seeds = 1                                # Number of random seeds
    mode = 'here_no_doodad'                   # Local execution (no cluster)
    exp_prefix = 'rlkit-reacher-oracle'       # Experiment name prefix

    print(" Starting Oracle Training for Sawyer Reacher Environment")
    print("=" * 60)
    print(f"Environment: {variant['env_class'].__name__}")
    print(f"Observation: {variant['observation_key']} (State-based)")
    print(f"Goal: {variant['desired_goal_key']}")
    print(f"Episodes: {variant['algo_kwargs']['td3_kwargs']['num_epochs']}")
    print(f"Episode Length: {variant['algo_kwargs']['td3_kwargs']['max_path_length']}")
    print(f"Success Threshold: {variant['env_kwargs']['indicator_threshold']}m")
    print("=" * 60)

    # Run experiments for each seed
    for seed in range(n_seeds):
        print(f"\n Starting training run {seed + 1}/{n_seeds}")
        
        run_experiment(
            her_td3_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            exp_id=seed,
            use_gpu=True,  # Uncomment if you have a GPU available
        )
        
        print(f"Completed training run {seed + 1}/{n_seeds}")

    print("\n All Oracle training runs completed!")
    print("\nResults will be saved in:")
    print("  - Logs: rlkit/data/")
    print("  - Videos: rlkit/data/*/video_*.mp4")
    print("  - Snapshots: rlkit/data/*/params.pkl")
