"""
Oracle training script for Sawyer Pick and Place Environment
Based on the pusher oracle example, adapted for pick-and-place-specific configuration.

This script trains a TD3+HER agent using state-based observations (Oracle),
providing an upper-bound baseline for comparison with vision-based methods.

The pick and place task involves:
- Picking up an object from the table
- Moving it to a target location
- Releasing it at the goal position

This Oracle version uses ground-truth state information for both observations
and goals, representing the performance ceiling for this task.
"""
import sys
import os

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_td3_experiment
# add export MUJOCO_GL=egl to avoid rendering issues
os.environ["MUJOCO_GL"] = "egl"

if __name__ == "__main__":
    # Oracle configuration for Pick and Place environment
    # This provides state-based observations as an upper bound baseline
    variant = dict(
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=301,                   
                num_steps_per_epoch=10000,        # Standard epoch length
                num_steps_per_eval=10000,         # Evaluation length
                max_path_length=150,             # Longer episodes for pick-and-place (more complex task)
                num_updates_per_env_step=4,      # Update frequency
                batch_size=128,                  # Batch size
                discount=0.99,                   # Discount factor
                min_num_steps_before_training=4000,  # More warm-up for complex task
                reward_scale=1.0,                # Reward scaling
                render=False,                    # No rendering during training
                tau=1e-2,                        # Soft update rate
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),                   # Large replay buffer for complex task
            fraction_goals_rollout_goals=0.1,    # Goal relabeling ratio
            fraction_goals_env_goals=0.5,        # Environment goal ratio
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],             # Q-function network size
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],             # Policy network size
        ),
        version='normal',
        es_kwargs=dict(
            max_sigma=.2,                        # Exploration noise
        ),
        exploration_type='ou',                   # Ornstein-Uhlenbeck noise
        observation_key='state_observation',    # Use state observations (Oracle)
        desired_goal_key='state_desired_goal',  # Use state goals
        init_camera=sawyer_pick_and_place_camera, # Camera configuration for videos
        do_state_exp=True,                      # Enable state-based experiments

        # Video and logging configuration
        save_video=True,                        # Save videos for visualization
        imsize=84,                             # Image size (for video only)

        # Snapshotting configuration
        snapshot_mode='gap_and_last',          # Save checkpoints
        snapshot_gap=50,                       # Save every 50 epochs

        # Environment configuration
        env_class=SawyerPickAndPlaceEnv,       # Pick and place environment
        env_kwargs=dict(
            # Reward configuration
            reward_type='hand_and_obj_distance', # Distance-based reward
            indicator_threshold=0.06,           # Success threshold (6cm)
            
            # Object initialization
            obj_init_positions=((0.0, 0.6, 0.02),),  # Single object start position
            random_init=True,                   # Randomize initial positions
            
            # Goal configuration
            fix_goal=False,                     # Dynamic goal positioning
            hide_goal_markers=False,            # Show goal markers for debugging
            
            # Reset configuration
            reset_free=False,                   # Use full environment resets
            oracle_reset_prob=0.0,              # No oracle resets
            
            # Hand workspace bounds (pick and place requires larger workspace)
            hand_low=(-0.2, 0.55, 0.05),       # Hand position lower bounds
            hand_high=(0.2, 0.75, 0.3),        # Hand position upper bounds
            
            # Object workspace bounds
            obj_low=(-0.15, 0.55, 0.02),       # Object position lower bounds  
            obj_high=(0.15, 0.7, 0.02),        # Object position upper bounds
        ),

        algorithm='Oracle-PickAndPlace',        # Algorithm identifier
    )

    # Experiment configuration
    n_seeds = 1                               # Number of random seeds
    mode = 'here_no_doodad'                  # Local execution (no cluster)
    exp_prefix = 'rlkit-pick-and-place-oracle' # Experiment name prefix

    print("🎯 Starting Oracle Training for Sawyer Pick and Place Environment")
    print("=" * 70)
    print(f"Environment: {variant['env_class'].__name__}")
    print(f"Observation: {variant['observation_key']} (State-based)")
    print(f"Goal: {variant['desired_goal_key']}")
    print(f"Episodes: {variant['algo_kwargs']['td3_kwargs']['num_epochs']}")
    print(f"Episode Length: {variant['algo_kwargs']['td3_kwargs']['max_path_length']}")
    print(f"Success Threshold: {variant['env_kwargs']['indicator_threshold']}m")
    print(f"Object Start: {variant['env_kwargs']['obj_init_positions']}")
    print(f"Hand Workspace: {variant['env_kwargs']['hand_low']} to {variant['env_kwargs']['hand_high']}")
    print("=" * 70)

    # Run experiments for each seed
    for seed in range(n_seeds):
        print(f"\n🚀 Starting training run {seed + 1}/{n_seeds}")
        
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
    print(f"\nTraining Summary:")
    print(f"  - Total epochs: {variant['algo_kwargs']['td3_kwargs']['num_epochs']}")
    print(f"  - Steps per epoch: {variant['algo_kwargs']['td3_kwargs']['num_steps_per_epoch']}")
    print(f"  - Total environment steps: {variant['algo_kwargs']['td3_kwargs']['num_epochs'] * variant['algo_kwargs']['td3_kwargs']['num_steps_per_epoch']:,}")
    print(f"  - Algorithm: TD3 + HER with state observations")
    print(f"  - Task: Pick object and place at goal location")
    
    print("\nResults will be saved in:")
    print("  - Logs: rlkit/data/")
    print("  - Videos: rlkit/data/*/video_*.mp4")
    print("  - Snapshots: rlkit/data/*/params.pkl")
    print("  - Progress: rlkit/data/*/progress.csv")
    
    print("\nNext steps:")
    print("  1. Monitor training progress in the logs")
    print("  2. Watch training videos to see policy behavior")
    print("  3. Compare with vision-based methods (RIG)")
    print("  4. Use this as upper-bound baseline for evaluations")
