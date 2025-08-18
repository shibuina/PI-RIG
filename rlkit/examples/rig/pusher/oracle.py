from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_td3_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=501,  # Reduced for faster feedback
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=50,  # Reduced to focus on quick success
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=1.0,
                render=False,
                tau=0.005,  # Fixed: was too high at 1e-2, causing instability
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # Increased from 0.1 for better HER
            fraction_goals_env_goals=0.0,  # Changed from 0.5 - focus on HER goals
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        version='normal',
        es_kwargs=dict(
            max_sigma=.3,  # Increased exploration for better coverage
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=True,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=25,  # More frequent snapshots for better monitoring

        env_class=SawyerPushAndReachXYEasyEnv,
        env_kwargs=dict(
            # hide_goal=True,  # Commented out - hiding goal makes task unnecessarily hard
            reward_info=dict(
                type="state_distance",
            ),
            # Make the task easier by allowing larger hand movements  
            pos_action_scale=3.0 / 100,  # Increased from default 2.0/100 for bigger actions
            # Reduce frame skip for better control
            frame_skip=25,  # Reduced from default 50 for more responsive control
        ),

        algorithm='HER_TD3_Improved',  # More accurate description
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'rlkit-pusher-her-td3-improved'

    for _ in range(n_seeds):
        run_experiment(
            her_td3_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            use_gpu=True,  # Turn on if you have a GPU
        )
