import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize48_default_architecture_with_more_hidden_layers
from rlkit.launchers.cvae_experiments import grill_her_td3_offpolicy_online_vae_full_experiment
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from rlkit.torch.vae.conditional_conv_vae import DeltaCVAE
from rlkit.torch.vae.conditional_vae_trainer import DeltaCVAETrainer
from rlkit.data_management.online_conditional_vae_replay_buffer import \
        OnlineConditionalVaeRelabelingBuffer

# Pick and place workspace parameters - larger workspace for manipulation
x_var = 0.2   # X workspace limit (wider than pusher/reacher)
y_var = 0.15  # Y workspace limit
z_var = 0.25  # Z workspace limit (more vertical space for lifting)

# Define 3D workspace bounds for pick and place
x_low = -x_var
x_high = x_var
y_low = 0.55  # Base Y position for pick and place
y_high = 0.75  # Smaller Y range, more focused workspace
z_low = 0.05  # Can go lower to grasp objects
z_high = z_low + z_var

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_pick_and_place_camera,
        env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            # Pick and place specific environment parameters
            obj_init_positions=((0.0, 0.6, 0.02),),  # Object spawn position
            random_init=True,                          # Random initialization
            fix_goal=False,                           # Dynamic goal positioning
            hide_goal_markers=True,                   # Hide visual goal markers for vision-based learning
            reset_free=False,                         # Use full environment resets
            oracle_reset_prob=0.0,                    # No oracle resets for fair comparison
            reward_type='hand_and_obj_distance',      # Distance-based reward function
            indicator_threshold=0.06,                 # Success threshold (6cm)
            
            # Workspace bounds - larger than pusher for pick and place
            hand_low=(x_low, y_low, z_low),          # Hand position lower bounds
            hand_high=(x_high, y_high, z_high),      # Hand position upper bounds
            obj_low=(x_low + 0.02, y_low + 0.02, 0.02),    # Object position bounds
            obj_high=(x_high - 0.02, y_high - 0.02, 0.02),  # Keep objects on table surface
            
            # Camera and rendering
            action_repeat=1,
            use_textures=True,
            init_camera=sawyer_pick_and_place_camera,
        ),

        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=0,
            ),
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=100,  # Longer episodes for pick and place complexity
            algo_kwargs=dict(
                batch_size=128,
                num_epochs=1001,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=1000,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=4000,
                vae_training_schedule=vae_schedules.never_train,
                oracle_data=False,
                vae_save_period=25,
                parallel_vae_train=False,
                dataset_path=None,
                rl_offpolicy_num_training_steps=0,
            ),
            td3_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1.0,
                tau=1e-2,
            ),
            replay_buffer_class=OnlineConditionalVaeRelabelingBuffer,
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
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
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                epsilon=0.06,  # Pick and place tolerance (6cm as standard)
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='ONLINE-VAE-SAC-BERNOULLI',
        ),
        
        train_vae_variant=dict(
            latent_sizes=6,  # 6D latent space for pick and place (hand + object positions)
            beta=10,
            beta_schedule_kwargs=dict(
                x_values=(0, 1000),
                y_values=(1, 100),
            ),
            context_schedule=1,
            num_epochs=1500,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=100000,
                n_random_steps=10,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_pick_and_place_tuning",
            ),
            vae_trainer_class=DeltaCVAETrainer,
            vae_class=DeltaCVAE,
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture_with_more_hidden_layers,
                decoder_distribution='gaussian_identity_variance',
            ),

            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=1e-3,
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
        region='us-west-2',

        logger_variant=dict(
            tensorboard=True,
        ),
    )

    # Hyperparameter search space optimized for pick and place task
    search_space = {
        'train_vae_variant.latent_sizes': [(6,), (8,), (10,)],  # Test different latent dimensions for complex task
        'train_vae_variant.context_schedule': [
            1.0,
        ],
        'train_vae_variant.beta_schedule_kwargs': [
            dict(x_values=(0, 1500,), y_values=(1, 50)),
        ],
        'train_vae_variant.algo_kwargs.batch_size': [128, ],
        'train_vae_variant.algo_kwargs.lr': [1e-4, ],
        'train_vae_variant.algo_kwargs.weight_decay': [1e-4, ],
        'grill_variant.algo_kwargs.num_trains_per_train_loop':[1000,],
        'grill_variant.algo_kwargs.batch_size': [128,],
        'grill_variant.exploration_noise': [0.3],
        'grill_variant.max_path_length': [100],  # Pick and place appropriate episode length
    }
    
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 5
    mode = 'local'  # Can be changed to 'ec2' for cloud training
    exp_prefix = 'ccrig-pick-and-place'

    print(" Starting CCRIG training for Pick and Place Environment")
    print(f"Experiment prefix: {exp_prefix}")
    print(f"Number of seeds: {n_seeds}")
    print(f"Mode: {mode}")
    print(f"Workspace bounds:")
    print(f"  Hand: X[{x_low:.2f}, {x_high:.2f}], Y[{y_low:.2f}, {y_high:.2f}], Z[{z_low:.2f}, {z_high:.2f}]")
    print(f"  Object spawn: (0.0, 0.6, 0.02)")
    print(f"  Success threshold: 6cm")
    print(f"  Episode length: 100 steps")
    
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        print(f"\nStarting experiment {exp_id+1} with latent size: {variant['train_vae_variant']['latent_sizes']}")
        for seed_id in range(n_seeds):
            print(f"  Running seed {seed_id+1}/{n_seeds}")
            run_experiment(
                grill_her_td3_offpolicy_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                region='us-west-2',
                use_gpu=True,
            )
    
    print("\nCCRIG Pick and Place training completed!")
