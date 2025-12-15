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
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture

# Configuration for resuming training
RESUME_TRAINING = True  # Set to True to resume from checkpoint
CHECKPOINT_PATH = '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit_original/data/08-26-08-26-skewfit-reacher-local-test/08-26-08-26-skewfit-reacher-local-test_2025_08_26_09_07_03_0000--s-45166'

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint files in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None, None
    
    # Look for progress.csv to determine last completed epoch
    progress_file = os.path.join(checkpoint_dir, 'progress.csv')
    latest_epoch = 0
    
    if os.path.exists(progress_file):
        try:
            # Simple approach: read the last line and extract epoch
            with open(progress_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    last_line = lines[-1].strip()
                    if last_line:
                        latest_epoch = int(last_line.split(',')[0])
            print(f"📊 Found progress.csv - Last completed epoch: {latest_epoch}")
        except Exception as e:
            print(f"⚠️  Could not read progress.csv: {e}")
    
    # Look for checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pkl') and ('params' in file or 'algorithm' in file):
            checkpoint_files.append(file)
    
    return latest_epoch, checkpoint_files


if __name__ == "__main__":
    # Base variant configuration
    variant = dict(
        algorithm='Skew-Fit',
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYZEnv-v1',  # Changed to reacher environment
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
            max_path_length=50,  # Suitable for reaching tasks
            algo_kwargs=dict(
                batch_size=256,  # Reduced for local testing
                num_epochs=300,  # Target 300 epochs total
                num_eval_steps_per_epoch=200,  # Reduced
                num_expl_steps_per_train_loop=200,  # Reduced
                num_trains_per_train_loop=200,  # Reduced
                min_num_steps_before_training=2000,  # Reduced
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
                max_size=int(50000),  # Reduced for local testing
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
            representation_size=4,  # Can be adjusted for reacher complexity
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=40,  # Number of training data points
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

    # Add resume functionality if specified
    if RESUME_TRAINING:
        latest_epoch, checkpoint_files = find_latest_checkpoint(CHECKPOINT_PATH)
        
        if latest_epoch > 0 and checkpoint_files:
            variant['resume_training'] = True
            variant['checkpoint_path'] = CHECKPOINT_PATH
            # Start from the next epoch after the last completed one
            variant['skewfit_variant']['algo_kwargs']['start_epoch'] = latest_epoch
            variant['skewfit_variant']['algo_kwargs']['num_epochs'] = 300
            
            # FIX for CSV key mismatch: Disable VAE training in resume mode
            # The VAE is already trained, so we just continue RL training
            variant['train_vae_variant']['num_epochs'] = 0
            variant['skewfit_variant']['algo_kwargs']['vae_training_schedule'] = vae_schedules.never_train
            variant['skewfit_variant']['algo_kwargs']['oracle_data'] = True
            
            print(f"🔄 RESUMING TRAINING from checkpoint: {CHECKPOINT_PATH}")
            print(f"📊 Last completed epoch: {latest_epoch}")
            print(f"📊 Continuing from epoch {latest_epoch} to reach 300 total epochs")
            print(f"📁 Available checkpoint files: {checkpoint_files}")
            print(f"🔧 VAE training: DISABLED (using saved VAE to avoid CSV key mismatch)")
        else:
            print(f"❌ ERROR: Could not find valid checkpoint at {CHECKPOINT_PATH}")
            print(f"🔄 Switching to fresh training...")
            variant['resume_training'] = False
            RESUME_TRAINING = False
    
    if not RESUME_TRAINING:
        variant['resume_training'] = False
        print("🆕 STARTING NEW TRAINING from scratch")

    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # Configure for local execution
    n_seeds = 1
    mode = 'here_no_doodad'  # Use local execution without doodad
    
    if RESUME_TRAINING:
        exp_prefix = '08-26-skewfit-reacher-resume-300epochs'
    else:
        exp_prefix = '08-26-skewfit-reacher-local-test'

    print("🚀 Starting Skew-Fit Reacher Experiment")
    print("=" * 60)
    print(f"Algorithm: {variant['algorithm']}")
    print(f"Environment: {variant['env_id']}")
    print(f"Image size: {variant['imsize']}")
    print(f"Max path length: {variant['skewfit_variant']['max_path_length']}")
    print(f"Representation size: {variant['train_vae_variant']['representation_size']}")
    print(f"Target epochs: {variant['skewfit_variant']['algo_kwargs']['num_epochs']}")
    
    if RESUME_TRAINING:
        print(f"Resume mode: ENABLED")
        print(f"Starting from epoch: {variant['skewfit_variant']['algo_kwargs'].get('start_epoch', 0)}")
        print(f"Checkpoint path: {CHECKPOINT_PATH}")
    else:
        print(f"Resume mode: DISABLED (fresh start)")
    
    print(f"Number of seeds: {n_seeds}")
    print(f"Execution mode: {mode}")
    print(f"Experiment prefix: {exp_prefix}")
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
