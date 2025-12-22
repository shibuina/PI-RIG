import sys
import os
import subprocess

# Set environment variables for headless rendering before importing mujoco
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
os.environ['EGL_DEVICE_ID'] = '0'  # Use GPU 0
os.environ['gpu_id'] = '0'  # Set GPU device for mujoco_env

# Try to start xvfb for virtual display if not already running
try:
    subprocess.run(['pkill', 'Xvfb'], capture_output=True)  # Kill existing Xvfb
    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ['DISPLAY'] = ':99'
    print("Started virtual display :99")
    print("Using EGL rendering with GPU")
except Exception as e:
    print(f"Could not start virtual display: {e}")
    # Fallback to osmesa for software rendering
    os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ.pop('EGL_DEVICE_ID', None)  # Remove EGL device
    os.environ.pop('DISPLAY', None)  # Remove display requirement
    print("Falling back to OSMesa software rendering")

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG/multiworld')

from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.launchers.physics_informed_rig_experiments import physics_informed_grill_her_td3_full_experiment
from rlkit.torch.vae.enhanced_p3_vae_trainer import EnhancedP3VAETrainer
from rlkit.torch.vae.physics_informed_goal_sampling import create_physics_informed_goal_sampler

if __name__ == "__main__":
    
    # Physics-informed RIG variant 
    physics_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        
        # Enhanced P³-VAE parameters 
        physics_type='pusher',           # Task type for physics constraints
        physics_weight=0.3,              # λ_physics  (Table 1: 0.3 for pusher)
        regularization_weight=0.01,      # λ_reg for latent regularization
        supervision_ratio=0.5,           # α: balance supervised/unsupervised data
        use_physics_constraints=True,    # Enable physics constraint integration
        use_physics_goal_sampling=True,  # Enable physics-informed goal sampling
        goal_sampling_candidates=10000,  # N candidates for goal sampling (Algorithm 1)
        
        grill_variant=dict(
            save_video=False,  # Disable video saving to avoid OpenGL issues
            save_video_period=50,
            
            # Policy and Q-function architectures 
            qf_kwargs=dict(
                hidden_sizes=[400, 300],            #  implementation details
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],            #  implementation details
            ),
            
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=300,                 # RL Episodes 
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,   # Min Steps Before Training
                    batch_size=128,
                    max_path_length=100,            # Episode Length 
                    discount=0.99,                  # Discount Factor γ 
                    num_updates_per_env_step=4,     # Updates per Env Step 
                    reward_scale=1,
                    # Physics-informed goal sampling integration
                    use_physics_goal_sampling=True,
                    goal_sampling_candidates=10000,  # N candidates (Algorithm 1)
                    physics_type='pusher',
                ),
                her_kwargs=dict(
                    # Physics-informed experience filtering 
                    use_physics_filtering=True,      # Filter experiences with P(z_T) > 0.5
                    physics_validation_threshold=0.5,
                ),
            ),
            
            replay_buffer_kwargs=dict(
                max_size=int(1e6),                  # Replay Buffer Size 
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            
            # Physics-informed reward computation 
            reward_params=dict(
                type='latent_distance',             # Standard RIG reward formulation
                use_physics_distance=False,        # Keep standard reward for compatibility
            ),
            
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=False,       # Use physics-informed goal sampling
                use_physics_goal_sampling=True,     # Enable Algorithm 1
                goal_sampling_candidates=10000,     # N candidates for sampling
                physics_type='pusher',              # Task-specific constraints
                # Physics validation parameters 
                workspace_bounds=0.8,               # Pusher workspace: 0.8×0.8 m
                max_velocity=2.0,                   # Max velocity: 2.0 m/s  
                contact_radius=0.1,                 # Contact threshold: 0.1 m
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            # Enhanced P³-VAE latent space configuration
            representation_size=8,       # Total: z_I (4) + z_E (4) dimensions
            z_I_dim=4,                  # Physics latent dimensions (pusher: hand_pos + puck_pos)
            z_E_dim=4,                  # Environmental latent dimensions
            beta=10.0 / 128,            # β value 
            num_epochs=300,             # VAE training epochs 
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            trainer_class=EnhancedP3VAETrainer,  # Enhanced P³-VAE trainer
            
            # Physics-informed training parameters 
            physics_type='pusher',                    # Task-specific physics constraints
            physics_weight=0.3,                      # λ_physics 
            regularization_weight=0.01,              # λ_reg for L2 regularization
            supervision_ratio=0.5,                   # α: supervised vs unsupervised ratio
            use_physics_constraints=True,            # Enable physics loss L_physics
            dt=0.1,                                  # Physics time step 
            grad_clip_value=5.0,                     # Gradient clipping for stability
            
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,                             # Dataset size 
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(
                    # Task-specific physics parameters 
                    contact_threshold=0.1,           # Contact threshold for pusher
                    friction_coefficient=0.3,       # Friction coefficient  
                    mass_hand=1.0,                  # Hand mass (kg)
                    mass_puck=0.5,                  # Puck mass (kg)
                    max_velocity=2.0,               # Max velocity (m/s)
                    workspace_bounds=0.8,           # Workspace: 0.8×0.8 m
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
                # Enhanced P³-VAE architecture parameters
                physics_type='pusher',
                z_I_dim=4,
                z_E_dim=4,
                hidden_dim=256,                     # Hidden layer dimensions
            ),
            algo_kwargs=dict(
                batch_size=128,                     # Batch size 
                lr=1e-3,                           # Learning rate 
                # Enhanced P³-VAE loss components
                physics_weight=0.3,                # λ_physics (pusher-specific)
                regularization_weight=0.01,       # λ_reg 
                conservation_weight=0.05,          # Conservation laws weight
                beta=10.0 / 128,                   # β for KL regularization
                supervision_ratio=0.5,             # α for semi-supervised learning
                use_physics_constraints=True,
                dt=0.1,                           # Physics time step
                grad_clip_value=5.0,              # Gradient clipping
            ),
            save_period=10,
        ),
        algorithm='Physics-Informed-RIG',
    )

    # Standard RIG baseline for comparison (no physics constraints)
    standard_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        
        # Standard RIG parameters (no physics)
        physics_type='none',
        use_physics_constraints=False,
        use_physics_goal_sampling=False,
        grill_variant=dict(
            save_video=False,  # Disable video saving to avoid OpenGL issues
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
            representation_size=4,              # Standard VAE latent size
            beta=10.0 / 128,                    # Same β as physics-informed for fair comparison
            num_epochs=300,                     # Same training epochs
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            
            # Standard VAE trainer (no physics constraints)
            use_physics_constraints=False,
            physics_weight=0.0,                 # Disable physics loss
            
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,                        # Same dataset size
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
                batch_size=128,                 # Same batch size
                lr=1e-3,                       # Same learning rate
                # No physics parameters for baseline
                physics_weight=0.0,
                use_physics_constraints=False,
            ),
            save_period=10,
        ),
        algorithm='Standard-RIG',
    )

    print("=" * 80)
    print("PHYSICS-INFORMED REPRESENTATION-GUIDED RL (PI-RIG) EXPERIMENT")
    print("=" * 80)
    print("Configuration:")
    print(f"  Environment: SawyerPushNIPS-v0 (Pusher Task)")
    print(f"  Enhanced P³-VAE: z_I={physics_variant['train_vae_variant']['z_I_dim']}D physics + z_E={physics_variant['train_vae_variant']['z_E_dim']}D environment")
    print(f"  Physics Weight: λ_physics={physics_variant['physics_weight']} ")
    print(f"  Goal Sampling: N={physics_variant['goal_sampling_candidates']} candidates (Algorithm 1)")
    print(f"  Training: {physics_variant['grill_variant']['algo_kwargs']['td3_kwargs']['num_epochs']} episodes")
    print("=" * 80)

    
    # Run physics-informed RIG experiment
    print("\n Running Physics-Informed RIG experiment...")
    print("   • Enhanced P³-VAE with physics constraints")
    print("   • Physics-informed goal sampling (Algorithm 1)")
    print("   • Semi-supervised learning with ground truth physics")
    print("   • HER with physics-informed experience filtering")
    
    run_experiment(
        physics_informed_grill_her_td3_full_experiment,
        exp_prefix='pi-rig-pusher-enhanced',
        mode='here_no_doodad',
        variant=physics_variant,
        use_gpu=True,
        exp_id=0,  # Experiment ID for reproducibility
    )

 
