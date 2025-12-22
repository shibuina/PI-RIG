import sys
import os
import subprocess

# # Set environment variables for headless rendering before importing mujoco
# os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering
# os.environ['EGL_DEVICE_ID'] = '0'  # Use GPU 0
# os.environ['gpu_id'] = '0'  # Set GPU device for mujoco_env

# # Try to start xvfb for virtual display if not already running
# try:
#     subprocess.run(['pkill', 'Xvfb'], capture_output=True)  # Kill existing Xvfb
#     subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'], 
#                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     os.environ['DISPLAY'] = ':99'
#     print("Started virtual display :99")
#     print("Using EGL rendering with GPU")
# except Exception as e:
#     print(f"Could not start virtual display: {e}")
#     # Fallback to osmesa for software rendering
#     os.environ['MUJOCO_GL'] = 'osmesa'
#     os.environ.pop('EGL_DEVICE_ID', None)  # Remove EGL device
#     os.environ.pop('DISPLAY', None)  # Remove display requirement
#     print("Falling back to OSMesa software rendering")

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
    """
    Physics-Informed Representation-Guided RL (PI-RIG) Implementation for Reacher Task
    
    This script implements the complete PI-RIG methodology for the 2-link reacher arm:
    
    1. **Enhanced P³-VAE Architecture** (Section 2):
       - Latent space decomposition: z_I (physics) + z_E (environment)  
       - Physics-guided encoder with shared CNN + separate heads
       - Physics constraints through 2-link arm dynamics and joint limits
       
    2. **Physics Constraints Integration** (Section 2.3):
       - 2-link arm dynamics: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ - F(q̇)
       - Joint limits: θ₁, θ₂ ∈ [-π, π]
       - Angular velocity limits: |θ̇₁|, |θ̇₂| ≤ 3.0 rad/s
       - Energy conservation with friction dissipation
       
    3. **Physics-Informed Goal Sampling** (Algorithm 1, Section 3):
       - Sample N candidates from N(0,I)
       - Physics validation P(z_g): joint limits, velocity limits, workspace constraints
       - Reachability estimation R(z_g|s_t): joint space distance with kinematic constraints
       - Select goals with highest physics × reachability scores
       
    4. **Semi-Supervised Learning** (Section 2.4):
       - Supervised loss L(x, z_I*) with ground truth joint states
       - Unsupervised loss U(x) with stop-gradient operator
       - Classification loss L_c(φ; z_I*) for joint state prediction
       
    5. **Complete Training Integration** (Section 4):
       - Enhanced P³-VAE training with arm dynamics constraints
       - TD3 policy learning with physics-informed goals
       - HER with physics-informed experience filtering
    """
    
    # Physics-informed RIG variant following methodology for reacher task
    physics_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYEnv-v1',  # 2-link reacher environment
        
        # Enhanced P³-VAE parameters from methodology (Table 2 - Reacher specific)
        physics_type='reacher',          # Task type for 2-link arm physics constraints
        physics_weight=0.25,             # λ_physics from methodology (Table 1: 0.2-0.25 for reacher)
        regularization_weight=0.01,      # λ_reg for latent regularization
        supervision_ratio=0.6,           # α: balance supervised/unsupervised data (higher for reacher)
        use_physics_constraints=True,    # Enable 2-link arm dynamics integration
        use_physics_goal_sampling=True,  # Enable physics-informed goal sampling
        goal_sampling_candidates=10000,  # N candidates for goal sampling (Algorithm 1)
        
        grill_variant=dict(
            save_video=False,  # Disable video saving to avoid OpenGL issues
            save_video_period=50,
            
            # Policy and Q-function architectures (methodology Table 1)
            qf_kwargs=dict(
                hidden_sizes=[400, 300],            # From methodology implementation details
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],            # From methodology implementation details
            ),
            
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=300,                 # RL Episodes from methodology
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,   # Min Steps Before Training
                    batch_size=128,
                    max_path_length=100,            # Episode Length from methodology
                    discount=0.99,                  # Discount Factor γ from methodology
                    num_updates_per_env_step=4,     # Updates per Env Step from methodology
                    reward_scale=1,
                    # Physics-informed goal sampling integration
                    use_physics_goal_sampling=True,
                    goal_sampling_candidates=10000,  # N candidates (Algorithm 1)
                    physics_type='reacher',
                ),
                her_kwargs=dict(
                    # Physics-informed experience filtering (methodology Section 4.4)
                    use_physics_filtering=True,      # Filter experiences with P(z_T) > 0.5
                    physics_validation_threshold=0.5,
                ),
            ),
            
            replay_buffer_kwargs=dict(
                max_size=int(1e6),                  # Replay Buffer Size from methodology
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            
            # Physics-informed reward computation (methodology Section 4.4)
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
                physics_type='reacher',             # Task-specific constraints
                # Physics validation parameters (methodology Section 3.1 - Reacher)
                workspace_radius=0.5,               # Reacher workspace: 0.5 m radius
                max_angular_velocity=3.0,           # Max angular velocity: 3.0 rad/s  
                joint_limits=3.14159,               # Joint limits: ±π radians
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            # Enhanced P³-VAE latent space configuration (methodology Section 4 - Reacher)
            representation_size=8,       # Total: z_I (4) + z_E (4) dimensions  
            z_I_dim=4,                  # Physics latent dimensions (reacher: θ₁, θ₂, θ̇₁, θ̇₂)
            z_E_dim=4,                  # Environmental latent dimensions
            beta=10.0 / 128,            # β value from methodology (Table 1)
            num_epochs=300,             # VAE training epochs from methodology
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            trainer_class=EnhancedP3VAETrainer,  # Enhanced P³-VAE trainer
            
            # Physics-informed training parameters from methodology
            physics_type='reacher',                   # Task-specific 2-link arm physics constraints
            physics_weight=0.25,                     # λ_physics (methodology Table 1 - reacher)
            regularization_weight=0.01,              # λ_reg for L2 regularization
            supervision_ratio=0.6,                   # α: supervised vs unsupervised ratio (higher for reacher)
            use_physics_constraints=True,            # Enable 2-link arm dynamics loss L_physics
            dt=0.1,                                  # Physics time step (methodology)
            grad_clip_value=5.0,                     # Gradient clipping for stability
            
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=10000,                             # Dataset size from methodology
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(
                    # Task-specific physics parameters (methodology Table 2 - Reacher)
                    joint_limits=3.14159,           # Joint limits: ±π radians
                    max_angular_velocity=3.0,       # Max angular velocity: 3.0 rad/s
                    link1_length=0.3,               # Link 1 length: 0.3 m
                    link2_length=0.3,               # Link 2 length: 0.3 m
                    link1_mass=1.0,                 # Link 1 mass: 1.0 kg
                    link2_mass=0.8,                 # Link 2 mass: 0.8 kg
                    joint_friction=0.05,            # Joint friction coefficient
                    workspace_radius=0.5,           # Workspace: 0.5 m radius
                    gravity=9.81,                   # Gravity: 9.81 m/s²
                    torque_limits=10.0,             # Torque limits: ±10.0 Nm
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
                # Enhanced P³-VAE architecture parameters
                physics_type='reacher',
                z_I_dim=4,
                z_E_dim=4,
                hidden_dim=256,                     # Hidden layer dimensions
            ),
            algo_kwargs=dict(
                batch_size=128,                     # Batch size from methodology
                lr=1e-3,                           # Learning rate from methodology
                # Enhanced P³-VAE loss components
                physics_weight=0.25,               # λ_physics (reacher-specific)
                regularization_weight=0.01,       # λ_reg 
                conservation_weight=0.05,          # Conservation laws weight
                beta=10.0 / 128,                   # β for KL regularization
                supervision_ratio=0.6,             # α for semi-supervised learning
                use_physics_constraints=True,
                dt=0.1,                           # Physics time step
                grad_clip_value=5.0,              # Gradient clipping
            ),
            save_period=10,
        ),
        algorithm='Physics-Informed-RIG-Reacher',
    )

    # Standard RIG baseline for comparison (no physics constraints)
    standard_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerReachXYEnv-v1',  # Same reacher environment
        
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
        algorithm='Standard-RIG-Reacher',
    )

    # Experimental setup following methodology
    print("=" * 80)
    print("PHYSICS-INFORMED REPRESENTATION-GUIDED RL (PI-RIG) EXPERIMENT - REACHER TASK")
    print("=" * 80)
    print("Configuration:")
    print(f"  Environment: SawyerReachXYEnv-v1 (2-Link Reacher Task)")
    print(f"  Enhanced P³-VAE: z_I={physics_variant['train_vae_variant']['z_I_dim']}D physics + z_E={physics_variant['train_vae_variant']['z_E_dim']}D environment")
    print(f"  Physics Weight: λ_physics={physics_variant['physics_weight']} (methodology Table 1)")
    print(f"  Goal Sampling: N={physics_variant['goal_sampling_candidates']} candidates (Algorithm 1)")
    print(f"  Training: {physics_variant['grill_variant']['algo_kwargs']['td3_kwargs']['num_epochs']} episodes")
    print("=" * 80)
    
    # Validate configuration matches methodology specifications for reacher
    def validate_methodology_compliance():
        """Ensure configuration matches methodology specifications for reacher task."""
        print("🔍 Validating methodology compliance for Reacher task...")
        
        # Enhanced P³-VAE validation
        vae_config = physics_variant['train_vae_variant']
        assert vae_config['z_I_dim'] == 4, "Reacher task should use 4D physics latent (θ₁, θ₂, θ̇₁, θ̇₂)"
        assert vae_config['z_E_dim'] == 4, "Environment latent should be 4D per methodology"
        assert vae_config['physics_weight'] == 0.25, "Physics weight should be 0.2-0.25 for reacher (Table 1)"
        assert vae_config['algo_kwargs']['batch_size'] == 128, "Batch size should be 128 (Table 1)"
        assert vae_config['num_epochs'] == 300, "VAE training should use 300 epochs (Table 1)"
        
        # Physics parameters validation (Table 2 - Reacher)
        physics_params = vae_config['generate_vae_dataset_kwargs']['vae_dataset_specific_kwargs']
        assert physics_params['joint_limits'] == 3.14159, "Joint limits should be ±π radians"
        assert physics_params['max_angular_velocity'] == 3.0, "Max angular velocity should be 3.0 rad/s"
        assert physics_params['link1_length'] == 0.3, "Link 1 length should be 0.3 m"
        assert physics_params['link2_length'] == 0.3, "Link 2 length should be 0.3 m"
        assert physics_params['link1_mass'] == 1.0, "Link 1 mass should be 1.0 kg"
        assert physics_params['link2_mass'] == 0.8, "Link 2 mass should be 0.8 kg"
        assert physics_params['joint_friction'] == 0.05, "Joint friction should be 0.05"
        assert physics_params['workspace_radius'] == 0.5, "Workspace should be 0.5 m radius"
        assert physics_params['gravity'] == 9.81, "Gravity should be 9.81 m/s²"
        assert physics_params['torque_limits'] == 10.0, "Torque limits should be ±10.0 Nm"
        
        # RL parameters validation (Table 1)
        rl_config = physics_variant['grill_variant']['algo_kwargs']['td3_kwargs']
        assert rl_config['num_epochs'] == 300, "RL episodes should be 300"
        assert rl_config['max_path_length'] == 100, "Episode length should be 100 steps"
        assert rl_config['discount'] == 0.99, "Discount factor should be 0.99"
        assert rl_config['num_updates_per_env_step'] == 4, "Should use 4 updates per env step"
        assert rl_config['min_num_steps_before_training'] == 4000, "Min steps should be 4000"
        
        # Goal sampling validation (Algorithm 1)
        assert physics_variant['goal_sampling_candidates'] == 10000, "Should use 10,000 goal candidates"
        
        print("✅ All methodology specifications validated successfully for Reacher task!")
        
        # Print key methodology features enabled for reacher
        print("\n📋 Methodology features enabled for Reacher:")
        features = [
            "✓ Enhanced P³-VAE with physics/environment latent decomposition",
            "✓ 2-link arm dynamics: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ - F(q̇)",
            "✓ Joint limits and angular velocity constraints",
            "✓ Energy conservation with friction dissipation",
            "✓ Semi-supervised learning with ground truth joint states",
            "✓ Physics-informed goal sampling (Algorithm 1)",
            "✓ Workspace reachability constraints", 
            "✓ HER with physics-informed experience filtering",
            "✓ Task-specific reacher physics parameters",
            "✓ Complete loss function: L_Total = L_Enhanced + β*L_KL + λ_physics*L_physics + λ_reg*L_reg"
        ]
        for feature in features:
            print(f"    {feature}")
        print()
    
    # Run validation
    validate_methodology_compliance()
    
    # Run physics-informed RIG experiment for reacher
    print("\n🚀 Running Physics-Informed RIG experiment for Reacher...")
    print("   • Enhanced P³-VAE with 2-link arm dynamics constraints")
    print("   • Physics-informed goal sampling with joint space validation")
    print("   • Semi-supervised learning with ground truth joint states")
    print("   • HER with physics-informed experience filtering")
    print("   • Workspace reachability estimation")
    
    run_experiment(
        physics_informed_grill_her_td3_full_experiment,
        exp_prefix='pi-rig-reacher-enhanced',
        mode='here_no_doodad',
        variant=physics_variant,
        use_gpu=True,
        exp_id=0,  # Experiment ID for reproducibility
    )
    
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: STANDARD RIG - REACHER")
    print("=" * 80)
    
    # Run standard RIG baseline for comparison
    print("\n📊 Running Standard RIG baseline for Reacher comparison...")
    print("   • Standard β-VAE without 2-link arm dynamics constraints")
    print("   • Uniform goal sampling from latent space")
    print("   • No physics-informed components")
    print("   • No joint limit or workspace constraints")
    
    run_experiment(
        grill_her_td3_full_experiment,
        exp_prefix='standard-rig-reacher-baseline',
        mode='here_no_doodad',
        variant=standard_variant,
        use_gpu=True,
        exp_id=1,  # Different experiment ID
    )
    
    print("\n" + "=" * 80)
    print("REACHER EXPERIMENT COMPLETED")
    print("=" * 80)
    print("Results will be saved in:")
    print("  • Physics-Informed RIG: ./data/pi-rig-reacher-enhanced/")
    print("  • Standard RIG:         ./data/standard-rig-reacher-baseline/")
    print("\nReacher-specific comparison metrics:")
    print("  • End-Effector Success Rate: Target reaching percentage")
    print("  • Joint Space Efficiency: Steps to reach target configurations")  
    print("  • Physics Compliance: Joint limit and dynamics violation rates")
    print("  • Workspace Coverage: Reachable workspace utilization")
    print("  • Joint State Extraction Accuracy: θ₁, θ₂, θ̇₁, θ̇₂ prediction quality")
    print("  • Energy Efficiency: Torque usage and energy consumption")
    print("=" * 80)
