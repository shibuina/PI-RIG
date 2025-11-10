"""
PINN-based Physics-Informed RIG Experiment Launcher
Uses Physics-Informed Neural Networks (PINNs) with PDEs for more rigorous physics constraints
"""

import sys
import os

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')

from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment
from rlkit.launchers.physics_informed_rig_experiments import physics_informed_grill_her_td3_full_experiment
from rlkit.torch.vae.pinn_vae_trainer import PINNVAETrainer


def pinn_grill_her_td3_full_experiment(variant):
    """
    PINN-enhanced RIG experiment.
    Replaces the standard physics-informed trainer with PINN trainer.
    """
    # Copy the variant and modify the trainer
    pinn_variant = variant.copy()
    
    # Ensure we use the PINN trainer
    if 'train_vae_variant' in pinn_variant:
        pinn_variant['train_vae_variant']['trainer_class'] = PINNVAETrainer
        
        # Add PINN-specific parameters to algo_kwargs
        if 'algo_kwargs' not in pinn_variant['train_vae_variant']:
            pinn_variant['train_vae_variant']['algo_kwargs'] = {}
            
        # Set PINN parameters
        pinn_variant['train_vae_variant']['algo_kwargs'].update({
            'pinn_weight': variant.get('pinn_weight', 0.1),
            'physics_type': variant.get('physics_type', 'pusher'),
            'dt': variant.get('dt', 0.1),
            'enable_contact_dynamics': variant.get('enable_contact_dynamics', True),
            'enable_friction': variant.get('enable_friction', True),
            'enable_conservation_laws': variant.get('enable_conservation_laws', True),
        })
    
    # Run the standard physics-informed experiment with PINN trainer
    return physics_informed_grill_her_td3_full_experiment(pinn_variant)


if __name__ == "__main__":
    from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
    
    # PINN-based RIG variant for pusher task
    pinn_variant = dict(
        imsize=84,
        init_camera=sawyer_init_camera_zoomed_in,
        env_id='SawyerPushNIPS-v0',
        
        # PINN-specific parameters
        pinn_weight=0.15,              # Weight for PINN loss terms
        physics_type='pusher',         # Type of physics to enforce
        dt=0.1,                       # Time step for physics simulation
        enable_contact_dynamics=True,  # Enable contact force modeling
        enable_friction=True,          # Enable friction modeling
        enable_conservation_laws=True, # Enable momentum/energy conservation
        
        # Standard physics-informed parameters (for compatibility)
        physics_weight=0.1,
        contact_weight=0.05,
        momentum_weight=0.05,
        temporal_consistency_weight=0.02,
        
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
            representation_size=8,  # Larger latent space for physics states
            beta=10.0 / 128,
            num_epochs=400,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            trainer_class=PINNVAETrainer,  # Use PINN trainer
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=15000,
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
                # PINN parameters
                pinn_weight=0.15,
                physics_type='pusher',
                dt=0.1,
                enable_contact_dynamics=True,
                enable_friction=True,
                enable_conservation_laws=True,
            ),
            save_period=10,
        ),
        algorithm='PINN-RIG-Pusher',
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
                    num_epochs=300,
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
            representation_size=8,  # Same latent size for fair comparison
            beta=10.0 / 128,
            num_epochs=400,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=15000,
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
        algorithm='Standard-RIG-Pusher',
    )

    def run_pinn_comparison():
        """Run PINN-RIG vs Standard RIG comparison."""
        
        print("=" * 80)
        print("🧠 PINN-BASED PHYSICS-INFORMED RIG vs STANDARD RIG COMPARISON")
        print("=" * 80)
        print("PINN Approach:")
        print("  • Uses Physics-Informed Neural Networks (PINNs)")
        print("  • Enforces PDEs (Newton's laws, conservation laws)")
        print("  • Contact dynamics modeling")
        print("  • Momentum and energy conservation")
        print("  • Kinematic constraints")
        print()
        print("Training setup:")
        print(f"  • Environment: {pinn_variant['env_id']}")
        print(f"  • Image size: {pinn_variant['imsize']}x{pinn_variant['imsize']}")
        print(f"  • VAE epochs: {pinn_variant['train_vae_variant']['num_epochs']}")
        print(f"  • RL epochs: {pinn_variant['grill_variant']['algo_kwargs']['td3_kwargs']['num_epochs']}")
        print(f"  • Physics type: {pinn_variant['physics_type']}")
        print(f"  • PINN weight: {pinn_variant['pinn_weight']}")
        print(f"  • Time step: {pinn_variant['dt']}")
        print("=" * 80)
        
        # Run PINN-based RIG
        print("\n🧠 Running PINN-based RIG experiment...")
        print("This approach uses Physics-Informed Neural Networks to enforce:")
        print("  - Newton's laws of motion")
        print("  - Conservation of momentum and energy")
        print("  - Contact dynamics")
        print("  - Kinematic constraints")
        print("  - Friction modeling")
        
        try:
            pinn_results = run_experiment(
                pinn_grill_her_td3_full_experiment,
                exp_prefix='rlkit-pusher-pinn-rig',
                mode='here_no_doodad',
                variant=pinn_variant,
                use_gpu=True,
            )
            print("✅ PINN-RIG training completed successfully!")
        except Exception as e:
            print(f"❌ PINN-RIG training failed: {e}")
            import traceback
            traceback.print_exc()
            pinn_results = None
        
        # Run standard RIG for comparison
        print("\n📊 Running Standard RIG experiment for comparison...")
        print("This is the baseline approach without physics constraints.")
        
        try:
            standard_results = run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix='rlkit-pusher-standard-rig',
                mode='here_no_doodad',
                variant=standard_variant,
                use_gpu=True,
            )
            print("✅ Standard RIG training completed successfully!")
        except Exception as e:
            print(f"❌ Standard RIG training failed: {e}")
            standard_results = None
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 PINN-RIG vs STANDARD RIG COMPARISON RESULTS")
        print("=" * 80)
        
        if pinn_results and standard_results:
            print("✅ Both approaches completed successfully!")
            print("\nExpected advantages of PINN-RIG:")
            print("  • More physically consistent latent representations")
            print("  • Better sample efficiency due to physics priors")
            print("  • More robust to unseen situations")
            print("  • Smoother and more realistic trajectories")
            print("  • Better generalization across different scenarios")
            
        elif pinn_results:
            print("✅ PINN-RIG completed successfully")
            print("❌ Standard RIG encountered issues")
            
        elif standard_results:
            print("❌ PINN-RIG encountered issues") 
            print("✅ Standard RIG completed successfully")
            
        else:
            print("❌ Both approaches encountered issues")
            
        print("\n📁 Results saved in:")
        print("  • rlkit/data/08-xx-rlkit-pusher-pinn-rig/")
        print("  • rlkit/data/08-xx-rlkit-pusher-standard-rig/")
        
        print("\n📈 Key metrics to compare:")
        print("  • VAE reconstruction quality")
        print("  • Physics constraint satisfaction")
        print("  • RL sample efficiency")
        print("  • Task success rate")
        print("  • Latent space structure")
        
        print("\n🎉 PINN-RIG comparison experiment completed!")
        print("=" * 80)
        
        return pinn_results, standard_results
    
    # Run the comparison
    run_pinn_comparison()
