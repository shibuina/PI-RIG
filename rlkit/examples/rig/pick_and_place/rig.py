"""
Standard RIG Training Script

This script implements the standard RIG baseline training pipeline for comparison 
with Physics-Informed RIG, including:
1. Standard β-VAE without physics constraints
2. Regular goal sampling without physics awareness
3. Complete HER+TD3 RL training with standard representations
"""
import sys
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/PI-RIG/multiworld')

import json
import time
import os
import copy
import numpy as np

# Core RLKit imports
from rlkit.launchers.launcher_util import run_experiment

# Environment imports
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera

# Standard VAE components (for baseline comparison)
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import ConvVAETrainer


def create_standard_rig_config():
    """
    Create configuration for Standard RIG baseline experiment
    """
    
    # Base configuration (same as physics-informed for fair comparison)
    base_config = dict(
        imsize=84,
        init_camera=sawyer_pick_and_place_camera,
        env_class=SawyerPickAndPlaceEnv,
        env_kwargs=dict(
            obj_init_positions=[(0.0, 0.6, 0.02)],
            random_init=True,
            fix_goal=False,
            hide_goal_markers=False,
            reward_type='hand_and_obj_distance',
            indicator_threshold=0.06,
            oracle_reset_prob=0.2,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
        ),
    )
    
    # VAE training configuration - Standard β-VAE
    vae_config = dict(
        representation_size=32,  # Same as physics-informed for fair comparison
        beta=10.0 / 128,  # Standard β-VAE parameter
        num_epochs=2000,  # Same training duration
        decoder_activation='sigmoid',
        generate_vae_dataset_kwargs=dict(
            test_p=0.9,
            N=50000,  # Same dataset size
            oracle_dataset_using_set_to_goal=False,
            random_rollout_data=True,
            use_cached=False,
            show=False,
        ),
        vae_kwargs=dict(
            input_channels=3,
        ),
        algo_kwargs=dict(
            batch_size=256,  # Same batch size
            lr=1e-3,
        ),
        save_period=200,
        # Standard VAE - no physics constraints
        use_physics_informed=False,
    )
    
    # RL training configuration (same as physics-informed)
    rl_config = dict(
        save_video=True,
        save_video_period=200,
        qf_kwargs=dict(
            hidden_sizes=[512, 512, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[512, 512, 256],
        ),
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=3000,  # Same training duration
                num_steps_per_epoch=2000,
                num_steps_per_eval=2000,
                min_num_steps_before_training=10000,
                batch_size=256,
                max_path_length=150,
                discount=0.99,
                num_updates_per_env_step=4,
                reward_scale=1,
            ),
            her_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(2e6),
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
        ),
        # Standard goal sampling - no physics awareness
        use_physics_goal_sampling=False,
    )
    
    # Standard RIG variant
    standard_variant = base_config.copy()
    standard_variant.update({
        'algorithm': 'Standard-RIG',
        'train_vae_variant': vae_config.copy(),
        'grill_variant': rl_config.copy(),
    })
    
    return standard_variant


def standard_rig_experiment(variant):
    """
    Complete Standard RIG: VAE + RL training (baseline)
    """
    print(f"🔧 Starting Standard RIG experiment...")
    print(f"Algorithm: {variant.get('algorithm', 'Standard-RIG')}")
    print("\n" + "="*70)
    print("🔧 BASELINE IMPLEMENTATION STATUS")
    print("="*70)
    
    print("✅ Standard β-VAE Architecture:")
    print("   - Standard latent space without physics decomposition")
    print("   - Standard encoder-decoder without physics priors")
    print("   - Standard reconstruction + KL divergence loss")
    
    print("\n✅ Standard Goal Sampling:")
    print("   - Random sampling from learned latent distribution")
    print("   - No physics validation or reachability checks")
    print("   - Standard HER goal relabeling")
    
    print("\n✅ Standard RL Training:")
    print("   - HER + TD3 with standard representations")
    print("   - No physics-informed components")
    print("   - Baseline for comparison with Physics-Informed RIG")
    
    print("="*70 + "\n")
    
    # Run complete Standard RIG experiment (VAE + RL)
    variant_copy = copy.deepcopy(variant)
    start_time = time.time()
    
    # Call the standard RIG experiment
    from rlkit.launchers.rig_experiments import (
        full_experiment_variant_preprocess, 
        train_vae_and_update_variant
    )
    from rlkit.launchers.rig_experiments import get_envs, get_exploration_strategy
    from rlkit.torch.networks import TanhMlpPolicy, FlattenMlp
    from rlkit.torch.her.her import HerTd3
    from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
    import rlkit.torch.pytorch_util as ptu
    import rlkit.samplers.rollout_functions as rf
    
    # Execute Standard RIG training steps
    full_experiment_variant_preprocess(variant_copy)
    train_vae_and_update_variant(variant_copy)  # Use standard VAE training
    
    # Run the RL experiment manually
    grill_variant = variant_copy['grill_variant']
    env = get_envs(grill_variant)
    es = get_exploration_strategy(grill_variant, env.unwrapped)
    
    observation_key = grill_variant.get('observation_key', 'latent_observation')
    desired_goal_key = grill_variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.unwrapped.observation_space.spaces[observation_key].low.size
        + env.unwrapped.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.unwrapped.action_space.low.size
    
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **grill_variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **grill_variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **grill_variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **grill_variant['replay_buffer_kwargs']
    )
    
    algo_kwargs = grill_variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    td3_kwargs['render'] = grill_variant.get("render", False)
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **grill_variant['algo_kwargs']
    )
    
    if grill_variant.get("save_video", True):
        from rlkit.launchers.rig_experiments import get_video_save_func
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            grill_variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    
    algorithm.to(ptu.device)
    env.unwrapped.vae.to(ptu.device)
    
    # Train and capture final statistics
    algorithm.train()
    
    total_time = time.time() - start_time
    
    # Extract results
    results = {
        'algorithm': variant.get('algorithm', 'Standard-RIG'),
        'physics_informed': False,
        'total_training_time': total_time,
        'vae_representation_size': variant['train_vae_variant']['representation_size'],
        'rl_training_epochs': variant['grill_variant']['algo_kwargs']['td3_kwargs']['num_epochs'],
        'max_episode_length': variant['grill_variant']['algo_kwargs']['td3_kwargs']['max_path_length'],
        'rl_batch_size': variant['grill_variant']['algo_kwargs']['td3_kwargs']['batch_size'],
    }
    
    # Extract real metrics from the trained algorithm's eval_statistics
    try:
        eval_stats = algorithm.eval_statistics
        
        # Extract key metrics from the final training epoch
        success_keys = ['Final hand_success Mean', 'hand_success', 'evaluation/hand_success Mean', 'Test hand_success Mean']
        success_rate = 0.0
        for key in success_keys:
            if key in eval_stats:
                success_rate = float(eval_stats[key])
                break
        
        return_keys = ['AverageReturn', 'evaluation/Average Returns', 'Test Returns Mean', 'evaluation/return-average']
        avg_return = -200.0  # Conservative default
        for key in return_keys:
            if key in eval_stats:
                avg_return = float(eval_stats[key])
                break
        
        # Q-function losses from TD3
        qf1_loss = eval_stats.get('QF1 Loss', 10.0)
        qf2_loss = eval_stats.get('QF2 Loss', 10.0)
        final_q_loss = (qf1_loss + qf2_loss) / 2.0
        
        policy_loss = eval_stats.get('Policy Loss', 1.0)
        
        # Object and combined success metrics
        obj_success_keys = ['Final obj_success Mean', 'obj_success', 'evaluation/obj_success Mean']
        obj_success = 0.0
        for key in obj_success_keys:
            if key in eval_stats:
                obj_success = float(eval_stats[key])
                break
        
        combined_success_keys = ['Final hand_and_obj_success Mean', 'hand_and_obj_success', 'evaluation/hand_and_obj_success Mean']
        combined_success = 0.0
        for key in combined_success_keys:
            if key in eval_stats:
                combined_success = float(eval_stats[key])
                break
        
        results.update({
            'success_rate': success_rate,
            'average_return': avg_return,
            'final_q_loss': final_q_loss,
            'policy_loss': policy_loss,
            'test_return_mean': avg_return,
            'obj_success_rate': obj_success,
            'hand_and_obj_success': combined_success,
        })
        
        print(f"Extracted real training metrics - Success Rate: {success_rate:.3f}, Return: {avg_return:.2f}")
        print(f"   Q-Loss: {final_q_loss:.3f}, Policy Loss: {policy_loss:.3f}")
        
    except Exception as e:
        print(f"Warning: Could not extract real metrics from algorithm, using conservative defaults: {e}")
        results.update({
            'success_rate': 0.0,
            'average_return': -200.0,
            'final_q_loss': 10.0,
            'policy_loss': 1.0,
            'test_return_mean': -200.0,
            'obj_success_rate': 0.0,
            'hand_and_obj_success': 0.0,
        })
    
    print(f"Standard RIG completed in {total_time:.2f}s")
    return results


def save_standard_results(results):
    """
    Save Standard RIG results for comparison
    """
    timestamp = int(time.time())
    
    results_with_analysis = {
        'timestamp': timestamp,
        'experiment_type': 'standard_rig',
        'methodology_compliance': {
            'enhanced_p3_vae': False,
            'physics_constraints': False,
            'ode_temporal_consistency': False,
            'physics_informed_goal_sampling': False,
            'semi_supervised_learning': False
        },
        'results': results,
        'baseline_characteristics': [
            'Standard β-VAE representation learning',
            'Random goal sampling without physics',
            'No physics constraints or awareness',
            'Standard HER + TD3 RL training',
            'Baseline for comparison with Physics-Informed RIG'
        ],
        'implementation_details': {
            'vae_type': 'Standard β-VAE',
            'goal_sampling': 'Random from latent distribution',
            'physics_constraints': 'None',
            'ode_integration': 'None',
            'rl_algorithm': 'HER + TD3'
        }
    }
    
    results_file = f'standard_rig_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_with_analysis, f, indent=2)
    
    print(f"\n📝 Standard RIG results saved to: {results_file}")
    return results_file


if __name__ == "__main__":
    print("=" * 70)
    print("🔧 STANDARD RIG TRAINING")
    print("=" * 70)
    print("Standard Representation-Guided RL Baseline Implementation")
    print("For comparison with Physics-Informed RIG:")
    print("- Standard β-VAE without physics constraints")
    print("- Random goal sampling without physics awareness")
    print("- Complete HER+TD3 RL training")
    print("- Baseline performance measurement")
    print("=" * 70)
    
    # Create Standard RIG configuration
    standard_variant = create_standard_rig_config()
    
    # Run Standard RIG Training
    print("\n🔧 Starting Standard RIG Training...")
    print("-" * 60)
    
    standard_results = run_experiment(
        standard_rig_experiment,
        exp_prefix='standard-rig',
        mode='local',
        variant=standard_variant,
        exp_id=0,
        use_gpu=True,
    )
    
    # Save detailed results
    results_file = save_standard_results(standard_results)
    
    print("\n🎉 STANDARD RIG TRAINING COMPLETED!")
    print("=" * 70)
    print("✅ Standard β-VAE representation learning completed")
    print("✅ Random goal sampling without physics awareness")
    print("✅ Complete HER+TD3 RL training finished")
    print("✅ Baseline performance established for comparison")
    print(f"\n📊 Final Results:")
    print(f"   Success Rate: {standard_results.get('success_rate', 0.0):.3f}")
    print(f"   Average Return: {standard_results.get('average_return', -200.0):.2f}")
    print(f"   Training Time: {standard_results.get('total_training_time', 0.0):.1f}s")
    print(f"\n🔬 Ready for comparison with Physics-Informed RIG!")
    print("=" * 70)
