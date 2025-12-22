import sys
import os
import argparse
import pickle
import torch
import numpy as np

# Add required paths
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/rlkit')
sys.path.insert(0, '/media/aiserver/New Volume/HDD_linux/bear/AIP/final_project_aip/multiworld')

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.vae_wrapper import VAEWrappedEnv


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb"))
    
    # Handle different file formats
    if 'policy' in data and 'env' in data:
        # Standard format
        policy = data['policy']
        env = data['env']
    elif 'eval_policy' in data:
        # RLKit params format - use eval_policy for evaluation
        policy = data['eval_policy']
        env = data['env']
    else:
        raise ValueError(f"Could not find policy in file. Available keys: {list(data.keys())}")
    print("Policy and environment loaded")
    print(f"Policy type: {type(policy)}")
    
    # Reconstruct the policy if it has the RLKit format (params + __kwargs)
    def reconstruct_policy(old_policy):
        """Reconstruct a properly initialized policy from saved RLKit format"""
        if hasattr(old_policy, 'params') and hasattr(old_policy, '__kwargs'):
            print("Reconstructing policy from saved parameters...")
            from rlkit.torch.networks import TanhMlpPolicy
            
            # Create a new policy instance with the saved parameters
            new_policy = TanhMlpPolicy(**old_policy.__kwargs)
            
            # Load the saved state dict
            new_policy.load_state_dict(old_policy.params)
            
            print(f"Policy reconstructed with architecture: input={old_policy.__kwargs.get('input_size')}, "
                  f"output={old_policy.__kwargs.get('output_size')}, "
                  f"hidden={old_policy.__kwargs.get('hidden_sizes')}")
            
            return new_policy
        else:
            return old_policy
    
    # Always reconstruct the policy to ensure proper PyTorch module initialization
    if hasattr(policy, 'params') and hasattr(policy, '__kwargs'):
        print("Reconstructing policy for proper module initialization...")
        policy = reconstruct_policy(policy)
    else:
        print("Policy appears to be in standard format")
    if args.gpu:
        ptu.set_gpu_mode(True)
        print(f"GPU mode enabled, device: {ptu.device}")
        try:
            # Check if policy is a proper PyTorch module
            if isinstance(policy, torch.nn.Module):
                # Initialize _modules if missing (can happen with unpickling issues)
                if not hasattr(policy, '_modules'):
                    policy._modules = torch.nn.ModuleDict()
                
                # Initialize other required attributes if missing
                if not hasattr(policy, '_parameters'):
                    policy._parameters = {}
                if not hasattr(policy, '_buffers'):
                    policy._buffers = {}
                if not hasattr(policy, '_non_persistent_buffers_set'):
                    policy._non_persistent_buffers_set = set()
                if not hasattr(policy, '_backward_hooks'):
                    policy._backward_hooks = {}
                if not hasattr(policy, '_forward_hooks'):
                    policy._forward_hooks = {}
                if not hasattr(policy, '_forward_pre_hooks'):
                    policy._forward_pre_hooks = {}
                if not hasattr(policy, '_state_dict_hooks'):
                    policy._state_dict_hooks = {}
                if not hasattr(policy, '_load_state_dict_pre_hooks'):
                    policy._load_state_dict_pre_hooks = {}
                
                policy.to(ptu.device)
                print(f"Policy moved to device: {ptu.device}")
            else:
                # Try manual parameter transfer for non-standard modules
                try:
                    if hasattr(policy, 'parameters'):
                        for param in policy.parameters():
                            if hasattr(param, 'to'):
                                param.data = param.data.to(ptu.device)
                        print(f"Policy parameters manually moved to device: {ptu.device}")
                    else:
                        print(f"Policy is not a PyTorch module (type: {type(policy)}), using CPU")
                        ptu.set_gpu_mode(False)
                except Exception as inner_e:
                    print(f"Manual parameter transfer also failed: {inner_e}")
                    ptu.set_gpu_mode(False)
        except Exception as e:
            print(f"Failed to move policy to GPU: {e}")
            print("Falling back to CPU mode")
            ptu.set_gpu_mode(False)
    if isinstance(env, VAEWrappedEnv):
        try:
            env.mode(args.mode)
            print(f"Environment mode set to: {args.mode}")
        except AttributeError as e:
            print(f"Could not set environment mode: {e}")
            print("Continuing without setting mode...")
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        try:
            env.enable_render()
            print("Environment rendering enabled")
        except Exception as e:
            print(f"Could not enable environment rendering: {e}")
            print("Continuing without explicit render enabling...")
    policy.train(False)
    
    # Enable goal visualization for pusher task
    print("Setting up goal visualization...")
    if hasattr(env, 'render_goals'):
        env.render_goals = True
        print("Goal rendering enabled")
    
    # Set rendering mode for better visualization
    if hasattr(env, 'render_mode'):
        env.render_mode = 'human'
        print("Render mode set to human")
    
    # Enable rendering if not in hide mode
    if not args.hide:
        try:
            if hasattr(env, 'viewer') and env.viewer is None:
                # Initialize the viewer for MuJoCo environments
                env.render()
            print("Environment viewer initialized for goal visualization")
        except Exception as e:
            print(f"Could not initialize viewer: {e}")
    
    print("Starting policy simulation...")
    
    # Test environment reset
    try:
        obs = env.reset()
        print(f"Environment reset successful. Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Print goal information clearly
        if isinstance(obs, dict):
            print("\n" + "="*60)
            print(" GOAL INFORMATION FOR THIS EPISODE")
            print("="*60)
            if 'desired_goal' in obs:
                goal = obs['desired_goal']
                print(f" GOAL POSITION: [{goal[0]:+.3f}, {goal[1]:+.3f}]")
                
                # Provide context
                x_desc = "RIGHT" if goal[0] > 0.05 else "LEFT" if goal[0] < -0.05 else "CENTER"
                y_desc = "FORWARD" if goal[1] > 0.6 else "BACKWARD" if goal[1] < 0.5 else "MIDDLE"
                print(f" Goal is {x_desc} and {y_desc} in the workspace")
            
            if 'observation' in obs and len(obs['observation']) >= 4:
                obj_pos = obs['observation'][2:4]
                print(f" OBJECT START: [{obj_pos[0]:+.3f}, {obj_pos[1]:+.3f}]")
                
                if 'desired_goal' in obs:
                    goal = obs['desired_goal']
                    distance = np.sqrt((goal[0] - obj_pos[0])**2 + (goal[1] - obj_pos[1])**2)
                    print(f" DISTANCE: {distance:.3f}m")
                    
                    # Direction guidance
                    dx, dy = goal[0] - obj_pos[0], goal[1] - obj_pos[1]
                    if abs(dx) > abs(dy):
                        direction = "PUSH RIGHT " if dx > 0 else "PUSH LEFT "
                    else:
                        direction = "PUSH FORWARD " if dy > 0 else "PUSH BACK "
                    print(f"  TASK: {direction}")
            
            print("WATCH: The robot should push the object to the goal!")
            print("LOOK FOR: Goal marker or different colored indicator")
            print("="*60 + "\n")
        
    except Exception as e:
        print(f"Environment reset failed: {e}")
        print("Trying to access underlying environment...")
        # Try to get the underlying environment
        if hasattr(env, 'wrapped_env'):
            underlying_env = env.wrapped_env
            print(f"Found wrapped_env: {type(underlying_env)}")
        elif hasattr(env, '_wrapped_env'):
            underlying_env = env._wrapped_env
            print(f"Found _wrapped_env: {type(underlying_env)}")
        else:
            print(f"Environment type: {type(env)}")
            print(f"Environment attributes: {[attr for attr in dir(env) if not attr.startswith('_')]}")
        
        # Try common environment attribute names
        for attr_name in ['env', 'base_env', 'unwrapped']:
            if hasattr(env, attr_name):
                underlying_env = getattr(env, attr_name)
                print(f"Found {attr_name}: {type(underlying_env)}")
                try:
                    obs = underlying_env.reset()
                    print(f"{attr_name} reset successful")
                    env = underlying_env
                    break
                except Exception as e3:
                    print(f"{attr_name} reset failed: {e3}")
                    continue
        else:
            print("Could not find working underlying environment")
            return
        
        try:
            obs = underlying_env.reset()
            print(f"Underlying environment reset successful. Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            env = underlying_env  # Use the underlying environment directly
        except Exception as e2:
            print(f"Underlying environment reset also failed: {e2}")
            return
    
    paths = []
    episode_count = 0
    max_episodes = 1000  # Limit episodes for testing
    
    while episode_count < max_episodes:
        try:
            print(f"Running episode {episode_count + 1}/{max_episodes}")
            
            # Set goal visualization for this episode
            if hasattr(env, 'set_goal_visualization'):
                env.set_goal_visualization(True)
            
            path = multitask_rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=not args.hide,
                observation_key='observation',
                desired_goal_key='desired_goal',
            )
            paths.append(path)
            episode_count += 1
            
            # Print goal information for debugging
            if 'desired_goal' in path and len(path['desired_goal']) > 0:
                print(f"Goal for episode {episode_count}: {path['desired_goal'][0]}")
            if 'observations' in path and len(path['observations']) > 0:
                final_obs = path['observations'][-1]
                if isinstance(final_obs, dict) and 'achieved_goal' in final_obs:
                    print(f"Final achieved goal: {final_obs['achieved_goal']}")
            
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics(paths)
            if hasattr(env, "get_diagnostics"):
                for k, v in env.get_diagnostics(paths).items():
                    logger.record_tabular(k, v)
            logger.dump_tabular()
            
            print(f"Episode {episode_count} completed. Path length: {len(path['actions'])}")
            
            # Add a small pause between episodes for better visualization
            if not args.hide:
                import time
                time.sleep(1.0)
            
        except Exception as e:
            print(f"Error during episode {episode_count + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"Simulation completed. Total episodes: {len(paths)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
