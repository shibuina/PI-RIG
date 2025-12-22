"""
Physics-Informed Goal Sampling Algorithm
Implementation of Algorithm 1 from the methodology section.

This module provides the missing goal sampling functionality for Physics-Informed RIG.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
from rlkit.torch import pytorch_util as ptu


class PhysicsValidator:
    """Physics validation functions P(z_g) for different tasks."""
    
    def __init__(self, physics_type='pusher'):
        self.physics_type = physics_type
        self._initialize_task_params()
    
    def _initialize_task_params(self):
        """Initialize task-specific physics parameters."""
        if self.physics_type == 'pusher':
            self.workspace_bounds = {'x': (-0.4, 0.4), 'y': (-0.4, 0.4)}
            self.table_bounds = {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}
            self.max_velocity = 2.0  # m/s
            self.contact_radius = 0.1  # m
            
        elif self.physics_type == 'pick_and_place':
            self.workspace_bounds = {'x': (-0.3, 0.3), 'y': (-0.3, 0.3), 'z': (0.0, 0.4)}
            self.table_height = 0.05  # m
            self.gripper_range = (0.0, 0.05)  # m
            self.max_velocity = 1.0  # m/s
            
        elif self.physics_type == 'reacher':
            self.joint_limits = (-np.pi, np.pi)  # radians
            self.max_angular_velocity = 3.0  # rad/s
            self.workspace_radius = 0.5  # m
            
    def validate_pusher(self, physics_state: torch.Tensor) -> torch.Tensor:
        """
        Validate pusher physics state.
        
        Args:
            physics_state: [batch_size, 4] containing [hand_x, hand_y, puck_x, puck_y]
            
        Returns:
            torch.Tensor: Physics validity scores [batch_size]
        """
        batch_size = physics_state.shape[0]
        scores = torch.ones(batch_size, device=physics_state.device)
        
        hand_pos = physics_state[:, :2]  # [hand_x, hand_y]
        puck_pos = physics_state[:, 2:4]  # [puck_x, puck_y]
        
        # 1. Hand position within workspace bounds
        hand_in_workspace = (
            (hand_pos[:, 0] >= self.workspace_bounds['x'][0]) & 
            (hand_pos[:, 0] <= self.workspace_bounds['x'][1]) &
            (hand_pos[:, 1] >= self.workspace_bounds['y'][0]) & 
            (hand_pos[:, 1] <= self.workspace_bounds['y'][1])
        ).float()
        
        # 2. Puck position within table bounds
        puck_on_table = (
            (puck_pos[:, 0] >= self.table_bounds['x'][0]) & 
            (puck_pos[:, 0] <= self.table_bounds['x'][1]) &
            (puck_pos[:, 1] >= self.table_bounds['y'][0]) & 
            (puck_pos[:, 1] <= self.table_bounds['y'][1])
        ).float()
        
        # 3. No collision between hand and puck
        distance = torch.norm(hand_pos - puck_pos, dim=1)
        no_collision = (distance >= self.contact_radius).float()
        
        # Combine all constraints (multiplicative)
        scores = hand_in_workspace * puck_on_table * no_collision
        
        return scores
    
    def validate_pick_and_place(self, physics_state: torch.Tensor) -> torch.Tensor:
        """
        Validate pick-and-place physics state.
        
        Args:
            physics_state: [batch_size, 7] containing [gripper_opening, hand_xyz, obj_xyz]
            
        Returns:
            torch.Tensor: Physics validity scores [batch_size]
        """
        batch_size = physics_state.shape[0]
        scores = torch.ones(batch_size, device=physics_state.device)
        
        gripper_opening = physics_state[:, 0]  # gripper opening
        hand_pos = physics_state[:, 1:4]       # hand position [x, y, z]
        obj_pos = physics_state[:, 4:7]        # object position [x, y, z]
        
        # 1. Gripper opening in valid range [0, 0.05]
        gripper_valid = (
            (gripper_opening >= self.gripper_range[0]) & 
            (gripper_opening <= self.gripper_range[1])
        ).float()
        
        # 2. Object above table surface
        obj_above_table = (obj_pos[:, 2] >= self.table_height).float()
        
        # 3. Hand within workspace bounds
        hand_in_workspace = (
            (hand_pos[:, 0] >= self.workspace_bounds['x'][0]) & 
            (hand_pos[:, 0] <= self.workspace_bounds['x'][1]) &
            (hand_pos[:, 1] >= self.workspace_bounds['y'][0]) & 
            (hand_pos[:, 1] <= self.workspace_bounds['y'][1]) &
            (hand_pos[:, 2] >= self.workspace_bounds['z'][0]) & 
            (hand_pos[:, 2] <= self.workspace_bounds['z'][1])
        ).float()
        
        # 4. Grasp feasibility (simplified: hand should be close to object if gripper is closed)
        hand_obj_distance = torch.norm(hand_pos - obj_pos, dim=1)
        closed_gripper_mask = gripper_opening < 0.02
        grasp_feasible = torch.where(
            closed_gripper_mask,
            (hand_obj_distance < 0.1).float(),  # Hand should be close if gripper closed
            torch.ones_like(hand_obj_distance)  # No constraint if gripper open
        )
        
        # Combine all constraints
        scores = gripper_valid * obj_above_table * hand_in_workspace * grasp_feasible
        
        return scores
    
    def validate_reacher(self, physics_state: torch.Tensor) -> torch.Tensor:
        """
        Validate reacher physics state.
        
        Args:
            physics_state: [batch_size, 4] containing [θ₁, θ₂, θ̇₁, θ̇₂]
            
        Returns:
            torch.Tensor: Physics validity scores [batch_size]
        """
        batch_size = physics_state.shape[0]
        scores = torch.ones(batch_size, device=physics_state.device)
        
        joint_angles = physics_state[:, :2]      # [θ₁, θ₂]
        joint_velocities = physics_state[:, 2:4] # [θ̇₁, θ̇₂]
        
        # 1. Joint angles within limits [-π, π]
        angles_valid = (
            (joint_angles >= self.joint_limits[0]) & 
            (joint_angles <= self.joint_limits[1])
        ).all(dim=1).float()
        
        # 2. Angular velocities within safe limits
        velocities_valid = (torch.abs(joint_velocities) <= self.max_angular_velocity).all(dim=1).float()
        
        # 3. End-effector within reachable workspace (simplified 2-link arm)
        # Forward kinematics: approximate reachable workspace as circle
        L1, L2 = 0.3, 0.3  # Link lengths
        max_reach = L1 + L2
        x_end = L1 * torch.cos(joint_angles[:, 0]) + L2 * torch.cos(joint_angles.sum(dim=1))
        y_end = L1 * torch.sin(joint_angles[:, 0]) + L2 * torch.sin(joint_angles.sum(dim=1))
        end_effector_distance = torch.sqrt(x_end**2 + y_end**2)
        workspace_valid = (end_effector_distance <= self.workspace_radius).float()
        
        # Combine all constraints
        scores = angles_valid * velocities_valid * workspace_valid
        
        return scores
    
    def validate(self, physics_state: torch.Tensor) -> torch.Tensor:
        """Main validation function P(φ_i)."""
        if self.physics_type == 'pusher':
            return self.validate_pusher(physics_state)
        elif self.physics_type == 'pick_and_place':
            return self.validate_pick_and_place(physics_state)
        elif self.physics_type == 'reacher':
            return self.validate_reacher(physics_state)
        else:
            # Default: return all valid
            return torch.ones(physics_state.shape[0], device=physics_state.device)


class ReachabilityEstimator:
    """Reachability estimation functions R(z_g|s_t) for different tasks."""
    
    def __init__(self, physics_type='pusher'):
        self.physics_type = physics_type
        self._initialize_reachability_params()
    
    def _initialize_reachability_params(self):
        """Extract lambda values from environment workspace bounds."""
        if self.physics_type == 'pusher':
            # Extract from workspace bounds: (-0.4, 0.4) x (-0.4, 0.4) = 0.8 x 0.8m
            workspace_x = 0.4 - (-0.4)  # 0.8m
            workspace_y = 0.4 - (-0.4)  # 0.8m
            self.max_hand_reach = max(workspace_x, workspace_y)  # 0.8m
            self.max_puck_push = 0.5  # Task-specific constraint for puck dynamics
            
        elif self.physics_type == 'pick_and_place':
            # Extract from workspace bounds: (-0.3, 0.3) x (-0.3, 0.3) x (0.0, 0.4) = 0.6 x 0.6 x 0.4m
            workspace_x = 0.3 - (-0.3)  # 0.6m
            workspace_y = 0.3 - (-0.3)  # 0.6m
            workspace_z = 0.4 - 0.0      # 0.4m
            self.max_hand_reach = max(workspace_x, workspace_y)  # 0.6m
            self.max_obj_displacement = workspace_z             # 0.4m
            
        elif self.physics_type == 'reacher':
            # Extract from kinematic limits
            self.max_joint_change = np.pi  # Maximum joint angle change
            self.max_velocity_change = 3.0  # Maximum velocity change
            
        else:
            # Default fallback values
            self.max_hand_reach = 1.0
            self.max_obj_displacement = 1.0
    
    def estimate_pusher_reachability(self, goal_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """
        Estimate reachability for pusher task.
        
        Args:
            goal_state: [batch_size, 4] goal physics state
            current_state: [batch_size, 4] current physics state
            
        Returns:
            torch.Tensor: Reachability scores [batch_size]
        """
        # Distance-based reachability (simplified)
        current_hand = current_state[:, :2]
        current_puck = current_state[:, 2:4]
        goal_hand = goal_state[:, :2]
        goal_puck = goal_state[:, 2:4]
        
        # Hand movement distance
        hand_distance = torch.norm(goal_hand - current_hand, dim=1)
        # Puck movement distance
        puck_distance = torch.norm(goal_puck - current_puck, dim=1)
        
        # Reachability decreases with distance (exponential decay)
        # Lambda values extracted from environment workspace bounds
        hand_reachability = torch.exp(-hand_distance / self.max_hand_reach)
        puck_reachability = torch.exp(-puck_distance / self.max_puck_push)
        
        # Combined reachability (geometric mean)
        return torch.sqrt(hand_reachability * puck_reachability)
    
    def estimate_pick_and_place_reachability(self, goal_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Estimate reachability for pick-and-place task."""
        current_hand = current_state[:, 1:4]  # hand position
        current_obj = current_state[:, 4:7]   # object position
        goal_hand = goal_state[:, 1:4]
        goal_obj = goal_state[:, 4:7]
        
        # Hand reachability (lambda extracted from workspace bounds)
        hand_distance = torch.norm(goal_hand - current_hand, dim=1)
        hand_reachability = torch.exp(-hand_distance / self.max_hand_reach)
        
        # Object reachability (depends on whether it's grasped)
        obj_distance = torch.norm(goal_obj - current_obj, dim=1)
        obj_reachability = torch.exp(-obj_distance / self.max_obj_displacement)
        
        return torch.sqrt(hand_reachability * obj_reachability)
    
    def estimate_reacher_reachability(self, goal_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Estimate reachability for reacher task."""
        # Joint space distance
        joint_distance = torch.norm(goal_state[:, :2] - current_state[:, :2], dim=1)
        velocity_distance = torch.norm(goal_state[:, 2:4] - current_state[:, 2:4], dim=1)
        
        # Reachability based on joint and velocity changes (lambda from kinematic limits)
        joint_reachability = torch.exp(-joint_distance / self.max_joint_change)
        velocity_reachability = torch.exp(-velocity_distance / self.max_velocity_change)
        
        return torch.sqrt(joint_reachability * velocity_reachability)
    
    def estimate(self, goal_state: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """Main reachability estimation function R(φ_i|s_t)."""
        if self.physics_type == 'pusher':
            return self.estimate_pusher_reachability(goal_state, current_state)
        elif self.physics_type == 'pick_and_place':
            return self.estimate_pick_and_place_reachability(goal_state, current_state)
        elif self.physics_type == 'reacher':
            return self.estimate_reacher_reachability(goal_state, current_state)
        else:
            # Default: uniform reachability
            return torch.ones(goal_state.shape[0], device=goal_state.device)


class PhysicsStateExtractor(torch.nn.Module):
    """
    Neural network that extracts interpretable physics states from compressed latent codes z_I.
    This implements the learned mapping from compressed physics representation to interpretable variables.
    """
    
    def __init__(self, z_I_dim, physics_type='pusher', hidden_dim=128):
        super(PhysicsStateExtractor, self).__init__()
        self.physics_type = physics_type
        self.z_I_dim = z_I_dim
        
        # Determine output dimensions based on physics type
        if physics_type == 'pusher':
            self.physics_dim = 4  # [hand_x, hand_y, puck_x, puck_y]
        elif physics_type == 'pick_and_place':
            self.physics_dim = 7  # [gripper_opening, hand_xyz, obj_xyz]
        elif physics_type == 'reacher':
            self.physics_dim = 4  # [θ₁, θ₂, θ̇₁, θ̇₂]
        else:
            self.physics_dim = z_I_dim  # Default: same as input
        
        # Learned decoder network: z_I -> interpretable physics variables
        self.physics_decoder = torch.nn.Sequential(
            torch.nn.Linear(z_I_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.physics_dim)
        )
        
        # Task-specific output constraints/activations
        self._initialize_output_constraints()
    
    def _initialize_output_constraints(self):
        """Initialize task-specific output constraints."""
        if self.physics_type == 'pick_and_place':
            # Gripper opening should be in [0, 0.05], positions in reasonable bounds
            self.gripper_range = (0.0, 0.05)
            self.position_range = (-0.5, 0.5)  # Workspace bounds
        elif self.physics_type == 'pusher':
            self.position_range = (-0.5, 0.5)  # Workspace bounds
        elif self.physics_type == 'reacher':
            self.angle_range = (-torch.pi, torch.pi)  # Joint limits
            self.velocity_range = (-3.0, 3.0)  # Angular velocity limits
    
    def forward(self, z_I: torch.Tensor) -> torch.Tensor:
        """
        Extract physics state from latent representation.
        
        Args:
            z_I: Physics latent variables [batch_size, z_I_dim]
            
        Returns:
            torch.Tensor: Physics state variables [batch_size, physics_dim]
        """
        # Decode through learned network
        raw_physics = self.physics_decoder(z_I)
        
        # Apply task-specific constraints
        if self.physics_type == 'pick_and_place':
            # Constrain gripper opening to [0, 0.05]
            gripper = torch.sigmoid(raw_physics[:, 0:1]) * self.gripper_range[1]
            # Constrain positions to workspace bounds
            positions = torch.tanh(raw_physics[:, 1:]) * self.position_range[1]
            physics_state = torch.cat([gripper, positions], dim=1)
            
        elif self.physics_type == 'pusher':
            # Constrain all positions to workspace bounds
            physics_state = torch.tanh(raw_physics) * self.position_range[1]
            
        elif self.physics_type == 'reacher':
            # Constrain joint angles to [-π, π]
            joint_angles = torch.tanh(raw_physics[:, :2]) * torch.pi
            # Constrain velocities to safe limits
            joint_velocities = torch.tanh(raw_physics[:, 2:4]) * 3.0
            physics_state = torch.cat([joint_angles, joint_velocities], dim=1)
            
        else:
            # No constraints for unknown physics types
            physics_state = raw_physics
        
        return physics_state


class PhysicsInformedGoalSampler:
    """
    Physics-Informed Goal Sampling Algorithm (Algorithm 1 from methodology).
    """
    
    def __init__(self, vae_model, physics_type='pusher', z_I_dim=4, use_learned_extraction=True):
        self.vae_model = vae_model
        self.physics_type = physics_type
        self.z_I_dim = z_I_dim
        self.use_learned_extraction = use_learned_extraction
        
        # Initialize physics validator and reachability estimator
        self.physics_validator = PhysicsValidator(physics_type)
        self.reachability_estimator = ReachabilityEstimator(physics_type)
        
        # Initialize physics state extractor
        if use_learned_extraction:
            self.physics_extractor = PhysicsStateExtractor(
                z_I_dim=z_I_dim, 
                physics_type=physics_type
            )
            # Move to same device as VAE model
            if hasattr(vae_model, 'parameters'):
                device = next(vae_model.parameters()).device
                self.physics_extractor.to(device)
        else:
            self.physics_extractor = None
        
    def extract_physics_state(self, z_I: torch.Tensor) -> torch.Tensor:
        """
        Extract physics state φ from latent representation z_I.
        
        Args:
            z_I: Physics latent variables [batch_size, z_I_dim]
            
        Returns:
            torch.Tensor: Physics state variables [batch_size, physics_dim]
        """
        if self.use_learned_extraction and self.physics_extractor is not None:
            # Use learned neural network decoder
            return self.physics_extractor(z_I)
        else:
            # Fallback: direct dimension slicing (naive approach)
            if self.physics_type == 'pusher':
                return z_I[:, :4]  # [hand_x, hand_y, puck_x, puck_y]
            elif self.physics_type == 'pick_and_place':
                return z_I[:, :7]  # [gripper_opening, hand_xyz, obj_xyz]
            elif self.physics_type == 'reacher':
                return z_I[:, :4]  # [θ₁, θ₂, θ̇₁, θ̇₂]
            else:
                return z_I
    
    def sample_physics_informed_goal(
        self, 
        current_state: torch.Tensor, 
        num_candidates: int = 10000,
        return_best_k: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Sample physics-informed goals using Algorithm 1.
        
        Args:
            current_state: Current physics state [physics_dim]
            num_candidates: Number of goal candidates to generate (N in algorithm)
            return_best_k: Number of best goals to return
            
        Returns:
            tuple: (selected_goals [return_best_k, z_I_dim], statistics_dict)
        """
        device = ptu.device
        
        # Step 1: Sample N candidates from N(0, I)
        z_candidates = torch.randn(num_candidates, self.z_I_dim, device=device)
        
        # Step 2: Extract physics states φ_i = extract_physics_state(z_g^(i))
        physics_states = self.extract_physics_state(z_candidates)
        
        # Step 3: Compute physics scores s_physics^(i) = P(φ_i)
        physics_scores = self.physics_validator.validate(physics_states)
        
        # Step 4: Compute reachability scores s_reach^(i) = R(φ_i | s_t)
        current_state_batch = current_state.unsqueeze(0).expand(num_candidates, -1)
        reachability_scores = self.reachability_estimator.estimate(physics_states, current_state_batch)
        
        # Step 5: Compute total scores s_total^(i) = s_physics^(i) × s_reach^(i)
        total_scores = physics_scores * reachability_scores
        
        # Step 6: Select goals with highest scores
        _, top_indices = torch.topk(total_scores, return_best_k)
        selected_goals = z_candidates[top_indices]
        
        # Compute statistics
        statistics = {
            'mean_physics_score': physics_scores.mean().item(),
            'mean_reachability_score': reachability_scores.mean().item(),
            'mean_total_score': total_scores.mean().item(),
            'max_total_score': total_scores.max().item(),
            'valid_goals_fraction': (physics_scores > 0.5).float().mean().item(),
            'reachable_goals_fraction': (reachability_scores > 0.5).float().mean().item(),
        }
        
        return selected_goals, statistics
    
    def sample_goal_batch(
        self, 
        current_states: torch.Tensor, 
        num_candidates: int = 1000
    ) -> torch.Tensor:
        """
        Sample physics-informed goals for a batch of current states.
        
        Args:
            current_states: Current physics states [batch_size, physics_dim]
            num_candidates: Number of candidates per state
            
        Returns:
            torch.Tensor: Selected goals [batch_size, z_I_dim]
        """
        batch_size = current_states.shape[0]
        selected_goals = torch.zeros(batch_size, self.z_I_dim, device=current_states.device)
        
        for i in range(batch_size):
            goals, _ = self.sample_physics_informed_goal(
                current_states[i], 
                num_candidates=num_candidates,
                return_best_k=1
            )
            selected_goals[i] = goals[0]
        
        return selected_goals


def create_physics_informed_goal_sampler(vae_model, physics_type='pusher', z_I_dim=4, use_learned_extraction=True):
    """
    Factory function to create a physics-informed goal sampler.
    
    Args:
        vae_model: Trained Enhanced P3-VAE model
        physics_type: Task type ('pusher', 'pick_and_place', 'reacher')
        z_I_dim: Dimensionality of physics latent space
        use_learned_extraction: Whether to use learned neural network for physics extraction
        
    Returns:
        PhysicsInformedGoalSampler: Configured goal sampler
    """
    return PhysicsInformedGoalSampler(vae_model, physics_type, z_I_dim, use_learned_extraction)
