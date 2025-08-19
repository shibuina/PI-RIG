"""
Physics configuration for Sawyer Pick and Place environment.

This module provides physics-specific utilities and configurations for 
the Pick and Place task, including:
- Contact detection between hand and object
- Gravity effects on objects
- Grasp stability constraints
- Observation space utilities

Observation structure for Pick and Place:
- Gripper position: [0] (0.0 to 0.04)
- Hand position: [1:4] (xyz coordinates)  
- Object position: [4:7] (xyz coordinates)
"""

import numpy as np
import torch
import torch.nn.functional as F


class PickAndPlacePhysics:
    """
    Physics utilities for Pick and Place environment
    """
    
    def __init__(self, contact_threshold=0.05, gravity_strength=9.81):
        """
        Args:
            contact_threshold: Distance threshold for hand-object contact (meters)
            gravity_strength: Gravity acceleration (m/s^2)
        """
        self.contact_threshold = contact_threshold
        self.gravity_strength = gravity_strength
        
        # Object properties (assuming small cube/sphere)
        self.object_mass = 0.1  # kg
        self.object_radius = 0.02  # meters
        
        # Environment bounds
        self.table_height = 0.05  # Height of table surface
        self.hand_bounds = {
            'low': np.array([-0.2, 0.55, 0.05]),
            'high': np.array([0.2, 0.75, 0.3])
        }
        self.obj_bounds = {
            'low': np.array([-0.2, 0.55, 0.02]),
            'high': np.array([0.2, 0.75, 0.3])
        }
    
    def extract_positions(self, observations):
        """
        Extract hand and object positions from observations.
        
        Args:
            observations: Tensor of shape (batch_size, 7) 
                         [gripper, hand_x, hand_y, hand_z, obj_x, obj_y, obj_z]
        
        Returns:
            hand_pos: Tensor of shape (batch_size, 3)
            obj_pos: Tensor of shape (batch_size, 3)
            gripper_pos: Tensor of shape (batch_size, 1)
        """
        gripper_pos = observations[:, 0:1]  # Gripper opening
        hand_pos = observations[:, 1:4]     # Hand xyz
        obj_pos = observations[:, 4:7]      # Object xyz
        
        return hand_pos, obj_pos, gripper_pos
    
    def compute_contact_forces(self, hand_pos, obj_pos, hand_vel=None, obj_vel=None):
        """
        Compute contact forces between hand and object.
        
        Args:
            hand_pos: Hand position (batch_size, 3)
            obj_pos: Object position (batch_size, 3)
            hand_vel: Hand velocity (batch_size, 3), optional
            obj_vel: Object velocity (batch_size, 3), optional
        
        Returns:
            contact_mask: Boolean mask for contact (batch_size,)
            contact_force: Contact force magnitude (batch_size,)
        """
        # Compute distance between hand and object
        distance = torch.norm(hand_pos - obj_pos, dim=1)
        contact_mask = distance < self.contact_threshold
        
        # Simple contact force model
        penetration = torch.clamp(self.contact_threshold - distance, min=0)
        contact_force = penetration * 1000.0  # Spring constant
        
        # Add damping if velocities available
        if hand_vel is not None and obj_vel is not None:
            relative_vel = torch.norm(hand_vel - obj_vel, dim=1)
            damping_force = relative_vel * 100.0  # Damping coefficient
            contact_force = contact_force + damping_force * contact_mask.float()
        
        return contact_mask, contact_force
    
    def compute_gravity_effects(self, obj_pos, obj_vel=None, in_contact=None):
        """
        Compute gravity effects on object.
        
        Args:
            obj_pos: Object position (batch_size, 3)
            obj_vel: Object velocity (batch_size, 3), optional
            in_contact: Contact mask (batch_size,), if object is in contact
        
        Returns:
            gravity_acceleration: Expected acceleration due to gravity (batch_size, 3)
            on_surface: Whether object is on table surface (batch_size,)
        """
        batch_size = obj_pos.shape[0]
        device = obj_pos.device
        
        # Gravity acts downward (negative z direction)
        gravity_acc = torch.zeros_like(obj_pos)
        gravity_acc[:, 2] = -self.gravity_strength
        
        # Check if object is on table surface
        on_surface = obj_pos[:, 2] <= (self.table_height + self.object_radius)
        
        # If object is supported (on surface or in contact), no gravity effect
        if in_contact is not None:
            supported = on_surface | in_contact
        else:
            supported = on_surface
        
        # Apply gravity only to unsupported objects
        gravity_acc = gravity_acc * (~supported).float().unsqueeze(1)
        
        return gravity_acc, on_surface
    
    def compute_grasp_stability(self, hand_pos, obj_pos, gripper_pos, hand_vel=None, obj_vel=None):
        """
        Compute grasp stability constraints.
        
        When object is grasped (gripper closed and in contact):
        - Object should follow hand motion
        - Relative motion should be minimal
        
        Args:
            hand_pos: Hand position (batch_size, 3)
            obj_pos: Object position (batch_size, 3)
            gripper_pos: Gripper opening (batch_size, 1)
            hand_vel: Hand velocity (batch_size, 3), optional
            obj_vel: Object velocity (batch_size, 3), optional
        
        Returns:
            grasp_mask: Boolean mask for grasping (batch_size,)
            stability_error: Grasp stability error (batch_size,)
        """
        # Detect grasping: gripper closed AND in contact
        contact_mask, _ = self.compute_contact_forces(hand_pos, obj_pos)
        gripper_closed = gripper_pos[:, 0] < 0.02  # Gripper mostly closed
        grasp_mask = contact_mask & gripper_closed
        
        # For grasped objects, compute stability
        if hand_vel is not None and obj_vel is not None:
            # Object should follow hand motion
            velocity_error = torch.norm(hand_vel - obj_vel, dim=1)
        else:
            # Use position-based stability (object should be close to hand)
            velocity_error = torch.norm(hand_pos - obj_pos, dim=1)
        
        # Apply error only to grasped objects
        stability_error = velocity_error * grasp_mask.float()
        
        return grasp_mask, stability_error
    
    def compute_momentum_conservation(self, velocities_t1, velocities_t2, masses=None):
        """
        Compute momentum conservation for hand-object interactions.
        
        Args:
            velocities_t1: Velocities at time t1 (batch_size, 6) [hand_vel, obj_vel]
            velocities_t2: Velocities at time t2 (batch_size, 6) [hand_vel, obj_vel]
            masses: Optional masses [hand_mass, obj_mass]
        
        Returns:
            momentum_violation: Momentum conservation violation (batch_size,)
        """
        if masses is None:
            # Default masses (hand is much heavier than object)
            hand_mass = 5.0  # kg (including robot arm)
            obj_mass = self.object_mass
            masses = torch.tensor([hand_mass, obj_mass], device=velocities_t1.device)
        
        # Split velocities
        hand_vel_t1, obj_vel_t1 = velocities_t1[:, :3], velocities_t1[:, 3:]
        hand_vel_t2, obj_vel_t2 = velocities_t2[:, :3], velocities_t2[:, 3:]
        
        # Calculate momentum at each time
        momentum_t1 = masses[0] * hand_vel_t1 + masses[1] * obj_vel_t1
        momentum_t2 = masses[0] * hand_vel_t2 + masses[1] * obj_vel_t2
        
        # Momentum conservation violation
        momentum_violation = torch.norm(momentum_t1 - momentum_t2, dim=1)
        
        return momentum_violation
    
    def compute_physics_losses(self, observations_seq, physics_weights):
        """
        Compute all physics losses for a sequence of observations.
        
        Args:
            observations_seq: Sequence of observations (batch_size, seq_len, 7)
            physics_weights: Dictionary of physics loss weights
        
        Returns:
            physics_losses: Dictionary of computed physics losses
        """
        batch_size, seq_len, obs_dim = observations_seq.shape
        device = observations_seq.device
        
        # Extract positions for all timesteps
        all_hand_pos = observations_seq[:, :, 1:4]  # (batch, seq_len, 3)
        all_obj_pos = observations_seq[:, :, 4:7]   # (batch, seq_len, 3)
        all_gripper_pos = observations_seq[:, :, 0:1]  # (batch, seq_len, 1)
        
        losses = {}
        
        # Temporal consistency loss
        if physics_weights.get('temporal_consistency', 0) > 0:
            position_diff = torch.diff(all_hand_pos, dim=1)  # (batch, seq_len-1, 3)
            temporal_loss = torch.mean(torch.norm(position_diff, dim=2)**2)
            losses['temporal_consistency'] = temporal_loss
        
        # Contact and grasp losses (computed pairwise)
        if seq_len >= 2:
            # Use middle timesteps for physics computation
            mid_idx = seq_len // 2
            hand_pos = all_hand_pos[:, mid_idx]
            obj_pos = all_obj_pos[:, mid_idx]
            gripper_pos = all_gripper_pos[:, mid_idx]
            
            # Contact loss
            if physics_weights.get('contact', 0) > 0:
                contact_mask, contact_force = self.compute_contact_forces(hand_pos, obj_pos)
                # Encourage realistic contact forces
                contact_loss = torch.mean(contact_force**2)
                losses['contact'] = contact_loss
            
            # Gravity loss
            if physics_weights.get('gravity', 0) > 0:
                contact_mask, _ = self.compute_contact_forces(hand_pos, obj_pos)
                gravity_acc, on_surface = self.compute_gravity_effects(obj_pos, in_contact=contact_mask)
                
                # For unsupported objects, expect downward acceleration
                if seq_len >= 3:
                    # Estimate acceleration from position differences
                    obj_acc = all_obj_pos[:, mid_idx+1] - 2*all_obj_pos[:, mid_idx] + all_obj_pos[:, mid_idx-1]
                    gravity_violation = torch.norm(obj_acc - gravity_acc, dim=1)
                    gravity_loss = torch.mean(gravity_violation**2)
                else:
                    gravity_loss = torch.tensor(0.0, device=device)
                losses['gravity'] = gravity_loss
            
            # Grasp stability loss
            if physics_weights.get('grasp_stability', 0) > 0:
                grasp_mask, stability_error = self.compute_grasp_stability(
                    hand_pos, obj_pos, gripper_pos
                )
                grasp_loss = torch.mean(stability_error**2)
                losses['grasp_stability'] = grasp_loss
        
        # Momentum conservation (requires velocities)
        if physics_weights.get('momentum_conservation', 0) > 0 and seq_len >= 2:
            # Estimate velocities from position differences
            hand_vel_1 = all_hand_pos[:, 1] - all_hand_pos[:, 0]
            obj_vel_1 = all_obj_pos[:, 1] - all_obj_pos[:, 0]
            hand_vel_2 = all_hand_pos[:, -1] - all_hand_pos[:, -2]
            obj_vel_2 = all_obj_pos[:, -1] - all_obj_pos[:, -2]
            
            velocities_1 = torch.cat([hand_vel_1, obj_vel_1], dim=1)
            velocities_2 = torch.cat([hand_vel_2, obj_vel_2], dim=1)
            
            momentum_violation = self.compute_momentum_conservation(velocities_1, velocities_2)
            momentum_loss = torch.mean(momentum_violation**2)
            losses['momentum_conservation'] = momentum_loss
        
        return losses


def get_pick_and_place_physics_config():
    """
    Get default physics configuration for Pick and Place environment.
    
    Returns:
        dict: Physics configuration with weights and parameters
    """
    return {
        'physics_loss_weights': {
            'temporal_consistency': 1.0,   # Smooth motion
            'momentum_conservation': 0.5,  # Momentum conservation during contact
            'gravity': 2.0,                # Strong gravity effects
            'contact': 1.5,                # Hand-object contact dynamics
            'grasp_stability': 2.0,        # Object follows hand when grasped
            'angular_momentum': 0.3,       # Object rotation (if using advanced trainer)
            'collision': 1.0,              # Collision response (if using advanced trainer)
        },
        'contact_threshold': 0.05,  # 5cm contact detection
        'gravity_strength': 9.81,   # Earth gravity
        'object_mass': 0.1,         # 100g object
        'gripper_closed_threshold': 0.02,  # Gripper closure threshold
    }


def create_pick_and_place_vae_trainer_kwargs(physics_config=None):
    """
    Create VAE trainer kwargs for Pick and Place environment.
    
    Args:
        physics_config: Physics configuration dict (uses default if None)
    
    Returns:
        dict: Trainer kwargs for physics-informed VAE
    """
    if physics_config is None:
        physics_config = get_pick_and_place_physics_config()
    
    return {
        'physics_loss_weights': physics_config['physics_loss_weights'],
        'contact_threshold': physics_config['contact_threshold'],
        'gravity_weight': physics_config['physics_loss_weights'].get('gravity', 0.0),
        'grasp_stability_weight': physics_config['physics_loss_weights'].get('grasp_stability', 0.0),
        'contact_weight': physics_config['physics_loss_weights'].get('contact', 0.0),
        'momentum_weight': physics_config['physics_loss_weights'].get('momentum_conservation', 0.0),
        'temporal_consistency_weight': physics_config['physics_loss_weights'].get('temporal_consistency', 0.0),
        'angular_momentum_weight': physics_config['physics_loss_weights'].get('angular_momentum', 0.0),
        'collision_response_weight': physics_config['physics_loss_weights'].get('collision', 0.0),
    }
