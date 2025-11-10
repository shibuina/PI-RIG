"""
Physics configuration for Sawyer Door Hook environment.

This module provides physics-specific utilities and configurations for 
the Door Opening task, including:
- Contact detection between hand and door handle
- Rotational dynamics of door
- Handle grasp constraints
- Angular velocity and position relationships

Observation structure for Door Hook:
- Hand position: [0:3] (xyz coordinates)  
- Door angle: [3] (radians, 0 to max_angle)
"""

import numpy as np
import torch
import torch.nn.functional as F


class DoorPhysics:
    """
    Physics utilities for Door Hook environment
    """
    
    def __init__(self, contact_threshold=0.03, max_angle=1.0472, door_mass=2.0):
        """
        Args:
            contact_threshold: Distance threshold for hand-handle contact (meters)
            max_angle: Maximum door opening angle (radians, ~60 degrees)
            door_mass: Mass of the door (kg)
        """
        self.contact_threshold = contact_threshold
        self.max_angle = max_angle
        self.door_mass = door_mass
        
        # Door properties
        self.door_width = 0.8  # meters
        self.door_height = 2.0  # meters
        self.handle_height = 1.0  # meters from ground
        self.handle_radius = 0.02  # meters
        
        # Physics constants
        self.door_friction = 0.1  # Rotational friction coefficient
        self.handle_stiffness = 5000.0  # N/m, for contact forces
        self.damping_coefficient = 50.0  # Ns/m, for velocity damping
        
        # Environment bounds for door task
        self.hand_bounds = {
            'low': np.array([-0.1, 0.45, 0.15]),
            'high': np.array([0.0, 0.65, 0.225])
        }
        
        # Handle position (approximate, varies with door angle)
        self.handle_base_pos = np.array([0.0, 0.6, 0.12])  # When door closed
        self.door_pivot = np.array([0.1, 0.6, 0.0])  # Door hinge location
    
    def extract_positions(self, observations):
        """
        Extract hand position and door angle from observations.
        
        Args:
            observations: Tensor of shape (batch_size, 4) 
                         [hand_x, hand_y, hand_z, door_angle]
        
        Returns:
            hand_pos: Tensor of shape (batch_size, 3)
            door_angle: Tensor of shape (batch_size, 1)
        """
        hand_pos = observations[:, 0:3]     # Hand xyz
        door_angle = observations[:, 3:4]   # Door angle
        
        return hand_pos, door_angle
    
    def compute_handle_position(self, door_angle):
        """
        Compute handle position based on door angle.
        
        Args:
            door_angle: Door opening angle (batch_size, 1)
        
        Returns:
            handle_pos: Handle position (batch_size, 3)
        """
        batch_size = door_angle.shape[0]
        device = door_angle.device
        
        # Base handle position when door is closed
        base_pos = torch.tensor(self.handle_base_pos, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Rotate handle position around door pivot
        cos_angle = torch.cos(door_angle)
        sin_angle = torch.sin(door_angle)
        
        # Relative position from pivot
        rel_pos = base_pos - torch.tensor(self.door_pivot, device=device).unsqueeze(0)
        
        # Apply rotation around z-axis (vertical door rotation)
        rotated_x = rel_pos[:, 0:1] * cos_angle - rel_pos[:, 1:2] * sin_angle
        rotated_y = rel_pos[:, 0:1] * sin_angle + rel_pos[:, 1:2] * cos_angle
        rotated_z = rel_pos[:, 2:3]  # Z unchanged for vertical rotation
        
        # Add back pivot position
        handle_pos = torch.cat([rotated_x, rotated_y, rotated_z], dim=1)
        handle_pos = handle_pos + torch.tensor(self.door_pivot, device=device).unsqueeze(0)
        
        return handle_pos
    
    def compute_contact_forces(self, hand_pos, door_angle, hand_vel=None, door_angular_vel=None):
        """
        Compute contact forces between hand and door handle.
        
        Args:
            hand_pos: Hand position (batch_size, 3)
            door_angle: Door angle (batch_size, 1)
            hand_vel: Hand velocity (batch_size, 3), optional
            door_angular_vel: Door angular velocity (batch_size, 1), optional
        
        Returns:
            contact_mask: Boolean mask for contact (batch_size,)
            contact_force: Contact force magnitude (batch_size,)
            torque_applied: Torque applied to door (batch_size,)
        """
        # Get current handle position
        handle_pos = self.compute_handle_position(door_angle)
        
        # Compute distance between hand and handle
        distance = torch.norm(hand_pos - handle_pos, dim=1)
        contact_mask = distance < self.contact_threshold
        
        # Contact force based on penetration
        penetration = torch.clamp(self.contact_threshold - distance, min=0)
        contact_force = penetration * self.handle_stiffness
        
        # Add damping if velocities available
        if hand_vel is not None:
            # Estimate handle velocity from door angular velocity
            if door_angular_vel is not None:
                # Handle velocity = angular_vel × distance_from_pivot
                pivot_to_handle = handle_pos - torch.tensor(self.door_pivot, device=handle_pos.device).unsqueeze(0)
                handle_speed = torch.norm(pivot_to_handle, dim=1, keepdim=True) * torch.abs(door_angular_vel)
                
                # Simplified relative velocity
                relative_vel = torch.norm(hand_vel, dim=1) - handle_speed.squeeze()
                damping_force = torch.abs(relative_vel) * self.damping_coefficient
                contact_force = contact_force + damping_force * contact_mask.float()
        
        # Compute torque applied to door (simplified)
        # Torque = force × lever arm (distance from pivot to handle)
        pivot_to_handle = handle_pos - torch.tensor(self.door_pivot, device=handle_pos.device).unsqueeze(0)
        lever_arm = torch.norm(pivot_to_handle, dim=1)
        torque_applied = contact_force * lever_arm * contact_mask.float()
        
        return contact_mask, contact_force, torque_applied
    
    def compute_door_dynamics(self, door_angle, torque_applied, door_angular_vel=None):
        """
        Compute door rotational dynamics.
        
        Args:
            door_angle: Current door angle (batch_size, 1)
            torque_applied: Applied torque (batch_size,)
            door_angular_vel: Current angular velocity (batch_size, 1), optional
        
        Returns:
            angular_acceleration: Door angular acceleration (batch_size, 1)
            friction_torque: Friction torque opposing motion (batch_size, 1)
        """
        batch_size = door_angle.shape[0]
        device = door_angle.device
        
        # Moment of inertia for door (simplified as rectangular plate)
        # I = (1/12) * m * (w^2 + h^2) for rotation about edge
        moment_of_inertia = (1/12) * self.door_mass * (self.door_width**2 + self.door_height**2)
        
        # Friction torque (opposes motion)
        if door_angular_vel is not None:
            friction_torque = -torch.sign(door_angular_vel) * self.door_friction
        else:
            friction_torque = torch.zeros(batch_size, 1, device=device)
        
        # Total torque
        total_torque = torque_applied.unsqueeze(1) + friction_torque
        
        # Angular acceleration = Torque / Moment of Inertia
        angular_acceleration = total_torque / moment_of_inertia
        
        return angular_acceleration, friction_torque
    
    def enforce_angle_constraints(self, door_angle, door_angular_vel=None):
        """
        Enforce physical constraints on door angle.
        
        Args:
            door_angle: Door angle (batch_size, 1)
            door_angular_vel: Angular velocity (batch_size, 1), optional
        
        Returns:
            constrained_angle: Angle within valid range (batch_size, 1)
            constraint_violation: Boolean mask for violations (batch_size,)
        """
        # Clamp angle to valid range [0, max_angle]
        constrained_angle = torch.clamp(door_angle, 0.0, self.max_angle)
        
        # Check for constraint violations
        constraint_violation = (door_angle < 0.0) | (door_angle > self.max_angle)
        
        # If at limits and velocity would increase violation, set velocity to zero
        if door_angular_vel is not None:
            at_min_limit = (constrained_angle <= 0.0) & (door_angular_vel < 0.0)
            at_max_limit = (constrained_angle >= self.max_angle) & (door_angular_vel > 0.0)
            
            # This would need to be applied externally to the velocity
            # We just return the information here
            velocity_constrained = at_min_limit | at_max_limit
        
        return constrained_angle, constraint_violation
    
    def compute_physics_loss(self, predicted_states, target_states, contact_mask=None):
        """
        Compute physics-based loss for VAE training.
        
        Args:
            predicted_states: Predicted observations (batch_size, 4)
            target_states: Target observations (batch_size, 4)
            contact_mask: Contact detection mask (batch_size,), optional
        
        Returns:
            physics_loss: Combined physics loss
            loss_components: Dictionary of individual loss components
        """
        pred_hand_pos, pred_door_angle = self.extract_positions(predicted_states)
        target_hand_pos, target_door_angle = self.extract_positions(target_states)
        
        # Position consistency loss
        position_loss = F.mse_loss(pred_hand_pos, target_hand_pos)
        
        # Angle consistency loss
        angle_loss = F.mse_loss(pred_door_angle, target_door_angle)
        
        # Physics constraint losses
        physics_losses = []
        
        # 1. Contact consistency
        if contact_mask is not None:
            pred_handle_pos = self.compute_handle_position(pred_door_angle)
            pred_contact_dist = torch.norm(pred_hand_pos - pred_handle_pos, dim=1)
            
            # If in contact, distance should be small
            contact_loss = torch.mean(contact_mask.float() * torch.clamp(pred_contact_dist - self.contact_threshold, min=0))
            physics_losses.append(contact_loss)
        
        # 2. Angle bounds constraint
        angle_bound_loss = torch.mean(
            torch.clamp(-pred_door_angle, min=0) +  # Negative angle penalty
            torch.clamp(pred_door_angle - self.max_angle, min=0)  # Exceeds max penalty
        )
        physics_losses.append(angle_bound_loss)
        
        # 3. Hand bounds constraint
        hand_bounds_low = torch.tensor(self.hand_bounds['low'], device=pred_hand_pos.device)
        hand_bounds_high = torch.tensor(self.hand_bounds['high'], device=pred_hand_pos.device)
        
        hand_bound_loss = torch.mean(
            torch.clamp(hand_bounds_low - pred_hand_pos, min=0) +
            torch.clamp(pred_hand_pos - hand_bounds_high, min=0)
        )
        physics_losses.append(hand_bound_loss)
        
        # Combine losses
        total_physics_loss = sum(physics_losses) if physics_losses else torch.tensor(0.0, device=predicted_states.device)
        
        # Total loss
        physics_loss = position_loss + angle_loss + 0.1 * total_physics_loss
        
        loss_components = {
            'position_loss': position_loss.item(),
            'angle_loss': angle_loss.item(),
            'physics_loss': total_physics_loss.item() if total_physics_loss.requires_grad else 0.0,
            'total_physics_loss': physics_loss.item(),
        }
        
        return physics_loss, loss_components


def get_door_physics_config():
    """
    Get default physics configuration for door opening task.
    
    Returns:
        dict: Physics configuration parameters
    """
    return {
        'contact_threshold': 0.03,  # 3cm for handle contact
        'max_angle': 1.0472,  # ~60 degrees in radians
        'door_mass': 2.0,  # kg
        'friction_coefficient': 0.1,
        'handle_stiffness': 5000.0,  # N/m
        'damping_coefficient': 50.0,  # Ns/m
        'physics_loss_weight': 0.1,
        'use_contact_detection': True,
        'use_angle_constraints': True,
        'use_torque_dynamics': True,
        'physics_loss_weights': {
            'contact_consistency': 0.1,
            'angle_bounds': 0.05,
            'hand_bounds': 0.05,
            'torque_dynamics': 0.1,
            'rotational_physics': 0.1,
        }
    }


def create_door_vae_trainer_kwargs(physics_config=None, use_physics=True):
    """
    Create VAE trainer kwargs with door-specific physics constraints.
    
    Args:
        physics_config: Physics configuration dict
        use_physics: Whether to use physics-informed training
    
    Returns:
        dict: VAE trainer keyword arguments
    """
    if physics_config is None:
        physics_config = get_door_physics_config()
    
    base_kwargs = {
        'lr': 1e-3,
        'weight_decay': 0.0,
        'beta': 10.0,
        'num_epochs': 2000,
        'batch_size': 128,
        'beta_schedule_kwargs': {
            'x_values': [0, 100, 200, 500],
            'y_values': [0, 0, 5, 10],
        },
    }
    
    if use_physics:
        # Add physics-specific parameters
        physics_kwargs = {
            'physics_informed': True,
            'physics_config': physics_config,
            'physics_loss_weight': physics_config.get('physics_loss_weight', 0.1),
            'use_contact_loss': physics_config.get('use_contact_detection', True),
            'use_constraint_loss': physics_config.get('use_angle_constraints', True),
        }
        base_kwargs.update(physics_kwargs)
    
    return base_kwargs


def validate_door_observations(observations):
    """
    Validate door environment observations for physics consistency.
    
    Args:
        observations: Tensor of observations (batch_size, 4)
    
    Returns:
        dict: Validation results
    """
    hand_pos, door_angle = DoorPhysics().extract_positions(observations)
    
    # Check bounds
    hand_bounds = DoorPhysics().hand_bounds
    hand_in_bounds = torch.all(
        (hand_pos >= torch.tensor(hand_bounds['low'], device=hand_pos.device)) &
        (hand_pos <= torch.tensor(hand_bounds['high'], device=hand_pos.device)),
        dim=1
    )
    
    angle_in_bounds = (door_angle >= 0.0) & (door_angle <= DoorPhysics().max_angle)
    
    # Compute physics metrics
    physics = DoorPhysics()
    handle_pos = physics.compute_handle_position(door_angle)
    hand_handle_dist = torch.norm(hand_pos - handle_pos, dim=1)
    
    validation_results = {
        'hand_in_bounds_rate': torch.mean(hand_in_bounds.float()).item(),
        'angle_in_bounds_rate': torch.mean(angle_in_bounds.float()).item(),
        'average_hand_handle_distance': torch.mean(hand_handle_dist).item(),
        'contact_rate': torch.mean((hand_handle_dist < physics.contact_threshold).float()).item(),
        'angle_range': {
            'min': torch.min(door_angle).item(),
            'max': torch.max(door_angle).item(),
            'mean': torch.mean(door_angle).item(),
        }
    }
    
    return validation_results
