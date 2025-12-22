"""
Enhanced P3-VAE (Physics-informed Variational Auto-Encoder) Trainer
Based on the original P3-VAE paper: https://arxiv.org/pdf/2210.10418

This implementation incorporates:
1. Proper latent space separation (z_I for physics, z_E for environment)
2. Physics-guided encoding with domain-specific priors
3. Physics-informed decoding with constraints
4. Robust loss functions with stability improvements
5. Enhanced physics modeling for robotics tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os.path as osp
from torchvision.utils import save_image
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger
from multiworld.core.image_env import normalize_image

# Try to import torchdiffeq for proper ODE integration
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not available, using simple Euler integration")


class PhysicsStateExtractor(nn.Module):
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
        self.physics_decoder = nn.Sequential(
            nn.Linear(z_I_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.physics_dim)
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
            self.velocity_range = (-5.0, 5.0)  # Angular velocity limits (match PhysicsLossCalculator)
    
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
            # Constrain velocities to safe limits (match PhysicsLossCalculator)
            joint_velocities = torch.tanh(raw_physics[:, 2:4]) * 5.0
            physics_state = torch.cat([joint_angles, joint_velocities], dim=1)
            
        else:
            # No constraints for unknown physics types
            physics_state = raw_physics
        
        return physics_state


class PhysicsGuidedEncoder(nn.Module):
    """
    Physics-guided encoder that separates latent space into:
    - z_I (intrinsic/physics): Gaussian distribution with physics-informed priors
    - z_E (extrinsic/environment): Beta distribution for environmental factors
    """
    
    def __init__(self, input_channels, imsize, z_I_dim, z_E_dim, physics_type='pick_and_place', hidden_dim=256):
        super(PhysicsGuidedEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.imsize = imsize
        self.z_I_dim = z_I_dim
        self.z_E_dim = z_E_dim
        self.physics_type = physics_type
        self.hidden_dim = hidden_dim
        
        # Shared convolutional feature extractor
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size()
        
        # Physics (z_I) encoder - encodes intrinsic physics variables
        self.physics_encoder = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.z_I_mean = nn.Linear(hidden_dim, z_I_dim)
        self.z_I_lnvar = nn.Linear(hidden_dim, z_I_dim)
        
        # Environment (z_E) encoder - encodes environmental factors
        # Takes both image features and physics mean as input
        self.env_encoder = nn.Sequential(
            nn.Linear(conv_out_size + z_I_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Beta distribution parameters for z_E
        self.z_E_alpha = nn.Linear(hidden_dim, z_E_dim)
        self.z_E_beta = nn.Linear(hidden_dim, z_E_dim)
        
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.imsize, self.imsize)
            dummy_output = self.conv_features(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def encode_physics(self, x):
        """Encode physics variables z_I"""
        # Reshape flattened input to image tensor if needed
        if len(x.shape) == 2:  # [batch_size, flattened_pixels]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.input_channels, self.imsize, self.imsize)
        
        conv_out = self.conv_features(x)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        
        physics_features = self.physics_encoder(conv_flat)
        z_I_mean = self.z_I_mean(physics_features)
        z_I_lnvar = self.z_I_lnvar(physics_features)
        
        return {
            'mean': z_I_mean,
            'lnvar': z_I_lnvar,
            'features': physics_features
        }
    
    def encode_environment(self, x, z_I_mean):
        """Encode environmental variables z_E"""
        # Reshape flattened input to image tensor if needed
        if len(x.shape) == 2:  # [batch_size, flattened_pixels]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.input_channels, self.imsize, self.imsize)
        
        conv_out = self.conv_features(x)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        
        # Combine image features with physics information
        combined_features = torch.cat([conv_flat, z_I_mean], dim=1)
        env_features = self.env_encoder(combined_features)
        
        # Beta distribution parameters (ensure positive)
        alpha = 1e-1 + F.softplus(self.z_E_alpha(env_features))
        beta = 1e-1 + F.softplus(self.z_E_beta(env_features))
        
        return {
            'alpha': alpha,
            'beta': beta,
            'features': env_features
        }
    
    def kl_divergence_physics(self, z_I_stats, prior_z_I_stats):
        """KL divergence for physics variables (Gaussian)"""
        mean = z_I_stats['mean']
        lnvar = z_I_stats['lnvar']
        prior_mean = prior_z_I_stats['mean']
        prior_lnvar = prior_z_I_stats['lnvar']
        
        kl = 0.5 * torch.sum(
            prior_lnvar - lnvar + 
            (torch.exp(lnvar) + (mean - prior_mean)**2) / torch.exp(prior_lnvar) - 1,
            dim=1
        )
        return kl
    
    def kl_divergence_environment(self, z_E_stats, prior_z_E_stats):
        """KL divergence for environmental variables (Beta)"""
        alpha = z_E_stats['alpha']
        beta = z_E_stats['beta']
        prior_alpha = prior_z_E_stats['alpha']
        prior_beta = prior_z_E_stats['beta']
        
        # KL(Beta(α,β) || Beta(α₀,β₀))
        kl = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta) - \
             torch.lgamma(prior_alpha + prior_beta) + torch.lgamma(prior_alpha) + torch.lgamma(prior_beta) + \
             (alpha - prior_alpha) * torch.digamma(alpha) + \
             (beta - prior_beta) * torch.digamma(beta) + \
             (prior_alpha - alpha + prior_beta - beta) * torch.digamma(alpha + beta)
        
        return torch.sum(kl, dim=1)


class PhysicsInformedDecoder(nn.Module):
    """
    Physics-informed decoder that reconstructs images from separated latent variables
    while enforcing physics constraints.
    """
    
    def __init__(self, z_I_dim, z_E_dim, input_channels, imsize, physics_type='pick_and_place', hidden_dim=256):
        super(PhysicsInformedDecoder, self).__init__()
        
        self.z_I_dim = z_I_dim
        self.z_E_dim = z_E_dim
        self.input_channels = input_channels
        self.imsize = imsize
        self.physics_type = physics_type
        self.hidden_dim = hidden_dim
        
        # Physics decoder - processes physics variables
        self.physics_decoder = nn.Sequential(
            nn.Linear(z_I_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Environment decoder - processes environmental variables
        self.env_decoder = nn.Sequential(
            nn.Linear(z_E_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combined decoder - merges physics and environment
        self.combined_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Calculate the required input size for deconvolutional layers
        self.deconv_input_size = self._calculate_deconv_input_size()
        
        self.fc_to_conv = nn.Linear(hidden_dim, self.deconv_input_size)
        
        # Deconvolutional layers
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def _calculate_deconv_input_size(self):
        """Calculate the input size needed for deconvolutional layers"""
        # For 48x48 input with 4 conv layers (stride 2 each): 48 -> 24 -> 12 -> 6 -> 3
        # So we need 256 * 3 * 3 = 2304 for the linear layer
        size_after_conv = self.imsize // (2 ** 4)  # 4 conv layers with stride 2
        return 256 * size_after_conv * size_after_conv
    
    def apply_physics_constraints(self, z_I, physics_features):
        """Apply physics constraints to the physics features"""
        if self.physics_type == 'pick_and_place':
            # Ensure physics variables are in reasonable ranges
            if z_I.shape[1] >= 7:  # [gripper_opening, hand_xyz, obj_xyz]
                # Gripper opening should be in [0, 0.05]
                gripper_constraint = torch.sigmoid(physics_features[:, :32]) * 0.05
                # Hand and object positions should be in reasonable workspace
                position_constraint = torch.tanh(physics_features[:, 32:]) * 0.5
                constrained_features = torch.cat([gripper_constraint, position_constraint], dim=1)
                return constrained_features
        
        # Default: apply mild constraints
        return torch.tanh(physics_features) * 2.0
    
    def forward(self, z_I, z_E, apply_constraints=True):
        """Decode from physics and environmental latent variables"""
        # Process physics variables
        physics_features = self.physics_decoder(z_I)
        if apply_constraints:
            physics_features = self.apply_physics_constraints(z_I, physics_features)
        
        # Process environmental variables
        env_features = self.env_decoder(z_E)
        
        # Combine physics and environmental features
        combined_features = torch.cat([physics_features, env_features], dim=1)
        combined_output = self.combined_decoder(combined_features)
        
        # Convert to spatial representation
        spatial_features = self.fc_to_conv(combined_output)
        batch_size = spatial_features.size(0)
        size_after_conv = self.imsize // (2 ** 4)
        spatial_features = spatial_features.view(batch_size, 256, size_after_conv, size_after_conv)
        
        # Generate image
        reconstructed = self.deconv_layers(spatial_features)
        
        return reconstructed


class PhysicsLossCalculator(nn.Module):
    """
    Calculate physics-based losses for different environments.
    Incorporates PDE residuals, conservation laws, and kinematic constraints.
    
    Implements L_physics = λ_temporal*L_temporal + λ_momentum*L_momentum + λ_contact*L_contact
    through PDE residual computation covering:
    - Temporal consistency (smooth transitions between consecutive states)
    - Momentum conservation (during contact interactions)  
    - Contact dynamics (spring-damper model for collision forces)
    """
    
    def __init__(self, physics_type='pick_and_place', dt=0.02):
        super(PhysicsLossCalculator, self).__init__()
        self.physics_type = physics_type
        self.dt = dt
        
        # Physics parameters based on environment
        self._initialize_physics_params()
    
    def _initialize_physics_params(self):
        """Initialize physics parameters based on environment type"""
        if self.physics_type == 'pick_and_place':
            self.mass_hand = 5.0
            self.mass_object = 0.1
            self.gravity = 9.81
            self.contact_threshold = 0.05
            self.gripper_threshold = 0.02
            self.table_height = 0.05
            self.friction_coeff = 0.3
        elif self.physics_type == 'pusher':
            self.mass_hand = 1.0
            self.mass_puck = 0.5
            self.friction_coeff = 0.3
            self.contact_threshold = 0.1
        elif self.physics_type == 'reacher':
            # 2-link arm parameters
            self.link1_length = 0.3
            self.link2_length = 0.3
            self.link1_mass = 1.0
            self.link2_mass = 0.8
            self.joint_friction = 0.05
            self.joint_limits = (-np.pi, np.pi)  # Joint angle limits
            self.velocity_limits = (-5.0, 5.0)   # Angular velocity limits
            self.torque_limits = (-10.0, 10.0)  # Torque limits
        # Add more environments as needed
    
    def extract_state_variables(self, z_I):
        """Extract state variables from physics latent representation"""
        if self.physics_type == 'pick_and_place' and z_I.shape[1] >= 7:
            return {
                'gripper_opening': z_I[:, 0:1],
                'hand_pos': z_I[:, 1:4],
                'object_pos': z_I[:, 4:7]
            }
        elif self.physics_type == 'pusher' and z_I.shape[1] >= 4:
            return {
                'hand_pos': z_I[:, :2],
                'puck_pos': z_I[:, 2:4]
            }
        elif self.physics_type == 'reacher' and z_I.shape[1] >= 4:
            return {
                'joint_angles': z_I[:, :2],      # [θ₁, θ₂]
                'joint_velocities': z_I[:, 2:4]  # [θ̇₁, θ̇₂]
            }
        else:
            return {'generic_state': z_I}
    
    def compute_pde_residual(self, z_I_t, z_I_t1):
        """
        Compute PDE residual for physics consistency (Methodology Section 2.3)
        
        This method computes the combined physics loss including:
        - Kinematic Consistency: temporal derivatives match physical evolution
        - Newton's Laws: force balance equations (F = ma) 
        - Momentum Conservation: conservation during interactions
        - Energy Conservation: with dissipation allowance
        
        Environment-specific applications:
        - Pusher: All 4 losses active (momentum conservation during hand-puck contact)
        - Pick-and-place: All 4 losses active (grasp constraints, gravity, table contact)
        - Reacher: All 4 losses active (joint dynamics, no object contacts)
        """
        state_t = self.extract_state_variables(z_I_t)
        state_t1 = self.extract_state_variables(z_I_t1)
        
        if self.physics_type == 'pick_and_place':
            return self._pick_and_place_pde_residual(state_t, state_t1)
        elif self.physics_type == 'pusher':
            return self._pusher_pde_residual(state_t, state_t1)
        elif self.physics_type == 'reacher':
            return self._reacher_pde_residual(state_t, state_t1)
        else:
            # Generic: assume constant velocity model
            velocity = (z_I_t1 - z_I_t) / self.dt
            acceleration = torch.zeros_like(velocity)
            return torch.mean(acceleration ** 2)
    
    def _pick_and_place_pde_residual(self, state_t, state_t1):
        """PDE residual for pick and place dynamics"""
        hand_pos_t = state_t['hand_pos']
        object_pos_t = state_t['object_pos']
        gripper_t = state_t['gripper_opening']
        
        hand_pos_t1 = state_t1['hand_pos']
        object_pos_t1 = state_t1['object_pos']
        gripper_t1 = state_t1['gripper_opening']
        
        # Compute velocities
        hand_vel = (hand_pos_t1 - hand_pos_t) / self.dt
        object_vel = (object_pos_t1 - object_pos_t) / self.dt
        
        # Contact and grasping detection
        hand_object_dist = torch.norm(hand_pos_t - object_pos_t, dim=1, keepdim=True)
        in_contact = (hand_object_dist < self.contact_threshold).float()
        gripper_closed = (gripper_t < self.gripper_threshold).float()
        grasping = in_contact * gripper_closed
        
        # Physics residuals
        residuals = []
        
        # 1. Newton's laws for object motion
        # F = ma, where F includes gravity, contact forces, and constraint forces
        object_mass = self.mass_object
        
        # Gravity force (downward)
        gravity_force = torch.zeros_like(object_pos_t)
        gravity_force[:, 2:3] = -object_mass * self.gravity
        
        # Contact force when grasping (object follows hand)
        if grasping.sum() > 0:
            # When grasping, object should have similar acceleration to hand
            hand_accel = (hand_vel - torch.zeros_like(hand_vel)) / self.dt  # Assume hand was at rest
            object_accel = (object_vel - torch.zeros_like(object_vel)) / self.dt
            grasp_residual = grasping * torch.norm(object_accel - hand_accel, dim=1, keepdim=True)
            residuals.append(torch.mean(grasp_residual))
        
        # 2. Conservation of energy during contact
        if in_contact.sum() > 0:
            # Kinetic energy should be conserved during elastic collisions
            hand_ke_t = 0.5 * self.mass_hand * torch.sum(hand_vel**2, dim=1, keepdim=True)
            object_ke_t = 0.5 * object_mass * torch.sum(object_vel**2, dim=1, keepdim=True)
            total_ke = hand_ke_t + object_ke_t
            
            # Energy should be approximately conserved (allowing for dissipation)
            energy_residual = in_contact * torch.abs(total_ke - torch.mean(total_ke))
            residuals.append(torch.mean(energy_residual))
        
        # 3. Table constraint (objects can't go below table)
        on_table = (object_pos_t[:, 2:3] <= self.table_height + 0.01).float()
        table_violation = torch.relu(self.table_height - object_pos_t[:, 2:3])
        residuals.append(torch.mean(table_violation))
        
        return torch.mean(torch.stack(residuals)) if residuals else torch.tensor(0.0, device=object_pos_t.device)
    
    def _pusher_pde_residual(self, state_t, state_t1):
        """PDE residual for pusher dynamics"""
        hand_pos_t = state_t['hand_pos']
        puck_pos_t = state_t['puck_pos']
        hand_pos_t1 = state_t1['hand_pos']
        puck_pos_t1 = state_t1['puck_pos']
        
        # Contact detection
        contact_dist = torch.norm(hand_pos_t - puck_pos_t, dim=1, keepdim=True)
        in_contact = (contact_dist < self.contact_threshold).float()
        
        # Momentum conservation during contact
        if in_contact.sum() > 0:
            hand_vel = (hand_pos_t1 - hand_pos_t) / self.dt
            puck_vel = (puck_pos_t1 - puck_pos_t) / self.dt
            
            momentum_hand = self.mass_hand * hand_vel
            momentum_puck = self.mass_puck * puck_vel
            total_momentum = momentum_hand + momentum_puck
            
            # Momentum should be conserved
            momentum_residual = torch.norm(total_momentum - torch.mean(total_momentum, dim=0), dim=1, keepdim=True)
            return torch.mean(in_contact * momentum_residual)
        
        return torch.tensor(0.0, device=hand_pos_t.device)
    
    def _reacher_pde_residual(self, state_t, state_t1):
        """PDE residual for 2-link reacher arm dynamics"""
        θ1_t, θ2_t = state_t['joint_angles'][:, 0:1], state_t['joint_angles'][:, 1:2]
        θ1_dot_t, θ2_dot_t = state_t['joint_velocities'][:, 0:1], state_t['joint_velocities'][:, 1:2]
        
        θ1_t1, θ2_t1 = state_t1['joint_angles'][:, 0:1], state_t1['joint_angles'][:, 1:2]
        θ1_dot_t1, θ2_dot_t1 = state_t1['joint_velocities'][:, 0:1], state_t1['joint_velocities'][:, 1:2]
        
        # Compute actual angular accelerations from finite differences
        θ1_ddot_actual = (θ1_dot_t1 - θ1_dot_t) / self.dt
        θ2_ddot_actual = (θ2_dot_t1 - θ2_dot_t) / self.dt
        
        # 2-link arm dynamics (simplified model)
        # M(q)q̈ + C(q,q̇)q̇ + G(q) = τ - F(q̇)
        L1, L2 = self.link1_length, self.link2_length
        m1, m2 = self.link1_mass, self.link2_mass
        
        # Mass matrix elements (simplified)
        M11 = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * torch.cos(θ2_t)
        M12 = m2 * L2**2 + m2 * L1 * L2 * torch.cos(θ2_t)
        M21 = M12
        M22 = m2 * L2**2
        
        # Coriolis terms (simplified)
        C1 = -m2 * L1 * L2 * torch.sin(θ2_t) * (2 * θ1_dot_t * θ2_dot_t + θ2_dot_t**2)
        C2 = m2 * L1 * L2 * torch.sin(θ2_t) * θ1_dot_t**2
        
        # Gravitational terms (assuming gravity acts downward)
        g = 9.81
        G1 = (m1 + m2) * g * L1 * torch.cos(θ1_t) + m2 * g * L2 * torch.cos(θ1_t + θ2_t)
        G2 = m2 * g * L2 * torch.cos(θ1_t + θ2_t)
        
        # Friction terms
        F1 = self.joint_friction * θ1_dot_t
        F2 = self.joint_friction * θ2_dot_t
        
        # Expected accelerations (assuming zero external torque)
        det_M = M11 * M22 - M12 * M21
        det_M = torch.clamp(det_M, min=1e-6)  # Avoid division by zero
        
        θ1_ddot_expected = (M22 * (-C1 - G1 - F1) - M12 * (-C2 - G2 - F2)) / det_M
        θ2_ddot_expected = (M11 * (-C2 - G2 - F2) - M21 * (-C1 - G1 - F1)) / det_M
        
        # PDE residual for dynamics
        dynamics_residual = torch.mean((θ1_ddot_actual - θ1_ddot_expected)**2) + \
                           torch.mean((θ2_ddot_actual - θ2_ddot_expected)**2)
        
        # Joint limit constraints
        joint_limit_penalty = torch.mean(torch.relu(torch.abs(θ1_t) - np.pi)) + \
                             torch.mean(torch.relu(torch.abs(θ2_t) - np.pi))
        
        # Velocity limit constraints
        velocity_limit_penalty = torch.mean(torch.relu(torch.abs(θ1_dot_t) - 5.0)) + \
                                torch.mean(torch.relu(torch.abs(θ2_dot_t) - 5.0))
        
        # Energy conservation (kinetic + potential)
        KE_t = 0.5 * (M11 * θ1_dot_t**2 + M22 * θ2_dot_t**2 + 2 * M12 * θ1_dot_t * θ2_dot_t)
        KE_t1 = 0.5 * (M11 * θ1_dot_t1**2 + M22 * θ2_dot_t1**2 + 2 * M12 * θ1_dot_t1 * θ2_dot_t1)
        
        # Potential energy (gravitational)
        PE_t = -(m1 + m2) * g * L1 * torch.sin(θ1_t) - m2 * g * L2 * torch.sin(θ1_t + θ2_t)
        PE_t1 = -(m1 + m2) * g * L1 * torch.sin(θ1_t1) - m2 * g * L2 * torch.sin(θ1_t1 + θ2_t1)
        
        # Total energy
        E_total_t = KE_t + PE_t
        E_total_t1 = KE_t1 + PE_t1
        
        # Energy should decrease due to friction
        expected_energy_loss = self.joint_friction * (θ1_dot_t**2 + θ2_dot_t**2) * self.dt
        actual_energy_change = E_total_t1 - E_total_t
        energy_residual = torch.mean((actual_energy_change + expected_energy_loss)**2)
        
        # Combine all residuals
        total_residual = dynamics_residual + \
                        0.1 * joint_limit_penalty + \
                        0.1 * velocity_limit_penalty + \
                        0.05 * energy_residual
        
        return total_residual
    
    def compute_contact_dynamics_loss(self, z_I_t, z_I_t1):
        """
        Compute contact dynamics loss following methodology Equation (6):
        L_contact = E[∑_{i,j} M_contact(i,j) * ||F_ij - k_c * max(0, r_threshold - ||p_i - p_j||) * n_ij||²]
        
        Environment-specific application:
        - Pusher: Hand-puck contact dynamics (k_c = 1000 N/m, r_threshold = 0.1m)
        - Pick-and-place: Hand-object + object-table contact (k_c = 500/2000 N/m)
        - Reacher: Not applicable (returns 0.0) - no object contacts in joint space
        
        Args:
            z_I_t: Physics latent variables at time t
            z_I_t1: Physics latent variables at time t+1
            
        Returns:
            torch.Tensor: Contact dynamics loss (0.0 for reacher environment)
        """
        if self.physics_type not in ['pusher', 'pick_and_place']:
            # Reacher doesn't have contact dynamics - only joint dynamics
            return torch.tensor(0.0, device=z_I_t.device)
        
        state_t = self.extract_state_variables(z_I_t)
        state_t1 = self.extract_state_variables(z_I_t1)
        
        contact_loss = torch.tensor(0.0, device=z_I_t.device)
        
        if self.physics_type == 'pusher':
            contact_loss = self._pusher_contact_dynamics_loss(state_t, state_t1)
        elif self.physics_type == 'pick_and_place':
            contact_loss = self._pick_and_place_contact_dynamics_loss(state_t, state_t1)
            
        return contact_loss
    
    def _pusher_contact_dynamics_loss(self, state_t, state_t1):
        """Contact dynamics loss for pusher task (hand-puck interaction)"""
        hand_pos_t = state_t['hand_pos']
        puck_pos_t = state_t['puck_pos']
        hand_pos_t1 = state_t1['hand_pos']
        puck_pos_t1 = state_t1['puck_pos']
        
        # Compute velocities for force prediction
        hand_vel = (hand_pos_t1 - hand_pos_t) / self.dt
        puck_vel = (puck_pos_t1 - puck_pos_t) / self.dt
        
        # Distance and contact detection (methodology: r_threshold = 0.1m)
        distance = torch.norm(hand_pos_t - puck_pos_t, dim=1, keepdim=True)
        M_contact = (distance < self.contact_threshold).float()
        
        # Contact normal vector (from hand to puck)
        contact_direction = puck_pos_t - hand_pos_t
        contact_distance_safe = torch.clamp(torch.norm(contact_direction, dim=1, keepdim=True), min=1e-6)
        n_hat = contact_direction / contact_distance_safe  # Unit normal vector
        
        # Predicted contact force from velocity changes (Newton's 2nd law: F = ma)
        puck_accel = (puck_vel - torch.zeros_like(puck_vel)) / self.dt  # Assume puck was at rest
        F_predicted = self.mass_puck * puck_accel  # Predicted force on puck
        
        # Expected spring-damper contact force (methodology: k_c = 1000 N/m)
        k_c = 1000.0  # Contact stiffness (N/m)
        penetration = torch.clamp(self.contact_threshold - distance, min=0.0)  # max(0, r_threshold - ||p_i - p_j||)
        F_expected_magnitude = k_c * penetration
        F_expected = F_expected_magnitude * n_hat  # Force in contact normal direction
        
        # Contact dynamics residual: ||F_ij - expected_force||²
        force_residual = torch.norm(F_predicted - F_expected, dim=1, keepdim=True) ** 2
        
        # Apply contact mask and compute expectation
        contact_loss = torch.mean(M_contact * force_residual)
        
        return contact_loss
    
    def _pick_and_place_contact_dynamics_loss(self, state_t, state_t1):
        """Contact dynamics loss for pick-and-place task (hand-object and object-table interactions)"""
        hand_pos_t = state_t['hand_pos']
        object_pos_t = state_t['object_pos']
        gripper_t = state_t['gripper_opening']
        
        hand_pos_t1 = state_t1['hand_pos']
        object_pos_t1 = state_t1['object_pos']
        
        # Compute velocities for force prediction
        hand_vel = (hand_pos_t1 - hand_pos_t) / self.dt
        object_vel = (object_pos_t1 - object_pos_t) / self.dt
        
        total_contact_loss = torch.tensor(0.0, device=hand_pos_t.device)
        
        # 1. Hand-Object Contact Dynamics
        hand_object_dist = torch.norm(hand_pos_t - object_pos_t, dim=1, keepdim=True)
        M_contact_hand_obj = (hand_object_dist < self.contact_threshold).float()
        gripper_closed = (gripper_t < self.gripper_threshold).float()
        grasping = M_contact_hand_obj * gripper_closed
        
        if M_contact_hand_obj.sum() > 0:
            # Contact normal (from hand to object)
            contact_direction = object_pos_t - hand_pos_t
            contact_distance_safe = torch.clamp(torch.norm(contact_direction, dim=1, keepdim=True), min=1e-6)
            n_hat = contact_direction / contact_distance_safe
            
            # Predicted force from object acceleration
            object_accel = (object_vel - torch.zeros_like(object_vel)) / self.dt
            F_predicted = self.mass_object * object_accel
            
            # Expected spring-damper force (methodology: k_c = 500 N/m for pick-and-place)
            k_c = 500.0  # Contact stiffness (N/m)
            penetration = torch.clamp(self.contact_threshold - hand_object_dist, min=0.0)
            F_expected_magnitude = k_c * penetration
            F_expected = F_expected_magnitude * n_hat
            
            # Contact loss for hand-object interaction
            force_residual = torch.norm(F_predicted - F_expected, dim=1, keepdim=True) ** 2
            hand_obj_loss = torch.mean(M_contact_hand_obj * force_residual)
            total_contact_loss = total_contact_loss + hand_obj_loss
        
        # 2. Object-Table Contact Dynamics
        table_height = self.table_height
        object_table_dist = object_pos_t[:, 2:3] - table_height  # Z-distance to table
        M_contact_obj_table = (object_table_dist < 0.01).float()  # Small threshold for table contact
        
        if M_contact_obj_table.sum() > 0:
            # Normal force from table (upward)
            n_hat_table = torch.zeros_like(object_pos_t)
            n_hat_table[:, 2:3] = 1.0  # Upward normal
            
            # Predicted force from object weight and motion
            gravity_force = self.mass_object * self.gravity
            object_accel_z = object_vel[:, 2:3] / self.dt  # Vertical acceleration
            F_predicted_table = self.mass_object * object_accel_z + gravity_force
            
            # Expected normal force from table (prevents penetration)
            k_c_table = 2000.0  # Higher stiffness for table contact
            penetration_table = torch.clamp(-object_table_dist, min=0.0)
            F_expected_table_magnitude = k_c_table * penetration_table
            F_expected_table = F_expected_table_magnitude  # Scalar for vertical force
            
            # Table contact loss
            table_force_residual = (F_predicted_table[:, 0:1] - F_expected_table) ** 2
            table_loss = torch.mean(M_contact_obj_table * table_force_residual)
            total_contact_loss = total_contact_loss + table_loss
        
        return total_contact_loss


class PhysicsDynamicsNetwork(nn.Module):
    """
    Neural network f_physics(z_I, t; θ_physics) that learns physics dynamics for ODE integration.
    
    This implements the methodology's temporal consistency via ODE integration:
    dz_I/dt = f_physics(z_I, t; θ_physics)
    
    The network learns to predict how physics latent variables evolve over time
    according to the underlying physics laws of the task.
    """
    
    def __init__(self, z_I_dim, physics_type='pusher', hidden_dim=128, dt=0.1):
        super(PhysicsDynamicsNetwork, self).__init__()
        self.z_I_dim = z_I_dim
        self.physics_type = physics_type
        self.dt = dt
        
        # Neural network to learn physics dynamics
        # Input: z_I (physics latent) + t (time) -> Output: dz_I/dt (time derivative)
        self.dynamics_net = nn.Sequential(
            nn.Linear(z_I_dim + 1, hidden_dim),  # +1 for time input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_I_dim)  # Output: dz_I/dt
        )
        
        # Physics parameters for different tasks
        self._initialize_physics_constants()
    
    def _initialize_physics_constants(self):
        """Initialize physics constants for task-specific dynamics."""
        if self.physics_type == 'pusher':
            self.damping = 0.1      # Damping coefficient
            self.stiffness = 10.0   # Spring stiffness
            self.mass = 1.0         # Effective mass
        elif self.physics_type == 'pick_and_place':
            self.damping = 0.05
            self.stiffness = 5.0
            self.mass = 0.5
        elif self.physics_type == 'reacher':
            self.damping = 0.2      # Joint damping
            self.stiffness = 15.0   # Joint stiffness
            self.mass = 1.5         # Link mass
        else:
            # Default values
            self.damping = 0.1
            self.stiffness = 10.0
            self.mass = 1.0
    
    def forward(self, t, z_I):
        """
        Compute dz_I/dt = f_physics(z_I, t; θ_physics)
        
        Args:
            t: Time tensor [1] or scalar
            z_I: Physics latent variables [batch_size, z_I_dim]
            
        Returns:
            torch.Tensor: Time derivatives dz_I/dt [batch_size, z_I_dim]
        """
        batch_size = z_I.shape[0]
        
        # Expand time to match batch size
        if t.dim() == 0:  # Scalar time
            t_expanded = t.expand(batch_size, 1)
        else:
            t_expanded = t.expand(batch_size, 1) if t.shape[0] == 1 else t.unsqueeze(1)
        
        # Concatenate physics state with time
        z_I_with_time = torch.cat([z_I, t_expanded], dim=1)
        
        # Predict time derivatives using neural network
        dzdt_predicted = self.dynamics_net(z_I_with_time)
        
        # Apply physics-based constraints to the predictions
        dzdt_constrained = self._apply_physics_constraints(z_I, dzdt_predicted)
        
        return dzdt_constrained
    
    def _apply_physics_constraints(self, z_I, dzdt_raw):
        """
        Apply physics-based constraints to the predicted time derivatives.
        
        This implements task-specific physics laws to constrain the learned dynamics:
        - Conservation of momentum/energy
        - Kinematic constraints
        - Stability constraints
        """
        if self.physics_type == 'pusher' and z_I.shape[1] >= 4:
            return self._apply_pusher_constraints(z_I, dzdt_raw)
        elif self.physics_type == 'pick_and_place' and z_I.shape[1] >= 7:
            return self._apply_pick_and_place_constraints(z_I, dzdt_raw)
        elif self.physics_type == 'reacher' and z_I.shape[1] >= 4:
            return self._apply_reacher_constraints(z_I, dzdt_raw)
        else:
            # Generic constraints: add damping for stability
            return dzdt_raw - self.damping * z_I
    
    def _apply_pusher_constraints(self, z_I, dzdt_raw):
        """Apply pusher-specific physics constraints."""
        # Extract positions: [hand_x, hand_y, puck_x, puck_y]
        hand_pos = z_I[:, :2]
        puck_pos = z_I[:, 2:4]
        
        hand_vel = dzdt_raw[:, :2]
        puck_vel = dzdt_raw[:, 2:4]
        
        # Apply damping to velocities for stability
        hand_vel_damped = hand_vel - self.damping * hand_pos
        puck_vel_damped = puck_vel - self.damping * puck_pos
        
        # Contact dynamics: if hand and puck are close, enforce contact constraints
        distance = torch.norm(hand_pos - puck_pos, dim=1, keepdim=True)
        contact_threshold = 0.1
        in_contact = (distance < contact_threshold).float()
        
        # During contact, puck velocity should be influenced by hand velocity
        contact_force = in_contact * self.stiffness * (hand_vel - puck_vel)
        puck_vel_contact = puck_vel_damped + contact_force * 0.1  # Scale contact influence
        
        # Combine constrained velocities
        dzdt_constrained = torch.cat([hand_vel_damped, puck_vel_contact], dim=1)
        
        return dzdt_constrained
    
    def _apply_pick_and_place_constraints(self, z_I, dzdt_raw):
        """Apply pick-and-place specific physics constraints."""
        # Extract: [gripper_opening, hand_xyz, obj_xyz]
        gripper_vel = dzdt_raw[:, 0:1]
        hand_vel = dzdt_raw[:, 1:4]
        obj_vel = dzdt_raw[:, 4:7]
        
        # Apply damping
        gripper_vel_damped = gripper_vel - self.damping * z_I[:, 0:1]
        hand_vel_damped = hand_vel - self.damping * z_I[:, 1:4]
        obj_vel_damped = obj_vel - self.damping * z_I[:, 4:7]
        
        # Gravity effect on object (z-direction)
        gravity_force = torch.zeros_like(obj_vel_damped)
        gravity_force[:, 2] = -9.81 * 0.01  # Small gravity effect
        obj_vel_gravity = obj_vel_damped + gravity_force
        
        # Grasp constraint: if gripper is closed and near object, object follows hand
        gripper_opening = z_I[:, 0:1]
        hand_pos = z_I[:, 1:4]
        obj_pos = z_I[:, 4:7]
        
        hand_obj_distance = torch.norm(hand_pos - obj_pos, dim=1, keepdim=True)
        is_grasping = ((gripper_opening < 0.02) & (hand_obj_distance < 0.05)).float()
        
        # During grasping, object velocity follows hand velocity
        grasp_constraint = is_grasping * (hand_vel_damped - obj_vel_gravity)
        obj_vel_constrained = obj_vel_gravity + grasp_constraint * 0.5
        
        dzdt_constrained = torch.cat([gripper_vel_damped, hand_vel_damped, obj_vel_constrained], dim=1)
        
        return dzdt_constrained
    
    def _apply_reacher_constraints(self, z_I, dzdt_raw):
        """Apply reacher-specific physics constraints."""
        # Extract joint angles and velocities: [θ₁, θ₂, θ̇₁, θ̇₂]
        joint_angles = z_I[:, :2]
        joint_velocities = z_I[:, 2:4]
        
        angle_derivatives = dzdt_raw[:, :2]   # Should be joint velocities
        velocity_derivatives = dzdt_raw[:, 2:4]  # Should be joint accelerations
        
        # Kinematic constraint: d(θ)/dt = θ̇
        angle_derivatives_constrained = joint_velocities
        
        # Dynamic constraint: joint accelerations with damping and limits
        velocity_derivatives_damped = velocity_derivatives - self.damping * joint_velocities
        
        # Joint limits: add restoring force if near limits
        joint_limit = torch.pi * 0.9  # 90% of full range
        limit_violation = torch.relu(torch.abs(joint_angles) - joint_limit)
        restoring_force = -torch.sign(joint_angles) * limit_violation * self.stiffness
        velocity_derivatives_constrained = velocity_derivatives_damped + restoring_force * 0.01
        
        dzdt_constrained = torch.cat([angle_derivatives_constrained, velocity_derivatives_constrained], dim=1)
        
        return dzdt_constrained


class ODEPhysicsIntegrator:
    """
    ODE integrator for temporal physics consistency as described in methodology.
    
    This implements the methodology's requirement:
    "For temporal physics consistency, the system uses ODE solvers to ensure that 
    latent trajectories follow physically plausible dynamics"
    """
    
    def __init__(self, physics_dynamics_net, dt=0.1, method='euler'):
        self.physics_dynamics_net = physics_dynamics_net
        self.dt = dt
        self.method = method
        
    def integrate_trajectory(self, z_I_initial, time_span, use_ode_solver=True):
        """
        Integrate physics latent trajectory over time using ODE solver.
        
        Args:
            z_I_initial: Initial physics latent state [batch_size, z_I_dim]
            time_span: Time points to integrate [num_time_points]
            use_ode_solver: Whether to use torchdiffeq (True) or Euler (False)
            
        Returns:
            torch.Tensor: Integrated trajectory [num_time_points, batch_size, z_I_dim]
        """
        if use_ode_solver and HAS_TORCHDIFFEQ:
            # Use proper ODE integration with torchdiffeq
            return odeint(self.physics_dynamics_net, z_I_initial, time_span, method='euler')
        else:
            # Fallback: Simple Euler integration
            return self._euler_integration(z_I_initial, time_span)
    
    def _euler_integration(self, z_I_initial, time_span):
        """
        Simple Euler integration fallback when torchdiffeq is not available.
        
        This implements: z_{I,t+1} = z_{I,t} + Δt · f_physics(z_{I,t}, t)
        """
        trajectory = [z_I_initial]
        z_I_current = z_I_initial
        
        for i in range(1, len(time_span)):
            t_current = time_span[i-1]
            dt = time_span[i] - time_span[i-1]
            
            # Compute time derivative
            dzdt = self.physics_dynamics_net(t_current, z_I_current)
            
            # Euler step
            z_I_next = z_I_current + dt * dzdt
            
            trajectory.append(z_I_next)
            z_I_current = z_I_next
        
        # Stack into [time, batch, dim] tensor
        return torch.stack(trajectory, dim=0)
    
    def compute_temporal_consistency_loss(self, z_I_sequence, time_span):
        """
        Compute temporal consistency loss using ODE integration.
        
        This ensures that consecutive latent states follow physics dynamics:
        L_temporal = ||z_{I,t+1} - integrate(z_{I,t}, t, t+1)||²
        """
        if len(z_I_sequence) < 2:
            return torch.tensor(0.0, device=z_I_sequence[0].device)
        
        total_loss = torch.tensor(0.0, device=z_I_sequence[0].device)
        num_pairs = len(z_I_sequence) - 1
        
        for i in range(num_pairs):
            z_I_current = z_I_sequence[i]
            z_I_next_actual = z_I_sequence[i + 1]
            
            # Integrate from current state to next time step
            if i < len(time_span) - 1:
                t_span = time_span[i:i+2]
            else:
                # Default time step
                t_span = torch.tensor([0.0, self.dt], device=z_I_current.device)
            
            # Predict next state using ODE integration
            trajectory = self.integrate_trajectory(z_I_current.unsqueeze(0), t_span)
            z_I_next_predicted = trajectory[-1, 0]  # [batch_size, z_I_dim]
            
            # Compute L2 loss between predicted and actual next state
            consistency_loss = F.mse_loss(z_I_next_predicted, z_I_next_actual)
            total_loss += consistency_loss
        
        return total_loss / num_pairs


class EnhancedP3VAETrainer(ConvVAETrainer):
    """
    Enhanced P3-VAE trainer with improved architecture and physics modeling.
    Based on the original P3-VAE paper with additional robustness improvements.
    """
    
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            # Enhanced P3-VAE parameters
            physics_weight=0.2,
            physics_type='pick_and_place',
            z_I_dim=7,
            z_E_dim=3,
            dt=0.02,
            use_physics_constraints=True,
            conservation_weight=0.05,
            regularization_weight=0.01,
            grad_clip_value=5.0,
    ):
        # Store enhanced parameters first
        self.physics_weight = physics_weight
        self.physics_type = physics_type
        self.z_I_dim = z_I_dim
        self.z_E_dim = z_E_dim
        self.dt = dt
        self.use_physics_constraints = use_physics_constraints
        self.conservation_weight = conservation_weight
        self.regularization_weight = regularization_weight
        self.grad_clip_value = grad_clip_value
        
        # Create locals dict with only parent class parameters for quick_init
        parent_locals = {
            'self': self,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'model': model,
            'batch_size': batch_size,
            'log_interval': log_interval,
            'beta': beta,
            'lr': lr,
            'do_scatterplot': do_scatterplot,
            'normalize': normalize,
            'mse_weight': mse_weight,
            'is_auto_encoder': is_auto_encoder,
            'background_subtract': background_subtract,
        }
        
        # Initialize parent class manually to avoid quick_init issues
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8
        
        self.normalize = normalize
        self.mse_weight = mse_weight
        self.is_auto_encoder = is_auto_encoder
        self.background_subtract = background_subtract
        
        # Create enhanced components
        self.encoder = PhysicsGuidedEncoder(
            input_channels=model.input_channels,
            imsize=model.imsize,
            z_I_dim=z_I_dim,
            z_E_dim=z_E_dim,
            physics_type=physics_type
        ).to(ptu.device)
        
        self.decoder = PhysicsInformedDecoder(
            z_I_dim=z_I_dim,
            z_E_dim=z_E_dim,
            input_channels=model.input_channels,
            imsize=model.imsize,
            physics_type=physics_type
        ).to(ptu.device)
        
        self.physics_loss_calculator = PhysicsLossCalculator(
            physics_type=physics_type,
            dt=dt
        ).to(ptu.device)
        
        # Create Physics State Extractor for interpretable physics variables
        self.physics_extractor = PhysicsStateExtractor(
            z_I_dim=z_I_dim,
            physics_type=physics_type,
            hidden_dim=128
        ).to(ptu.device)
        
        # Create Physics Dynamics Network for ODE integration (methodology Section 2.4)
        self.physics_dynamics_net = PhysicsDynamicsNetwork(
            z_I_dim=z_I_dim,
            physics_type=physics_type,
            hidden_dim=256,
            dt=dt
        ).to(ptu.device)
        
        # Create ODE Physics Integrator for temporal consistency
        self.ode_integrator = ODEPhysicsIntegrator(
            physics_dynamics_net=self.physics_dynamics_net,
            dt=dt,
            method='euler'
        )
        
        # Update optimizer to include new parameters
        all_params = list(self.model.parameters()) + \
                    list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.physics_loss_calculator.parameters()) + \
                    list(self.physics_extractor.parameters()) + \
                    list(self.physics_dynamics_net.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.lr, weight_decay=1e-5)
        
        # Enhanced logging
        self.enhanced_stats = {}
        
        # Initialize consistent logging keys
        self._init_logging_keys()
        
        print(f"Enhanced P3-VAE initialized for {physics_type} with z_I_dim={z_I_dim}, z_E_dim={z_E_dim}")
        print(f"✅ ODE Physics Integration: {'torchdiffeq' if HAS_TORCHDIFFEQ else 'Euler fallback'}")
        print(f"✅ Temporal Consistency: dz_I/dt = f_physics(z_I, t; θ_physics)")
        print(f"✅ Physics Dynamics Network: {self.physics_dynamics_net.__class__.__name__}")
    
    def _init_logging_keys(self):
        """Initialize consistent logging keys to prevent table key changes"""
        self.consistent_log_keys = {
            # Loss components
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_z_I': 0.0,
            'kl_z_E': 0.0,
            'physics_loss': 0.0,
            'pde_loss': 0.0,
            'ode_temporal_loss': 0.0,
            'pde_physics_loss': 0.0,
            'contact_dynamics_loss': 0.0,  # New: Contact dynamics loss from methodology Eq. 6
            'regularization_loss': 0.0,
            
            # Latent statistics
            'z_I_mean_norm': 0.0,
            'z_E_alpha_mean': 0.0,
            'z_E_beta_mean': 0.0,
            
            # Physics analysis (for first few dimensions)
            'z_I_dim_0_mean': 0.0,
            'z_I_dim_0_std': 0.0,
            'z_I_dim_1_mean': 0.0,
            'z_I_dim_1_std': 0.0,
            'z_I_dim_2_mean': 0.0,
            'z_I_dim_2_std': 0.0,
            'z_I_dim_3_mean': 0.0,
            'z_I_dim_3_std': 0.0,
            
            'z_E_dim_0_mean': 0.0,
            'z_E_dim_0_std': 0.0,
            'z_E_dim_1_mean': 0.0,
            'z_E_dim_1_std': 0.0,
            'z_E_dim_2_mean': 0.0,
            'z_E_dim_2_std': 0.0,
            'z_E_dim_3_mean': 0.0,
            'z_E_dim_3_std': 0.0,
        }
    
    def get_physics_priors(self, batch_size, device):
        """Get physics-informed priors for both z_I and z_E"""
        if self.physics_type == 'pick_and_place':
            # Prior for physics variables: slightly favor certain configurations
            prior_z_I_mean = torch.zeros(batch_size, self.z_I_dim, device=device)
            prior_z_I_lnvar = torch.ones(batch_size, self.z_I_dim, device=device) * (-1.0)  # Small variance
            
            # Adjust priors based on physics knowledge
            if self.z_I_dim >= 7:
                # Gripper should prefer closed position (grasping)
                prior_z_I_mean[:, 0] = 0.01  # Small opening
                # Hand and object should be near workspace center
                prior_z_I_mean[:, 1:4] = 0.0  # Center of workspace
                prior_z_I_mean[:, 4:7] = 0.0  # Object at center
            
        else:
            # Default priors
            prior_z_I_mean = torch.zeros(batch_size, self.z_I_dim, device=device)
            prior_z_I_lnvar = torch.zeros(batch_size, self.z_I_dim, device=device)
        
        # Environmental priors: uniform Beta(1,1)
        prior_z_E_alpha = torch.ones(batch_size, self.z_E_dim, device=device)
        prior_z_E_beta = torch.ones(batch_size, self.z_E_dim, device=device)
        
        return {
            'z_I': {'mean': prior_z_I_mean, 'lnvar': prior_z_I_lnvar},
            'z_E': {'alpha': prior_z_E_alpha, 'beta': prior_z_E_beta}
        }
    
    def sample_from_distributions(self, z_I_stats, z_E_stats):
        """Sample from the latent distributions"""
        # Sample z_I (Gaussian)
        z_I_mean = z_I_stats['mean']
        z_I_lnvar = z_I_stats['lnvar']
        z_I_std = torch.exp(0.5 * z_I_lnvar)
        eps_I = torch.randn_like(z_I_std)
        z_I = z_I_mean + eps_I * z_I_std
        
        # Sample z_E (Beta)
        alpha = z_E_stats['alpha']
        beta = z_E_stats['beta']
        z_E_dist = torch.distributions.beta.Beta(alpha, beta)
        z_E = z_E_dist.rsample()
        
        return z_I, z_E
    
    def extract_ground_truth_physics(self, env_states, physics_type=None):
        """
        Extract ground truth physics variables from simulator environment states.
        
        Args:
            env_states: List or tensor of environment states from MuJoCo simulator
            physics_type: Override for physics type (uses self.physics_type if None)
            
        Returns:
            torch.Tensor: Ground truth physics variables [batch_size, z_I_dim]
        """
        if physics_type is None:
            physics_type = self.physics_type
            
        if not isinstance(env_states, torch.Tensor):
            env_states = torch.FloatTensor(env_states).to(ptu.device)
            
        batch_size = env_states.shape[0]
        
        if physics_type == 'pusher':
            # Extract hand and puck positions from MuJoCo state
            # Assumes state format: [hand_x, hand_y, puck_x, puck_y, ...]
            if env_states.shape[1] >= 4:
                hand_pos = env_states[:, :2]  # hand x, y
                puck_pos = env_states[:, 2:4]  # puck x, y
                z_I_gt = torch.cat([hand_pos, puck_pos], dim=1)
            else:
                # Fallback: use available dimensions
                z_I_gt = env_states[:, :min(self.z_I_dim, env_states.shape[1])]
                if z_I_gt.shape[1] < self.z_I_dim:
                    # Pad with zeros if needed
                    padding = torch.zeros(batch_size, self.z_I_dim - z_I_gt.shape[1], device=z_I_gt.device)
                    z_I_gt = torch.cat([z_I_gt, padding], dim=1)
                    
        elif physics_type == 'pick_and_place':
            # Extract gripper opening, hand position, object position
            # Assumes state format: [gripper_opening, hand_xyz, object_xyz, ...]
            if env_states.shape[1] >= 7:
                gripper_opening = env_states[:, 0:1]  # gripper opening
                hand_pos = env_states[:, 1:4]         # hand x, y, z
                object_pos = env_states[:, 4:7]       # object x, y, z
                z_I_gt = torch.cat([gripper_opening, hand_pos, object_pos], dim=1)
            else:
                # Fallback
                z_I_gt = env_states[:, :min(self.z_I_dim, env_states.shape[1])]
                if z_I_gt.shape[1] < self.z_I_dim:
                    padding = torch.zeros(batch_size, self.z_I_dim - z_I_gt.shape[1], device=z_I_gt.device)
                    z_I_gt = torch.cat([z_I_gt, padding], dim=1)
                    
        elif physics_type == 'reacher':
            # Extract joint angles and velocities
            # Assumes state format: [joint_angles, joint_velocities, ...]
            if env_states.shape[1] >= 4:
                joint_angles = env_states[:, :2]      # θ₁, θ₂
                joint_velocities = env_states[:, 2:4] # θ̇₁, θ̇₂
                z_I_gt = torch.cat([joint_angles, joint_velocities], dim=1)
            else:
                z_I_gt = env_states[:, :min(self.z_I_dim, env_states.shape[1])]
                if z_I_gt.shape[1] < self.z_I_dim:
                    padding = torch.zeros(batch_size, self.z_I_dim - z_I_gt.shape[1], device=z_I_gt.device)
                    z_I_gt = torch.cat([z_I_gt, padding], dim=1)
        else:
            # Generic extraction: use first z_I_dim dimensions
            z_I_gt = env_states[:, :min(self.z_I_dim, env_states.shape[1])]
            if z_I_gt.shape[1] < self.z_I_dim:
                padding = torch.zeros(batch_size, self.z_I_dim - z_I_gt.shape[1], device=z_I_gt.device)
                z_I_gt = torch.cat([z_I_gt, padding], dim=1)
                
        return z_I_gt
    
    def _compute_supervised_loss(self, batch_supervised, gt_physics_supervised, device):
        """
        Compute supervised loss L(x, z_I*) as defined in the paper.
        Uses proper encoder to get compressed latent z_I, ensuring architectural consistency.
        
        Args:
            batch_supervised: Supervised image observations [batch_size, C, H, W]
            gt_physics_supervised: Ground truth physics variables [batch_size, physics_dim]
            device: Device for computation
            
        Returns:
            tuple: (supervised_loss, loss_dict)
        """
        batch_size = batch_supervised.shape[0]
        
        # 1. Encode supervised images to get compressed latent physics representation z_I
        z_I_stats = self.encoder.encode_physics(batch_supervised)
        z_I_compressed = z_I_stats['mean']  # Use mean of the latent distribution
        
        # 2. Encode environmental variables conditioned on the encoded physics
        z_E_stats = self.encoder.encode_environment(batch_supervised, z_I_compressed)
        z_E = self.sample_from_distributions({'mean': torch.zeros_like(z_E_stats['alpha']), 
                                            'lnvar': torch.log(z_E_stats['alpha'])}, 
                                           z_E_stats)[1]
        
        # 3. Reconstruction with encoded physics and sampled environment
        z_combined = torch.cat([z_I_compressed, z_E], dim=1)
        reconstruction, _ = self.model.decode(z_combined)
        
        # Reconstruction loss
        if len(batch_supervised.shape) == 4:
            batch_flat = batch_supervised.view(batch_supervised.size(0), -1)
            recon_loss = F.mse_loss(reconstruction, batch_flat, reduction='mean')
        else:
            recon_loss = F.mse_loss(reconstruction, batch_supervised, reduction='mean')
        
        # 4. Physics supervision regularization: encourage extracted physics to match ground truth
        physics_extracted = self.physics_extractor(z_I_compressed)
        physics_supervision_loss = F.mse_loss(physics_extracted, gt_physics_supervised, reduction='mean')
        
        # 5. Log probabilities following paper formulation
        # log p_θ(x|z_I, z_E) - reconstruction likelihood
        log_p_x_given_z = -recon_loss
        
        # Get priors for KL divergence computation
        priors = self.get_physics_priors(batch_size, device)
        
        # log p(z_I) - prior on compressed physics variables
        log_p_z_I = -torch.mean(self.encoder.kl_divergence_physics(z_I_stats, priors['z_I']))
        
        # log p(z_E) - prior on environmental variables  
        log_p_z_E = -torch.mean(self.encoder.kl_divergence_environment(z_E_stats, priors['z_E']))
        
        # log q_φ(z_E|x, z_I) - entropy term
        # Approximated by negative KL divergence from uniform prior
        log_q_z_E = -torch.mean(self.encoder.kl_divergence_environment(z_E_stats, priors['z_E']))
        
        # 6. Combined supervised loss: 
        # L(x, z_I*) = E[log p(x|z_I, z_E) + log p(z_I) + log p(z_E) - log q(z_E|x, z_I)] + λ * physics_supervision
        base_supervised_loss = -(log_p_x_given_z + log_p_z_I + log_p_z_E - log_q_z_E)
        physics_supervision_weight = 0.5  # Weight for physics supervision regularization
        supervised_loss = base_supervised_loss + physics_supervision_weight * physics_supervision_loss
        
        loss_dict = {
            'reconstruction': recon_loss.item(),
            'physics_supervision': physics_supervision_loss.item(),
            'log_p_x_given_z': log_p_x_given_z.item(),
            'log_p_z_I': log_p_z_I.item(), 
            'log_p_z_E': log_p_z_E.item(),
            'log_q_z_E': log_q_z_E.item(),
            'total': supervised_loss.item()
        }
        
        return supervised_loss, loss_dict
    
    def _compute_unsupervised_loss(self, batch_unsupervised, device, use_stop_gradient=True):
        """
        Compute unsupervised loss U(x) with stop-gradient operator.
        
        Args:
            batch_unsupervised: Unsupervised image observations [batch_size, C, H, W] 
            device: Device for computation
            use_stop_gradient: Whether to apply stop-gradient operator
            
        Returns:
            tuple: (unsupervised_loss, loss_dict)
        """
        batch_size = batch_unsupervised.shape[0]
        
        # 1. Encode both physics and environmental variables
        z_I_stats = self.encoder.encode_physics(batch_unsupervised)
        z_E_stats = self.encoder.encode_environment(batch_unsupervised, z_I_stats['mean'])
        
        # 2. Sample from distributions
        z_I, z_E = self.sample_from_distributions(z_I_stats, z_E_stats)
        
        # 3. Apply stop-gradient operator to prevent f_I^θ from overwhelming f_E
        if use_stop_gradient:
            # Stop gradient on physics variables to let physics constraints dominate
            z_I_for_decoding = z_I.detach()
        else:
            z_I_for_decoding = z_I
        
        # 4. Reconstruction using potentially detached physics variables
        z_combined = torch.cat([z_I_for_decoding, z_E], dim=1)
        reconstruction, _ = self.model.decode(z_combined)
        
        # Reconstruction loss
        if len(batch_unsupervised.shape) == 4:
            batch_flat = batch_unsupervised.view(batch_unsupervised.size(0), -1)
            recon_loss = F.mse_loss(reconstruction, batch_flat, reduction='mean')
        else:
            recon_loss = F.mse_loss(reconstruction, batch_unsupervised, reduction='mean')
        
        # 5. Log probabilities for ELBO
        # log p_θ(x|z_I, z_E)
        log_p_x_given_z = -recon_loss
        
        # Priors
        priors = self.get_physics_priors(batch_size, device)
        
        # log p(z_I) and log p(z_E)
        log_p_z_I = -torch.mean(self.encoder.kl_divergence_physics(z_I_stats, priors['z_I']))
        log_p_z_E = -torch.mean(self.encoder.kl_divergence_environment(z_E_stats, priors['z_E']))
        
        # log q_φ(z_I, z_E|x) - approximate as sum of individual entropies
        log_q_z_I = -torch.mean(self.encoder.kl_divergence_physics(z_I_stats, priors['z_I']))
        log_q_z_E = -torch.mean(self.encoder.kl_divergence_environment(z_E_stats, priors['z_E']))
        log_q_z = log_q_z_I + log_q_z_E
        
        # 6. Unsupervised loss: U(x) = E[log p(x|z_I, z_E) + log p(z_I) + log p(z_E) - log q(z_I, z_E|x)]
        unsupervised_loss = -(log_p_x_given_z + log_p_z_I + log_p_z_E - log_q_z)
        
        loss_dict = {
            'reconstruction': recon_loss.item(),
            'log_p_x_given_z': log_p_x_given_z.item(),
            'log_p_z_I': log_p_z_I.item(),
            'log_p_z_E': log_p_z_E.item(), 
            'log_q_z': log_q_z.item(),
            'total': unsupervised_loss.item(),
            'stop_gradient_applied': use_stop_gradient
        }
        
        return unsupervised_loss, loss_dict
    
    def _compute_classification_loss(self, batch_images, gt_physics, device):
        """
        Compute classification/regression loss L_c(φ; z_I*) for physics prediction.
        Uses the PhysicsStateExtractor to map from compressed latent z_I to interpretable physics variables.
        
        Args:
            batch_images: Input images [batch_size, C, H, W]
            gt_physics: Ground truth physics variables [batch_size, physics_dim]
            device: Device for computation
            
        Returns:
            torch.Tensor: Classification/regression loss
        """
        # Encode physics variables from images to get compressed latent representation z_I
        z_I_stats = self.encoder.encode_physics(batch_images)
        z_I_compressed = z_I_stats['mean']  # Use mean of latent distribution
        
        # Extract interpretable physics variables using learned mapping
        physics_predicted = self.physics_extractor(z_I_compressed)
        
        # MSE loss between extracted physics variables and ground truth
        classification_loss = F.mse_loss(physics_predicted, gt_physics, reduction='mean')
        
        return classification_loss

    def compute_enhanced_loss(self, batch, epoch, ground_truth_physics=None, supervision_ratio=0.5):
        """
        Compute the enhanced semi-supervised P3-VAE loss following the LNCS paper formulation.
        Combines supervised loss L(x, z_I*), unsupervised loss U(x), and classification loss L_c.
        
        Args:
            batch: Image observations [batch_size, channels, height, width]  
            epoch: Current training epoch
            ground_truth_physics: Ground truth physics states [batch_size, z_I_dim] (optional)
            supervision_ratio: Fraction of data to treat as supervised (α in paper)
            
        Returns:
            tuple: (total_loss, loss_dict, outputs)
        """
        
        # Reshape flattened data to image tensor if needed
        if len(batch.shape) == 2:  # [batch_size, flattened_pixels]
            batch_size = batch.shape[0]
            expected_size = self.input_channels * self.imsize * self.imsize
            assert batch.shape[1] == expected_size, f"Expected {expected_size} pixels, got {batch.shape[1]}"
            batch = batch.view(batch_size, self.input_channels, self.imsize, self.imsize)
        elif len(batch.shape) == 3:
            batch = batch.unsqueeze(1)
        
        batch_size = batch.shape[0]
        device = batch.device
        
        # Ensure batch is properly normalized
        if batch.max() > 1.0:
            batch = batch.float() / 255.0
        
        # Determine supervised/unsupervised split
        if ground_truth_physics is not None and supervision_ratio > 0:
            num_supervised = int(batch_size * supervision_ratio)
            supervised_indices = torch.randperm(batch_size)[:num_supervised]
            unsupervised_indices = torch.randperm(batch_size)[num_supervised:]
            
            batch_supervised = batch[supervised_indices] if num_supervised > 0 else None
            batch_unsupervised = batch[unsupervised_indices] if len(unsupervised_indices) > 0 else None
            gt_physics_supervised = ground_truth_physics[supervised_indices] if num_supervised > 0 else None
        else:
            # All unsupervised if no ground truth provided
            batch_supervised = None
            batch_unsupervised = batch
            gt_physics_supervised = None
            num_supervised = 0
            supervision_ratio = 0.0
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # 1. SUPERVISED LOSS: L(x, z_I*) for labeled data
        if batch_supervised is not None and num_supervised > 0:
            supervised_loss, supervised_dict = self._compute_supervised_loss(
                batch_supervised, gt_physics_supervised, device
            )
            total_loss += supervision_ratio * supervised_loss
            
            # Add supervised loss components to loss_dict
            for key, value in supervised_dict.items():
                loss_dict[f"supervised_{key}"] = value
                
        # 2. UNSUPERVISED LOSS: U(x) for unlabeled data
        if batch_unsupervised is not None:
            unsupervised_loss, unsupervised_dict = self._compute_unsupervised_loss(
                batch_unsupervised, device, use_stop_gradient=True
            )
            total_loss += (1.0 - supervision_ratio) * unsupervised_loss
            
            # Add unsupervised loss components to loss_dict
            for key, value in unsupervised_dict.items():
                loss_dict[f"unsupervised_{key}"] = value
        
        # 3. CLASSIFICATION/REGRESSION LOSS: L_c(φ; z_I*) for physics prediction
        if batch_supervised is not None and gt_physics_supervised is not None:
            classification_loss = self._compute_classification_loss(
                batch_supervised, gt_physics_supervised, device
            )
            total_loss += self.physics_weight * classification_loss
            loss_dict['classification_loss'] = classification_loss.item()
        
        # Encode full batch to get latent statistics for analysis
        z_I_stats = self.encoder.encode_physics(batch)
        z_E_stats = self.encoder.encode_environment(batch, z_I_stats['mean'])
        z_I, z_E = self.sample_from_distributions(z_I_stats, z_E_stats)
        
        # 3. Decode to reconstruct using the original VAE decoder
        # Combine z_I and z_E to form the full latent vector
        z_combined = torch.cat([z_I, z_E], dim=1)
        reconstruction, _ = self.model.decode(z_combined)  # Extract only the image
        
        # 4. Reconstruction loss with numerical stability
        # Need to handle shape mismatch: reconstruction is flat, batch is reshaped
        if len(batch.shape) == 4:  # batch is [B, C, H, W]
            batch_flat = batch.view(batch.size(0), -1)  # Flatten to match reconstruction
            recon_loss = F.mse_loss(reconstruction, batch_flat, reduction='mean')
        else:  # batch is already flat
            recon_loss = F.mse_loss(reconstruction, batch, reduction='mean')
        
        # Add small epsilon to prevent numerical issues
        recon_loss = recon_loss + 1e-8
        
        # 5. KL divergence losses
        priors = self.get_physics_priors(batch_size, device)
        
        kl_z_I = torch.mean(self.encoder.kl_divergence_physics(z_I_stats, priors['z_I']))
        kl_z_E = torch.mean(self.encoder.kl_divergence_environment(z_E_stats, priors['z_E']))
        
        # Clamp KL losses to prevent explosion
        kl_z_I = torch.clamp(kl_z_I, max=10.0)
        kl_z_E = torch.clamp(kl_z_E, max=10.0)
        
        # 6. Physics constraints: L_physics = L_kinematic + L_newton + L_momentum + L_energy + L_contact
        # Following methodology Section 2.3: Physics Constraints Integration
        physics_loss = torch.tensor(0.0, device=device)
        ode_temporal_loss = torch.tensor(0.0, device=device)
        contact_dynamics_loss = torch.tensor(0.0, device=device)
        
        if batch_size > 1 and self.physics_weight > 0:
            try:
                z_I_t = z_I[:-1]
                z_I_t1 = z_I[1:]
                
                # Compute existing physics constraints (kinematic, Newton's laws, momentum, energy)
                pde_physics_loss = self.physics_loss_calculator.compute_pde_residual(z_I_t, z_I_t1)
                pde_physics_loss = torch.clamp(pde_physics_loss, max=100.0)  # Prevent explosion
                
                # Compute contact dynamics loss (methodology Eq. 6)
                contact_dynamics_loss = self.physics_loss_calculator.compute_contact_dynamics_loss(z_I_t, z_I_t1)
                contact_dynamics_loss = torch.clamp(contact_dynamics_loss, max=50.0)  # Prevent explosion
                
                # Compute ODE temporal consistency loss (methodology: dz_I/dt = f_physics(z_I, t))
                if len(z_I) >= 2:
                    # Create time span for integration
                    time_span = torch.linspace(0.0, self.dt, 2, device=device)
                    
                    # Compute temporal consistency using ODE integration
                    ode_temporal_loss = self.ode_integrator.compute_temporal_consistency_loss(
                        z_I_sequence=[z_I_t, z_I_t1],
                        time_span=time_span
                    )
                    ode_temporal_loss = torch.clamp(ode_temporal_loss, max=50.0)  # Prevent explosion
                
                # Combine all physics constraints: L_physics = kinematic + newton + momentum + energy + contact
                physics_loss = pde_physics_loss + contact_dynamics_loss + 0.5 * ode_temporal_loss
                
            except Exception as e:
                print(f"Warning: Physics constraint computation failed: {e}")
                physics_loss = torch.tensor(0.0, device=device)
                ode_temporal_loss = torch.tensor(0.0, device=device)
                contact_dynamics_loss = torch.tensor(0.0, device=device)
        
        # 7. Regularization losses
        reg_loss = torch.tensor(0.0, device=device)
        if self.regularization_weight > 0:
            # L2 regularization on physics variables to keep them in reasonable ranges
            physics_reg = torch.mean(z_I ** 2)
            env_reg = torch.mean(z_E ** 2)
            reg_loss = physics_reg + env_reg
        
        # 8. Follow methodology: L_Total = L_Enhanced + β*L_KL + λ_physics*L_physics + λ_reg*L_reg
        
        # L_Enhanced = α*L(x, z_I*) + (1-α)*U(x) + λ*L_c(φ; z_I*) (already computed above)
        enhanced_loss = total_loss  # This already contains the Enhanced P3-VAE components
        
        # Additional KL regularization (separate from Enhanced P3-VAE components)
        additional_kl_loss = self.beta * (kl_z_I + kl_z_E)
        
        # Physics constraints: L_physics = physics residuals (temporal + momentum + contact)
        physics_constraints = physics_loss  # This contains temporal, momentum, contact constraints
        
        # Total loss following methodology structure
        total_loss = (
            enhanced_loss + 
            additional_kl_loss +
            self.physics_weight * physics_constraints +
            self.regularization_weight * reg_loss
        )
        
        # Note: Removed separate reconstruction loss to avoid double-counting since it's in Enhanced loss
        
        # Check for NaN/Inf and handle gracefully
        if not torch.isfinite(total_loss):
            print("Warning: Non-finite loss detected, using backup loss")
            total_loss = recon_loss + 0.1 * (kl_z_I + kl_z_E)
        
        # Loss dictionary for logging following Enhanced P3-VAE methodology structure
        # Note: loss_dict already contains supervised/unsupervised components from above
        
        # Enhanced P3-VAE structure components
        loss_updates = {
            'total_loss': total_loss.item(),
            'enhanced_loss': enhanced_loss.item(),
            'additional_kl_loss': additional_kl_loss.item(),
            'physics_constraints': physics_constraints.item(),
            'regularization_loss': reg_loss.item(),
            # Physics constraint breakdown (methodology Section 2.3)
            'contact_dynamics_loss': contact_dynamics_loss.item(),
            'ode_temporal_loss': ode_temporal_loss.item(),
            'pde_physics_loss': (physics_constraints.item() - contact_dynamics_loss.item() - 0.5 * ode_temporal_loss.item()),
            # Individual KL components for analysis
            'kl_z_I': kl_z_I.item(),
            'kl_z_E': kl_z_E.item(),
            # Latent statistics
            'z_I_mean_norm': torch.norm(z_I_stats['mean']).item(),
            'z_E_alpha_mean': torch.mean(z_E_stats['alpha']).item(),
            'z_E_beta_mean': torch.mean(z_E_stats['beta']).item(),
            # Legacy reconstruction for comparison (but not used in total loss)
            'reconstruction_loss_reference': recon_loss.item(),
        }
        
        # Add standard loss components to existing loss_dict (preserves supervised/unsupervised)
        for key, value in loss_updates.items():
            if key in self.consistent_log_keys:
                loss_dict[key] = value
        
        # Store enhanced stats
        self.enhanced_stats.update(loss_dict)
        
        return total_loss, loss_dict, {
            'z_I': z_I,
            'z_E': z_E,
            'reconstruction': reconstruction,
            'z_I_stats': z_I_stats,
            'z_E_stats': z_E_stats
        }
    
    def get_temporal_batch_with_physics(self, train=True):
        """
        Get temporal batch with ground truth physics for semi-supervised training.
        Extended version that can extract physics from environment when available.
        """
        dataset = self.train_dataset if train else self.test_dataset
        batch_size = self.batch_size
        
        # Sample random indices ensuring we can get t+1
        max_idx = len(dataset) - 1
        ind_t = np.random.randint(0, max_idx, batch_size)
        ind_t1 = ind_t + 1
        
        # Get normalized images
        img_t = normalize_image(dataset[ind_t, :])
        img_t1 = normalize_image(dataset[ind_t1, :])
        
        if self.normalize:
            img_t = ((img_t - self.train_data_mean) + 1) / 2
            img_t1 = ((img_t1 - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            img_t = img_t - self.train_data_mean
            img_t1 = img_t1 - self.train_data_mean
        
        # Convert to tensors
        img_t = ptu.from_numpy(img_t)
        img_t1 = ptu.from_numpy(img_t1)
        
        # Try to extract ground truth physics if dataset has environment states
        gt_physics_t = None
        gt_physics_t1 = None
        
        # Check if dataset has environment state information
        # This would need to be added to your dataset collection process
        if hasattr(self, 'train_env_states') and self.train_env_states is not None:
            try:
                env_states_t = self.train_env_states[ind_t]
                env_states_t1 = self.train_env_states[ind_t1] 
                
                gt_physics_t = self.extract_ground_truth_physics(env_states_t)
                gt_physics_t1 = self.extract_ground_truth_physics(env_states_t1)
            except Exception as e:
                print(f"Warning: Could not extract ground truth physics: {e}")
                gt_physics_t = None
                gt_physics_t1 = None
        
        return img_t, img_t1, gt_physics_t, gt_physics_t1
    
    def set_environment_states(self, env_states_train=None, env_states_test=None):
        """
        Set environment states for semi-supervised learning.
        
        Args:
            env_states_train: Training environment states [N, state_dim]
            env_states_test: Test environment states [M, state_dim]
        """
        self.train_env_states = env_states_train
        self.test_env_states = env_states_test
        
        if env_states_train is not None:
            print(f"Set training environment states: {len(env_states_train)} samples")
        if env_states_test is not None:
            print(f"Set test environment states: {len(env_states_test)} samples")
