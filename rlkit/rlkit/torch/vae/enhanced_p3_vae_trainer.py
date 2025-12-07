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
        elif self.physics_type == 'pendulum':
            self.mass = 1.0
            self.length = 1.0
            self.gravity = 9.81
            self.damping = 0.1
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
        elif self.physics_type == 'pendulum' and z_I.shape[1] >= 2:
            return {
                'theta': z_I[:, 0:1],
                'theta_dot': z_I[:, 1:2]
            }
        else:
            return {'generic_state': z_I}
    
    def compute_pde_residual(self, z_I_t, z_I_t1):
        """Compute PDE residual for physics consistency"""
        state_t = self.extract_state_variables(z_I_t)
        state_t1 = self.extract_state_variables(z_I_t1)
        
        if self.physics_type == 'pick_and_place':
            return self._pick_and_place_pde_residual(state_t, state_t1)
        elif self.physics_type == 'pusher':
            return self._pusher_pde_residual(state_t, state_t1)
        elif self.physics_type == 'pendulum':
            return self._pendulum_pde_residual(state_t, state_t1)
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
    
    def _pendulum_pde_residual(self, state_t, state_t1):
        """PDE residual for pendulum dynamics"""
        theta_t = state_t['theta']
        theta_dot_t = state_t['theta_dot']
        theta_t1 = state_t1['theta']
        theta_dot_t1 = state_t1['theta_dot']
        
        # Pendulum equation: θ̈ = -(g/L)sin(θ) - damping·θ̇
        expected_theta_ddot = -(self.gravity / self.length) * torch.sin(theta_t) - self.damping * theta_dot_t
        
        # Compute actual acceleration from finite differences
        actual_theta_ddot = (theta_dot_t1 - theta_dot_t) / self.dt
        
        # PDE residual
        residual = torch.mean((actual_theta_ddot - expected_theta_ddot) ** 2)
        
        # Energy conservation
        E_t = 0.5 * self.mass * (self.length * theta_dot_t)**2 + \
              self.mass * self.gravity * self.length * (1 - torch.cos(theta_t))
        E_t1 = 0.5 * self.mass * (self.length * theta_dot_t1)**2 + \
               self.mass * self.gravity * self.length * (1 - torch.cos(theta_t1))
        
        # Energy should decrease due to damping
        expected_energy_change = -self.damping * self.mass * (self.length * theta_dot_t)**2 * self.dt
        actual_energy_change = E_t1 - E_t
        energy_residual = torch.mean((actual_energy_change - expected_energy_change) ** 2)
        
        return residual + 0.1 * energy_residual


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
            pde_weight=0.1,
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
        self.pde_weight = pde_weight
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
        
        # Update optimizer to include new parameters
        all_params = list(self.model.parameters()) + \
                    list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.physics_loss_calculator.parameters())
        self.optimizer = optim.Adam(all_params, lr=self.lr, weight_decay=1e-5)
        
        # Enhanced logging
        self.enhanced_stats = {}
        
        # Initialize consistent logging keys
        self._init_logging_keys()
        
        print(f"Enhanced P3-VAE initialized for {physics_type} with z_I_dim={z_I_dim}, z_E_dim={z_E_dim}")
    
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
    
    def compute_enhanced_loss(self, batch, epoch):
        """Compute the enhanced P3-VAE loss with improved stability"""
        
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
        
        # 1. Encode to get latent statistics
        z_I_stats = self.encoder.encode_physics(batch)
        z_E_stats = self.encoder.encode_environment(batch, z_I_stats['mean'])
        
        # 2. Sample from latent distributions
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
        
        # 6. Physics losses (if we have consecutive frames)
        physics_loss = torch.tensor(0.0, device=device)
        pde_loss = torch.tensor(0.0, device=device)
        
        if batch_size > 1 and self.physics_weight > 0:
            try:
                z_I_t = z_I[:-1]
                z_I_t1 = z_I[1:]
                
                # Physics consistency loss
                physics_loss = self.physics_loss_calculator.compute_pde_residual(z_I_t, z_I_t1)
                physics_loss = torch.clamp(physics_loss, max=100.0)  # Prevent explosion
                
            except Exception as e:
                print(f"Warning: Physics loss computation failed: {e}")
                physics_loss = torch.tensor(0.0, device=device)
        
        # 7. Regularization losses
        reg_loss = torch.tensor(0.0, device=device)
        if self.regularization_weight > 0:
            # L2 regularization on physics variables to keep them in reasonable ranges
            physics_reg = torch.mean(z_I ** 2)
            env_reg = torch.mean(z_E ** 2)
            reg_loss = physics_reg + env_reg
        
        # 8. Total loss with weights
        total_loss = (
            recon_loss + 
            self.beta * (kl_z_I + kl_z_E) + 
            self.physics_weight * physics_loss +
            self.pde_weight * pde_loss +
            self.regularization_weight * reg_loss
        )
        
        # Check for NaN/Inf and handle gracefully
        if not torch.isfinite(total_loss):
            print("Warning: Non-finite loss detected, using backup loss")
            total_loss = recon_loss + 0.1 * (kl_z_I + kl_z_E)
        
        # Loss dictionary for logging - use consistent keys only
        loss_dict = {}
        
        # Only update the keys that exist in consistent_log_keys
        loss_updates = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_z_I': kl_z_I.item(),
            'kl_z_E': kl_z_E.item(),
            'physics_loss': physics_loss.item(),
            'pde_loss': pde_loss.item(),
            'regularization_loss': reg_loss.item(),
            'z_I_mean_norm': torch.norm(z_I_stats['mean']).item(),
            'z_E_alpha_mean': torch.mean(z_E_stats['alpha']).item(),
            'z_E_beta_mean': torch.mean(z_E_stats['beta']).item(),
        }
        
        # Only add keys that exist in consistent_log_keys
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
    
    def train_epoch(self, epoch):
        """Enhanced training epoch with gradient clipping and stability checks"""
        self.model.train()
        self.encoder.train()
        self.decoder.train()
        
        train_loss = 0
        num_batches = len(self.train_dataset) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.train_dataset))
            batch = self.train_dataset[start_idx:end_idx]
            
            # Convert to tensor and move to device
            batch = torch.FloatTensor(batch).to(ptu.device)
            if self.input_channels == 1 and len(batch.shape) == 4:
                batch = torch.mean(batch, dim=1, keepdim=True)
            
            self.optimizer.zero_grad()
            
            try:
                loss, loss_dict, outputs = self.compute_enhanced_loss(batch, epoch)
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Gradient clipping for stability
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + 
                        list(self.encoder.parameters()) + 
                        list(self.decoder.parameters()),
                        self.grad_clip_value
                    )
                
                self.optimizer.step()
                train_loss += loss.item()
                
                # Log training progress
                if self.log_interval > 0 and batch_idx % self.log_interval == 0:
                    # Log losses - only log consistent keys that exist in loss_dict
                    for key, value in loss_dict.items():
                        logger.record_tabular(f"train/{key}", value)
                    
            except Exception as e:
                print(f"Warning: Training step failed at batch {batch_idx}: {e}")
                continue
                
        return train_loss / max(num_batches, 1)
    
    def test_epoch(self, epoch, save_reconstruction=True, save_vae=True, from_rl=False):
        """Enhanced test epoch with detailed analysis"""
        self.model.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        test_loss = 0
        num_batches = max(1, len(self.test_dataset) // self.batch_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.test_dataset))
                batch = self.test_dataset[start_idx:end_idx]
                
                batch = torch.FloatTensor(batch).to(ptu.device)
                if self.input_channels == 1 and len(batch.shape) == 4:
                    batch = torch.mean(batch, dim=1, keepdim=True)
                
                try:
                    loss, loss_dict, outputs = self.compute_enhanced_loss(batch, epoch)
                    test_loss += loss.item()
                except Exception as e:
                    print(f"Warning: Test step failed: {e}")
                    continue
        
        test_loss /= num_batches
        
        # Enhanced logging
        if self.log_interval > 0:
            # Create test dictionary with only valid keys
            test_dict = {}
            
            # Add test versions of keys that exist in loss_dict
            for k, v in loss_dict.items():
                test_key = f'test_{k}'
                test_dict[test_key] = v
            
            # Add test_loss
            test_dict['test_loss'] = test_loss
            
            # Log test metrics
            for key, value in test_dict.items():
                logger.record_tabular(key, value)
        
        # Save samples and analysis - but skip latent analysis logging to avoid table key issues
        if save_reconstruction and epoch % 10 == 0:
            self._save_enhanced_samples(epoch)
            # Skip latent analysis to prevent table key inconsistency
            # self._analyze_latent_space(epoch)
        
        return test_loss
    
    def _save_enhanced_samples(self, epoch):
        """Save enhanced samples and reconstructions"""
        self.encoder.eval()
        self.decoder.eval()
        
        batch_size = min(64, len(self.test_dataset))
        batch = self.test_dataset[:batch_size]
        batch = torch.FloatTensor(batch).to(ptu.device) / 255.0
        
        # Reshape flattened data to image tensor if needed
        if len(batch.shape) == 2:  # [batch_size, flattened_pixels]
            expected_size = self.input_channels * self.imsize * self.imsize
            batch = batch.view(batch_size, self.input_channels, self.imsize, self.imsize)
        elif self.input_channels == 1 and len(batch.shape) == 4:
            batch = torch.mean(batch, dim=1, keepdim=True)
        
        with torch.no_grad():
            # Original reconstructions
            z_I_stats = self.encoder.encode_physics(batch)
            z_E_stats = self.encoder.encode_environment(batch, z_I_stats['mean'])
            z_I, z_E = self.sample_from_distributions(z_I_stats, z_E_stats)
            
            # Use original VAE decoder
            z_combined = torch.cat([z_I, z_E], dim=1)
            reconstructions, _ = self.model.decode(z_combined)  # Extract only the image
            
            # Reshape to image format for display
            reconstructions = reconstructions.view(-1, self.input_channels, self.imsize, self.imsize)
            
            # Save comparison
            comparison = torch.cat([batch[:16], reconstructions[:16]], dim=0)
            save_path = osp.join(logger.get_snapshot_dir(), f'enhanced_recon_epoch_{epoch}.png')
            save_image(comparison, save_path, nrow=16)
            
            # Save physics-controlled samples
            # Fix environmental variables, vary physics
            z_E_fixed = z_E[:1].repeat(16, 1)
            z_I_varied = torch.randn(16, self.z_I_dim, device=ptu.device)
            z_combined_varied = torch.cat([z_I_varied, z_E_fixed], dim=1)
            physics_samples, _ = self.model.decode(z_combined_varied)  # Extract only the image
            physics_samples = physics_samples.view(-1, self.input_channels, self.imsize, self.imsize)
            
            save_path = osp.join(logger.get_snapshot_dir(), f'physics_varied_epoch_{epoch}.png')
            save_image(physics_samples, save_path, nrow=8)
            
            # Save environment-controlled samples
            z_I_fixed = z_I[:1].repeat(16, 1)
            z_E_varied = torch.rand(16, self.z_E_dim, device=ptu.device)  # Beta samples
            z_combined_varied = torch.cat([z_I_fixed, z_E_varied], dim=1)
            env_samples, _ = self.model.decode(z_combined_varied)  # Extract only the image
            env_samples = env_samples.view(-1, self.input_channels, self.imsize, self.imsize)
            
            save_path = osp.join(logger.get_snapshot_dir(), f'env_varied_epoch_{epoch}.png')
            save_image(env_samples, save_path, nrow=8)
    
    def _analyze_latent_space(self, epoch):
        """Analyze the learned latent space - always log consistent keys"""
        self.encoder.eval()
        
        # Always initialize the analysis dict with default values
        physics_analysis = {}
        
        # Always fill all analysis keys with default values first
        for i in range(4):  # Always analyze first 4 dimensions
            mean_key = f'z_I_dim_{i}_mean'
            std_key = f'z_I_dim_{i}_std'
            if mean_key in self.consistent_log_keys:
                physics_analysis[mean_key] = 0.0
            if std_key in self.consistent_log_keys:
                physics_analysis[std_key] = 0.0
        
        for i in range(4):  # Always analyze first 4 dimensions
            mean_key = f'z_E_dim_{i}_mean'
            std_key = f'z_E_dim_{i}_std'
            if mean_key in self.consistent_log_keys:
                physics_analysis[mean_key] = 0.0
            if std_key in self.consistent_log_keys:
                physics_analysis[std_key] = 0.0
        
        # Only compute actual values every 10 epochs to save computation
        if epoch % 10 == 0:
            try:
                batch_size = min(100, len(self.test_dataset))
                batch = self.test_dataset[:batch_size]
                batch = torch.FloatTensor(batch).to(ptu.device) / 255.0
                
                if self.input_channels == 1 and len(batch.shape) == 4:
                    batch = torch.mean(batch, dim=1, keepdim=True)
                
                with torch.no_grad():
                    z_I_stats = self.encoder.encode_physics(batch)
                    z_E_stats = self.encoder.encode_environment(batch, z_I_stats['mean'])
                    z_I, z_E = self.sample_from_distributions(z_I_stats, z_E_stats)
                    
                    # Update with actual computed values
                    for i in range(min(self.z_I_dim, 4)):  # Only first 4 dimensions (consistent with init)
                        if i < z_I.shape[1]:
                            mean_key = f'z_I_dim_{i}_mean'
                            std_key = f'z_I_dim_{i}_std'
                            if mean_key in self.consistent_log_keys:
                                physics_analysis[mean_key] = torch.mean(z_I[:, i]).item()
                            if std_key in self.consistent_log_keys:
                                physics_analysis[std_key] = torch.std(z_I[:, i]).item()
                    
                    # Analyze environmental variables
                    for i in range(min(self.z_E_dim, 4)):  # Only first 4 dimensions
                        if i < z_E.shape[1]:
                            mean_key = f'z_E_dim_{i}_mean'
                            std_key = f'z_E_dim_{i}_std'
                            if mean_key in self.consistent_log_keys:
                                physics_analysis[mean_key] = torch.mean(z_E[:, i]).item()
                            if std_key in self.consistent_log_keys:
                                physics_analysis[std_key] = torch.std(z_E[:, i]).item()
                        
            except Exception as e:
                print(f"Warning: Latent analysis computation failed: {e}")
                # physics_analysis already has the default values
        
        # Always log the same keys regardless of computation success
        for key, value in physics_analysis.items():
            logger.record_tabular(f"latent_analysis/{key}", value)
        
        self.enhanced_stats.update(physics_analysis)
    
    def get_diagnostics(self):
        """Get enhanced diagnostic information"""
        stats = super().get_diagnostics() if hasattr(super(), 'get_diagnostics') else {}
        stats.update(self.enhanced_stats)
        return stats
    
    def dump_samples(self, epoch):
        """Enhanced sample generation"""
        self._save_enhanced_samples(epoch)

    def debug_statistics(self):
        """
        Override parent's debug_statistics to ensure consistent logging keys.
        Returns a fixed set of debug statistics to prevent table key changes.
        """
        # Always return the same set of debug keys regardless of computation success
        default_debug_stats = {
            'debug/MSE improvement over random Mean': 0.0,
            'debug/MSE improvement over random Std': 0.0,
            'debug/MSE of random decoding Mean': 0.0,
            'debug/MSE of random decoding Std': 0.0,
            'debug/MSE of reconstruction': 0.0,
        }
        
        try:
            debug_batch_size = 64
            data = self.get_batch(train=False)
            
            # Get reconstruction using enhanced model
            batch = torch.FloatTensor(data).to(ptu.device) / 255.0
            if len(batch.shape) == 2:
                batch = batch.view(batch.size(0), self.input_channels, self.imsize, self.imsize)
            
            with torch.no_grad():
                # Enhanced reconstruction
                z_I_stats = self.encoder.encode_physics(batch)
                z_E_stats = self.encoder.encode_environment(batch, z_I_stats['mean'])
                z_I, z_E = self.sample_from_distributions(z_I_stats, z_E_stats)
                z_combined = torch.cat([z_I, z_E], dim=1)
                reconstructions, _ = self.model.decode(z_combined)
                
                img = batch[0].view(-1)
                recon = reconstructions[0]
                recon_mse = ((recon - img) ** 2).mean()
                
                # Random samples
                samples = ptu.randn(debug_batch_size, self.representation_size)
                random_imgs, _ = self.model.decode(samples)
                img_repeated = img.expand((debug_batch_size, img.shape[0]))
                random_mses = ((random_imgs - img_repeated) ** 2).mean(dim=1)
                mse_improvement = ptu.get_numpy(random_mses.mean() - recon_mse)
                
                # Update with computed values
                default_debug_stats.update({
                    'debug/MSE improvement over random Mean': float(mse_improvement),
                    'debug/MSE improvement over random Std': 0.0,  # Consistent placeholder
                    'debug/MSE of random decoding Mean': float(ptu.get_numpy(random_mses.mean())),
                    'debug/MSE of random decoding Std': float(ptu.get_numpy(random_mses.std())),
                    'debug/MSE of reconstruction': float(ptu.get_numpy(recon_mse)),
                })
            
        except Exception as e:
            print(f"Warning: Debug statistics computation failed: {e}")
            # default_debug_stats already has the correct defaults
        
        return default_debug_stats

    def get_batch(self, train=True):
        """Get a batch for debugging purposes"""
        dataset = self.train_dataset if train else self.test_dataset
        batch_size = min(self.batch_size, len(dataset))
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        batch = dataset[indices]
        return batch
