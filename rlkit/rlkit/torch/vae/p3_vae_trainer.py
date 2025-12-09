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

# Try to import torchdiffeq for proper ODE integration, fallback to simple integration
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not available, using simple Euler integration")


class P3VAETrainer(ConvVAETrainer):
    """
    P3-VAE (Physics-informed VAE) trainer based on the Romain3Ch216/p3VAE methodology.
    Separates latent space into physics variables (z_I) and environmental variables (z_E)
    with explicit physics-informed encoding and decoding.
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
            # P3-VAE specific parameters
            physics_weight=0.1  # Weight for physics loss
            , physics_type='pusher'  # 'pusher', 'pendulum', 'manipulation'
            , z_I_dim=4  # Dimension of intrinsic (physics) latent variables
            , z_E_dim=4  # Dimension of environmental latent variables  
            , dt=0.1  # Time step for physics simulation
            , use_beta_distribution=True  # Use Beta distribution for z_E
            , enable_contact_dynamics=True
            , enable_friction=True
            , enable_conservation_laws=True
            , physics_regularization_weight=0.01  # Weight for physics regularization
    ):
        # Store P3-VAE specific parameters first
        self.physics_weight = physics_weight
        self.physics_type = physics_type
        self.z_I_dim = z_I_dim
        self.z_E_dim = z_E_dim
        self.dt = dt
        self.use_beta_distribution = use_beta_distribution
        self.enable_contact_dynamics = enable_contact_dynamics
        self.enable_friction = enable_friction
        self.enable_conservation_laws = enable_conservation_laws
        self.physics_regularization_weight = physics_regularization_weight
        
        # Initialize parent class manually to avoid quick_init conflicts
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
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        self.eval_statistics = None
        self.vae_logger_stats_for_rl = {}

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self.is_auto_encoder = is_auto_encoder
        
        # Initialize physics parameters based on task
        self._initialize_physics_parameters()
        
        # P3-VAE logging
        self.p3_vae_stats = {}
        
    def _initialize_physics_parameters(self):
        """Initialize physics parameters based on the task type."""
        
        if self.physics_type == 'pusher':
            # Pusher task: hand pushing a puck on a table
            self.mass_hand = 1.0
            self.mass_puck = 0.5
            self.friction_coeff = 0.3
            self.restitution = 0.8
            self.contact_stiffness = 1000.0
            # Physics variables: [hand_x, hand_y, puck_x, puck_y] or [hand_vx, hand_vy, puck_vx, puck_vy]
            self.physics_variables = ['hand_x', 'hand_y', 'puck_x', 'puck_y']
            
        elif self.physics_type == 'reacher':
            # Reacher task: robotic arm reaching to target positions
            self.arm_mass = 5.0  # Total arm mass
            self.joint_damping = 0.1  # Joint damping coefficient
            self.joint_friction = 0.05  # Joint friction
            self.gravity = 9.81
            self.arm_length_1 = 0.4  # First link length
            self.arm_length_2 = 0.3  # Second link length
            # Joint limits (radians)
            self.joint_limit_low = [-3.14, -1.57, -3.14, -1.57]  # Joint angle limits
            self.joint_limit_high = [3.14, 1.57, 3.14, 1.57]
            # Physics variables: [joint1_angle, joint2_angle, joint1_vel, joint2_vel]
            self.physics_variables = ['joint1_angle', 'joint2_angle', 'joint1_vel', 'joint2_vel']
            
        elif self.physics_type == 'pendulum':
            # Pendulum: simple harmonic motion
            self.mass = 1.0
            self.length = 1.0
            self.gravity = 9.81
            self.damping = 0.1
            # Physics variables: [theta, theta_dot]
            self.physics_variables = ['theta', 'theta_dot']
            
        elif self.physics_type == 'manipulation':
            # General manipulation: gripper + object
            self.mass_gripper = 2.0
            self.mass_object = 1.0
            self.friction_coeff = 0.4
            self.contact_stiffness = 500.0
            # Physics variables: [gripper_x, gripper_y, gripper_z, object_x, object_y, object_z]
            self.physics_variables = ['gripper_x', 'gripper_y', 'gripper_z', 'object_x', 'object_y', 'object_z']
            
        elif self.physics_type == 'pick_and_place':
            # Pick and place task: hand grasping and moving objects
            self.mass_hand = 5.0  # Hand + robot arm mass
            self.mass_object = 0.1  # Small object mass (100g)
            self.friction_coeff = 0.3
            self.contact_stiffness = 1000.0
            self.contact_threshold = 0.05  # 5cm contact detection
            self.gripper_closed_threshold = 0.02  # Gripper closure threshold
            self.table_height = 0.05
            # Physics variables: [gripper_opening, hand_x, hand_y, hand_z, obj_x, obj_y, obj_z]
            self.physics_variables = ['gripper_opening', 'hand_x', 'hand_y', 'hand_z', 'obj_x', 'obj_y', 'obj_z']
            
        # Common parameters
        self.gravity = getattr(self, 'gravity', 9.81)
        
    def encode_z_I(self, x):
        """
        Encode physics (intrinsic) variables z_I from input.
        These variables have direct physical meaning.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            dict: z_I statistics (mean, lnvar for normal distribution)
        """
        # Use a physics-specific encoder network
        # This should be implemented in the model architecture
        if hasattr(self.model, 'encode_physics'):
            z_I_stats = self.model.encode_physics(x)
        else:
            # Fallback: use part of standard encoder
            mu, logvar = self.model.encode(x)
            z_I_mean = mu[:, :self.z_I_dim]
            z_I_lnvar = logvar[:, :self.z_I_dim]
            z_I_stats = {'mean': z_I_mean, 'lnvar': z_I_lnvar}
            
        return z_I_stats
    
    def encode_z_E(self, x, z_I_stats):
        """
        Encode environmental variables z_E from input and z_I.
        These variables capture environmental factors (lighting, texture, etc.).
        
        Args:
            x: Input images [batch_size, channels, height, width]
            z_I_stats: Physics variables statistics
            
        Returns:
            dict: z_E statistics (alpha, beta for Beta distribution or mean, lnvar for normal)
        """
        # Combine input with z_I mean for environmental encoding
        if hasattr(self.model, 'encode_environment'):
            z_E_stats = self.model.encode_environment(x, z_I_stats['mean'])
        else:
            # Fallback: use part of standard encoder
            mu, logvar = self.model.encode(x)
            # Use the remaining dimensions for environmental variables
            z_E_mean = mu[:, self.z_I_dim:]
            z_E_lnvar = logvar[:, self.z_I_dim:]
            
            if self.use_beta_distribution:
                # Convert to Beta distribution parameters with numerical stability
                # Use softplus for positive values with clipping for stability
                alpha = 1.0 + F.softplus(z_E_mean).clamp(max=10.0)
                beta = 1.0 + F.softplus(z_E_lnvar).clamp(max=10.0)
                z_E_stats = {'alpha': alpha, 'beta': beta}
            else:
                # Normal distribution
                z_E_stats = {'mean': z_E_mean, 'lnvar': z_E_lnvar}
                
        return z_E_stats
    
    def draw_z_I(self, z_I_stats):
        """Sample from z_I distribution (normal)."""
        mean = z_I_stats['mean']
        lnvar = z_I_stats['lnvar']
        std = torch.exp(0.5 * lnvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def draw_z_E(self, z_E_stats):
        """Sample from z_E distribution (Beta or normal)."""
        if self.use_beta_distribution:
            alpha = z_E_stats['alpha']
            beta = z_E_stats['beta']
            dist = torch.distributions.beta.Beta(alpha, beta)
            return dist.rsample()
        else:
            mean = z_E_stats['mean']
            lnvar = z_E_stats['lnvar']
            std = torch.exp(0.5 * lnvar)
            eps = torch.randn_like(std)
            return mean + eps * std
    
    def get_priors(self, batch_size, device):
        """Get prior distributions for z_I and z_E."""
        # z_I prior: standard normal
        prior_z_I_stats = {
            'mean': torch.zeros(batch_size, self.z_I_dim, device=device),
            'lnvar': torch.zeros(batch_size, self.z_I_dim, device=device)
        }
        
        # z_E prior: uniform Beta(1,1) or standard normal
        if self.use_beta_distribution:
            prior_z_E_stats = {
                'alpha': torch.ones(batch_size, self.z_E_dim, device=device),
                'beta': torch.ones(batch_size, self.z_E_dim, device=device)
            }
        else:
            prior_z_E_stats = {
                'mean': torch.zeros(batch_size, self.z_E_dim, device=device),
                'lnvar': torch.zeros(batch_size, self.z_E_dim, device=device)
            }
            
        return prior_z_I_stats, prior_z_E_stats
    
    def kl_divergence_z_I(self, z_I_stats, prior_z_I_stats):
        """Compute KL divergence for z_I (normal distributions)."""
        mean = z_I_stats['mean']
        lnvar = z_I_stats['lnvar']
        prior_mean = prior_z_I_stats['mean']
        prior_lnvar = prior_z_I_stats['lnvar']
        
        # KL(q(z_I) || p(z_I)) for normal distributions
        kl = 0.5 * torch.sum(
            prior_lnvar - lnvar + 
            (torch.exp(lnvar) + (mean - prior_mean)**2) / torch.exp(prior_lnvar) - 1,
            dim=1
        )
        return kl
    
    def kl_divergence_z_E(self, z_E_stats, prior_z_E_stats):
        """Compute KL divergence for z_E (Beta or normal distributions)."""
        if self.use_beta_distribution:
            alpha = z_E_stats['alpha']
            beta = z_E_stats['beta']
            prior_alpha = prior_z_E_stats['alpha']
            prior_beta = prior_z_E_stats['beta']
            
            # KL(Beta(alpha, beta) || Beta(1, 1)) = KL(Beta(alpha, beta) || Uniform)
            # For Beta(a,b) || Beta(a0,b0): KL = ln(B(a0,b0)/B(a,b)) + (a-a0)*digamma(a) + (b-b0)*digamma(b) + (a0-a+b0-b)*digamma(a+b)
            # For uniform prior Beta(1,1): KL = ln(1/B(a,b)) + (a-1)*digamma(a) + (b-1)*digamma(b) + (2-a-b)*digamma(a+b)
            kl = -torch.lgamma(alpha) - torch.lgamma(beta) + torch.lgamma(alpha + beta) + \
                 (alpha - 1) * torch.digamma(alpha) + (beta - 1) * torch.digamma(beta) + \
                 (2 - alpha - beta) * torch.digamma(alpha + beta)
            return torch.sum(kl, dim=1)
        else:
            # Standard normal KL divergence
            mean = z_E_stats['mean']
            lnvar = z_E_stats['lnvar']
            prior_mean = prior_z_E_stats['mean']
            prior_lnvar = prior_z_E_stats['lnvar']
            
            kl = 0.5 * torch.sum(
                prior_lnvar - lnvar + 
                (torch.exp(lnvar) + (mean - prior_mean)**2) / torch.exp(prior_lnvar) - 1,
                dim=1
            )
            return kl
    
    def extract_physics_state(self, z_I):
        """
        Extract physics state variables from z_I.
        This is the key P3-VAE innovation: z_I directly represents physics variables.
        
        Args:
            z_I: Physics latent variables [batch_size, z_I_dim]
            
        Returns:
            dict: Physics state variables with clear physical interpretation
        """
        batch_size = z_I.shape[0]
        
        if self.physics_type == 'pusher':
            if self.z_I_dim >= 4:
                # Direct mapping: [hand_x, hand_y, puck_x, puck_y]
                return {
                    'hand_pos': z_I[:, :2],  # hand position
                    'puck_pos': z_I[:, 2:4],  # puck position
                    'hand_x': z_I[:, 0:1],
                    'hand_y': z_I[:, 1:2], 
                    'puck_x': z_I[:, 2:3],
                    'puck_y': z_I[:, 3:4],
                }
            else:
                # Reduced case
                return {
                    'hand_pos': z_I[:, :2],
                    'puck_pos': z_I[:, :2],  # Assume same as hand for simplicity
                }
                
        elif self.physics_type == 'pendulum':
            if self.z_I_dim >= 2:
                # Direct mapping: [theta, theta_dot]
                theta = z_I[:, 0:1]
                theta_dot = z_I[:, 1:2]
                return {
                    'angle': theta,
                    'angular_velocity': theta_dot,
                    'x_pos': self.length * torch.sin(theta),
                    'y_pos': -self.length * torch.cos(theta),
                    'theta': theta,
                    'theta_dot': theta_dot,
                }
            else:
                # Only angle
                theta = z_I[:, 0:1]
                return {
                    'angle': theta,
                    'angular_velocity': torch.zeros_like(theta),
                    'theta': theta,
                }
                
        elif self.physics_type == 'manipulation':
            if self.z_I_dim >= 6:
                # [gripper_x, gripper_y, gripper_z, object_x, object_y, object_z]
                return {
                    'gripper_pos': z_I[:, :3],
                    'object_pos': z_I[:, 3:6],
                    'gripper_x': z_I[:, 0:1],
                    'gripper_y': z_I[:, 1:2],
                    'gripper_z': z_I[:, 2:3],
                    'object_x': z_I[:, 3:4],
                    'object_y': z_I[:, 4:5],
                    'object_z': z_I[:, 5:6],
                }
            else:
                # Reduced case  
                return {
                    'gripper_pos': z_I[:, :3] if self.z_I_dim >= 3 else z_I,
                    'object_pos': z_I[:, :3] if self.z_I_dim >= 3 else z_I,
                }
                
        elif self.physics_type == 'reacher':
            if self.z_I_dim >= 4:
                # Direct mapping: [joint1_angle, joint2_angle, joint1_vel, joint2_vel]
                return {
                    'joint1_angle': z_I[:, 0:1],
                    'joint2_angle': z_I[:, 1:2],
                    'joint1_vel': z_I[:, 2:3],
                    'joint2_vel': z_I[:, 3:4],
                    'q1': z_I[:, 0:1],
                    'q2': z_I[:, 1:2],
                    'q1_dot': z_I[:, 2:3],
                    'q2_dot': z_I[:, 3:4],
                    'theta1': z_I[:, 0:1],
                    'theta2': z_I[:, 1:2],
                    'theta1_dot': z_I[:, 2:3],
                    'theta2_dot': z_I[:, 3:4],
                }
            else:
                # Reduced case - only joint angles
                return {
                    'joint1_angle': z_I[:, 0:1],
                    'joint2_angle': z_I[:, 1:2] if self.z_I_dim >= 2 else torch.zeros_like(z_I[:, 0:1]),
                    'joint1_vel': torch.zeros_like(z_I[:, 0:1]),
                    'joint2_vel': torch.zeros_like(z_I[:, 0:1]),
                }
                
        elif self.physics_type == 'pick_and_place':
            if self.z_I_dim >= 7:
                # Direct mapping: [gripper_opening, hand_x, hand_y, hand_z, obj_x, obj_y, obj_z]
                return {
                    'gripper_opening': z_I[:, 0:1],
                    'hand_pos': z_I[:, 1:4],
                    'object_pos': z_I[:, 4:7],
                    'gripper_pos': z_I[:, 0:1],
                    'hand_x': z_I[:, 1:2],
                    'hand_y': z_I[:, 2:3],
                    'hand_z': z_I[:, 3:4],
                    'obj_x': z_I[:, 4:5],
                    'obj_y': z_I[:, 5:6],
                    'obj_z': z_I[:, 6:7],
                }
            else:
                # Reduced case
                return {
                    'gripper_opening': z_I[:, 0:1] if self.z_I_dim >= 1 else torch.zeros((z_I.shape[0], 1), device=z_I.device),
                    'hand_pos': z_I[:, 1:4] if self.z_I_dim >= 4 else z_I[:, :3] if self.z_I_dim >= 3 else z_I,
                    'object_pos': z_I[:, 4:7] if self.z_I_dim >= 7 else z_I[:, 1:4] if self.z_I_dim >= 4 else z_I,
                }
        
        else:
            # Default: treat z_I as generic physics variables
            return {
                'physics_vars': z_I,
                'positions': z_I,
            }
    
    def physics_forward_model(self, state_t, dt=None):
        """
        Forward physics model to predict next state.
        This enforces physics consistency in z_I space.
        
        Args:
            state_t: Current physics state
            dt: Time step (default: self.dt)
            
        Returns:
            dict: Predicted next state
        """
        if dt is None:
            dt = self.dt
            
        if self.physics_type == 'pusher':
            return self._pusher_forward_model(state_t, dt)
        elif self.physics_type == 'reacher':
            return self._reacher_forward_model(state_t, dt)
        elif self.physics_type == 'pendulum':
            return self._pendulum_forward_model(state_t, dt)
        elif self.physics_type == 'manipulation':
            return self._manipulation_forward_model(state_t, dt)
        elif self.physics_type == 'pick_and_place':
            return self._pick_and_place_forward_model(state_t, dt)
        else:
            # Default: assume constant velocity
            return state_t
    
    def _pusher_forward_model(self, state_t, dt):
        """Forward model for pusher dynamics."""
        hand_pos = state_t['hand_pos']  # [batch, 2]
        puck_pos = state_t['puck_pos']  # [batch, 2]
        
        # Simple dynamics: assume hand moves puck on contact
        hand_puck_dist = torch.norm(hand_pos - puck_pos, dim=1, keepdim=True)
        contact_threshold = 0.1
        
        # If in contact, puck follows hand
        in_contact = (hand_puck_dist < contact_threshold).float()
        
        # For simplicity, assume hand velocity is encoded implicitly
        # and puck moves towards hand when in contact
        next_puck_pos = puck_pos + in_contact * (hand_pos - puck_pos) * 0.1
        
        return {
            'hand_pos': hand_pos,  # Hand position controlled externally
            'puck_pos': next_puck_pos,
            'hand_x': hand_pos[:, 0:1],
            'hand_y': hand_pos[:, 1:2],
            'puck_x': next_puck_pos[:, 0:1], 
            'puck_y': next_puck_pos[:, 1:2],
        }
    
    def _reacher_forward_model(self, state_t, dt):
        """Forward model for reacher arm dynamics."""
        # State variables: [joint1_angle, joint2_angle, joint1_vel, joint2_vel]
        joint1_angle = state_t.get('joint1_angle', state_t.get('q1', state_t.get('theta1')))
        joint2_angle = state_t.get('joint2_angle', state_t.get('q2', state_t.get('theta2')))
        joint1_vel = state_t.get('joint1_vel', state_t.get('q1_dot', state_t.get('theta1_dot')))
        joint2_vel = state_t.get('joint2_vel', state_t.get('q2_dot', state_t.get('theta2_dot')))
        
        # Simplified arm dynamics with gravity and damping
        # For a 2-link arm, the dynamics are complex, so we use a simplified model
        
        # Gravity effects (simplified)
        gravity_torque1 = -self.gravity * (self.arm_mass * self.arm_length_1 * 0.5) * torch.cos(joint1_angle)
        gravity_torque2 = -self.gravity * (self.arm_mass * self.arm_length_2 * 0.5) * torch.cos(joint1_angle + joint2_angle)
        
        # Damping torques
        damping_torque1 = -self.joint_damping * joint1_vel
        damping_torque2 = -self.joint_damping * joint2_vel
        
        # Simplified angular accelerations (ignoring coupling terms for simplicity)
        # In reality, these would involve the full inertia matrix
        inertia1 = self.arm_mass * (self.arm_length_1 ** 2) / 3.0  # Simplified inertia
        inertia2 = self.arm_mass * (self.arm_length_2 ** 2) / 3.0
        
        joint1_acc = (gravity_torque1 + damping_torque1) / inertia1
        joint2_acc = (gravity_torque2 + damping_torque2) / inertia2
        
        # Euler integration
        next_joint1_vel = joint1_vel + joint1_acc * dt
        next_joint2_vel = joint2_vel + joint2_acc * dt
        next_joint1_angle = joint1_angle + joint1_vel * dt
        next_joint2_angle = joint2_angle + joint2_vel * dt
        
        # Apply joint limits if enabled
        if hasattr(self, 'enable_joint_limits') and self.enable_joint_limits:
            next_joint1_angle = torch.clamp(next_joint1_angle, 
                                          self.joint_limit_low[0], self.joint_limit_high[0])
            next_joint2_angle = torch.clamp(next_joint2_angle,
                                          self.joint_limit_low[1], self.joint_limit_high[1])
        
        # Compute end-effector position for auxiliary outputs
        end_x = (self.arm_length_1 * torch.cos(next_joint1_angle) + 
                self.arm_length_2 * torch.cos(next_joint1_angle + next_joint2_angle))
        end_y = (self.arm_length_1 * torch.sin(next_joint1_angle) + 
                self.arm_length_2 * torch.sin(next_joint1_angle + next_joint2_angle))
        
        return {
            'joint1_angle': next_joint1_angle,
            'joint2_angle': next_joint2_angle,
            'joint1_vel': next_joint1_vel,
            'joint2_vel': next_joint2_vel,
            'q1': next_joint1_angle,
            'q2': next_joint2_angle,
            'q1_dot': next_joint1_vel,
            'q2_dot': next_joint2_vel,
            'end_effector_x': end_x,
            'end_effector_y': end_y,
            'joint1_acc': joint1_acc,
            'joint2_acc': joint2_acc,
        }
    
    def _pendulum_forward_model(self, state_t, dt):
        """Forward model for pendulum dynamics."""
        theta = state_t['theta']  # [batch, 1]
        theta_dot = state_t['theta_dot']  # [batch, 1]
        
        # Pendulum equation: theta_ddot = -(g/L) * sin(theta) - damping * theta_dot
        theta_ddot = -(self.gravity / self.length) * torch.sin(theta) - self.damping * theta_dot
        
        # Euler integration
        next_theta_dot = theta_dot + theta_ddot * dt
        next_theta = theta + theta_dot * dt
        
        return {
            'theta': next_theta,
            'theta_dot': next_theta_dot,
            'angle': next_theta,
            'angular_velocity': next_theta_dot,
            'x_pos': self.length * torch.sin(next_theta),
            'y_pos': -self.length * torch.cos(next_theta),
        }
    
    def _manipulation_forward_model(self, state_t, dt):
        """Forward model for manipulation dynamics."""
        gripper_pos = state_t['gripper_pos']  # [batch, 3]
        object_pos = state_t['object_pos']  # [batch, 3]
        
        # Simple grasping dynamics
        gripper_object_dist = torch.norm(gripper_pos - object_pos, dim=1, keepdim=True)
        grasp_threshold = 0.05
        
        # If close enough, object follows gripper
        in_grasp = (gripper_object_dist < grasp_threshold).float()
        next_object_pos = object_pos + in_grasp * (gripper_pos - object_pos) * 0.5
        
        return {
            'gripper_pos': gripper_pos,  # Controlled externally
            'object_pos': next_object_pos,
            'gripper_x': gripper_pos[:, 0:1],
            'gripper_y': gripper_pos[:, 1:2],
            'gripper_z': gripper_pos[:, 2:3],
            'object_x': next_object_pos[:, 0:1],
            'object_y': next_object_pos[:, 1:2],
            'object_z': next_object_pos[:, 2:3],
        }
    
    def _pick_and_place_forward_model(self, state_t, dt):
        """Forward model for pick and place dynamics."""
        gripper_opening = state_t['gripper_opening']  # [batch, 1]
        hand_pos = state_t['hand_pos']  # [batch, 3]
        object_pos = state_t['object_pos']  # [batch, 3]
        
        # Compute hand-object distance
        hand_object_dist = torch.norm(hand_pos - object_pos, dim=1, keepdim=True)
        
        # Contact detection
        in_contact = (hand_object_dist < self.contact_threshold).float()
        
        # Grasping detection (gripper closed and in contact)
        gripper_closed = (gripper_opening < self.gripper_closed_threshold).float()
        grasping = in_contact * gripper_closed
        
        # Object dynamics
        next_object_pos = object_pos.clone()
        
        # If grasping, object follows hand motion
        if grasping.sum() > 0:
            # Object is carried by hand
            object_hand_offset = object_pos - hand_pos
            # Maintain relative position when grasping
            next_object_pos = next_object_pos + grasping * (hand_pos - object_pos + object_hand_offset) * 0.3
        else:
            # Apply gravity to unsupported objects
            # Check if object is on table (z > table_height)
            on_table = (object_pos[:, 2:3] <= self.table_height + 0.02).float()
            
            # Apply gravity to objects not on table and not grasped
            gravity_effect = (1.0 - on_table) * (1.0 - grasping)
            gravity_acc = -self.gravity * dt * dt * 0.5  # Simple gravity integration
            gravity_delta = gravity_effect * gravity_acc
            
            # Update z position with gravity (avoiding in-place operation)
            next_object_pos_z = next_object_pos[:, 2:3] + gravity_delta
            
            # Prevent object from falling below table (avoiding in-place operation)
            next_object_pos_z_clamped = torch.clamp(next_object_pos_z, 
                                                   min=self.table_height, max=None)
            
            # Create new tensor with updated z coordinate
            next_object_pos = torch.cat([
                next_object_pos[:, :2],  # x, y unchanged
                next_object_pos_z_clamped  # z with gravity and clamping
            ], dim=1)
        
        # Contact forces: if in contact but not grasping, apply repulsion
        contact_force = in_contact * (1.0 - grasping)
        if contact_force.sum() > 0:
            # Push object away from hand slightly
            hand_to_object = object_pos - hand_pos
            hand_to_object_norm = torch.norm(hand_to_object, dim=1, keepdim=True)
            hand_to_object_unit = hand_to_object / (hand_to_object_norm + 1e-6)
            repulsion = contact_force * hand_to_object_unit * 0.01  # Small repulsion
            next_object_pos = next_object_pos + repulsion
        
        return {
            'gripper_opening': gripper_opening,  # Controlled externally
            'hand_pos': hand_pos,  # Controlled externally
            'object_pos': next_object_pos,
            'gripper_pos': gripper_opening,
            'hand_x': hand_pos[:, 0:1],
            'hand_y': hand_pos[:, 1:2],
            'hand_z': hand_pos[:, 2:3],
            'obj_x': next_object_pos[:, 0:1],
            'obj_y': next_object_pos[:, 1:2],
            'obj_z': next_object_pos[:, 2:3],
        }
    
    def compute_physics_loss(self, z_I_t, z_I_t1):
        """
        Compute physics consistency loss between consecutive time steps.
        This is the core P3-VAE physics enforcement.
        
        Args:
            z_I_t: Physics variables at time t [batch_size, z_I_dim]
            z_I_t1: Physics variables at time t+1 [batch_size, z_I_dim]
            
        Returns:
            torch.Tensor: Physics loss
        """
        # Extract physics states
        state_t = self.extract_physics_state(z_I_t)
        state_t1 = self.extract_physics_state(z_I_t1)
        
        # Predict next state using physics model
        predicted_state_t1 = self.physics_forward_model(state_t)
        
        # Compute physics loss as MSE between predicted and actual next state
        physics_loss = 0.0
        
        for key in predicted_state_t1:
            if key in state_t1:
                pred_var = predicted_state_t1[key]
                actual_var = state_t1[key]
                if pred_var.shape == actual_var.shape:
                    physics_loss += F.mse_loss(pred_var, actual_var)
        
        # Add conservation laws if enabled
        if self.enable_conservation_laws:
            physics_loss += self._compute_conservation_losses(state_t, state_t1)
            
        return physics_loss
    
    def _compute_conservation_losses(self, state_t, state_t1):
        """Compute conservation law violations (energy, momentum)."""
        conservation_loss = 0.0
        
        if self.physics_type == 'pendulum':
            # Energy conservation for pendulum
            theta_t = state_t['theta']
            theta_dot_t = state_t['theta_dot']
            theta_t1 = state_t1['theta']
            theta_dot_t1 = state_t1['theta_dot']
            
            # Total energy = kinetic + potential
            E_t = 0.5 * self.mass * (self.length * theta_dot_t)**2 + \
                  self.mass * self.gravity * self.length * (1 - torch.cos(theta_t))
            E_t1 = 0.5 * self.mass * (self.length * theta_dot_t1)**2 + \
                   self.mass * self.gravity * self.length * (1 - torch.cos(theta_t1))
                   
            # Energy should be approximately conserved (allowing for damping)
            energy_loss = F.mse_loss(E_t1, E_t * (1 - self.damping * self.dt))
            conservation_loss += energy_loss
            
        elif self.physics_type == 'pusher':
            # Momentum conservation during contact
            hand_pos_t = state_t['hand_pos']
            puck_pos_t = state_t['puck_pos']
            hand_pos_t1 = state_t1['hand_pos']
            puck_pos_t1 = state_t1['puck_pos']
            
            # Check if contact occurred
            dist_t = torch.norm(hand_pos_t - puck_pos_t, dim=1, keepdim=True)
            dist_t1 = torch.norm(hand_pos_t1 - puck_pos_t1, dim=1, keepdim=True)
            contact_mask = ((dist_t < 0.1) | (dist_t1 < 0.1)).float()
            
            # During contact, momentum should be conserved
            if contact_mask.sum() > 0:
                # Simplified momentum conservation check
                hand_vel_t = hand_pos_t1 - hand_pos_t
                puck_vel_t = puck_pos_t1 - puck_pos_t
                
                momentum_t = self.mass_hand * hand_vel_t + self.mass_puck * puck_vel_t
                # In next step, total momentum should be conserved
                momentum_loss = torch.mean(contact_mask * torch.norm(momentum_t, dim=1, keepdim=True))
                conservation_loss += momentum_loss
                
        elif self.physics_type == 'pick_and_place':
            # Momentum conservation during contact/grasping
            hand_pos_t = state_t['hand_pos']
            object_pos_t = state_t['object_pos']
            hand_pos_t1 = state_t1['hand_pos']
            object_pos_t1 = state_t1['object_pos']
            gripper_t = state_t['gripper_opening']
            gripper_t1 = state_t1['gripper_opening']
            
            # Check if grasping occurred
            dist_t = torch.norm(hand_pos_t - object_pos_t, dim=1, keepdim=True)
            dist_t1 = torch.norm(hand_pos_t1 - object_pos_t1, dim=1, keepdim=True)
            gripper_closed_t = (gripper_t < self.gripper_closed_threshold).float()
            gripper_closed_t1 = (gripper_t1 < self.gripper_closed_threshold).float()
            
            contact_t = (dist_t < self.contact_threshold).float()
            contact_t1 = (dist_t1 < self.contact_threshold).float()
            grasping_t = contact_t * gripper_closed_t
            grasping_t1 = contact_t1 * gripper_closed_t1
            
            # When grasping, object should follow hand motion
            if grasping_t1.sum() > 0:
                object_vel = object_pos_t1 - object_pos_t
                hand_vel = hand_pos_t1 - hand_pos_t
                grasp_consistency = torch.norm(object_vel - hand_vel, dim=1, keepdim=True)
                grasp_loss = torch.mean(grasping_t1 * grasp_consistency)
                conservation_loss += grasp_loss
            
            # Gravity effects: unsupported objects should fall
            on_table_t = (object_pos_t[:, 2:3] <= self.table_height + 0.02).float()
            on_table_t1 = (object_pos_t1[:, 2:3] <= self.table_height + 0.02).float()
            unsupported = (1.0 - on_table_t) * (1.0 - grasping_t)
            
            if unsupported.sum() > 0:
                # Objects should fall when unsupported
                expected_fall = -0.5 * self.gravity * (self.dt ** 2)
                actual_z_change = object_pos_t1[:, 2:3] - object_pos_t[:, 2:3]
                gravity_violation = torch.abs(actual_z_change - expected_fall)
                gravity_loss = torch.mean(unsupported * gravity_violation)
                conservation_loss += gravity_loss
                
        return conservation_loss
    
    def compute_loss(self, batch, epoch, **kwargs):
        """
        Compute the total P3-VAE loss.
        
        Args:
            batch: Batch of data
            epoch: Current epoch
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        x = batch
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Encode z_I (physics variables)
        z_I_stats = self.encode_z_I(x)
        z_I = self.draw_z_I(z_I_stats)
        
        # 2. Encode z_E (environmental variables)
        z_E_stats = self.encode_z_E(x, z_I_stats)
        z_E = self.draw_z_E(z_E_stats)
        
        # 3. Decode to reconstruct image
        if hasattr(self.model, 'decode_from_physics_env'):
            x_recon = self.model.decode_from_physics_env(z_I, z_E)
        else:
            # Fallback: concatenate z_I and z_E
            z_combined = torch.cat([z_I, z_E], dim=1)
            x_recon, _ = self.model.decode(z_combined)  # decode returns (recon, params)
        
        # 4. Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
        
        # 5. KL divergence losses
        prior_z_I_stats, prior_z_E_stats = self.get_priors(batch_size, device)
        kl_z_I = torch.mean(self.kl_divergence_z_I(z_I_stats, prior_z_I_stats))
        kl_z_E = torch.mean(self.kl_divergence_z_E(z_E_stats, prior_z_E_stats))
        
        # 6. Physics loss (if we have consecutive frames)
        physics_loss = 0.0
        if batch_size > 1:
            # Assume consecutive frames in batch for physics loss
            z_I_t = z_I[:-1]  # All but last
            z_I_t1 = z_I[1:]  # All but first
            physics_loss = self.compute_physics_loss(z_I_t, z_I_t1)
        
        # 7. Physics regularization (encourage meaningful physics variables)
        physics_reg_loss = 0.0
        if self.physics_regularization_weight > 0:
            # Encourage z_I to be in reasonable ranges for physics variables
            if self.physics_type == 'pusher':
                # Hand and puck positions should be in reasonable range, e.g., [-1, 1]
                pos_reg = torch.mean(torch.relu(torch.abs(z_I) - 2.0))
                physics_reg_loss += pos_reg
            elif self.physics_type == 'pendulum':
                # Angle should be in [-pi, pi], angular velocity reasonable
                angle_reg = torch.mean(torch.relu(torch.abs(z_I[:, 0:1]) - np.pi))
                vel_reg = torch.mean(torch.relu(torch.abs(z_I[:, 1:2]) - 10.0))
                physics_reg_loss += angle_reg + vel_reg
            elif self.physics_type == 'pick_and_place':
                # Gripper opening should be in [0, 0.04], positions reasonable
                if self.z_I_dim >= 1:
                    gripper_reg = torch.mean(torch.relu(z_I[:, 0:1] - 0.05) + torch.relu(-z_I[:, 0:1]))
                    physics_reg_loss += gripper_reg
                if self.z_I_dim >= 7:
                    # Hand and object positions should be in reasonable range [-0.5, 0.5]
                    pos_reg = torch.mean(torch.relu(torch.abs(z_I[:, 1:7]) - 1.0))
                    physics_reg_loss += pos_reg
        
        # 8. Total loss
        total_loss = (
            recon_loss + 
            self.beta * kl_z_I + 
            self.beta * kl_z_E + 
            self.physics_weight * physics_loss +
            self.physics_regularization_weight * physics_reg_loss
        )
        
        # Logging
        loss_dict = {
            'reconstruction_loss': recon_loss.item(),
            'kl_z_I': kl_z_I.item(),
            'kl_z_E': kl_z_E.item(),
            'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            'physics_regularization_loss': physics_reg_loss.item() if isinstance(physics_reg_loss, torch.Tensor) else physics_reg_loss,
            'total_loss': total_loss.item(),
        }
        
        # Store P3-VAE specific stats
        self.p3_vae_stats.update(loss_dict)
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        num_batches = len(self.train_dataset) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.train_dataset))
            batch = self.train_dataset[start_idx:end_idx]
            
            # Convert to tensor and normalize
            batch = torch.FloatTensor(batch).to(ptu.device) / 255.0
            if self.input_channels == 1:
                batch = torch.mean(batch, dim=1, keepdim=True)
            
            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_loss(batch, epoch)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
            if self.log_interval > 0 and batch_idx % self.log_interval == 0:
                logger.record_dict(loss_dict, prefix='train/')
                
        return train_loss / num_batches
    
    def test_epoch(self, epoch, save_reconstruction=True, save_vae=True, from_rl=False):
        """Test for one epoch."""
        self.model.eval()
        test_loss = 0
        num_batches = len(self.test_dataset) // self.batch_size
        
        # Handle case where test dataset is too small
        if num_batches == 0:
            num_batches = 1
            effective_batch_size = len(self.test_dataset)
        else:
            effective_batch_size = self.batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * effective_batch_size  
                end_idx = min(start_idx + effective_batch_size, len(self.test_dataset))
                batch = self.test_dataset[start_idx:end_idx]
                
                # Convert to tensor and normalize
                batch = torch.FloatTensor(batch).to(ptu.device) / 255.0
                if self.input_channels == 1:
                    batch = torch.mean(batch, dim=1, keepdim=True)
                
                loss, loss_dict = self.compute_loss(batch, epoch)
                test_loss += loss.item()
                
        test_loss /= num_batches
        
        # Log test statistics
        if self.log_interval > 0:
            logger.record_dict({
                'test_loss': test_loss,
                **{f'test_{k}': v for k, v in loss_dict.items()}
            })
            
        return test_loss
    
    def get_diagnostics(self):
        """Get diagnostic information."""
        stats = super().get_diagnostics() if hasattr(super(), 'get_diagnostics') else {}
        stats.update(self.p3_vae_stats)
        return stats
    
    def end_epoch(self, epoch):
        """End of epoch processing."""
        if hasattr(super(), 'end_epoch'):
            super().end_epoch(epoch)
        
        # Additional P3-VAE specific end-of-epoch processing
        if epoch % 10 == 0:
            # Save physics variable analysis
            self._analyze_physics_variables()
    
    def _analyze_physics_variables(self):
        """Analyze the learned physics variables."""
        self.model.eval()
        with torch.no_grad():
            # Sample a few images
            batch_size = min(32, len(self.test_dataset))
            batch = self.test_dataset[:batch_size]
            batch = torch.FloatTensor(batch).to(ptu.device) / 255.0
            if self.input_channels == 1:
                batch = torch.mean(batch, dim=1, keepdim=True)
            
            # Encode physics variables
            z_I_stats = self.encode_z_I(batch)
            z_I = self.draw_z_I(z_I_stats)
            
            # Extract physics state
            physics_state = self.extract_physics_state(z_I)
            
            # Log statistics of physics variables
            physics_analysis = {}
            for key, value in physics_state.items():
                if isinstance(value, torch.Tensor):
                    physics_analysis[f'physics_{key}_mean'] = torch.mean(value).item()
                    physics_analysis[f'physics_{key}_std'] = torch.std(value).item()
                    physics_analysis[f'physics_{key}_min'] = torch.min(value).item()
                    physics_analysis[f'physics_{key}_max'] = torch.max(value).item()
            
            logger.record_dict(physics_analysis, prefix='physics_analysis/')
            self.p3_vae_stats.update(physics_analysis)
    
    def dump_samples(self, epoch):
        """Generate and save samples from the P3-VAE model."""
        self.model.eval()
        
        # Sample from priors
        batch_size = 64
        device = ptu.device
        
        # Sample z_I from normal distribution (physics)
        z_I = torch.randn(batch_size, self.z_I_dim).to(device)
        
        # Sample z_E from appropriate distribution (environmental)
        if self.use_beta_distribution:
            # Sample from Beta(1, 1) = Uniform[0,1]
            z_E = torch.rand(batch_size, self.z_E_dim).to(device)
        else:
            # Sample from normal distribution
            z_E = torch.randn(batch_size, self.z_E_dim).to(device)
        
        # Combine and decode
        z_combined = torch.cat([z_I, z_E], dim=1)
        
        with torch.no_grad():
            if hasattr(self.model, 'decode_from_physics_env'):
                sample = self.model.decode_from_physics_env(z_I, z_E)
            else:
                sample, _ = self.model.decode(z_combined)
        
        # Save samples
        save_dir = osp.join(logger.get_snapshot_dir(), f's{epoch}.png')
        save_image(
            sample.data.view(batch_size, self.input_channels, self.imsize, self.imsize),
            save_dir
        )
    
    def ode_integration(self, func, y0, t, method='euler'):
        """
        ODE integration using torchdiffeq if available, otherwise simple Euler method.
        
        Args:
            func: Function that computes dy/dt = func(t, y)
            y0: Initial conditions [batch_size, state_dim]
            t: Time points [num_steps]
            method: Integration method ('euler', 'rk4', 'dopri5')
            
        Returns:
            torch.Tensor: Solution at time points [num_steps, batch_size, state_dim]
        """
        if HAS_TORCHDIFFEQ and method != 'euler':
            # Use proper ODE solver
            return odeint(func, y0, t, method=method)
        else:
            # Fallback to simple Euler integration
            return self.ode(func, y0, t)
    
    def ode(self, func, y0, t, *args):
        """
        Simple ODE integration using Euler method.
        Compatible with scipy.integrate.odeint interface.
        
        Args:
            func: Function that computes dy/dt = func(y, t, *args)
            y0: Initial conditions [batch_size, state_dim]
            t: Time points [num_steps]
            *args: Additional arguments to pass to func
            
        Returns:
            torch.Tensor: Solution at time points [num_steps, batch_size, state_dim]
        """
        if not isinstance(y0, torch.Tensor):
            y0 = torch.tensor(y0, dtype=torch.float32, device=ptu.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=ptu.device)
            
        if len(y0.shape) == 1:
            y0 = y0.unsqueeze(0)  # Add batch dimension
            
        solutions = [y0]
        y_current = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dydt = func(y_current, t[i-1], *args)
            if not isinstance(dydt, torch.Tensor):
                dydt = torch.tensor(dydt, dtype=torch.float32, device=ptu.device)
            y_current = y_current + dydt * dt
            solutions.append(y_current)
            
        return torch.stack(solutions, dim=0)
