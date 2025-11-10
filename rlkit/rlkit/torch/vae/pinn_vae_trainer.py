import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger
from multiworld.core.image_env import normalize_image


class PINNVAETrainer(ConvVAETrainer):
    """
    Physics-Informed Neural Network (PINN) VAE trainer.
    Uses PDEs to enforce physics constraints in the latent space.
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
            # PINN-specific parameters
            pinn_weight=0.1,
            physics_type='pusher',  # 'pusher', 'pendulum', 'manipulation'
            dt=0.1,  # Time step for physics simulation
            enable_contact_dynamics=True,
            enable_friction=True,
            enable_conservation_laws=True,
    ):
        # Store PINN-specific parameters first
        self.pinn_weight = pinn_weight
        self.physics_type = physics_type
        self.dt = dt
        self.enable_contact_dynamics = enable_contact_dynamics
        self.enable_friction = enable_friction
        self.enable_conservation_laws = enable_conservation_laws
        
        # Initialize parent class manually to avoid quick_init conflicts
        # Copy the parent class initialization logic
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
        
        # PINN logging
        self.pinn_stats = {}
        
    def _initialize_physics_parameters(self):
        """Initialize physics parameters based on the task type."""
        
        if self.physics_type == 'pusher':
            # Pusher task: hand pushing a puck on a table
            self.mass_hand = 1.0
            self.mass_puck = 0.5
            self.friction_coeff = 0.3
            self.restitution = 0.8
            self.contact_stiffness = 1000.0
            
        elif self.physics_type == 'pendulum':
            # Pendulum: simple harmonic motion
            self.mass = 1.0
            self.length = 1.0
            self.gravity = 9.81
            self.damping = 0.1
            
        elif self.physics_type == 'manipulation':
            # General manipulation: gripper + object
            self.mass_gripper = 2.0
            self.mass_object = 1.0
            self.friction_coeff = 0.4
            self.contact_stiffness = 500.0
            
        # Common parameters
        self.gravity = getattr(self, 'gravity', 9.81)
        
    def extract_physics_state(self, z):
        """
        Extract physics state variables from latent representation.
        
        Args:
            z: Latent representation [batch_size, latent_dim]
            
        Returns:
            dict: Physics state variables (positions, velocities, etc.)
        """
        batch_size, latent_dim = z.shape
        
        if self.physics_type == 'pusher':
            # For pusher: [hand_x, hand_y, puck_x, puck_y, hand_vx, hand_vy, puck_vx, puck_vy]
            # Assume latent dim is 8 or split evenly
            mid = latent_dim // 2
            
            positions = z[:, :mid]  # [hand_x, hand_y, puck_x, puck_y]
            velocities = z[:, mid:]  # [hand_vx, hand_vy, puck_vx, puck_vy]
            
            if mid >= 4:
                hand_pos = positions[:, :2]
                puck_pos = positions[:, 2:4] if mid >= 4 else positions[:, 2:3].repeat(1, 2)
                hand_vel = velocities[:, :2]
                puck_vel = velocities[:, 2:4] if mid >= 4 else velocities[:, 2:3].repeat(1, 2)
            else:
                # Simplified case with fewer dimensions
                hand_pos = positions[:, :2] if mid >= 2 else positions
                puck_pos = positions[:, :2] if mid >= 2 else positions
                hand_vel = velocities[:, :2] if mid >= 2 else velocities
                puck_vel = velocities[:, :2] if mid >= 2 else velocities
                
            return {
                'hand_pos': hand_pos,
                'puck_pos': puck_pos, 
                'hand_vel': hand_vel,
                'puck_vel': puck_vel,
                'positions': positions,
                'velocities': velocities
            }
            
        elif self.physics_type == 'pendulum':
            # For pendulum: [theta, theta_dot, ...]
            angle = z[:, 0:1]
            angular_vel = z[:, 1:2] if latent_dim > 1 else torch.zeros_like(angle)
            
            return {
                'angle': angle,
                'angular_velocity': angular_vel,
                'x_pos': torch.sin(angle),
                'y_pos': -torch.cos(angle),
            }
            
        elif self.physics_type == 'manipulation':
            # For manipulation: similar to pusher but more general
            mid = latent_dim // 2
            positions = z[:, :mid]
            velocities = z[:, mid:]
            
            gripper_pos = positions[:, :3] if mid >= 3 else positions
            object_pos = positions[:, 3:6] if mid >= 6 else positions[:, :3]
            gripper_vel = velocities[:, :3] if mid >= 3 else velocities
            object_vel = velocities[:, 3:6] if mid >= 6 else velocities[:, :3]
            
            return {
                'gripper_pos': gripper_pos,
                'object_pos': object_pos,
                'gripper_vel': gripper_vel,
                'object_vel': object_vel,
                'positions': positions,
                'velocities': velocities
            }
            
        else:
            # Default: assume first half positions, second half velocities
            mid = latent_dim // 2
            return {
                'positions': z[:, :mid],
                'velocities': z[:, mid:],
            }
    
    def compute_pde_residuals(self, state_t, state_t1):
        """
        Compute PDE residuals for physics laws.
        
        Args:
            state_t: Physics state at time t
            state_t1: Physics state at time t+1
            
        Returns:
            dict: PDE residual losses
        """
        residuals = {}
        
        if self.physics_type == 'pusher':
            residuals.update(self._compute_pusher_pde_residuals(state_t, state_t1))
        elif self.physics_type == 'pendulum':
            residuals.update(self._compute_pendulum_pde_residuals(state_t, state_t1))
        elif self.physics_type == 'manipulation':
            residuals.update(self._compute_manipulation_pde_residuals(state_t, state_t1))
            
        return residuals
    
    def _compute_pusher_pde_residuals(self, state_t, state_t1):
        """Compute PDE residuals for pusher task."""
        residuals = {}
        
        # Extract states
        hand_pos_t = state_t['hand_pos']
        puck_pos_t = state_t['puck_pos']
        hand_vel_t = state_t['hand_vel']
        puck_vel_t = state_t['puck_vel']
        
        hand_pos_t1 = state_t1['hand_pos']
        puck_pos_t1 = state_t1['puck_pos']
        hand_vel_t1 = state_t1['hand_vel']
        puck_vel_t1 = state_t1['puck_vel']
        
        # 1. Newton's first law: F = ma (for free motion)
        # When no contact, objects should move with constant velocity
        contact_dist = torch.norm(hand_pos_t - puck_pos_t, dim=1, keepdim=True)
        contact_threshold = 0.1
        no_contact_mask = (contact_dist > contact_threshold).float()
        
        # Free motion: velocity should be approximately constant
        hand_accel = (hand_vel_t1 - hand_vel_t) / self.dt
        puck_accel = (puck_vel_t1 - puck_vel_t) / self.dt
        
        # PDE: dv/dt = 0 (no forces)
        residuals['newton_first_hand'] = torch.mean(
            no_contact_mask * torch.norm(hand_accel, dim=1, keepdim=True)**2
        )
        residuals['newton_first_puck'] = torch.mean(
            no_contact_mask * torch.norm(puck_accel, dim=1, keepdim=True)**2
        )
        
        # 2. Kinematic constraint: dx/dt = v
        hand_pos_predicted = hand_pos_t + hand_vel_t * self.dt
        puck_pos_predicted = puck_pos_t + puck_vel_t * self.dt
        
        residuals['kinematics_hand'] = torch.mean(
            torch.norm(hand_pos_t1 - hand_pos_predicted, dim=1)**2
        )
        residuals['kinematics_puck'] = torch.mean(
            torch.norm(puck_pos_t1 - puck_pos_predicted, dim=1)**2
        )
        
        # 3. Conservation of momentum during contact
        if self.enable_conservation_laws:
            contact_mask = (contact_dist <= contact_threshold).float()
            
            # Total momentum before and after
            momentum_before = (self.mass_hand * hand_vel_t + self.mass_puck * puck_vel_t)
            momentum_after = (self.mass_hand * hand_vel_t1 + self.mass_puck * puck_vel_t1)
            
            momentum_conservation = torch.norm(momentum_after - momentum_before, dim=1, keepdim=True)
            residuals['momentum_conservation'] = torch.mean(
                contact_mask * momentum_conservation**2
            )
        
        # 4. Energy constraints
        if self.enable_conservation_laws:
            # Kinetic energy
            ke_hand_t = 0.5 * self.mass_hand * torch.sum(hand_vel_t**2, dim=1, keepdim=True)
            ke_puck_t = 0.5 * self.mass_puck * torch.sum(puck_vel_t**2, dim=1, keepdim=True)
            ke_total_t = ke_hand_t + ke_puck_t
            
            ke_hand_t1 = 0.5 * self.mass_hand * torch.sum(hand_vel_t1**2, dim=1, keepdim=True)
            ke_puck_t1 = 0.5 * self.mass_puck * torch.sum(puck_vel_t1**2, dim=1, keepdim=True)
            ke_total_t1 = ke_hand_t1 + ke_puck_t1
            
            # During free motion, energy should be conserved (ignoring friction for simplicity)
            energy_change = torch.abs(ke_total_t1 - ke_total_t)
            residuals['energy_conservation'] = torch.mean(
                no_contact_mask * energy_change**2
            )
        
        return residuals
    
    def _compute_pendulum_pde_residuals(self, state_t, state_t1):
        """Compute PDE residuals for pendulum task."""
        residuals = {}
        
        # Extract states
        theta_t = state_t['angle']
        omega_t = state_t['angular_velocity']
        theta_t1 = state_t1['angle']
        omega_t1 = state_t1['angular_velocity']
        
        # 1. Pendulum equation: d²θ/dt² = -(g/L)sin(θ) - b*dθ/dt
        # Where b is damping coefficient
        angular_accel = (omega_t1 - omega_t) / self.dt
        
        # Expected acceleration from physics
        gravity_term = -(self.gravity / self.length) * torch.sin(theta_t)
        damping_term = -self.damping * omega_t
        expected_accel = gravity_term + damping_term
        
        # PDE residual
        residuals['pendulum_equation'] = torch.mean(
            (angular_accel - expected_accel)**2
        )
        
        # 2. Kinematic constraint: dθ/dt = ω
        theta_predicted = theta_t + omega_t * self.dt
        residuals['angular_kinematics'] = torch.mean(
            (theta_t1 - theta_predicted)**2
        )
        
        # 3. Energy conservation (total energy should be conserved)
        if self.enable_conservation_laws:
            # Potential energy: mgh = mg*L*(1 - cos(θ))
            pe_t = self.mass * self.gravity * self.length * (1 - torch.cos(theta_t))
            pe_t1 = self.mass * self.gravity * self.length * (1 - torch.cos(theta_t1))
            
            # Kinetic energy: (1/2)*I*ω² where I = mL²
            I = self.mass * self.length**2
            ke_t = 0.5 * I * omega_t**2
            ke_t1 = 0.5 * I * omega_t1**2
            
            energy_t = pe_t + ke_t
            energy_t1 = pe_t1 + ke_t1
            
            # Energy should decrease due to damping
            energy_change = energy_t1 - energy_t
            expected_energy_loss = -self.damping * omega_t**2 * self.dt
            
            residuals['energy_consistency'] = torch.mean(
                (energy_change - expected_energy_loss)**2
            )
        
        return residuals
    
    def _compute_manipulation_pde_residuals(self, state_t, state_t1):
        """Compute PDE residuals for general manipulation task."""
        residuals = {}
        
        # Extract states
        gripper_pos_t = state_t['gripper_pos']
        object_pos_t = state_t['object_pos']
        gripper_vel_t = state_t['gripper_vel']
        object_vel_t = state_t['object_vel']
        
        gripper_pos_t1 = state_t1['gripper_pos']
        object_pos_t1 = state_t1['object_pos']
        gripper_vel_t1 = state_t1['gripper_vel']
        object_vel_t1 = state_t1['object_vel']
        
        # 1. Gravity effect on object (assuming z is vertical)
        if gripper_pos_t.shape[1] >= 3:  # 3D case
            object_accel = (object_vel_t1 - object_vel_t) / self.dt
            expected_gravity_accel = torch.zeros_like(object_accel)
            expected_gravity_accel[:, 2] = -self.gravity  # z-direction
            
            # When object is not grasped, it should fall
            grasp_dist = torch.norm(gripper_pos_t - object_pos_t, dim=1, keepdim=True)
            not_grasped_mask = (grasp_dist > 0.05).float()
            
            gravity_residual = object_accel[:, 2:3] - expected_gravity_accel[:, 2:3]
            residuals['gravity'] = torch.mean(
                not_grasped_mask * gravity_residual**2
            )
        
        # 2. Kinematic constraints
        gripper_pos_predicted = gripper_pos_t + gripper_vel_t * self.dt
        object_pos_predicted = object_pos_t + object_vel_t * self.dt
        
        residuals['kinematics_gripper'] = torch.mean(
            torch.norm(gripper_pos_t1 - gripper_pos_predicted, dim=1)**2
        )
        residuals['kinematics_object'] = torch.mean(
            torch.norm(object_pos_t1 - object_pos_predicted, dim=1)**2
        )
        
        # 3. Grasping constraints
        grasp_dist = torch.norm(gripper_pos_t - object_pos_t, dim=1, keepdim=True)
        grasped_mask = (grasp_dist <= 0.05).float()
        
        # When grasped, object should move with gripper
        if torch.sum(grasped_mask) > 0:
            relative_motion = torch.norm(
                (object_vel_t1 - object_vel_t) - (gripper_vel_t1 - gripper_vel_t), 
                dim=1, keepdim=True
            )
            residuals['grasp_constraint'] = torch.mean(
                grasped_mask * relative_motion**2
            )
        
        return residuals
    
    def compute_pinn_loss(self, z_t, z_t1):
        """
        Compute PINN loss by enforcing PDEs in latent space.
        
        Args:
            z_t: Latent codes at time t [batch_size, latent_dim]
            z_t1: Latent codes at time t+1 [batch_size, latent_dim]
            
        Returns:
            dict: PINN loss components
        """
        # Extract physics states
        state_t = self.extract_physics_state(z_t)
        state_t1 = self.extract_physics_state(z_t1)
        
        # Compute PDE residuals
        residuals = self.compute_pde_residuals(state_t, state_t1)
        
        # Convert to losses with weights
        pinn_losses = {}
        total_loss = 0.0
        
        for key, residual in residuals.items():
            if torch.isnan(residual) or torch.isinf(residual):
                # Skip invalid residuals
                continue
                
            loss_value = residual
            pinn_losses[key] = loss_value
            total_loss += loss_value
        
        pinn_losses['total'] = total_loss
        return pinn_losses
    
    def get_temporal_batch(self, train=True):
        """Get temporal batch for PINN training."""
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
            
        return ptu.from_numpy(img_t), ptu.from_numpy(img_t1)
    
    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        """Enhanced training with PINN loss."""
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        pinn_losses = {
            'total': [],
            'newton_first_hand': [],
            'newton_first_puck': [],
            'kinematics_hand': [],
            'kinematics_puck': [],
            'momentum_conservation': [],
            'energy_conservation': [],
            'pendulum_equation': [],
            'angular_kinematics': [],
            'energy_consistency': [],
            'gravity': [],
            'kinematics_gripper': [],
            'kinematics_object': [],
            'grasp_constraint': [],
        }
        
        for batch_idx in range(batches):
            if sample_batch is not None:
                # Standard VAE training on RL data
                data = sample_batch(self.batch_size)
                next_obs = data['next_obs']
                
                self.optimizer.zero_grad()
                reconstructions, obs_distribution_params, latent_distribution_params = self.model(next_obs)
                log_prob = self.model.logprob(next_obs, obs_distribution_params)
                kle = self.model.kl_divergence(latent_distribution_params)
                total_loss = self.beta * kle - log_prob
                
            else:
                # PINN training on temporal data
                img_t, img_t1 = self.get_temporal_batch()
                
                self.optimizer.zero_grad()
                
                # Encode both time steps
                z_t_params = self.model.encode(img_t)
                z_t1_params = self.model.encode(img_t1)
                
                # Sample latent codes
                z_t = self.model.rsample(z_t_params)
                z_t1 = self.model.rsample(z_t1_params)
                
                # Standard VAE losses
                recon_t, obs_dist_t, _ = self.model(img_t)
                recon_t1, obs_dist_t1, _ = self.model(img_t1)
                
                log_prob_t = self.model.logprob(img_t, obs_dist_t)
                log_prob_t1 = self.model.logprob(img_t1, obs_dist_t1)
                kle_t = self.model.kl_divergence(z_t_params)
                kle_t1 = self.model.kl_divergence(z_t1_params)
                
                vae_loss = self.beta * (kle_t + kle_t1) - (log_prob_t + log_prob_t1)
                
                # PINN loss
                pinn_loss_dict = self.compute_pinn_loss(z_t, z_t1)
                pinn_total = pinn_loss_dict.get('total', 0.0)
                
                total_loss = vae_loss + self.pinn_weight * pinn_total
                
                # Logging
                for key, value in pinn_loss_dict.items():
                    if key in pinn_losses:
                        val = value.item() if hasattr(value, 'item') else value
                        if not (np.isnan(val) or np.isinf(val)):
                            pinn_losses[key].append(val)
                
                log_prob = (log_prob_t + log_prob_t1) / 2
                kle = (kle_t + kle_t1) / 2
            
            total_loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            losses.append(total_loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tTotal Loss: {:.6f}\tVAE Loss: {:.6f}\tPINN: {:.6f}'.format(
                    epoch, batch_idx, batches, total_loss.item(), 
                    vae_loss.item() if 'vae_loss' in locals() else 0.0,
                    (self.pinn_weight * pinn_total).item() if 'pinn_total' in locals() else 0.0))
        
        # Update logging stats
        if from_rl:
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Train VAE Log Prob'] = np.mean(log_probs)
            self.vae_logger_stats_for_rl['Train VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Train VAE Loss'] = np.mean(losses)
            
            # Add PINN stats
            for key, values in pinn_losses.items():
                if values:
                    self.vae_logger_stats_for_rl[f'Train PINN {key}'] = np.mean(values)
        else:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/loss", np.mean(losses))
            logger.record_tabular("train/log_prob", np.mean(log_probs))
            logger.record_tabular("train/kl", np.mean(kles))
            
            # Log PINN losses
            for key, values in pinn_losses.items():
                if values:
                    logger.record_tabular(f"train/pinn_{key}", np.mean(values))
