import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from rlkit.torch.vae.physics_informed_vae_trainer import PhysicsInformedConvVAETrainer
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger


class AdvancedPhysicsInformedConvVAETrainer(PhysicsInformedConvVAETrainer):
    """
    Advanced physics-informed VAE trainer with additional physics constraints
    for complex robotic manipulation environments like pick-and-place, door opening, etc.
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
            physics_weight=0.1,
            contact_weight=0.05,
            momentum_weight=0.05,
            temporal_consistency_weight=0.02,
            # New physics constraints
            gravity_weight=0.0,
            grasp_stability_weight=0.0,
            angular_momentum_weight=0.0,
            hinge_constraint_weight=0.0,
            collision_response_weight=0.0,
    ):
        # Initialize parent class
        super().__init__(
            train_dataset,
            test_dataset,
            model,
            batch_size,
            log_interval,
            beta,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            physics_weight,
            contact_weight,
            momentum_weight,
            temporal_consistency_weight,
        )
        
        # Store additional physics weights
        self.gravity_weight = gravity_weight
        self.grasp_stability_weight = grasp_stability_weight
        self.angular_momentum_weight = angular_momentum_weight
        self.hinge_constraint_weight = hinge_constraint_weight
        self.collision_response_weight = collision_response_weight
        
    def compute_advanced_physics_loss(self, z_t, z_t1, img_t, img_t1, contact_mask=None):
        """
        Compute advanced physics loss terms for complex manipulation tasks.
        
        Args:
            z_t: Latent representation at time t [batch_size, latent_dim]
            z_t1: Latent representation at time t+1 [batch_size, latent_dim]
            img_t: Images at time t [batch_size, h*w*c]
            img_t1: Images at time t+1 [batch_size, h*w*c]
            contact_mask: Binary mask indicating contact events [batch_size, 1]
        
        Returns:
            dict: Advanced physics loss components
        """
        batch_size = z_t.shape[0]
        latent_dim = z_t.shape[1]
        losses = {}
        
        # 1. Gravity consistency loss
        if self.gravity_weight > 0:
            # Objects should fall when not supported
            # Look for vertical motion patterns in latent space
            # Assume first dimension represents vertical position
            vertical_pos_t = z_t[:, 0:1]  # First dimension
            vertical_pos_t1 = z_t1[:, 0:1]
            
            # Gravity should cause downward acceleration
            vertical_change = vertical_pos_t1 - vertical_pos_t
            
            # Detect if object is likely unsupported (high image change in lower region)
            img_t_reshaped = img_t.reshape(batch_size, self.imsize, self.imsize, self.input_channels)
            img_t1_reshaped = img_t1.reshape(batch_size, self.imsize, self.imsize, self.input_channels)
            
            # Look at bottom half of image for ground contact
            bottom_region_t = img_t_reshaped[:, self.imsize//2:, :, :]
            bottom_region_t1 = img_t1_reshaped[:, self.imsize//2:, :, :]
            
            ground_contact = torch.mean(torch.abs(bottom_region_t1 - bottom_region_t), dim=[1, 2, 3])
            unsupported_mask = (ground_contact < 0.05).float().unsqueeze(1)  # Low change = no ground contact
            
            # Apply gravity constraint: unsupported objects should fall
            gravity_loss = torch.mean(
                unsupported_mask * torch.clamp(-vertical_change, min=0)  # Penalize upward motion when unsupported
            )
            losses['gravity'] = self.gravity_weight * gravity_loss
        
        # 2. Grasp stability loss
        if self.grasp_stability_weight > 0:
            # During grasping, object and hand should move together
            # Look for correlated movement in latent space
            if contact_mask is not None:
                # Assume latent dimensions are [obj_pos, hand_pos, obj_vel, hand_vel]
                mid_dim = latent_dim // 4
                obj_pos_change = z_t1[:, :mid_dim] - z_t[:, :mid_dim]
                hand_pos_change = z_t1[:, mid_dim:2*mid_dim] - z_t[:, mid_dim:2*mid_dim]
                
                # During contact, object and hand should move together
                grasp_consistency = torch.norm(obj_pos_change - hand_pos_change, dim=1)
                grasp_loss = torch.mean(contact_mask.squeeze() * grasp_consistency)
                
                losses['grasp_stability'] = self.grasp_stability_weight * grasp_loss
        
        # 3. Angular momentum conservation (for rotational tasks)
        if self.angular_momentum_weight > 0:
            # For tasks like door opening - angular velocity should be consistent
            # Assume later dimensions represent angular quantities
            if latent_dim >= 6:
                angular_vel_t = z_t[:, -2:]  # Last 2 dimensions for angular velocity
                angular_vel_t1 = z_t1[:, -2:]
                
                # Angular momentum should be conserved unless external torque is applied
                angular_change = torch.norm(angular_vel_t1 - angular_vel_t, dim=1)
                
                # Allow changes during contact (external torque application)
                if contact_mask is not None:
                    angular_loss = torch.mean((1 - contact_mask.squeeze()) * angular_change)
                else:
                    angular_loss = torch.mean(angular_change)
                
                losses['angular_momentum'] = self.angular_momentum_weight * angular_loss
        
        # 4. Hinge constraint (for door-like mechanisms)
        if self.hinge_constraint_weight > 0:
            # Door should rotate around fixed hinge point
            # Constrain certain latent dimensions to represent valid hinge motion
            if latent_dim >= 4:
                # Assume dimensions represent [door_angle, hinge_x, hinge_y, door_angular_vel]
                hinge_x_t = z_t[:, 1:2]
                hinge_y_t = z_t[:, 2:3]
                hinge_x_t1 = z_t1[:, 1:2]
                hinge_y_t1 = z_t1[:, 2:3]
                
                # Hinge position should remain fixed
                hinge_stability = torch.mean(
                    torch.norm(torch.cat([hinge_x_t1 - hinge_x_t, hinge_y_t1 - hinge_y_t], dim=1), dim=1)
                )
                
                losses['hinge_constraint'] = self.hinge_constraint_weight * hinge_stability
        
        # 5. Collision response (for environments with walls/obstacles)
        if self.collision_response_weight > 0:
            # During collisions, velocity should reverse appropriately
            # Look for sudden velocity changes correlated with contact
            if latent_dim >= 4:
                mid_dim = latent_dim // 2
                vel_t = z_t[:, mid_dim:]
                vel_t1 = z_t1[:, mid_dim:]
                
                # Detect collisions from large velocity changes
                vel_change_magnitude = torch.norm(vel_t1 - vel_t, dim=1)
                collision_threshold = 0.1
                collision_detected = (vel_change_magnitude > collision_threshold).float().unsqueeze(1)
                
                # During collisions, velocity should change in physically plausible way
                # Simple check: velocity should not increase during collision
                speed_t = torch.norm(vel_t, dim=1)
                speed_t1 = torch.norm(vel_t1, dim=1)
                speed_increase = torch.clamp(speed_t1 - speed_t, min=0)
                
                collision_loss = torch.mean(collision_detected.squeeze() * speed_increase)
                losses['collision_response'] = self.collision_response_weight * collision_loss
        
        return losses
    
    def compute_physics_loss(self, z_t, z_t1, contact_mask=None, img_t=None, img_t1=None):
        """
        Enhanced physics loss computation including advanced constraints.
        """
        # Get basic physics losses from parent
        basic_losses = super().compute_physics_loss(z_t, z_t1, contact_mask)
        
        # Add advanced physics losses if image data is available
        if img_t is not None and img_t1 is not None:
            advanced_losses = self.compute_advanced_physics_loss(z_t, z_t1, img_t, img_t1, contact_mask)
            basic_losses.update(advanced_losses)
        
        return basic_losses
    
    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        """
        Enhanced training with advanced physics-informed loss.
        """
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        physics_losses = {
            'temporal': [],
            'momentum': [],
            'contact': [],
            'gravity': [],
            'grasp_stability': [],
            'angular_momentum': [],
            'hinge_constraint': [],
            'collision_response': [],
            'total_physics': []
        }
        
        for batch_idx in range(batches):
            if sample_batch is not None:
                # Standard VAE training for RL data
                data = sample_batch(self.batch_size)
                next_obs = data['next_obs']
                
                self.optimizer.zero_grad()
                reconstructions, obs_distribution_params, latent_distribution_params = self.model(next_obs)
                log_prob = self.model.logprob(next_obs, obs_distribution_params)
                kle = self.model.kl_divergence(latent_distribution_params)
                vae_loss = self.beta * kle - log_prob
                
                total_loss = vae_loss
                
            else:
                # Get temporal batch for physics-informed training
                img_t, img_t1 = self.get_temporal_batch()
                
                self.optimizer.zero_grad()
                
                # Encode both time steps
                z_t_params = self.model.encode(img_t)
                z_t1_params = self.model.encode(img_t1)
                
                # Sample latent codes
                z_t = self.model.rsample(z_t_params)
                z_t1 = self.model.rsample(z_t1_params)
                
                # Decode both
                recon_t, obs_dist_t, _ = self.model(img_t)
                recon_t1, obs_dist_t1, _ = self.model(img_t1)
                
                # Standard VAE losses
                log_prob_t = self.model.logprob(img_t, obs_dist_t)
                log_prob_t1 = self.model.logprob(img_t1, obs_dist_t1)
                kle_t = self.model.kl_divergence(z_t_params)
                kle_t1 = self.model.kl_divergence(z_t1_params)
                
                vae_loss = self.beta * (kle_t + kle_t1) - (log_prob_t + log_prob_t1)
                
                # Enhanced physics-informed loss with image data
                contact_mask = self.detect_contact_from_images(img_t, img_t1)
                physics_loss_dict = self.compute_physics_loss(z_t, z_t1, contact_mask, img_t, img_t1)
                
                physics_total = sum(physics_loss_dict.values())
                total_loss = vae_loss + self.physics_weight * physics_total
                
                # Logging
                for key, value in physics_loss_dict.items():
                    if key in physics_losses:
                        physics_losses[key].append(value.item() if hasattr(value, 'item') else value)
                physics_losses['total_physics'].append(physics_total.item() if hasattr(physics_total, 'item') else physics_total)
                
                log_prob = (log_prob_t + log_prob_t1) / 2
                kle = (kle_t + kle_t1) / 2
            
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            log_probs.append(log_prob.item())
            kles.append(kle.item())
            
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tTotal Loss: {:.6f}\tVAE Loss: {:.6f}'.format(
                    epoch, batch_idx, batches, total_loss.item(), vae_loss.item()))
        
        # Update logging stats
        if from_rl:
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Train VAE Log Prob'] = np.mean(log_probs)
            self.vae_logger_stats_for_rl['Train VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Train VAE Loss'] = np.mean(losses)
            
            # Add physics stats
            for key, values in physics_losses.items():
                if values:
                    self.vae_logger_stats_for_rl[f'Train Physics {key}'] = np.mean(values)
        else:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/loss", np.mean(losses))
            logger.record_tabular("train/log_prob", np.mean(log_probs))
            logger.record_tabular("train/kl", np.mean(kles))
            
            # Log physics losses
            for key, values in physics_losses.items():
                if values:
                    logger.record_tabular(f"train/physics_{key}", np.mean(values))
