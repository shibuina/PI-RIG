import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger
from multiworld.core.image_env import normalize_image


class PhysicsInformedConvVAETrainer(ConvVAETrainer):
    """
    Physics-informed VAE trainer for robotic manipulation tasks.
    Adds physics constraints to the standard VAE loss.
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
    ):
        # Store physics-specific parameters first
        self.physics_weight = physics_weight
        self.contact_weight = contact_weight  
        self.momentum_weight = momentum_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # Manually initialize the parent class to avoid quick_init issues
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
        
        # Physics-related logging
        self.physics_stats = {}
        
    def compute_physics_loss(self, z_t, z_t1, contact_mask=None):
        """
        Compute physics-informed loss terms.
        
        Args:
            z_t: Latent representation at time t [batch_size, latent_dim]
            z_t1: Latent representation at time t+1 [batch_size, latent_dim]
            contact_mask: Binary mask indicating contact events [batch_size, 1]
        
        Returns:
            dict: Physics loss components
        """
        batch_size = z_t.shape[0]
        latent_dim = z_t.shape[1]
        
        losses = {}
        
        # 1. Temporal consistency loss (smooth transitions)
        if self.temporal_consistency_weight > 0:
            temporal_diff = z_t1 - z_t
            temporal_loss = torch.mean(torch.norm(temporal_diff, dim=1))
            losses['temporal'] = self.temporal_consistency_weight * temporal_loss
        
        # 2. Momentum conservation loss (simplified)
        if self.momentum_weight > 0:
            # Assume first half of latent dim represents positions, second half velocities
            mid_dim = latent_dim // 2
            
            # Extract position and velocity components
            pos_t = z_t[:, :mid_dim]
            vel_t = z_t[:, mid_dim:]
            pos_t1 = z_t1[:, :mid_dim]
            vel_t1 = z_t1[:, mid_dim:]
            
            # Simple momentum conservation: velocity should be consistent
            # unless there's a contact event
            if contact_mask is not None:
                # During contact, allow velocity changes
                momentum_loss = torch.mean(
                    (1 - contact_mask) * torch.norm(vel_t1 - vel_t, dim=1)
                )
            else:
                # Without contact info, penalize large velocity changes
                momentum_loss = torch.mean(torch.norm(vel_t1 - vel_t, dim=1))
            
            losses['momentum'] = self.momentum_weight * momentum_loss
        
        # 3. Contact dynamics loss
        if self.contact_weight > 0 and contact_mask is not None:
            # During contact, positions should change less dramatically
            # (object can't pass through each other)
            mid_dim = latent_dim // 2
            pos_t = z_t[:, :mid_dim]
            pos_t1 = z_t1[:, :mid_dim]
            
            pos_change = torch.norm(pos_t1 - pos_t, dim=1)
            # During contact, penalize large position changes
            contact_loss = torch.mean(contact_mask.squeeze() * pos_change)
            
            losses['contact'] = self.contact_weight * contact_loss
        
        return losses
    
    def detect_contact_from_images(self, img_t, img_t1):
        """
        Simple contact detection from image differences.
        This is a heuristic - in practice you might want more sophisticated detection.
        
        Args:
            img_t: Flattened images at time t [batch_size, height*width*channels]
            img_t1: Flattened images at time t+1 [batch_size, height*width*channels]
        """
        # Reshape flattened images back to (batch, c, h, w) for processing
        batch_size = img_t.shape[0]
        h, w, c = self.imsize, self.imsize, self.input_channels
        
        # Reshape from flat to (batch, h*w*c) -> (batch, h, w, c) -> (batch, c, h, w)
        img_t_4d = img_t.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        img_t1_4d = img_t1.reshape(batch_size, h, w, c).permute(0, 3, 1, 2)
        
        # Compute image difference
        img_diff = torch.abs(img_t1_4d - img_t_4d)
        
        # Contact is detected when there's significant change in specific regions
        # For pusher task: look for changes in the central region where hand-puck contact occurs
        center_region = img_diff[:, :, h//4:3*h//4, w//4:3*w//4]
        
        # Contact detected when average change in center region exceeds threshold
        contact_score = torch.mean(center_region, dim=[1, 2, 3])
        contact_mask = (contact_score > 0.1).float().unsqueeze(1)  # Threshold
        
        return contact_mask
    
    def get_temporal_batch(self, train=True):
        """
        Get a batch of temporal pairs (t, t+1) for physics-informed training.
        Returns flattened images in the same format as get_batch().
        """
        dataset = self.train_dataset if train else self.test_dataset
        batch_size = self.batch_size
        
        # Sample random indices, ensuring we can get t+1
        max_idx = len(dataset) - 1
        ind_t = np.random.randint(0, max_idx, batch_size)
        ind_t1 = ind_t + 1
        
        # Get normalized images (this flattens them)
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
        """
        Enhanced training with physics-informed loss.
        """
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        physics_losses = {
            'temporal': [],
            'momentum': [],
            'contact': [],
            'total_physics': []
        }
        
        for batch_idx in range(batches):
            if sample_batch is not None:
                # Standard VAE training
                data = sample_batch(self.batch_size)
                next_obs = data['next_obs']
                
                # Standard VAE loss
                self.optimizer.zero_grad()
                reconstructions, obs_distribution_params, latent_distribution_params = self.model(next_obs)
                log_prob = self.model.logprob(next_obs, obs_distribution_params)
                kle = self.model.kl_divergence(latent_distribution_params)
                vae_loss = self.beta * kle - log_prob
                
                # No physics loss for RL data (for now)
                total_loss = vae_loss
                
            else:
                # Get temporal batch for physics-informed training
                img_t, img_t1 = self.get_temporal_batch()
                
                # Forward pass for both time steps
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
                
                # Physics-informed loss
                contact_mask = self.detect_contact_from_images(img_t, img_t1)
                physics_loss_dict = self.compute_physics_loss(z_t, z_t1, contact_mask)
                
                physics_total = sum(physics_loss_dict.values())
                total_loss = vae_loss + self.physics_weight * physics_total
                
                # Logging
                for key, value in physics_loss_dict.items():
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
                if values:  # Only log if we have physics losses
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
    
    def test_epoch(self, epoch, save_reconstruction=True, save_scatterplot=True, 
                   save_vae=True, from_rl=False):
        """
        Test epoch with physics-aware evaluation.
        """
        # For testing, we'll use the standard approach but add physics evaluation
        # Note: parent method doesn't have save_scatterplot parameter
        super().test_epoch(epoch, save_reconstruction, save_vae, from_rl)
        
        # Additional physics-specific testing could be added here
        # For now, we keep it simple and rely on the standard test procedure
