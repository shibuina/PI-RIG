import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from rlkit.torch import pytorch_util as ptu
from rlkit.core import logger
from multiworld.core.image_env import normalize_image
import torch.autograd as autograd

class PhysicsHead(nn.Module):
    """
    Map the VAE latetn space z to (q,p).
    """
    def __init__(self, z_dim, q_dim = 1, p_dim = 1, hidden = 128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, q_dim)
        )
        self.p = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p_dim)
        )

    def forward(self, z):
        q = self.q(z)
        p = self.p(z)
        return q, p
    
class Hamiltonian(nn.Module):
    """
    MLP H(q,p) -> scaler energy.
    """
    def __init__(self, q_dim = 1, p_dim = 1, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim + p_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, q, p):
        qp = torch.cat([q, p], dim=-1)
        H = self.net(qp)
        return H
    
class HamiltonVAETrainer(ConvVAETrainer):
    """
    Hamiltonian VAE Trainer 
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
            q_dim=1,
            p_dim=1,
            hidden=128,
            dt=0.05,
            lambda_dyn=1.0,
            lambda_energy=1.0,
            energy_mode='const',  # 'const', 'decay', 'none'
            rollout_K=0,
    ):
        
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
        self.batch_size = batch_size
        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        self.evaluation_statistics = None
        self.vae_logger_stats_for_rl = {}

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self.is_auto_encoder = is_auto_encoder

        z_dim = getattr(model, 'representation_size', self.representation_size)
        self.q_dim, self.p_dim = q_dim, p_dim
        self.dt = dt
        self.lambda_dyn = float(lambda_dyn)
        self.lambda_energy = float(lambda_energy)
        self.energy_mode = str(energy_mode)
        self.rollout_K = int(rollout_K)

        self.physics_head = PhysicsHead(z_dim, q_dim, p_dim, hidden).to(ptu.device)
        self.hamiltonian = Hamiltonian(q_dim, p_dim, hidden).to(ptu.device)

        self.optimizer.add_param_group({'params': list(self.physics_head.parameters())})
        self.optimizer.add_param_group({'params': list(self.hamiltonian.parameters())})

    def ham_grad(self, q, p):
        """
        Compute dH/dq, dH/dp using autograd
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        H = self.hamiltonian(q, p).sum()
        dH_dq, dH_dp = autograd.grad(H, (q, p), create_graph=True)
        return dH_dq, dH_dp
    
    def leapfrog(self, q, p, dt):
        """
        Leapfrog integrator step
        """
        dH_dq, dH_dp = self.ham_grad(q, p)
        p_half = p - 0.5 * dt * dH_dq
        dH_dq_half, dH_dp_half = self.ham_grad(q, p_half)
        q_next = q + dt * dH_dp_half
        dH_dq_next, dH_dp_next = self.ham_grad(q_next, p_half)
        p_next = p_half - 0.5 * dt * dH_dq_next

        return q_next, p_next
    
    def ham_losses(self, z_t, z_t1):
        """
        Hamiltonian losses
        """
        q_t, p_t = self.physics_head(z_t)
        q_t1_enc, p_t1_enc = self.physics_head(z_t1)

        q_t1_roll, p_t1_roll = self.leapfrog(q_t, p_t, self.dt)
        L_dyn = F.mse_loss(q_t1_roll, q_t1_enc) + F.mse_loss(p_t1_roll, p_t1_enc)

        L_roll = torch.tensor(0.).to(ptu.device) #optional rollout loss

        L_energy = torch.tensor(0.).to(ptu.device) #optional energy regularization

        if self.energy_mode != 'none':
            H_t = self.hamiltonian(q_t, p_t)
            H_t1 = self.hamiltonian(q_t1_enc.detach(), p_t1_enc.detach())
            if self.energy_mode == 'const':
                L_energy = F.mse_loss(H_t1, H_t)
            elif self.energy_mode == 'decay':
                inc = torch.clamp(H_t1 - H_t, min=0.)
                L_energy = torch.mean(inc**2)

        L_total = self.lambda_dyn * (L_dyn + L_roll) + self.lambda_energy * L_energy 
        
        return {'ham_dyn': L_dyn,
                'ham_roll': L_roll,
                'ham_energy': L_energy,
                'ham_total': L_total}
    
    def get_temporal_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        batch_size = self.batch_size

        max_idx = len(dataset) - 1 
        ind_t = np.random.randint(0, max_idx, size=batch_size)
        ind_t1 = ind_t + 1

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
        
        self.model.train()
        losses = []
        log_probs = []
        kles = []

        for batch_idx in range(batches):
            if sample_batch is not None:
                #standard VAE training
                data = sample_batch(self.batch_size)
                next_obs = data['next_obs']

                self.optimizer.zero_grad()
                reconstruction, obs_distribution_params, latent_distribution_params = self.model(next_obs)
                log_probs = self.model.logprob(next_obs, obs_distribution_params)
                kles = self.model.kl_divergence(latent_distribution_params)
                total_loss = self.beta * kles - log_probs

            else: 
                #Hamiltionian VAE training
                img_t, img_t1 = self.get_temporal_batch(train=True)

                self.optimizer.zero_grad()

                z_t_params = self.model.encode(img_t)
                z_t1_params = self.model.encode(img_t1)
                z_t = self.model.rsample(z_t_params)
                z_t1 = self.model.rsample(z_t1_params)

                recon_t, obs_dist_t, _ = self.model(img_t)
                recon_t1, obs_dist_t1, _ = self.model(img_t1)

                log_prob_t = self.model.logprob(img_t, obs_dist_t)
                log_prob_t1 = self.model.logprob(img_t1, obs_dist_t1)
                kld_t = self.model.kl_divergence(z_t_params)
                kld_t1 = self.model.kl_divergence(z_t1_params)

                vae_loss = self.beta * (kld_t + kld_t1) - (log_prob_t + log_prob_t1)

                ham = self.ham_losses(z_t, z_t1)
                total_loss = vae_loss + ham['ham_total']

                logger.record_tabular('train/ham_dyn', ham['ham_dyn'].item())
                logger.record_tabular('train/ham_roll', ham['ham_roll'].item())
                
                log_prob = (log_prob_t + log_prob_t1) / 2
                kld = (kld_t + kld_t1) / 2

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
            losses.append(total_loss.item())

            if sample_batch is not None:
                pass
            else:
                log_probs.append(log_prob.item())
                kles.append(kld.item())

            if self.log_interval and batch_idx % self.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{batches}]"
                      f"Total: {total_loss.item():.6f} ")
                
        if from_rl:
            if log_probs and kles:
                self.vae_logger_stats_for_rl['Train VAE Log Prob'] = float(np.mean(log_probs))
                self.vae_logger_stats_for_rl['Train VAE KL'] = float(np.mean(kles))
            self.vae_logger_stats_for_rl['Train VAE Loss'] = float(np.mean(losses))
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
        else:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/loss", float(np.mean(losses)))
            if log_probs and kles:
                logger.record_tabular("train/log_prob", float(np.mean(log_probs)))
                logger.record_tabular("train/kl", float(np.mean(kles)))


    def test_epoch(self, epoch, save_reconstruction=True, save_scatterplot=True,
                   save_vae=True, from_rl=False):
        # Use parent test (recon only). You can extend with Hamilton metrics later.
        super().test_epoch(epoch, save_reconstruction, save_vae, from_rl)    
