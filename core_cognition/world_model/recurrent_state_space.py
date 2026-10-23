import numpy as np
from typing import Tuple, Dict, Any, Optional

class RecurrentStateSpaceModel:
    """
    Recurrent State Space Model (RSSM) for self-supervised predictive world modeling.
    Maintains:
      - Deterministic recurrent state: h_t = f(h_t-1, z_t-1, a_t-1)
      - Stochastic prior: p(z_t | h_t) ~ N(mu_prior, sigma_prior)
      - Stochastic posterior: q(z_t | h_t, x_t) ~ N(mu_post, sigma_post)
      - Observation decoder: x_hat_t = g(h_t, z_t)
    """
    def __init__(self, obs_dim: int, action_dim: int, deter_dim: int = 32, stoch_dim: int = 16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # Deterministic transition weights (GRU-like transition)
        self.W_h = np.random.randn(deter_dim, deter_dim + stoch_dim + action_dim) * 0.05
        self.b_h = np.zeros(deter_dim)

        # Prior distribution network weights: (mu, log_std) from h
        self.W_prior_mu = np.random.randn(stoch_dim, deter_dim) * 0.05
        self.W_prior_log_std = np.random.randn(stoch_dim, deter_dim) * 0.05

        # Posterior distribution network weights: (mu, log_std) from h and obs
        self.W_post_mu = np.random.randn(stoch_dim, deter_dim + obs_dim) * 0.05
        self.W_post_log_std = np.random.randn(stoch_dim, deter_dim + obs_dim) * 0.05

        # Observation decoder weights: reconstructs obs from (h, z)
        self.W_dec = np.random.randn(obs_dim, deter_dim + stoch_dim) * 0.05
        self.b_dec = np.zeros(obs_dim)

        # Current internal states
        self.h = np.zeros(deter_dim)
        self.z = np.zeros(stoch_dim)

    def transition_deterministic(self, prev_h: np.ndarray, prev_z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Computes next deterministic state h_t = tanh(W_h * [h, z, a] + b)."""
        concat = np.concatenate([prev_h, prev_z, action])
        h_next = np.tanh(np.dot(self.W_h, concat) + self.b_h)
        return h_next

    def get_prior(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates prior Gaussian parameters mu_prior and std_prior from h."""
        mu = np.dot(self.W_prior_mu, h)
        log_std = np.clip(np.dot(self.W_prior_log_std, h), -5.0, 2.0)
        std = np.exp(log_std)
        return mu, std

    def get_posterior(self, h: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates posterior Gaussian parameters mu_post and std_post from h and obs."""
        concat = np.concatenate([h, obs])
        mu = np.dot(self.W_post_mu, concat)
        log_std = np.clip(np.dot(self.W_post_log_std, concat), -5.0, 2.0)
        std = np.exp(log_std)
        return mu, std

    def sample_stochastic(self, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Reparameterization trick: z = mu + std * eps."""
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode_observation(self, h: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Reconstructs observation from latent states: x_hat = W_dec * [h, z] + b."""
        concat = np.concatenate([h, z])
        return np.dot(self.W_dec, concat) + self.b_dec

    def step(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """
        Executes one predictive step given actual observation and executed action.
        Computes prediction, posterior update, reconstruction, and KL divergence.
        """
        # 1. Update deterministic state
        self.h = self.transition_deterministic(self.h, self.z, action)

        # 2. Get prior and posterior
        mu_prior, std_prior = self.get_prior(self.h)
        mu_post, std_post = self.get_posterior(self.h, obs)

        # 3. Sample posterior stochastic state
        self.z = self.sample_stochastic(mu_post, std_post)

        # 4. Decode reconstruction
        obs_recon = self.decode_observation(self.h, self.z)

        # 5. Calculate analytical KL Divergence between two Gaussians:
        # KL(q(post) || p(prior)) = sum( log(std_prior/std_post) + (std_post^2 + (mu_post - mu_prior)^2)/(2*std_prior^2) - 0.5 )
        kl = np.sum(
            np.log(std_prior / (std_post + 1e-12)) +
            (std_post**2 + (mu_post - mu_prior)**2) / (2.0 * std_prior**2 + 1e-12) - 0.5
        )
        kl = max(0.0, float(kl))

        recon_error = float(np.mean((obs - obs_recon) ** 2))

        return {
            "h": self.h,
            "z": self.z,
            "mu_prior": mu_prior,
            "mu_post": mu_post,
            "reconstruction": obs_recon,
            "recon_error": recon_error,
            "kl_divergence": kl
        }

    def rollout_imagination(self, action_sequence: np.ndarray) -> np.ndarray:
        """
        Imagines future observations purely in latent space without actual environment interaction.
        action_sequence: shape (horizon, action_dim)
        Returns imagined observation trajectory of shape (horizon, obs_dim).
        """
        horizon = len(action_sequence)
        imagined_obs = np.zeros((horizon, self.obs_dim))

        h_sim = np.copy(self.h)
        z_sim = np.copy(self.z)

        for t in range(horizon):
            h_sim = self.transition_deterministic(h_sim, z_sim, action_sequence[t])
            mu_prior, std_prior = self.get_prior(h_sim)
            z_sim = mu_prior  # Use mean for deterministic planning
            imagined_obs[t] = self.decode_observation(h_sim, z_sim)

        return imagined_obs

    def reset(self):
        self.h = np.zeros(self.deter_dim)
        self.z = np.zeros(self.stoch_dim)
