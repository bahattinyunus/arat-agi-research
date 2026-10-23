import numpy as np
from typing import Dict, Any, Optional

class EpistemicCuriosityEngine:
    """
    Epistemic Curiosity Engine computing intrinsic information gain and Bayesian surprise.
    Drives autonomous exploration according to Karl Friston's epistemic value and Schmidhuber's curiosity:
    R_intrinsic = beta_kl * D_KL(q(z | h, x) || p(z | h)) + beta_recon * ||x - x_hat||^2
    """
    def __init__(self, beta_kl: float = 1.0, beta_recon: float = 0.5, novelty_decay: float = 0.99):
        self.beta_kl = beta_kl
        self.beta_recon = beta_recon
        self.novelty_decay = novelty_decay

        self.cumulative_intrinsic_reward = 0.0
        self.curiosity_history = []

    def compute_intrinsic_reward(self, kl_divergence: float, recon_error: float) -> float:
        """
        Calculates intrinsic exploration reward.
        High KL divergence indicates significant model revision (information gain).
        High reconstruction error indicates novel, unmodeled sensory input.
        """
        r_info_gain = self.beta_kl * kl_divergence
        r_surprise = self.beta_recon * recon_error
        r_intrinsic = r_info_gain + r_surprise

        self.cumulative_intrinsic_reward = (
            self.cumulative_intrinsic_reward * self.novelty_decay + r_intrinsic
        )
        self.curiosity_history.append(r_intrinsic)

        return float(r_intrinsic)

    def blend_reward(self, extrinsic_reward: float, intrinsic_reward: float, curiosity_weight: float = 0.3) -> float:
        """Blends goal-directed extrinsic reward with exploratory intrinsic curiosity."""
        curiosity_weight = np.clip(curiosity_weight, 0.0, 1.0)
        return float((1.0 - curiosity_weight) * extrinsic_reward + curiosity_weight * intrinsic_reward)
