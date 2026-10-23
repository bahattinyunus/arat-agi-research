import numpy as np
from typing import Optional, Tuple

class STDPSynapse:
    """
    Spike-Timing-Dependent Plasticity (STDP) Synapse Matrix.
    Models biological Hebbian plasticity between pre-synaptic and post-synaptic neurons:
    Delta_w = A_plus * exp(-Delta_t / tau_plus) if Delta_t > 0 (LTP: pre before post)
    Delta_w = -A_minus * exp(Delta_t / tau_minus) if Delta_t < 0 (LTD: post before pre)
    """
    def __init__(
        self,
        num_pre: int,
        num_post: int,
        w_min: float = 0.0,
        w_max: float = 1.0,
        a_plus: float = 0.015,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        init_weight_mean: float = 0.3
    ):
        self.num_pre = num_pre
        self.num_post = num_post
        self.w_min = w_min
        self.w_max = w_max
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        # Initialize weights with normal distribution around mean
        self.weights = np.clip(
            np.random.normal(init_weight_mean, 0.05, size=(num_pre, num_post)),
            w_min,
            w_max
        )

        # Pre and post synaptic spike traces
        self.pre_trace = np.zeros(num_pre)
        self.post_trace = np.zeros(num_post)

    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Updates traces and synaptic weights based on spike events.
        pre_spikes: 1D binary vector of size num_pre
        post_spikes: 1D binary vector of size num_post
        """
        # Decay traces
        self.pre_trace *= np.exp(-dt / self.tau_plus)
        self.post_trace *= np.exp(-dt / self.tau_minus)

        # Increment traces on new spikes
        self.pre_trace += pre_spikes.astype(float)
        self.post_trace += post_spikes.astype(float)

        # LTP: Post-synaptic spike occurs -> increase weights from recently active pre-synaptic neurons
        # Delta_w += A_plus * pre_trace (when post fires)
        post_mask = (post_spikes > 0).astype(float)
        if np.any(post_mask):
            delta_ltp = self.a_plus * np.outer(self.pre_trace, post_mask)
            self.weights += delta_ltp

        # LTD: Pre-synaptic spike occurs -> decrease weights to recently active post-synaptic neurons
        # Delta_w -= A_minus * post_trace (when pre fires)
        pre_mask = (pre_spikes > 0).astype(float)
        if np.any(pre_mask):
            delta_ltd = self.a_minus * np.outer(pre_mask, self.post_trace)
            self.weights -= delta_ltd

        # Enforce weight bounds
        self.weights = np.clip(self.weights, self.w_min, self.w_max)
        return self.weights

    def compute_current(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Computes post-synaptic input current I = W^T * pre_spikes."""
        return np.dot(self.weights.T, pre_spikes)
