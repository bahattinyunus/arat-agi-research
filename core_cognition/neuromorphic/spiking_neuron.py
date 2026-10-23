import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.
    tau_m * dV/dt = -(V - V_rest) + R * I_syn
    Fires spike S(t) = 1 when V >= V_th, resets to V_reset, obeys refractory period.
    """
    def __init__(
        self,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        v_th: float = -50.0,
        tau_m: float = 20.0,
        r_membrane: float = 10.0,
        refractory_period: float = 2.0
    ):
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.tau_m = tau_m
        self.r_membrane = r_membrane
        self.refractory_period = refractory_period

        self.v = v_rest
        self.refractory_timer = 0.0
        self.spike_history: List[int] = []
        self.voltage_history: List[float] = []

    def step(self, i_syn: float, dt: float = 1.0) -> int:
        """Simulates one time step dt (ms) with input synaptic current i_syn (nA)."""
        spike = 0

        if self.refractory_timer > 0.0:
            self.refractory_timer -= dt
            self.v = self.v_reset
        else:
            # Euler integration of membrane potential
            dv = (-(self.v - self.v_rest) + self.r_membrane * i_syn) * (dt / self.tau_m)
            self.v += dv

            if self.v >= self.v_th:
                spike = 1
                self.v = self.v_reset
                self.refractory_timer = self.refractory_period

        self.spike_history.append(spike)
        self.voltage_history.append(self.v)
        return spike

    def reset(self):
        self.v = self.v_rest
        self.refractory_timer = 0.0
        self.spike_history.clear()
        self.voltage_history.clear()


class IzhikevichNeuron:
    """
    Izhikevich Simple Model of Spiking Neurons.
    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    if v >= 30: v <- c, u <- u + d
    Can produce regular spiking, fast spiking, bursting, or chattering depending on (a, b, c, d).
    """
    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.v = c
        self.u = b * c
        self.spike_history: List[int] = []
        self.v_history: List[float] = []

    def step(self, current_i: float, dt: float = 0.5) -> int:
        spike = 0
        
        # 0.5 * dt sub-step integration for numerical stability
        for _ in range(2):
            dv = (0.04 * (self.v ** 2) + 5.0 * self.v + 140.0 - self.u + current_i) * dt
            du = (self.a * (self.b * self.v - self.u)) * dt
            self.v += dv
            self.u += du

        if self.v >= 30.0:
            spike = 1
            self.v = self.c
            self.u += self.d

        self.spike_history.append(spike)
        self.v_history.append(self.v)
        return spike

    def reset(self):
        self.v = self.c
        self.u = self.b * self.c
        self.spike_history.clear()
        self.v_history.clear()


class PoissonSpikeEncoder:
    """
    Encodes continuous sensor floating-point inputs into Poisson spike trains.
    Rate proportional to normalized input amplitude.
    """
    def __init__(self, max_freq_hz: float = 100.0, dt_ms: float = 1.0):
        self.max_freq_hz = max_freq_hz
        self.dt_ms = dt_ms

    def encode(self, values: np.ndarray, duration_steps: int) -> np.ndarray:
        """
        values: 1D array of floats normalized between [0, 1]
        Returns: binary matrix of shape (len(values), duration_steps)
        """
        values = np.clip(values, 0.0, 1.0)
        num_inputs = len(values)
        spike_matrix = np.zeros((num_inputs, duration_steps), dtype=int)

        # Probability of spike per step: P = r * dt
        prob_per_step = (values * self.max_freq_hz * (self.dt_ms / 1000.0))
        
        for t in range(duration_steps):
            rand = np.random.uniform(0.0, 1.0, size=num_inputs)
            spike_matrix[:, t] = (rand < prob_per_step).astype(int)

        return spike_matrix
