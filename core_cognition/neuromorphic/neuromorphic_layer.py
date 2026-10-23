import numpy as np
from typing import List, Dict, Any, Optional
from .spiking_neuron import LIFNeuron, PoissonSpikeEncoder
from .stdp_synapse import STDPSynapse

class SpikingNeuralLayer:
    """
    Neuromorphic computing layer combining Poisson sensory encoding,
    plastic STDP synapses, and Leaky Integrate-and-Fire neurons.
    Operates on event-driven principles with ultra-low dynamic power dissipation.
    """
    def __init__(self, num_inputs: int, num_neurons: int, enable_plasticity: bool = True):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.enable_plasticity = enable_plasticity

        self.encoder = PoissonSpikeEncoder(max_freq_hz=120.0, dt_ms=1.0)
        self.synapses = STDPSynapse(num_pre=num_inputs, num_post=num_neurons)
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]

    def process_time_window(self, input_features: np.ndarray, duration_steps: int = 20, dt: float = 1.0) -> Dict[str, Any]:
        """
        Processes continuous input features through the spiking layer over a temporal window.
        Returns:
            - output_spikes: binary matrix (num_neurons, duration_steps)
            - spike_counts: 1D array of total spikes per neuron
            - firing_rates_hz: average firing rates
            - energy_metric: relative dynamic energy consumption proportional to spike count
        """
        # 1. Encode sensory stream to spike trains
        input_spikes = self.encoder.encode(input_features, duration_steps)

        output_spikes = np.zeros((self.num_neurons, duration_steps), dtype=int)
        
        for t in range(duration_steps):
            pre_t = input_spikes[:, t]

            # 2. Compute synaptic current
            synaptic_currents = self.synapses.compute_current(pre_t)

            # 3. Step individual LIF neurons
            post_t = np.zeros(self.num_neurons, dtype=int)
            for i, neuron in enumerate(self.neurons):
                post_t[i] = neuron.step(i_syn=float(synaptic_currents[i]), dt=dt)
            output_spikes[:, t] = post_t

            # 4. Apply STDP plasticity if enabled
            if self.enable_plasticity:
                self.synapses.step(pre_spikes=pre_t, post_spikes=post_t, dt=dt)

        spike_counts = np.sum(output_spikes, axis=1)
        duration_s = (duration_steps * dt) / 1000.0
        firing_rates = spike_counts / max(1e-5, duration_s)

        # Dynamic energy metric: in biological/neuromorphic circuits (e.g. TrueNorth, Loihi),
        # static power is minimal and energy is dissipated primarily per spike event (pJ/spike).
        total_spikes = int(np.sum(input_spikes) + np.sum(output_spikes))
        energy_pj = total_spikes * 2.5 # approx 2.5 picojoules per synaptic event

        return {
            "output_spikes": output_spikes,
            "spike_counts": spike_counts,
            "firing_rates_hz": firing_rates,
            "total_spikes": total_spikes,
            "energy_picojoules": energy_pj,
            "synaptic_weights": self.synapses.weights
        }

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()
