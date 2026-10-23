import pytest
import numpy as np
from core_cognition.neuromorphic import (
    LIFNeuron,
    IzhikevichNeuron,
    PoissonSpikeEncoder,
    STDPSynapse,
    SpikingNeuralLayer
)

def test_lif_neuron_leak_and_spike():
    neuron = LIFNeuron(v_rest=-70.0, v_reset=-75.0, v_th=-50.0, tau_m=10.0, r_membrane=10.0)

    # Sub-threshold input: no spike
    s1 = neuron.step(i_syn=0.5, dt=1.0)
    assert s1 == 0
    assert neuron.v > -70.0 # Depolarized

    # Super-threshold input: should spike
    spiked = False
    for _ in range(10):
        if neuron.step(i_syn=5.0, dt=1.0) == 1:
            spiked = True
            break
    assert spiked
    # After spike, voltage resets to v_reset
    assert neuron.v == -75.0

def test_izhikevich_neuron():
    neuron = IzhikevichNeuron()
    # Apply continuous depolarizing current
    spikes = [neuron.step(current_i=10.0) for _ in range(100)]
    assert sum(spikes) > 0 # Must generate spikes

def test_poisson_spike_encoder():
    encoder = PoissonSpikeEncoder(max_freq_hz=100.0, dt_ms=1.0)
    inputs = np.array([0.0, 0.5, 1.0])
    trains = encoder.encode(inputs, duration_steps=100)

    assert trains.shape == (3, 100)
    # Zero input should produce near zero spikes
    assert np.sum(trains[0]) == 0
    # High input should produce more spikes than medium input
    assert np.sum(trains[2]) >= np.sum(trains[1])

def test_stdp_synapse_plasticity():
    synapse = STDPSynapse(num_pre=2, num_post=2, a_plus=0.05, a_minus=0.04)
    w_initial = np.copy(synapse.weights)

    # Pre neuron 0 fires, Post neuron 0 fires in next step (LTP)
    synapse.step(pre_spikes=np.array([1, 0]), post_spikes=np.array([0, 0]), dt=1.0)
    synapse.step(pre_spikes=np.array([0, 0]), post_spikes=np.array([1, 0]), dt=1.0)

    # Synapse (0, 0) should be potentiated
    assert synapse.weights[0, 0] > w_initial[0, 0]

def test_spiking_neural_layer_processing():
    layer = SpikingNeuralLayer(num_inputs=4, num_neurons=3, enable_plasticity=True)
    features = np.array([0.8, 0.2, 0.9, 0.1])

    res = layer.process_time_window(features, duration_steps=30, dt=1.0)
    assert res["output_spikes"].shape == (3, 30)
    assert len(res["spike_counts"]) == 3
    assert res["energy_picojoules"] >= 0.0
