import pytest
import numpy as np
from core_cognition.quantum_cognition import (
    QuantumCognitiveState,
    LindbladCognitiveDynamics,
    QuantumDecisionAgent
)

def test_quantum_state_initialization_pure_and_mixed():
    # Pure state initialization
    pure_vec = [1.0, 0.0]
    state_pure = QuantumCognitiveState(dimension=2, state_vector=pure_vec)
    assert state_pure.is_valid()
    assert np.isclose(state_pure.purity(), 1.0)
    assert np.isclose(state_pure.von_neumann_entropy(), 0.0, atol=1e-5)

    # Maximally mixed state initialization
    state_mixed = QuantumCognitiveState(dimension=2)
    assert state_mixed.is_valid()
    assert np.isclose(state_mixed.purity(), 0.5)
    assert np.isclose(state_mixed.von_neumann_entropy(), np.log(2.0), atol=1e-5)

def test_born_rule_measurement_and_collapse():
    # Superposition |+> = (|0> + |1>) / sqrt(2)
    plus_vec = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
    state = QuantumCognitiveState(dimension=2, state_vector=plus_vec)
    
    probs = state.get_probabilities()
    assert np.isclose(probs[0], 0.5, atol=1e-4)
    assert np.isclose(probs[1], 0.5, atol=1e-4)

    outcome, collapsed = state.measure()
    assert outcome in [0, 1]
    assert collapsed.is_valid()
    assert np.isclose(collapsed.purity(), 1.0, atol=1e-5)
    
    # After collapse, measured outcome probability must be 1.0
    post_probs = collapsed.get_probabilities()
    assert np.isclose(post_probs[outcome], 1.0, atol=1e-5)

def test_lindblad_open_system_dynamics_trace_preservation():
    # Construct a 2-level cognitive system
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    dynamics = LindbladCognitiveDynamics(hamiltonian=H)

    # Add dephasing Lindblad jump operator
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    dynamics.add_lindblad_operator(rate=0.2, operator=sigma_z)

    init_state = QuantumCognitiveState(dimension=2, state_vector=[1.0, 0.0])
    
    current_state = init_state
    for _ in range(20):
        current_state = dynamics.step_rk4(current_state, dt=0.05)
        assert current_state.is_valid()
        # Trace must remain exactly 1.0 throughout open dynamics
        assert np.isclose(np.real(np.trace(current_state.rho)), 1.0, atol=1e-5)

    # Under dephasing, off-diagonal coherence decays, entropy increases
    assert current_state.von_neumann_entropy() > init_state.von_neumann_entropy()

def test_quantum_decision_agent_evolution_and_epistemic_belief():
    hypotheses = ["ExploreArea", "TargetObject", "AvoidObstacle"]
    agent = QuantumDecisionAgent(state_labels=hypotheses, oscillation_freq=1.5, decoherence_rate=0.1)

    init_belief = agent.get_epistemic_belief()
    assert len(init_belief) == 3
    assert np.isclose(np.sum(init_belief), 1.0)

    # Evolve cognitive mind
    agent.evolve(dt=0.1, steps=5)
    evolved_belief = agent.get_epistemic_belief()
    assert len(evolved_belief) == 3
    assert np.isclose(np.sum(evolved_belief), 1.0)

    # Inject sensory evidence favoring TargetObject (index 1)
    agent.inject_evidence_potential(state_idx=1, strength=3.0)
    agent.evolve(dt=0.1, steps=10)

    # Collapse decision
    decision = agent.project_decision()
    assert decision["decision_label"] in hypotheses
    assert decision["decision_index"] in [0, 1, 2]
    assert np.isclose(decision["purity"], 1.0, atol=1e-4)

def test_quantum_order_effect_non_commutativity():
    hypotheses = ["A", "B", "C"]
    agent = QuantumDecisionAgent(state_labels=hypotheses, oscillation_freq=2.0, decoherence_rate=0.05)
    
    # Evolve into a coherent superposition state
    agent.evolve(dt=0.1, steps=8)

    order_res = agent.test_order_effect(question_a_idx=0, question_b_idx=1)
    assert "p_A_then_B" in order_res
    assert "p_B_then_A" in order_res
    assert order_res["p_A_then_B"] >= 0.0
    assert order_res["p_B_then_A"] >= 0.0
