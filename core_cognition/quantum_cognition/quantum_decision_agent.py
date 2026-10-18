import numpy as np
from typing import List, Dict, Any, Optional
from .quantum_state import QuantumCognitiveState, LindbladCognitiveDynamics

class QuantumDecisionAgent:
    """
    Quantum Decision Agent modeling cognitive processes with quantum probability amplitudes.
    Exhibits quantum interference, belief superposition, and context-dependent measurement collapse.
    Interfaces directly with Active Inference agents and the Epistemic Message Broker.
    """
    def __init__(self, state_labels: List[str], oscillation_freq: float = 1.0, decoherence_rate: float = 0.05):
        self.state_labels = state_labels
        self.dimension = len(state_labels)
        
        # Initial superposition state (equal distribution across hypotheses)
        init_vec = np.ones(self.dimension, dtype=complex) / np.sqrt(self.dimension)
        self.state = QuantumCognitiveState(dimension=self.dimension, state_vector=init_vec)
        
        # Build Cognitive Hamiltonian: diagonal = state energies / biases, off-diagonal = cognitive coupling
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        for i in range(self.dimension):
            H[i, i] = 0.5 * (i + 1) # Potential energy gradient
            for j in range(i + 1, self.dimension):
                coupling = oscillation_freq / (abs(i - j) + 1.0)
                H[i, j] = coupling
                H[j, i] = coupling
        self.hamiltonian = H
        
        # Dynamics engine
        self.dynamics = LindbladCognitiveDynamics(hamiltonian=self.hamiltonian)
        
        # Add dephasing / decoherence Lindblad operators (environmental / sensory noise)
        for i in range(self.dimension):
            L_dephase = np.zeros((self.dimension, self.dimension), dtype=complex)
            L_dephase[i, i] = 1.0
            self.dynamics.add_lindblad_operator(rate=decoherence_rate, operator=L_dephase)

    def evolve(self, dt: float, steps: int = 1) -> QuantumCognitiveState:
        """Evolves the internal cognitive state over time step dt."""
        for _ in range(steps):
            self.state = self.dynamics.step_rk4(self.state, dt)
        return self.state

    def inject_evidence_potential(self, state_idx: int, strength: float = 2.0):
        """
        Shifts the cognitive Hamiltonian to favor a specific state hypothesis,
        analogous to sensory evidence tilting cognitive potential well.
        """
        if 0 <= state_idx < self.dimension:
            self.hamiltonian[state_idx, state_idx] -= strength
            self.dynamics.H = self.hamiltonian

    def project_decision(self) -> Dict[str, Any]:
        """
        Performs quantum measurement / decision collapse according to the Born rule.
        Returns the chosen decision label, index, and post-measurement state entropy.
        """
        outcome_idx, collapsed_state = self.state.measure()
        chosen_label = self.state_labels[outcome_idx]
        self.state = collapsed_state

        return {
            "decision_index": outcome_idx,
            "decision_label": chosen_label,
            "purity": self.state.purity(),
            "entropy": self.state.von_neumann_entropy(),
            "probabilities": self.state.get_probabilities().tolist()
        }

    def get_epistemic_belief(self) -> np.ndarray:
        """
        Extracts classical probability distribution q(s) for Active Inference integration.
        q(s) = diag(rho)
        """
        return self.state.get_probabilities()

    def test_order_effect(self, question_a_idx: int, question_b_idx: int) -> Dict[str, float]:
        """
        Demonstrates quantum cognitive non-commutativity: P(A then B) != P(B then A).
        Constructs projection operators for questions A and B, testing order dependence.
        """
        P_A = np.zeros((self.dimension, self.dimension), dtype=complex)
        P_A[question_a_idx, question_a_idx] = 1.0

        P_B = np.zeros((self.dimension, self.dimension), dtype=complex)
        P_B[question_b_idx, question_b_idx] = 1.0

        rho = self.state.rho

        # Order A then B: P(B | A) * P(A) = Tr(P_B P_A rho P_A P_B)
        p_A_then_B = np.real(np.trace(P_B @ P_A @ rho @ P_A @ P_B))

        # Order B then A: P(A | B) * P(B) = Tr(P_A P_B rho P_B P_A)
        p_B_then_A = np.real(np.trace(P_A @ P_B @ rho @ P_B @ P_A))

        return {
            "p_A_then_B": float(p_A_then_B),
            "p_B_then_A": float(p_B_then_A),
            "discrepancy": float(abs(p_A_then_B - p_B_then_A))
        }
