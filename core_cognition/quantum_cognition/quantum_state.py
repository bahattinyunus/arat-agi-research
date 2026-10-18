import numpy as np
from typing import List, Tuple, Optional

class QuantumCognitiveState:
    """
    Represents a quantum cognitive state in finite-dimensional complex Hilbert space C^d.
    Can represent pure states (|psi>) or mixed states via Density Matrix (rho).
    Implements von Neumann entropy, Born rule measurement, and state collapse.
    """
    def __init__(self, dimension: int, state_vector: Optional[np.ndarray] = None):
        self.dimension = dimension
        
        if state_vector is not None:
            # Initialize from pure state vector
            vec = np.array(state_vector, dtype=complex).flatten()
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            else:
                vec = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
            self.psi = vec
            self.rho = np.outer(self.psi, np.conj(self.psi))
        else:
            # Initialize as maximally mixed state (uniform epistemic uncertainty)
            self.psi = None
            self.rho = np.eye(dimension, dtype=complex) / float(dimension)

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray) -> 'QuantumCognitiveState':
        """Creates a quantum state directly from a density matrix rho."""
        dim = rho.shape[0]
        instance = cls(dimension=dim)
        # Enforce Hermiticity and Unit Trace
        rho_herm = (rho + np.conj(rho.T)) / 2.0
        trace_val = np.trace(rho_herm)
        if abs(trace_val) > 1e-12:
            rho_herm = rho_herm / trace_val
        instance.rho = rho_herm
        instance.psi = None
        return instance

    def is_valid(self) -> bool:
        """Checks if rho satisfies density matrix axioms: Hermiticity, Tr(rho)=1, positive semi-definite."""
        is_hermitian = np.allclose(self.rho, np.conj(self.rho.T), atol=1e-6)
        is_unit_trace = np.isclose(np.real(np.trace(self.rho)), 1.0, atol=1e-6)
        eigenvals = np.linalg.eigvalsh(self.rho)
        is_positive = np.all(eigenvals >= -1e-6)
        return bool(is_hermitian and is_unit_trace and is_positive)

    def purity(self) -> float:
        """Purity gamma = Tr(rho^2). Returns 1.0 for pure state, 1/d for maximally mixed."""
        return float(np.real(np.trace(np.dot(self.rho, self.rho))))

    def von_neumann_entropy(self) -> float:
        """Calculates von Neumann entropy: S(rho) = -Tr(rho * ln(rho)) = -sum(lambda_i * ln(lambda_i))."""
        eigenvals = np.linalg.eigvalsh(self.rho)
        # Filter non-zero positive eigenvalues
        pos_eigenvals = eigenvals[eigenvals > 1e-12]
        entropy = -np.sum(pos_eigenvals * np.log(pos_eigenvals))
        return float(np.real(entropy))

    def get_probabilities(self) -> np.ndarray:
        """Returns classical diagonal probability distribution P(s_i) = rho_ii."""
        probs = np.real(np.diag(self.rho))
        probs = np.clip(probs, 0.0, 1.0)
        return probs / (np.sum(probs) + 1e-12)

    def measure(self, measurement_operators: Optional[List[np.ndarray]] = None) -> Tuple[int, 'QuantumCognitiveState']:
        """
        Performs generalized projective or POVM measurement.
        If measurement_operators is None, measures in standard computational basis (|i><i|).
        Returns:
            (outcome_index, collapsed_post_measurement_state)
        """
        if measurement_operators is None:
            # Standard computational basis projectors
            measurement_operators = []
            for i in range(self.dimension):
                proj = np.zeros((self.dimension, self.dimension), dtype=complex)
                proj[i, i] = 1.0
                measurement_operators.append(proj)

        # Compute Born rule probabilities: P(m) = Tr(M_m * rho * M_m^\dagger)
        probs = []
        for M in measurement_operators:
            M_dag = np.conj(M.T)
            p = np.real(np.trace(M @ self.rho @ M_dag))
            probs.append(max(0.0, p))
            
        probs = np.array(probs)
        total_p = np.sum(probs)
        if total_p > 1e-12:
            probs = probs / total_p
        else:
            probs = np.ones(len(measurement_operators)) / len(measurement_operators)

        # Sample outcome according to Born rule
        outcome = int(np.random.choice(len(measurement_operators), p=probs))

        # Collapse state: rho' = (M_outcome * rho * M_outcome^\dagger) / Tr(...)
        M_selected = measurement_operators[outcome]
        collapsed_rho = M_selected @ self.rho @ np.conj(M_selected.T)
        norm = np.trace(collapsed_rho)
        if abs(norm) > 1e-12:
            collapsed_rho = collapsed_rho / norm
        else:
            collapsed_rho = np.eye(self.dimension, dtype=complex) / self.dimension

        return outcome, QuantumCognitiveState.from_density_matrix(collapsed_rho)


class LindbladCognitiveDynamics:
    r"""
    Simulates open quantum cognitive systems via the Lindblad Master Equation:
    d(rho)/dt = -i [H, rho] + sum_k gamma_k ( L_k rho L_k^\dagger - 0.5 {L_k^\dagger L_k, rho} )
    Models cognitive oscillations (deliberation) under environmental noise and decoherence.
    """
    def __init__(self, hamiltonian: np.ndarray, dissipation_operators: Optional[List[Tuple[float, np.ndarray]]] = None):
        """
        hamiltonian: Hermitian matrix H (d x d) representing internal cognitive evolution.
        dissipation_operators: list of tuples (gamma_k, L_k), where L_k are jump/decoherence operators.
        """
        self.H = np.array(hamiltonian, dtype=complex)
        self.dimension = self.H.shape[0]
        self.dissipation_operators = dissipation_operators or []

    def add_lindblad_operator(self, rate: float, operator: np.ndarray):
        """Adds a Lindblad jump operator with decay/decoherence rate."""
        self.dissipation_operators.append((float(rate), np.array(operator, dtype=complex)))

    def derivative(self, rho: np.ndarray) -> np.ndarray:
        """Computes d(rho)/dt according to Lindblad equation."""
        # Unitary component: -i [H, rho] = -i (H * rho - rho * H)
        commutator = self.H @ rho - rho @ self.H
        d_rho = -1j * commutator

        # Dissipative component: sum_k gamma_k ( L_k rho L_k^\dagger - 0.5 (L_k^\dagger L_k rho + rho L_k^\dagger L_k) )
        for gamma, L in self.dissipation_operators:
            L_dag = np.conj(L.T)
            L_dag_L = L_dag @ L
            anti_commutator = L_dag_L @ rho + rho @ L_dag_L
            dissipator = L @ rho @ L_dag - 0.5 * anti_commutator
            d_rho += gamma * dissipator

        return d_rho

    def step_rk4(self, state: QuantumCognitiveState, dt: float) -> QuantumCognitiveState:
        """
        Advances quantum density matrix by time step dt using 4th-order Runge-Kutta (RK4).
        Preserves trace normalization and Hermiticity.
        """
        rho = state.rho
        
        k1 = self.derivative(rho)
        k2 = self.derivative(rho + 0.5 * dt * k1)
        k3 = self.derivative(rho + 0.5 * dt * k2)
        k4 = self.derivative(rho + dt * k3)

        rho_next = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Enforce Hermiticity and Tr(rho) = 1
        rho_next = (rho_next + np.conj(rho_next.T)) / 2.0
        tr = np.trace(rho_next)
        if abs(tr) > 1e-12:
            rho_next = rho_next / tr

        return QuantumCognitiveState.from_density_matrix(rho_next)
