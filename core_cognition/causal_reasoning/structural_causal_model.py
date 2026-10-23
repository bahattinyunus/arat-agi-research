import numpy as np
from typing import Dict, Callable, Any, Optional, List
from .causal_graph import CausalDAG

class StructuralCausalModel:
    """
    Structural Causal Model (SCM) enabling Judea Pearl's Causal Hierarchy:
    Level 1: Association P(Y | X)
    Level 2: Intervention P(Y | do(X = x))
    Level 3: Counterfactuals P(Y_x | E = e) (Abduction -> Action -> Prediction)
    """
    def __init__(self):
        self.dag = CausalDAG()
        # Endogenous mechanisms: f_i(parents_dict, noise) -> float
        self.mechanisms: Dict[str, Callable[[Dict[str, float], float], float]] = {}
        # Invertible mechanism solvers for abduction: solve_noise(val, parents_dict) -> noise
        self.noise_solvers: Dict[str, Callable[[float, Dict[str, float]], float]] = {}
        # Noise distributions: callable() -> float
        self.noise_samplers: Dict[str, Callable[[], float]] = {}

    def add_variable(
        self,
        name: str,
        parents: List[str],
        mechanism: Callable[[Dict[str, float], float], float],
        noise_sampler: Callable[[], float] = lambda: float(np.random.normal(0, 1)),
        noise_solver: Optional[Callable[[float, Dict[str, float]], float]] = None
    ):
        """Adds an endogenous variable with its parents and causal generating mechanism."""
        self.dag.add_node(name)
        for p in parents:
            self.dag.add_edge(p, name)

        self.mechanisms[name] = mechanism
        self.noise_samplers[name] = noise_sampler

        # Default linear additive noise solver if none provided: noise = val - f(parents, 0)
        if noise_solver is not None:
            self.noise_solvers[name] = noise_solver
        else:
            self.noise_solvers[name] = lambda val, par, m=mechanism: val - m(par, 0.0)

    def sample_observational(self, num_samples: int = 1) -> Dict[str, np.ndarray]:
        """Draws observational samples from the generative causal model."""
        order = self.dag.topological_sort()
        samples: Dict[str, List[float]] = {var: [] for var in order}

        for _ in range(num_samples):
            current_values: Dict[str, float] = {}
            for var in order:
                u = self.noise_samplers[var]()
                val = self.mechanisms[var](current_values, u)
                current_values[var] = val
                samples[var].append(val)

        return {var: np.array(vals) for var, vals in samples.items()}

    def do_intervention(self, interventions: Dict[str, float], num_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Executes causal intervention do(X = x) via graph surgery (Pearl's Mutilation).
        Removes all incoming causal dependencies to intervened variables and fixes their values.
        """
        order = self.dag.topological_sort()
        samples: Dict[str, List[float]] = {var: [] for var in order}

        for _ in range(num_samples):
            current_values: Dict[str, float] = {}
            for var in order:
                if var in interventions:
                    val = float(interventions[var])
                else:
                    u = self.noise_samplers[var]()
                    val = self.mechanisms[var](current_values, u)
                current_values[var] = val
                samples[var].append(val)

        return {var: np.array(vals) for var, vals in samples.items()}

    def counterfactual_query(
        self,
        evidence: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        target_var: str
    ) -> float:
        """
        Executes Pearl's 3-Step Counterfactual Algorithm:
        Step 1 (Abduction): Compute exogenous noise U given factual evidence E = e.
        Step 2 (Action): Apply intervention do(X = x') to the structural equations.
        Step 3 (Prediction): Evaluate target variable Y using modified equations and abducted U.
        """
        order = self.dag.topological_sort()

        # Step 1: Abduction
        # Compute noise for each variable given observed evidence
        abducted_noise: Dict[str, float] = {}
        for var in order:
            val = evidence[var]
            parents_vals = {p: evidence[p] for p in self.dag.parents.get(var, set())}
            abducted_noise[var] = self.noise_solvers[var](val, parents_vals)

        # Step 2 & 3: Action & Prediction
        counterfactual_values: Dict[str, float] = {}
        for var in order:
            if var in counterfactual_intervention:
                val = float(counterfactual_intervention[var])
            else:
                u = abducted_noise[var]
                parents_vals = {p: counterfactual_values[p] for p in self.dag.parents.get(var, set())}
                val = self.mechanisms[var](parents_vals, u)
            counterfactual_values[var] = val

        return counterfactual_values[target_var]
