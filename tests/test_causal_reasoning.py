import pytest
import numpy as np
from core_cognition.causal_reasoning import CausalDAG, StructuralCausalModel

def test_causal_dag_topological_sort_and_ancestors():
    dag = CausalDAG()
    dag.add_edge("X", "Y")
    dag.add_edge("Z", "X")
    dag.add_edge("Z", "Y")

    order = dag.topological_sort()
    assert order.index("Z") < order.index("X")
    assert order.index("X") < order.index("Y")

    ancestors_y = dag.get_ancestors("Y")
    assert ancestors_y == {"Z", "X", "Y"}

def test_d_separation_chain_fork_collider():
    dag = CausalDAG()
    # Chain: A -> B -> C
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    # Fork: D <- B -> E
    dag.add_edge("B", "D")
    dag.add_edge("B", "E")
    # Collider: D -> F <- E
    dag.add_edge("D", "F")
    dag.add_edge("E", "F")

    # In chain A -> B -> C, conditioning on B d-separates A and C
    assert dag.is_d_separated("A", "C", conditioning_set={"B"})
    assert not dag.is_d_separated("A", "C", conditioning_set=set())

    # In fork D <- B -> E, conditioning on common cause B d-separates D and E
    # (if collider F is not conditioned on)
    assert dag.is_d_separated("D", "E", conditioning_set={"B"})
    # Without conditioning on B, the fork path D <- B -> E is active
    assert not dag.is_d_separated("D", "E", conditioning_set=set())

    # Conditioning on collider F activates the path D -> F <- E (Berkson's Paradox)
    assert not dag.is_d_separated("D", "E", conditioning_set={"B", "F"})

def test_scm_observational_and_intervention():
    # Confounded model:
    # Z (weather) -> X (sprinkler), Z -> Y (wet ground)
    # X -> Y
    scm = StructuralCausalModel()

    # Z = U_z (binary-like 0 or 1 with continuous noise)
    scm.add_variable(
        name="Z",
        parents=[],
        mechanism=lambda par, u: 1.0 if u > 0.0 else 0.0,
        noise_sampler=lambda: float(np.random.normal(0, 1))
    )

    # X = 0.8 * Z + U_x
    scm.add_variable(
        name="X",
        parents=["Z"],
        mechanism=lambda par, u: float(0.8 * par["Z"] + u),
        noise_sampler=lambda: float(np.random.normal(0, 0.1))
    )

    # Y = 2.0 * X + 1.5 * Z + U_y
    scm.add_variable(
        name="Y",
        parents=["X", "Z"],
        mechanism=lambda par, u: float(2.0 * par["X"] + 1.5 * par["Z"] + u),
        noise_sampler=lambda: float(np.random.normal(0, 0.1))
    )

    # 1. Observational
    obs = scm.sample_observational(num_samples=20)
    assert len(obs["X"]) == 20
    assert len(obs["Y"]) == 20

    # 2. Interventional do(X = 3.0)
    # In graph mutilation, X no longer depends on Z
    interv = scm.do_intervention({"X": 3.0}, num_samples=20)
    assert np.allclose(interv["X"], 3.0)
    # Y should average around 2.0 * 3.0 + 1.5 * E[Z] = 6.0 + 0.75 = 6.75
    mean_y = np.mean(interv["Y"])
    assert 6.0 < mean_y < 7.5

def test_scm_counterfactual_abduction_action_prediction():
    # Linear structural model:
    # X = U_x
    # Y = 3 * X + U_y
    scm = StructuralCausalModel()

    scm.add_variable(
        name="X",
        parents=[],
        mechanism=lambda par, u: float(u),
        noise_sampler=lambda: 2.0,
        noise_solver=lambda val, par: val
    )

    scm.add_variable(
        name="Y",
        parents=["X"],
        mechanism=lambda par, u: float(3.0 * par["X"] + u),
        noise_sampler=lambda: 1.0,
        noise_solver=lambda val, par: val - 3.0 * par["X"]
    )

    # Factual evidence: X = 2, Y = 7 (so abducted U_y = 7 - 3*2 = 1)
    evidence = {"X": 2.0, "Y": 7.0}

    # Counterfactual question: What would Y have been if X had been 5?
    # Prediction: Y = 3 * 5 + U_y = 15 + 1 = 16
    cf_y = scm.counterfactual_query(
        evidence=evidence,
        counterfactual_intervention={"X": 5.0},
        target_var="Y"
    )

    assert np.isclose(cf_y, 16.0, atol=1e-5)
