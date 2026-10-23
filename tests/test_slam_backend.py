import pytest
import numpy as np
from autonomous_control.slam_backend import PoseGraphOptimizer

def test_pose_graph_node_and_edge_addition():
    graph = PoseGraphOptimizer()
    n0 = graph.add_node((0.0, 0.0, 0.0))
    n1 = graph.add_node((1.0, 0.0, 0.0))
    assert n0 == 0
    assert n1 == 1

    graph.add_edge(n0, n1, (1.0, 0.0, 0.0))
    assert len(graph.edges) == 1

def test_pose_graph_residual_and_jacobian_computation():
    graph = PoseGraphOptimizer()
    graph.add_node((0.0, 0.0, 0.0))
    graph.add_node((1.1, 0.0, 0.0)) # 0.1m discrepancy from measurement

    edge = graph.edges or [None]
    graph.add_edge(0, 1, (1.0, 0.0, 0.0))

    e, J_i, J_j = graph.compute_edge_residual_and_jacobian(graph.edges[0])
    assert np.isclose(e[0], 0.1, atol=1e-4) # Error in x
    assert J_i.shape == (3, 3)
    assert J_j.shape == (3, 3)

def test_pose_graph_loop_closure_gauss_newton_optimization():
    graph = PoseGraphOptimizer()
    # Create a 4-node square loop: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    # Add noisy initial node poses with accumulated drift
    graph.add_node((0.0, 0.0, 0.0))       # Pose 0 (Fixed anchor)
    graph.add_node((1.05, 0.02, 0.0))     # Pose 1
    graph.add_node((1.08, 1.05, np.pi/2)) # Pose 2
    graph.add_node((0.04, 1.03, np.pi))   # Pose 3

    # Odometry edges
    graph.add_edge(0, 1, (1.0, 0.0, 0.0))
    graph.add_edge(1, 2, (0.0, 1.0, np.pi/2))
    graph.add_edge(2, 3, (0.0, 1.0, np.pi/2))
    # Loop closure edge from Pose 3 to Pose 0 (relative dx=0, dy=1, dtheta=-pi in Pose 3 frame)
    graph.add_edge(3, 0, (0.0, 1.0, -np.pi), information_diag=(200.0, 200.0, 1000.0))

    # Optimize graph
    final_error = graph.optimize(max_iterations=10)
    poses = graph.get_poses()

    # Anchor pose 0 must remain at origin
    assert np.isclose(poses[0][0], 0.0, atol=1e-3)
    assert np.isclose(poses[0][1], 0.0, atol=1e-3)

    # Pose 1 should converge close to (1.0, 0.0)
    assert np.isclose(poses[1][0], 1.0, atol=0.08)
    assert np.isclose(poses[1][1], 0.0, atol=0.08)
