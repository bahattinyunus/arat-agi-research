import pytest
import numpy as np
from autonomous_control.mppi_control import MPPIController

def test_mppi_rollout_dynamics():
    mppi = MPPIController(horizon=10, dt=0.1)
    init_pose = (0.0, 0.0, 0.0)
    u_seq = np.zeros((10, 2))
    u_seq[:, 0] = 1.0 # 1.0 m/s straight line

    traj = mppi.rollout_dynamics(init_pose, u_seq)
    assert traj.shape == (11, 3)
    # After 10 steps of dt=0.1 at v=1.0, x should reach 1.0m
    assert np.isclose(traj[-1, 0], 1.0, atol=1e-4)
    assert np.isclose(traj[-1, 1], 0.0, atol=1e-4)

def test_mppi_trajectory_cost_evaluation():
    mppi = MPPIController(horizon=5, dt=0.1)
    traj_clear = np.array([
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.6, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    goal = (1.0, 0.0)
    obstacles = [(0.4, 0.0)] # Obstacle directly on the trajectory

    cost_collision = mppi.compute_trajectory_cost(traj_clear, goal, obstacles, robot_radius=0.3)
    cost_no_obs = mppi.compute_trajectory_cost(traj_clear, goal, [], robot_radius=0.3)

    assert cost_collision > cost_no_obs + 400.0 # Collision penalty applied

def test_mppi_optimize_control_convergence():
    mppi = MPPIController(horizon=12, num_samples=50, dt=0.1)
    current_pose = (0.0, 0.0, 0.0)
    goal = (2.0, 0.0)
    obstacles = [(1.0, 0.0)] # Obstacle right in front

    v_cmd, w_cmd, best_traj = mppi.optimize_control(current_pose, goal, obstacles)

    assert v_cmd >= 0.0
    # Controller should turn to steer around obstacle rather than driving straight through
    assert abs(w_cmd) > 0.0
    assert best_traj.shape == (13, 3)
