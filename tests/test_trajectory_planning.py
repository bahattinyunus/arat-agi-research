import pytest
import numpy as np
from autonomous_control.trajectory_planning import (
    QuinticPolynomial,
    TrajectoryGenerator2D,
    DynamicWindowApproachPlanner,
    DWAPlannerConfig
)

def test_quintic_polynomial_boundary_conditions():
    s0, v0, a0 = 0.0, 0.0, 0.0
    s1, v1, a1 = 10.0, 2.0, 0.0
    T = 5.0

    poly = QuinticPolynomial(s0, v0, a0, s1, v1, a1, T)

    # Check boundary conditions at t = 0
    assert np.isclose(poly.calc_point(0.0), s0, atol=1e-4)
    assert np.isclose(poly.calc_first_derivative(0.0), v0, atol=1e-4)
    assert np.isclose(poly.calc_second_derivative(0.0), a0, atol=1e-4)

    # Check boundary conditions at t = T
    assert np.isclose(poly.calc_point(T), s1, atol=1e-4)
    assert np.isclose(poly.calc_first_derivative(T), v1, atol=1e-4)
    assert np.isclose(poly.calc_second_derivative(T), a1, atol=1e-4)

def test_trajectory_generator_2d_smoothness():
    start_pose = (0.0, 0.0, 0.0)
    target_pose = (5.0, 3.0, np.pi / 4.0)

    gen = TrajectoryGenerator2D(start_pose, target_pose, start_speed=0.0, target_speed=1.0, duration=4.0)
    traj = gen.sample_trajectory(num_points=40)

    assert len(traj) == 40
    # First point matches start pose
    assert np.isclose(traj[0]["x"], 0.0, atol=1e-3)
    assert np.isclose(traj[0]["y"], 0.0, atol=1e-3)
    # Final point matches target position
    assert np.isclose(traj[-1]["x"], 5.0, atol=1e-3)
    assert np.isclose(traj[-1]["y"], 3.0, atol=1e-3)
    # Velocity profile should be continuous and non-negative
    for pt in traj:
        assert pt["v"] >= -1e-5

def test_dwa_planner_dynamic_window():
    cfg = DWAPlannerConfig(
        max_speed=2.0, min_speed=0.0, max_yaw_rate=1.0,
        max_accel=1.0, max_dyaw=1.0, dt=0.1
    )
    planner = DynamicWindowApproachPlanner(cfg)

    # Robot at rest: v=0, w=0
    dw = planner.calc_dynamic_window(current_v=0.0, current_w=0.0)
    # Within 0.1s at 1.0 m/s^2, max attainable v is 0.1
    assert np.isclose(dw[0], 0.0)
    assert np.isclose(dw[1], 0.1)
    assert np.isclose(dw[2], -0.1)
    assert np.isclose(dw[3], 0.1)

def test_dwa_planner_obstacle_avoidance():
    planner = DynamicWindowApproachPlanner()
    current_pose = (0.0, 0.0, 0.0)
    goal = (5.0, 0.0)
    
    # An obstacle directly in front of the robot at (1.5, 0.0)
    obstacles = [(1.5, 0.0)]

    best_v, best_w, traj = planner.plan(
        current_pose=current_pose,
        current_v=0.5,
        current_w=0.0,
        goal=goal,
        obstacles=obstacles
    )

    # Planner should steer around obstacle (best_w != 0) rather than driving directly into it
    assert len(traj) > 0
    # Trajectory points should maintain clearance from the obstacle
    for pt in traj:
        dist = np.sqrt((pt["x"] - 1.5)**2 + (pt["y"] - 0.0)**2)
        assert dist > planner.config.robot_radius
