import pytest
import numpy as np
from autonomous_control.sensor_fusion.unscented_kalman_filter import (
    UnscentedKalmanFilter,
    UKFSensorFusionNode
)
from autonomous_control.sensor_fusion.ekf_fusion_node import EKFSensorFusionNode

def test_unscented_kalman_filter_sigma_point_generation():
    ukf = UnscentedKalmanFilter(n=3, z=2, alpha=1e-3, beta=2.0, kappa=0.0)
    ukf.x = np.array([[1.0], [2.0], [0.5]])
    ukf.P = np.diag([0.1, 0.2, 0.05])

    sigmas = ukf.generate_sigma_points(ukf.x, ukf.P)
    # Expected number of sigma points: 2n + 1 = 7
    assert sigmas.shape == (3, 7)
    # Center sigma point should equal the mean x
    assert np.allclose(sigmas[:, 0].reshape((3, 1)), ukf.x)
    # Weights sum to 1
    assert np.isclose(np.sum(ukf.wm), 1.0, atol=1e-5)

def test_ukf_prediction_and_update_linear():
    ukf = UnscentedKalmanFilter(n=2, z=1)
    ukf.x = np.array([[0.0], [1.0]]) # [pos, vel]
    ukf.P = np.eye(2) * 0.1
    ukf.Q = np.eye(2) * 0.01
    ukf.R = np.eye(1) * 0.05

    def f_linear(x, u):
        dt = 0.1
        F = np.array([[1.0, dt], [0.0, 1.0]])
        return F @ x

    def h_linear(x):
        return np.array([x[0, 0]])

    x_pred, P_pred = ukf.predict(f_linear)
    assert np.isclose(x_pred[0, 0], 0.1, atol=1e-4)

    # Measurement z = 0.12
    x_upd, P_upd = ukf.update(np.array([0.12]), h_linear)
    # Pos must be refined towards measurement
    assert 0.09 < x_upd[0, 0] < 0.13
    # Covariance should decrease after measurement
    assert P_upd[0, 0] < P_pred[0, 0]

def test_ukf_sensor_fusion_node_convergence():
    node = UKFSensorFusionNode(dt=0.1, initial_pose=(0.0, 0.0, 0.0))
    
    # Drive straight with v=1.0 m/s
    for i in range(10):
        meas = (float(0.1 * (i + 1)), 0.0) if i % 2 == 0 else None
        est = node.process_step(linear_vel=1.0, angular_vel=0.0, meas_pos=meas)

    est = node.get_state_estimate()
    # At t=1.0s, x should be around 1.0m, y around 0.0, theta around 0.0
    assert np.isclose(est["x"], 1.0, atol=0.15)
    assert np.isclose(est["y"], 0.0, atol=0.1)
    assert np.isclose(est["theta"], 0.0, atol=0.1)
    assert all(var > 0.0 for var in est["covariance"])

def test_ukf_vs_ekf_consistency():
    dt = 0.1
    ekf_node = EKFSensorFusionNode(dt=dt, initial_pose=(0.0, 0.0, 0.0))
    ukf_node = UKFSensorFusionNode(dt=dt, initial_pose=(0.0, 0.0, 0.0))

    # Drive along a curve: v=1.5 m/s, w=0.3 rad/s
    for step in range(8):
        v = 1.5
        w = 0.3
        meas = (step * 0.14, step * 0.02) if step % 2 == 0 else None
        
        ekf_est = ekf_node.process_step(linear_vel=v, angular_vel=w, meas_pos=meas)
        ukf_est = ukf_node.process_step(linear_vel=v, angular_vel=w, meas_pos=meas)

        # Compare estimates: EKF and UKF should track closely
        assert np.isclose(ekf_est["x"], ukf_est["x"], atol=0.2)
        assert np.isclose(ekf_est["y"], ukf_est["y"], atol=0.2)
