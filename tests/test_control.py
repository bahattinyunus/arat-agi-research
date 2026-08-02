import unittest
import numpy as np
from autonomous_control.sensor_fusion import KalmanFilter, ExtendedKalmanFilter, SensorFusionNode
from autonomous_control.kinematic_loops import PIDController, DifferentialDriveKinematics

class TestControl(unittest.TestCase):
    def test_kalman_filter_tracking(self):
        kf = KalmanFilter(n=2, z=1)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[0.1]])
        
        kf.x = np.array([[0.0], [1.0]])
        
        x_pred, _ = kf.predict()
        self.assertAlmostEqual(x_pred[0, 0], 1.0)
        
        x_upd, _ = kf.update([1.1])
        self.assertTrue(x_upd[0, 0] > 1.0 and x_upd[0, 0] <= 1.1)

    def test_extended_kalman_filter(self):
        # 1D State: state x = [x_pos]^T (dimension=1)
        # Non-linear process: f(x, u) = x + 0.1 * sin(x) + u
        # Jacobian F_j = 1 + 0.1 * cos(x)
        # Non-linear measurement: h(x) = x^2
        # Jacobian H_j = 2 * x
        
        ekf = ExtendedKalmanFilter(n=1, z=1)
        ekf.Q = np.array([[0.01]])
        ekf.R = np.array([[0.05]])
        ekf.x = np.array([[1.0]]) # position 1.0
        
        def f_func(x, u):
            val = x[0, 0]
            ctrl = u[0, 0] if u is not None else 0.0
            return np.array([[val + 0.1 * np.sin(val) + ctrl]])
            
        def F_jac(x, u):
            val = x[0, 0]
            return np.array([[1.0 + 0.1 * np.cos(val)]])
            
        def h_func(x):
            val = x[0, 0]
            return np.array([[val ** 2]])
            
        def H_jac(x):
            val = x[0, 0]
            return np.array([[2.0 * val]])
            
        # Run prediction
        ekf.predict_nonlinear(f_func, F_jac, u=np.array([[0.5]]))
        # x_next = 1.0 + 0.1 * sin(1) + 0.5 = 1.0 + 0.0841 + 0.5 = 1.5841
        self.assertAlmostEqual(ekf.x[0, 0], 1.584147, delta=1e-4)
        
        # Run update with measurement z = 2.5 (should be close to sqrt(2.5) ~ 1.58)
        ekf.update_nonlinear(z_meas=[2.50], h_func=h_func, H_jacobian_func=H_jac)
        self.assertTrue(ekf.x[0, 0] > 0.0)

    def test_sensor_fusion_node(self):
        node = SensorFusionNode(dt=0.1)
        for _ in range(5):
            estimate = node.process_step(imu_accel=1.0)
            self.assertTrue(estimate["position"] > 0)
            
        estimate = node.process_step(imu_accel=1.0, lidar_pos=0.2)
        self.assertAlmostEqual(estimate["position"], 0.2, delta=0.5)

    def test_pid_controller(self):
        # PID with Kff=1.5
        pid = PIDController(kp=2.0, ki=0.5, kd=0.1, kff=1.5, output_min=-20.0, output_max=20.0)
        
        # Setpoint = 10, measurement = 8, dt = 0.1, feedforward_input = 2.0
        # Expected feedback term is similar to previous test (~4.1)
        # Expected feedforward = (Kff * Setpoint) + feedforward_input = (1.5 * 10) + 2.0 = 17.0
        # Total output should be clamped at 20.0
        output = pid.compute(setpoint=10.0, measurement=8.0, dt=0.1, feedforward_input=2.0)
        self.assertAlmostEqual(output, 20.0)

    def test_kinematics(self):
        kin = DifferentialDriveKinematics(wheel_radius=0.1, wheel_base=0.5)
        omega_l, omega_r = kin.inverse_kinematics(v=1.0, omega=0.0)
        self.assertAlmostEqual(omega_l, 10.0)
        self.assertAlmostEqual(omega_r, 10.0)
        
        x, y, theta = kin.forward_kinematics(omega_l=10.0, omega_r=10.0, dt=1.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(theta, 0.0)

if __name__ == "__main__":
    unittest.main()
