import unittest
import numpy as np
from autonomous_control.sensor_fusion import KalmanFilter, SensorFusionNode
from autonomous_control.kinematic_loops import PIDController, DifferentialDriveKinematics

class TestControl(unittest.TestCase):
    def test_kalman_filter_tracking(self):
        # 1D position tracking (n=2: pos, vel. z=1: pos)
        kf = KalmanFilter(n=2, z=1)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # State transition
        kf.H = np.array([[1.0, 0.0]])             # Measurement mapping
        kf.Q = np.eye(2) * 0.01
        kf.R = np.array([[0.1]])
        
        kf.x = np.array([[0.0], [1.0]])  # Initial position 0, velocity 1
        
        # Predict one step
        x_pred, _ = kf.predict()
        self.assertAlmostEqual(x_pred[0, 0], 1.0)
        self.assertAlmostEqual(x_pred[1, 0], 1.0)
        
        # Update with position measurement 1.1
        x_upd, _ = kf.update([1.1])
        # Position should adjust closer to measurement
        self.assertTrue(x_upd[0, 0] > 1.0 and x_upd[0, 0] <= 1.1)

    def test_sensor_fusion_node(self):
        node = SensorFusionNode(dt=0.1)
        
        # Push 5 constant acceleration steps with IMU
        for _ in range(5):
            estimate = node.process_step(imu_accel=1.0)
            self.assertTrue(estimate["position"] > 0)
            
        # Push a LiDAR update step correcting position
        estimate = node.process_step(imu_accel=1.0, lidar_pos=0.2)
        self.assertAlmostEqual(estimate["position"], 0.2, delta=0.5)

    def test_pid_controller(self):
        # PID: Kp=2.0, Ki=0.5, Kd=0.1
        pid = PIDController(kp=2.0, ki=0.5, kd=0.1, output_min=-10.0, output_max=10.0)
        
        # Setpoint = 10, measurement = 8 (error = 2)
        output = pid.compute(setpoint=10.0, measurement=8.0, dt=0.1)
        
        # Proportional contribution alone should be Kp * Error = 2 * 2 = 4.0
        # Integral = Error * dt = 2 * 0.1 = 0.2 -> Ki * Integral = 0.5 * 0.2 = 0.1
        # Total output should be around 4.1 (plus derivative contribution)
        self.assertTrue(output > 4.0)
        self.assertTrue(output <= 10.0)

    def test_kinematics(self):
        kin = DifferentialDriveKinematics(wheel_radius=0.1, wheel_base=0.5)
        
        # Test Inverse Kinematics
        # Linear velocity = 1.0 m/s, Angular velocity = 0.0 rad/s
        omega_l, omega_r = kin.inverse_kinematics(v=1.0, omega=0.0)
        # Left and right wheel should have equal angular velocities: v / r = 1.0 / 0.1 = 10.0 rad/s
        self.assertAlmostEqual(omega_l, 10.0)
        self.assertAlmostEqual(omega_r, 10.0)
        
        # Test Forward Kinematics
        x, y, theta = kin.forward_kinematics(omega_l=10.0, omega_r=10.0, dt=1.0)
        # The robot should move straight ahead: pos_x = 1.0m, pos_y = 0.0m, theta = 0.0
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(theta, 0.0)

if __name__ == "__main__":
    unittest.main()
