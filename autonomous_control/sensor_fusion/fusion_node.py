import numpy as np
from .kalman_filter import KalmanFilter

class SensorFusionNode:
    """
    Simulates high-frequency fusion of IMU (acceleration) and LiDAR (position) data
    to estimate the 1D/2D state (position, velocity, acceleration) of an autonomous robot.
    """
    def __init__(self, dt=0.01):
        self.dt = dt
        # State vector x = [position_x, velocity_x]^T (2 dimensions)
        # Measurement vector z = [position_x] (from LiDAR)
        # Control vector u = [acceleration_x] (from IMU)
        self.filter = KalmanFilter(n=2, z=1, m=1)

        # Transition matrix: position = position + velocity*dt, velocity = velocity
        self.filter.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])

        # Control matrix: position = 0.5*accel*dt^2, velocity = accel*dt
        self.filter.B = np.array([
            [0.5 * (self.dt ** 2)],
            [self.dt]
        ])

        # Measurement matrix: we only measure position
        self.filter.H = np.array([[1.0, 0.0]])

        # Process noise covariance (trust in process transitions)
        self.filter.Q = np.array([
            [1e-4, 1e-5],
            [1e-5, 1e-4]
        ])

        # Measurement noise covariance (trust in LiDAR sensor)
        self.filter.R = np.array([[0.05]])

        # Prior covariance
        self.filter.P = np.eye(2) * 1.0

    def process_step(self, imu_accel, lidar_pos=None):
        """
        Process a single sensor fusion step:
        - Always run prediction with IMU acceleration as control input
        - If LiDAR position is available (lower frequency), run update step
        """
        # 1. Prediction using IMU
        control_input = np.array([[imu_accel]])
        self.filter.predict(control_input)

        # 2. Update if LiDAR measurement is available
        if lidar_pos is not None:
            measurement = np.array([[lidar_pos]])
            self.filter.update(measurement)

        return self.get_state_estimate()

    def get_state_estimate(self):
        """Returns current estimated (position, velocity)"""
        pos = float(self.filter.x[0, 0])
        vel = float(self.filter.x[1, 0])
        variance = (float(self.filter.P[0, 0]), float(self.filter.P[1, 1]))
        return {
            "position": pos,
            "velocity": vel,
            "covariance": variance
        }
