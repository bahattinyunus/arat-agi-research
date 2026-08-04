import numpy as np
from .kalman_filter import ExtendedKalmanFilter

class EKFSensorFusionNode:
    """
    Non-linear Extended Kalman Filter (EKF) node for tracking a 2D differential drive robot.
    State: x = [pos_x, pos_y, heading_theta]^T (3 dimensions)
    Control Input: u = [linear_velocity, angular_velocity]^T (2 dimensions)
    Measurements: z = [measured_x, measured_y]^T (2 dimensions)
    """
    def __init__(self, dt=0.1, initial_pose=(0.0, 0.0, 0.0)):
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(n=3, z=2)
        
        # Initialize state estimate and covariance
        self.ekf.x = np.array(initial_pose, dtype=float).reshape((3, 1))
        self.ekf.P = np.eye(3) * 0.1
        
        # Process noise covariance
        self.ekf.Q = np.array([
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-2]
        ])
        
        # Measurement noise covariance (LiDAR/GPS position measurement noise)
        self.ekf.R = np.array([
            [0.02, 0.0],
            [0.0, 0.02]
        ])

    def state_transition(self, x, u):
        """
        Non-linear state transition function f(x, u):
        x_next = x + v * cos(theta) * dt
        y_next = y + v * sin(theta) * dt
        theta_next = theta + omega * dt
        """
        pos_x = x[0, 0]
        pos_y = x[1, 0]
        theta = x[2, 0]
        
        if u is not None:
            v = u[0, 0]
            omega = u[1, 0]
        else:
            v = 0.0
            omega = 0.0
            
        x_next = pos_x + v * np.cos(theta) * self.dt
        y_next = pos_y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        
        # Normalize theta to [-pi, pi]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])

    def transition_jacobian(self, x, u):
        """
        Jacobian matrix of state transition F_j:
        [ 1,  0,  -v * sin(theta) * dt ]
        [ 0,  1,   v * cos(theta) * dt ]
        [ 0,  0,   1                   ]
        """
        theta = x[2, 0]
        if u is not None:
            v = u[0, 0]
        else:
            v = 0.0
            
        F_j = np.array([
            [1.0, 0.0, -v * np.sin(theta) * self.dt],
            [0.0, 1.0,  v * np.cos(theta) * self.dt],
            [0.0, 0.0,  1.0]
        ])
        return F_j

    def measurement_function(self, x):
        """
        Non-linear measurement function h(x) = [pos_x, pos_y]^T
        """
        return x[0:2, 0]

    def measurement_jacobian(self, x):
        """
        Jacobian matrix of measurement function H_j:
        [ 1,  0,  0 ]
        [ 0,  1,  0 ]
        """
        H_j = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        return H_j

    def process_step(self, linear_vel, angular_vel, meas_pos=None):
        """
        Executes an EKF step:
        - Predict next state using control inputs (linear velocity, angular velocity)
        - Update state estimate if position measurement is available
        """
        u = np.array([[linear_vel], [angular_vel]])
        
        # 1. Prediction step
        self.ekf.predict_nonlinear(self.state_transition, self.transition_jacobian, u)
        
        # 2. Correction step (if measurement is available)
        if meas_pos is not None:
            z_meas = np.array(meas_pos).reshape((2, 1))
            self.ekf.update_nonlinear(z_meas, self.measurement_function, self.measurement_jacobian)
            
        return self.get_state_estimate()

    def get_state_estimate(self):
        """Returns the current state estimate and variance dictionary"""
        return {
            "x": float(self.ekf.x[0, 0]),
            "y": float(self.ekf.x[1, 0]),
            "theta": float(self.ekf.x[2, 0]),
            "covariance": self.ekf.P.diagonal().tolist()
        }
