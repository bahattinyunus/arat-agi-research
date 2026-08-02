import numpy as np

class KalmanFilter:
    """
    Multidimensional Kalman Filter for tracking and sensor fusion.
    Models the state x and covariance P of a system.
    """
    def __init__(self, n, z, m=0):
        self.n = n  # Dimensions of the state vector
        self.z = z  # Dimensions of the measurement vector
        self.m = m  # Dimensions of the control input vector

        self.x = np.zeros((n, 1))           # State estimate
        self.P = np.eye(n)                  # Estimate covariance
        self.F = np.eye(n)                  # State transition matrix
        self.B = np.zeros((n, m)) if m > 0 else None  # Control matrix
        self.H = np.zeros((z, n))           # Measurement matrix
        self.Q = np.eye(n)                  # Process noise covariance
        self.R = np.eye(z)                  # Measurement noise covariance

    def predict(self, u=None):
        """
        Predicts state x_k and covariance P_k using the process model:
        x = F * x + B * u
        P = F * P * F^T + Q
        """
        # x_k|k-1 = F_k * x_k-1|k-1 (+ B_k * u_k)
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
            
        # P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x, self.P

    def update(self, z_meas):
        """
        Updates state and covariance using the measurement z_meas:
        y = z_meas - H * x  (Innovation/residual)
        S = H * P * H^T + R (Residual covariance)
        K = P * H^T * S^-1  (Optimal Kalman Gain)
        x = x + K * y       (Updated state estimate)
        P = (I - K * H) * P (Updated estimate covariance)
        """
        # Ensure measurement is shaped (z, 1)
        z_meas = np.array(z_meas).reshape((self.z, 1))

        # Innovation (Residual): y = z - H * x
        y = z_meas - np.dot(self.H, self.x)

        # Residual Covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman Gain: K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update State: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Update Covariance: P = (I - K * H) * P
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x, self.P
