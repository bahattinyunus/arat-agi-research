import numpy as np

class KalmanFilter:
    """
    Multidimensional Linear Kalman Filter for tracking and sensor fusion.
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
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
            
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x, self.P

    def update(self, z_meas):
        z_meas = np.array(z_meas).reshape((self.z, 1))
        y = z_meas - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x, self.P

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for non-linear state estimation.
    Uses Jacobians to linearize state transitions and measurements.
    """
    def __init__(self, n, z):
        self.n = n  # Dimension of state vector
        self.z = z  # Dimension of measurement vector
        
        self.x = np.zeros((n, 1))   # State estimate
        self.P = np.eye(n)          # Estimate covariance
        self.Q = np.eye(n)          # Process noise covariance
        self.R = np.eye(z)          # Measurement noise covariance

    def predict_nonlinear(self, f_func, F_jacobian_func, u=None):
        """
        Predicts next state using non-linear transition function:
        x = f(x, u)
        P = F_j * P * F_j^T + Q
        """
        # 1. State prediction via non-linear function
        self.x = f_func(self.x, u).reshape((self.n, 1))
        
        # 2. Linearize process model at current state using Jacobian
        F_j = F_jacobian_func(self.x, u)
        
        # 3. Covariance prediction
        self.P = np.dot(np.dot(F_j, self.P), F_j.T) + self.Q
        return self.x, self.P

    def update_nonlinear(self, z_meas, h_func, H_jacobian_func):
        """
        Updates state using non-linear measurement function:
        y = z - h(x)
        S = H_j * P * H_j^T + R
        K = P * H_j^T * S^-1
        x = x + K * y
        P = (I - K * H_j) * P
        """
        z_meas = np.array(z_meas).reshape((self.z, 1))
        
        # 1. Measurement residual using non-linear function
        y = z_meas - h_func(self.x).reshape((self.z, 1))
        
        # 2. Linearize measurement model using Jacobian
        H_j = H_jacobian_func(self.x)
        
        # 3. Compute Kalman Gain
        S = np.dot(np.dot(H_j, self.P), H_j.T) + self.R
        K = np.dot(np.dot(self.P, H_j.T), np.linalg.inv(S))
        
        # 4. State correction
        self.x = self.x + np.dot(K, y)
        
        # 5. Covariance correction
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, H_j), self.P)
        return self.x, self.P
