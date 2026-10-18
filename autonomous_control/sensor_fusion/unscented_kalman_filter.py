import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) utilizing the Scaled Unscented Transform (Julier & Uhlmann).
    Estimates non-linear state distributions without requiring Jacobian derivation.
    """
    def __init__(self, n: int, z: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n  # State dimension
        self.z = z  # Measurement dimension
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lambda_param = (alpha ** 2) * (n + kappa) - n
        
        # State and covariance
        self.x = np.zeros((n, 1))
        self.P = np.eye(n)
        self.Q = np.eye(n)
        self.R = np.eye(z)
        
        # Precompute weights for mean and covariance
        self.num_sigmas = 2 * n + 1
        self.wm = np.zeros(self.num_sigmas)
        self.wc = np.zeros(self.num_sigmas)
        
        c = self.n + self.lambda_param
        self.wm[0] = self.lambda_param / c
        self.wc[0] = self.lambda_param / c + (1.0 - alpha ** 2 + beta)
        
        for i in range(1, self.num_sigmas):
            self.wm[i] = 1.0 / (2.0 * c)
            self.wc[i] = 1.0 / (2.0 * c)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generates 2n+1 sigma points:
        chi_0 = x
        chi_i = x + (sqrt((n + lambda) * P))_i
        chi_i+n = x - (sqrt((n + lambda) * P))_i
        """
        sigma_points = np.zeros((self.n, self.num_sigmas))
        sigma_points[:, 0] = x.flatten()
        
        scale = self.n + self.lambda_param
        # Ensure symmetric positive definite covariance for Cholesky
        P_sym = 0.5 * (P + P.T) + np.eye(self.n) * 1e-9
        try:
            L = np.linalg.cholesky(scale * P_sym)
        except np.linalg.LinAlgError:
            # Fallback to eigen decomposition if numerical jitter affects Cholesky
            eigvals, eigvecs = np.linalg.eigh(P_sym)
            eigvals = np.maximum(eigvals, 1e-9)
            L = eigvecs @ np.diag(np.sqrt(scale * eigvals))
            
        for i in range(self.n):
            col = L[:, i]
            sigma_points[:, i + 1] = x.flatten() + col
            sigma_points[:, i + 1 + self.n] = x.flatten() - col
            
        return sigma_points

    def predict(self, f_func: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray], u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates sigma points through non-linear state transition f(chi, u)
        and computes predicted mean and covariance.
        """
        # 1. Generate sigma points from prior
        sigmas = self.generate_sigma_points(self.x, self.P)
        
        # 2. Transform through non-linear dynamic model
        sigmas_f = np.zeros_like(sigmas)
        for i in range(self.num_sigmas):
            sigmas_f[:, i] = f_func(sigmas[:, i].reshape((self.n, 1)), u).flatten()
            
        # 3. Calculate predicted mean
        x_pred = np.zeros((self.n, 1))
        for i in range(self.num_sigmas):
            x_pred += self.wm[i] * sigmas_f[:, i].reshape((self.n, 1))
            
        # 4. Calculate predicted covariance
        P_pred = np.copy(self.Q)
        for i in range(self.num_sigmas):
            diff = (sigmas_f[:, i].reshape((self.n, 1)) - x_pred)
            P_pred += self.wc[i] * np.outer(diff, diff)
            
        self.x = x_pred
        self.P = P_pred
        self.last_sigmas_f = sigmas_f
        return self.x, self.P

    def update(self, z_meas: np.ndarray, h_func: Callable[[np.ndarray], np.ndarray], angle_wrap_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates state and covariance given non-linear observation z = h(x).
        """
        z_meas = np.array(z_meas).reshape((self.z, 1))
        
        # Re-generate or use propagated sigma points
        sigmas = self.generate_sigma_points(self.x, self.P)
        
        # 1. Transform sigma points through measurement function
        sigmas_h = np.zeros((self.z, self.num_sigmas))
        for i in range(self.num_sigmas):
            sigmas_h[:, i] = h_func(sigmas[:, i].reshape((self.n, 1))).flatten()
            
        # 2. Predicted measurement mean
        z_pred = np.zeros((self.z, 1))
        for i in range(self.num_sigmas):
            z_pred += self.wm[i] * sigmas_h[:, i].reshape((self.z, 1))
            
        # 3. Innovation covariance S and Cross-covariance Pxz
        S = np.copy(self.R)
        Pxz = np.zeros((self.n, self.z))
        
        for i in range(self.num_sigmas):
            z_diff = sigmas_h[:, i].reshape((self.z, 1)) - z_pred
            x_diff = sigmas[:, i].reshape((self.n, 1)) - self.x
            
            if angle_wrap_idx is not None and 0 <= angle_wrap_idx < self.n:
                x_diff[angle_wrap_idx, 0] = np.arctan2(np.sin(x_diff[angle_wrap_idx, 0]), np.cos(x_diff[angle_wrap_idx, 0]))
                
            S += self.wc[i] * np.outer(z_diff, z_diff)
            Pxz += self.wc[i] * np.outer(x_diff, z_diff)
            
        # 4. Kalman gain
        K = Pxz @ np.linalg.inv(S)
        
        # 5. Residual
        residual = z_meas - z_pred
        
        # 6. Correct state and covariance
        self.x = self.x + K @ residual
        if angle_wrap_idx is not None and 0 <= angle_wrap_idx < self.n:
            self.x[angle_wrap_idx, 0] = np.arctan2(np.sin(self.x[angle_wrap_idx, 0]), np.cos(self.x[angle_wrap_idx, 0]))
            
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # Maintain symmetry
        
        return self.x, self.P


class UKFSensorFusionNode:
    """
    UKF-based 2D Pose and Sensor Fusion Node for differential drive robotics.
    State: x = [pos_x, pos_y, theta]^T (3 dimensions)
    Control: u = [v, omega]^T
    Measurements: z = [pos_x, pos_y]^T
    """
    def __init__(self, dt: float = 0.1, initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.dt = dt
        self.ukf = UnscentedKalmanFilter(n=3, z=2, alpha=1e-3, beta=2.0, kappa=0.0)
        
        self.ukf.x = np.array(initial_pose, dtype=float).reshape((3, 1))
        self.ukf.P = np.eye(3) * 0.1
        
        self.ukf.Q = np.array([
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-2]
        ])
        
        self.ukf.R = np.array([
            [0.02, 0.0],
            [0.0, 0.02]
        ])

    def state_transition(self, x: np.ndarray, u: Optional[np.ndarray]) -> np.ndarray:
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
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])

    def measurement_model(self, x: np.ndarray) -> np.ndarray:
        return x[0:2, 0]

    def process_step(self, linear_vel: float, angular_vel: float, meas_pos: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        u = np.array([[linear_vel], [angular_vel]])
        
        # 1. Prediction step
        self.ukf.predict(self.state_transition, u)
        
        # 2. Correction step
        if meas_pos is not None:
            z_meas = np.array(meas_pos).reshape((2, 1))
            self.ukf.update(z_meas, self.measurement_model, angle_wrap_idx=2)
            
        return self.get_state_estimate()

    def get_state_estimate(self) -> Dict[str, Any]:
        return {
            "x": float(self.ukf.x[0, 0]),
            "y": float(self.ukf.x[1, 0]),
            "theta": float(self.ukf.x[2, 0]),
            "covariance": self.ukf.P.diagonal().tolist()
        }
