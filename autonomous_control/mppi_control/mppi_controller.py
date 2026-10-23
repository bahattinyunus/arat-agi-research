import numpy as np
from typing import Callable, Tuple, List, Dict, Any, Optional

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) Controller for non-linear stochastic trajectory optimization.
    Samples K parallel rollouts in control space, evaluates trajectory cost landscapes,
    and computes the information-theoretically optimal control update:
    u_t* = u_t + sum_k( w_k * epsilon_t^k )
    w_k = exp(-1/lambda * (S_k - min_S)) / sum_j(...)
    """
    def __init__(
        self,
        horizon: int = 15,
        num_samples: int = 40,
        dt: float = 0.1,
        temperature_lambda: float = 0.5,
        control_noise_cov: Tuple[float, float] = (0.2, 0.4), # [std_v, std_w]
        v_limits: Tuple[float, float] = (0.0, 1.8),
        w_limits: Tuple[float, float] = (-1.5, 1.5)
    ):
        self.H = horizon
        self.K = num_samples
        self.dt = dt
        self.lambda_param = temperature_lambda
        self.noise_std = np.array(control_noise_cov)
        self.v_limits = v_limits
        self.w_limits = w_limits

        # Nominal control sequence of shape (H, 2) where col 0 is v, col 1 is w
        self.U = np.zeros((self.H, 2))

    def rollout_dynamics(
        self,
        initial_pose: Tuple[float, float, float],
        u_sequence: np.ndarray
    ) -> np.ndarray:
        """Propagates differential drive kinematics forward in time: shape (H+1, 3)."""
        trajectory = np.zeros((self.H + 1, 3))
        trajectory[0] = initial_pose

        x, y, theta = initial_pose
        for t in range(self.H):
            v, w = u_sequence[t]
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            # Normalize angle
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            trajectory[t + 1] = [x, y, theta]

        return trajectory

    def compute_trajectory_cost(
        self,
        trajectory: np.ndarray,
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]],
        robot_radius: float = 0.35
    ) -> float:
        """
        Evaluates stage costs: distance to goal, heading orientation, and obstacle repulsion.
        """
        cost = 0.0

        for t in range(1, self.H + 1):
            px, py, pth = trajectory[t]

            # 1. Distance to goal
            dist_goal = np.sqrt((px - goal[0])**2 + (py - goal[1])**2)
            cost += dist_goal * 2.0

            # 2. Obstacle collision repulsion
            for ox, oy in obstacles:
                d_obs = np.sqrt((px - ox)**2 + (py - oy)**2)
                if d_obs < robot_radius:
                    cost += 500.0 # Collision penalty
                elif d_obs < robot_radius + 0.5:
                    cost += 10.0 / (d_obs - robot_radius + 1e-3)

        # Terminal cost
        final_x, final_y, _ = trajectory[-1]
        cost += 5.0 * np.sqrt((final_x - goal[0])**2 + (final_y - goal[1])**2)

        return float(cost)

    def optimize_control(
        self,
        current_pose: Tuple[float, float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[float, float, np.ndarray]:
        """
        Runs one step of MPPI optimization and returns (v_cmd, w_cmd, best_trajectory).
        """
        # Sample perturbations epsilon of shape (K, H, 2)
        noise = np.random.normal(0.0, 1.0, size=(self.K, self.H, 2)) * self.noise_std

        # Perturbed control candidates
        perturbed_U = self.U + noise
        # Clamp to physical motor limits
        perturbed_U[:, :, 0] = np.clip(perturbed_U[:, :, 0], self.v_limits[0], self.v_limits[1])
        perturbed_U[:, :, 1] = np.clip(perturbed_U[:, :, 1], self.w_limits[0], self.w_limits[1])

        costs = np.zeros(self.K)
        trajectories = []

        # Evaluate cost for each sampled trajectory
        for k in range(self.K):
            traj = self.rollout_dynamics(current_pose, perturbed_U[k])
            trajectories.append(traj)
            cost = self.compute_trajectory_cost(traj, goal, obstacles)
            costs[k] = cost

        # Compute information-theoretic softmax weights
        min_cost = np.min(costs)
        exp_weights = np.exp(-1.0 / self.lambda_param * (costs - min_cost))
        weights = exp_weights / (np.sum(exp_weights) + 1e-12)

        # Weighted update to nominal control sequence
        delta_U = np.zeros((self.H, 2))
        for k in range(self.K):
            delta_U += weights[k] * (perturbed_U[k] - self.U)

        self.U += delta_U
        self.U[:, 0] = np.clip(self.U[:, 0], self.v_limits[0], self.v_limits[1])
        self.U[:, 1] = np.clip(self.U[:, 1], self.w_limits[0], self.w_limits[1])

        # Best trajectory corresponds to minimum cost rollout
        best_idx = int(np.argmin(costs))
        best_traj = trajectories[best_idx]

        # Extract first action to execute
        v_exec = float(self.U[0, 0])
        w_exec = float(self.U[0, 1])

        # Recede horizon (shift left by 1)
        self.U[:-1] = self.U[1:]
        self.U[-1] = [0.0, 0.0]

        return v_exec, w_exec, best_traj
