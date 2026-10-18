import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class DWAPlannerConfig:
    def __init__(
        self,
        max_speed: float = 1.8,
        min_speed: float = 0.0,
        max_yaw_rate: float = 1.5,
        max_accel: float = 1.0,
        max_dyaw: float = 2.0,
        v_resolution: float = 0.1,
        yaw_rate_resolution: float = 0.05,
        dt: float = 0.1,
        predict_time: float = 2.0,
        to_goal_cost_gain: float = 0.15,
        speed_cost_gain: float = 0.8,
        obstacle_cost_gain: float = 1.2,
        robot_radius: float = 0.35
    ):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_dyaw = max_dyaw
        self.v_resolution = v_resolution
        self.yaw_rate_resolution = yaw_rate_resolution
        self.dt = dt
        self.predict_time = predict_time
        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_cost_gain = speed_cost_gain
        self.obstacle_cost_gain = obstacle_cost_gain
        self.robot_radius = robot_radius


class DynamicWindowApproachPlanner:
    """
    Dynamic Window Approach (DWA) local motion planner for mobile robots.
    Evaluates reachable velocity windows considering kinematic and dynamic limits,
    simulates forward paths, and selects optimal (v, omega) minimizing cost.
    """
    def __init__(self, config: Optional[DWAPlannerConfig] = None):
        self.config = config or DWAPlannerConfig()

    def calc_dynamic_window(self, current_v: float, current_w: float) -> Tuple[float, float, float, float]:
        """Calculates dynamic window [Vs] intersect [Vd]."""
        # Specification limits
        Vs = [self.config.min_speed, self.config.max_speed,
              -self.config.max_yaw_rate, self.config.max_yaw_rate]

        # Dynamic limits based on acceleration
        Vd = [
            current_v - self.config.max_accel * self.config.dt,
            current_v + self.config.max_accel * self.config.dt,
            current_w - self.config.max_dyaw * self.config.dt,
            current_w + self.config.max_dyaw * self.config.dt
        ]

        # Intersection
        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3])
        ]
        return dw[0], dw[1], dw[2], dw[3]

    def predict_trajectory(self, x_init: float, y_init: float, theta_init: float, v: float, w: float) -> np.ndarray:
        """Simulates robot trajectory for velocity pair (v, w) over prediction horizon."""
        time_steps = int(self.config.predict_time / self.config.dt)
        trajectory = np.zeros((time_steps, 3))
        x, y, theta = x_init, y_init, theta_init

        for i in range(time_steps):
            x += v * np.cos(theta) * self.config.dt
            y += v * np.sin(theta) * self.config.dt
            theta += w * self.config.dt
            trajectory[i, 0] = x
            trajectory[i, 1] = y
            trajectory[i, 2] = theta

        return trajectory

    def calc_to_goal_cost(self, trajectory: np.ndarray, goal: Tuple[float, float]) -> float:
        """Calculates cost based on heading angle difference to goal."""
        final_x = trajectory[-1, 0]
        final_y = trajectory[-1, 1]
        final_theta = trajectory[-1, 2]

        goal_x, goal_y = goal
        dx = goal_x - final_x
        dy = goal_y - final_y
        error_angle = np.arctan2(dy, dx)
        cost_angle = error_angle - final_theta
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))
        return float(cost)

    def calc_obstacle_cost(self, trajectory: np.ndarray, obstacles: List[Tuple[float, float]]) -> float:
        """Calculates distance to closest obstacle. Returns inf if collision occurs."""
        if not obstacles:
            return 0.0

        min_dist = float("inf")
        for ox, oy in obstacles:
            dists = np.sqrt((trajectory[:, 0] - ox)**2 + (trajectory[:, 1] - oy)**2)
            closest = float(np.min(dists))
            if closest < self.config.robot_radius:
                return float("inf")
            if closest < min_dist:
                min_dist = closest

        return 1.0 / (min_dist + 1e-4)

    def plan(
        self,
        current_pose: Tuple[float, float, float],
        current_v: float,
        current_w: float,
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[float, float, List[Dict[str, float]]]:
        """
        Evaluates dynamic window, scores trajectories, and selects best velocity commands (v*, w*).
        """
        dw = self.calc_dynamic_window(current_v, current_w)
        min_cost = float("inf")
        best_v = 0.0
        best_w = 0.0
        best_traj = []

        v_vals = np.arange(dw[0], dw[1] + 1e-5, self.config.v_resolution)
        w_vals = np.arange(dw[2], dw[3] + 1e-5, self.config.yaw_rate_resolution)

        x, y, theta = current_pose

        for v in v_vals:
            for w in w_vals:
                traj = self.predict_trajectory(x, y, theta, v, w)
                
                goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.config.max_speed - v
                obs_cost = self.calc_obstacle_cost(traj, obstacles)

                final_cost = (
                    self.config.to_goal_cost_gain * goal_cost +
                    self.config.speed_cost_gain * speed_cost +
                    self.config.obstacle_cost_gain * obs_cost
                )

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_v = float(v)
                    best_w = float(w)
                    best_traj = traj

        traj_list = []
        if len(best_traj) > 0:
            for row in best_traj:
                traj_list.append({"x": float(row[0]), "y": float(row[1]), "theta": float(row[2])})

        return best_v, best_w, traj_list
