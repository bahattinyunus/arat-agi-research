import numpy as np
from typing import List, Tuple, Dict, Any

class QuinticPolynomial:
    """
    Quintic polynomial 1D trajectory generator:
    s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Solves boundary conditions:
    s(0) = s0,  s'(0) = v0,  s''(0) = a0
    s(T) = s1,  s'(T) = v1,  s''(T) = a1
    Minimizes jerk integral over [0, T].
    """
    def __init__(self, s0: float, v0: float, a0: float, s1: float, v1: float, a1: float, T: float):
        self.s0 = s0
        self.v0 = v0
        self.a0 = a0
        self.s1 = s1
        self.v1 = v1
        self.a1 = a1
        self.T = max(1e-4, T)

        # Coefficients
        self.c0 = s0
        self.c1 = v0
        self.c2 = 0.5 * a0

        # Linear system for c3, c4, c5
        T2 = self.T ** 2
        T3 = self.T ** 3
        T4 = self.T ** 4
        T5 = self.T ** 5

        A = np.array([
            [T3, T4, T5],
            [3.0 * T2, 4.0 * T3, 5.0 * T4],
            [6.0 * self.T, 12.0 * T2, 20.0 * T3]
        ])

        b = np.array([
            s1 - self.c0 - self.c1 * self.T - self.c2 * T2,
            v1 - self.c1 - 2.0 * self.c2 * self.T,
            a1 - 2.0 * self.c2
        ])

        try:
            x = np.linalg.solve(A, b)
            self.c3 = x[0]
            self.c4 = x[1]
            self.c5 = x[2]
        except np.linalg.LinAlgError:
            self.c3 = 0.0
            self.c4 = 0.0
            self.c5 = 0.0

    def calc_point(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        return self.c0 + self.c1 * t + self.c2 * (t**2) + self.c3 * (t**3) + self.c4 * (t**4) + self.c5 * (t**5)

    def calc_first_derivative(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        return self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * (t**2) + 4.0 * self.c4 * (t**3) + 5.0 * self.c5 * (t**4)

    def calc_second_derivative(self, t: float) -> float:
        t = np.clip(t, 0.0, self.T)
        return 2.0 * self.c2 + 6.0 * self.c3 * t + 12.0 * self.c4 * (t**2) + 20.0 * self.c5 * (t**3)


class TrajectoryGenerator2D:
    """
    Generates smooth 2D trajectories for differential drive robots by coupling two
    QuinticPolynomial generators along X and Y axes.
    """
    def __init__(self, start_pose: Tuple[float, float, float], target_pose: Tuple[float, float, float],
                 start_speed: float = 0.0, target_speed: float = 0.0, duration: float = 5.0):
        self.duration = duration
        
        x0, y0, th0 = start_pose
        x1, y1, th1 = target_pose

        vx0 = start_speed * np.cos(th0)
        vy0 = start_speed * np.sin(th0)
        vx1 = target_speed * np.cos(th1)
        vy1 = target_speed * np.sin(th1)

        self.poly_x = QuinticPolynomial(x0, vx0, 0.0, x1, vx1, 0.0, duration)
        self.poly_y = QuinticPolynomial(y0, vy0, 0.0, y1, vy1, 0.0, duration)

    def sample_trajectory(self, num_points: int = 50) -> List[Dict[str, float]]:
        times = np.linspace(0.0, self.duration, num_points)
        trajectory = []
        for t in times:
            x = self.poly_x.calc_point(t)
            y = self.poly_y.calc_point(t)
            vx = self.poly_x.calc_first_derivative(t)
            vy = self.poly_y.calc_first_derivative(t)
            theta = np.arctan2(vy, vx) if (vx**2 + vy**2) > 1e-4 else 0.0
            linear_v = np.sqrt(vx**2 + vy**2)

            trajectory.append({
                "time": float(t),
                "x": float(x),
                "y": float(y),
                "theta": float(theta),
                "v": float(linear_v)
            })
        return trajectory
