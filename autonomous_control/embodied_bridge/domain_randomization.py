import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .messages import Twist, LaserScan

class Sim2RealDomainRandomizer:
    """
    Sim2Real Domain Randomization Engine.
    Injects realistic physical hardware perturbations, sensor noise distributions,
    actuation latencies, motor deadbands, and wheel slip to prepare neural/active-inference
    policies for deployment on physical Arat robotic hardware.
    """
    def __init__(
        self,
        wheel_slip_std: float = 0.05,
        imu_bias_drift_std: float = 0.002,
        lidar_noise_std: float = 0.03,
        lidar_dropout_rate: float = 0.02,
        latency_steps: int = 1,
        motor_deadband: float = 0.04
    ):
        self.wheel_slip_std = wheel_slip_std
        self.imu_bias_drift_std = imu_bias_drift_std
        self.lidar_noise_std = lidar_noise_std
        self.lidar_dropout_rate = lidar_dropout_rate
        self.latency_steps = latency_steps
        self.motor_deadband = motor_deadband

        # Dynamic bias states
        self.gyro_bias = 0.0
        self.accel_bias = 0.0
        self.command_queue: List[Twist] = []

    def randomize_kinematics(self, target_v: float, target_w: float) -> Tuple[float, float]:
        """
        Simulates motor deadband and non-linear wheel slip.
        """
        # Motor deadband: very small command signals cannot overcome static friction
        v_act = 0.0 if abs(target_v) < self.motor_deadband else target_v
        w_act = 0.0 if abs(target_w) < self.motor_deadband else target_w

        # Wheel slip perturbations (multiplicative and additive)
        slip_factor_l = 1.0 + np.random.normal(0.0, self.wheel_slip_std)
        slip_factor_r = 1.0 + np.random.normal(0.0, self.wheel_slip_std)

        effective_v = v_act * (0.5 * (slip_factor_l + slip_factor_r))
        effective_w = w_act * (slip_factor_r - slip_factor_l + 1.0)

        return float(effective_v), float(effective_w)

    def randomize_lidar(self, true_ranges: List[float], max_range: float = 12.0) -> List[float]:
        """
        Injects Gaussian distance noise and specular dropouts (returns max_range on dropouts).
        """
        noisy_ranges = []
        for r in true_ranges:
            if np.random.uniform(0.0, 1.0) < self.lidar_dropout_rate:
                noisy_ranges.append(float(max_range))  # Ray dropped out / absorption
            else:
                noise = np.random.normal(0.0, self.lidar_noise_std)
                measured = np.clip(r + noise, 0.05, max_range)
                noisy_ranges.append(float(measured))
        return noisy_ranges

    def randomize_imu(self, true_accel: float, true_gyro: float, dt: float = 0.1) -> Tuple[float, float]:
        """
        Simulates random walk bias drift and measurement white noise for IMU sensors.
        """
        self.accel_bias += np.random.normal(0.0, self.imu_bias_drift_std)
        self.gyro_bias += np.random.normal(0.0, self.imu_bias_drift_std)

        noisy_accel = true_accel + self.accel_bias + np.random.normal(0.0, 0.05)
        noisy_gyro = true_gyro + self.gyro_bias + np.random.normal(0.0, 0.01)

        return float(noisy_accel), float(noisy_gyro)

    def queue_command_with_latency(self, cmd: Twist) -> Twist:
        """
        Simulates actuation transmission and compute latency over DDS/CAN bus.
        """
        self.command_queue.append(cmd)
        if len(self.command_queue) > self.latency_steps:
            return self.command_queue.pop(0)
        return Twist(0.0, 0.0, 0.0)
