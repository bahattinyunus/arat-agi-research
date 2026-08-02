import numpy as np

class DifferentialDriveKinematics:
    """
    Implements Forward and Inverse Kinematics for a differential drive mobile robot.
    Tracks the robot pose (x, y, theta) and computes wheel velocities.
    """
    def __init__(self, wheel_radius=0.1, wheel_base=0.5, initial_pose=(0.0, 0.0, 0.0)):
        self.r = wheel_radius       # Wheel radius (meters)
        self.L = wheel_base         # Distance between wheels (meters)
        
        # Robot pose: x (m), y (m), theta (rad)
        self.x, self.y, self.theta = initial_pose

    def inverse_kinematics(self, v, omega):
        """
        Translates target linear velocity v (m/s) and angular velocity omega (rad/s)
        into individual wheel angular velocities (rad/s):
        omega_left = (v - omega * L / 2) / r
        omega_right = (v + omega * L / 2) / r
        """
        v_l = v - (omega * self.L / 2.0)
        v_r = v + (omega * self.L / 2.0)
        
        omega_l = v_l / self.r
        omega_r = v_r / self.r
        
        return omega_l, omega_r

    def forward_kinematics(self, omega_l, omega_r, dt):
        """
        Updates pose from wheel angular velocities (rad/s) using Euler integration:
        v = r * (omega_right + omega_left) / 2
        omega = r * (omega_right - omega_left) / L
        """
        if dt <= 0:
            return self.x, self.y, self.theta

        # Convert wheel angular velocities to linear wheel velocities
        v_l = omega_l * self.r
        v_r = omega_r * self.r

        # Robot velocities in body frame
        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.L

        # Update pose in map frame
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        return self.x, self.y, self.theta

    def get_pose(self):
        return {"x": self.x, "y": self.y, "theta": self.theta}
