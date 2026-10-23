import time
from typing import List, Dict, Any, Optional

class Header:
    def __init__(self, frame_id: str = "base_link", stamp: Optional[float] = None):
        self.stamp = stamp or time.time()
        self.frame_id = frame_id

    def to_dict(self) -> Dict[str, Any]:
        return {"stamp": self.stamp, "frame_id": self.frame_id}


class Twist:
    """Standard geometry_msgs/Twist equivalent for robotic velocity commands."""
    def __init__(self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0):
        self.linear_x = float(linear_x)
        self.linear_y = float(linear_y)
        self.angular_z = float(angular_z)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linear": {"x": self.linear_x, "y": self.linear_y, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": self.angular_z}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Twist':
        lin = data.get("linear", {})
        ang = data.get("angular", {})
        return cls(
            linear_x=lin.get("x", 0.0),
            linear_y=lin.get("y", 0.0),
            angular_z=ang.get("z", 0.0)
        )


class Odometry:
    """Standard nav_msgs/Odometry equivalent with pose and twist."""
    def __init__(
        self,
        x: float, y: float, theta: float,
        v_linear: float, w_angular: float,
        frame_id: str = "odom",
        child_frame_id: str = "base_link"
    ):
        self.header = Header(frame_id=frame_id)
        self.child_frame_id = child_frame_id
        self.pose = {"x": float(x), "y": float(y), "theta": float(theta)}
        self.twist = {"v": float(v_linear), "w": float(w_angular)}
        self.pose_covariance = [0.01] * 9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "child_frame_id": self.child_frame_id,
            "pose": self.pose,
            "twist": self.twist,
            "covariance": self.pose_covariance
        }


class LaserScan:
    """Standard sensor_msgs/LaserScan equivalent for 2D LiDAR ranges."""
    def __init__(
        self,
        ranges: List[float],
        angle_min: float = -3.14159,
        angle_max: float = 3.14159,
        angle_increment: float = 0.01745,
        range_min: float = 0.1,
        range_max: float = 12.0
    ):
        self.header = Header(frame_id="laser_frame")
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = [float(r) for r in ranges]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "angle_increment": self.angle_increment,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "ranges": self.ranges
        }
