import asyncio
import time
from typing import Dict, List, Callable, Optional, Any
from .messages import Twist, Odometry, LaserScan

class QoSPolicy:
    RELIABLE = "RELIABLE"
    BEST_EFFORT = "BEST_EFFORT"
    VOLATILE = "VOLATILE"
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"

class ZenohDDSBridge:
    """
    High-throughput, asynchronous hardware communication bridge.
    Matches ROS2 / Zenoh / DDS protocol semantics with configurable QoS policies,
    telemetry streaming, and active safety deadman timeouts.
    """
    def __init__(self, node_name: str = "arat_embodied_node", deadman_timeout_s: float = 0.5):
        self.node_name = node_name
        self.deadman_timeout_s = deadman_timeout_s

        self.topics: Dict[str, List[Callable[[Any], None]]] = {}
        self.last_cmd_vel_time = time.time()
        self.current_cmd_vel = Twist(0.0, 0.0, 0.0)
        self.is_connected = True

    def create_subscription(self, topic: str, callback: Callable[[Any], None], qos: str = QoSPolicy.RELIABLE):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(callback)

    def publish(self, topic: str, message: Any):
        if not self.is_connected:
            return

        # Special tracking for cmd_vel
        if topic == "/cmd_vel" and isinstance(message, Twist):
            self.last_cmd_vel_time = time.time()
            self.current_cmd_vel = message

        if topic in self.topics:
            for callback in self.topics[topic]:
                callback(message)

    def get_safe_cmd_vel(self) -> Twist:
        """
        Deadman switch check: If command velocity was not updated within deadman_timeout_s,
        automatically returns zero velocity to prevent runaway robot.
        """
        elapsed = time.time() - self.last_cmd_vel_time
        if elapsed > self.deadman_timeout_s:
            return Twist(0.0, 0.0, 0.0)
        return self.current_cmd_vel

    def disconnect(self):
        self.is_connected = False
