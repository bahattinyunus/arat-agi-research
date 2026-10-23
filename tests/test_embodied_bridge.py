import pytest
import time
import numpy as np
from autonomous_control.embodied_bridge import (
    Twist,
    Odometry,
    LaserScan,
    ZenohDDSBridge,
    Sim2RealDomainRandomizer
)

def test_messages_serialization():
    t = Twist(linear_x=1.2, linear_y=0.0, angular_z=-0.3)
    d = t.to_dict()
    assert d["linear"]["x"] == 1.2
    assert d["angular"]["z"] == -0.3

    t2 = Twist.from_dict(d)
    assert t2.linear_x == 1.2
    assert t2.angular_z == -0.3

    odom = Odometry(x=1.0, y=2.0, theta=0.5, v_linear=1.0, w_angular=0.1)
    d_odom = odom.to_dict()
    assert d_odom["pose"]["x"] == 1.0
    assert d_odom["twist"]["w"] == 0.1

    scan = LaserScan(ranges=[1.0, 2.0, 3.0])
    d_scan = scan.to_dict()
    assert len(d_scan["ranges"]) == 3

def test_zenoh_dds_bridge_pubsub_and_deadman():
    bridge = ZenohDDSBridge(deadman_timeout_s=0.1)
    received = []

    bridge.create_subscription("/odom", lambda msg: received.append(msg))
    odom = Odometry(0.0, 0.0, 0.0, 0.0, 0.0)
    bridge.publish("/odom", odom)
    assert len(received) == 1

    # Publish cmd_vel
    cmd = Twist(linear_x=1.5, angular_z=0.2)
    bridge.publish("/cmd_vel", cmd)
    assert bridge.get_safe_cmd_vel().linear_x == 1.5

    # Wait beyond deadman timeout (0.12s)
    time.sleep(0.12)
    safe_cmd = bridge.get_safe_cmd_vel()
    # Deadman switch must reset velocity to 0.0
    assert safe_cmd.linear_x == 0.0
    assert safe_cmd.angular_z == 0.0

def test_sim2real_domain_randomization():
    randomizer = Sim2RealDomainRandomizer(
        wheel_slip_std=0.05,
        lidar_noise_std=0.02,
        lidar_dropout_rate=0.1,
        motor_deadband=0.05
    )

    # 1. Motor deadband: commands below 0.05 should produce 0 velocity
    v_act, w_act = randomizer.randomize_kinematics(target_v=0.02, target_w=0.01)
    assert v_act == 0.0
    assert w_act == 0.0

    # Commands above deadband should execute with slip
    v_act2, w_act2 = randomizer.randomize_kinematics(target_v=1.0, target_w=0.5)
    assert v_act2 > 0.0

    # 2. LiDAR noise & dropouts
    true_ranges = [1.0, 2.0, 3.0, 4.0, 5.0]
    noisy_ranges = randomizer.randomize_lidar(true_ranges, max_range=12.0)
    assert len(noisy_ranges) == 5

    # 3. IMU noise & bias drift
    acc, gyro = randomizer.randomize_imu(true_accel=0.0, true_gyro=0.0)
    assert isinstance(acc, float)
    assert isinstance(gyro, float)
