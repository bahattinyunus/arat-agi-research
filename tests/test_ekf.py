import unittest
import numpy as np
from autonomous_control.sensor_fusion.ekf_fusion_node import EKFSensorFusionNode

class TestEKF(unittest.TestCase):
    def test_ekf_fusion_node(self):
        # Initialize EKF at origin (0, 0, 0)
        node = EKFSensorFusionNode(dt=0.1, initial_pose=(0.0, 0.0, 0.0))
        
        # Test initial state
        state = node.get_state_estimate()
        self.assertAlmostEqual(state["x"], 0.0)
        self.assertAlmostEqual(state["y"], 0.0)
        self.assertAlmostEqual(state["theta"], 0.0)
        
        # Run prediction step with linear velocity = 1.0 m/s, angular velocity = 0.0 rad/s
        # x should propagate to x + v*dt = 0.1
        state_pred = node.process_step(linear_vel=1.0, angular_vel=0.0)
        self.assertAlmostEqual(state_pred["x"], 0.1)
        self.assertAlmostEqual(state_pred["y"], 0.0)
        self.assertAlmostEqual(state_pred["theta"], 0.0)
        
        # Run update step with measurement at (0.2, 0.05)
        # Position should adjust towards the measurement
        state_corr = node.process_step(linear_vel=1.0, angular_vel=0.0, meas_pos=(0.2, 0.05))
        
        self.assertTrue(state_corr["x"] > 0.1) # should pull state towards measurement x=0.2
        self.assertTrue(state_corr["y"] > 0.0) # should pull state towards measurement y=0.05
        
        # Check covariance
        self.assertEqual(len(state_corr["covariance"]), 3)
        for cov in state_corr["covariance"]:
            self.assertTrue(cov > 0.0)

if __name__ == "__main__":
    unittest.main()
