import unittest
import numpy as np
from core_cognition.reasoning_engine import ActiveInferenceAgent
from core_cognition.episodic_memory import SemanticMemory, WorkingMemory

class TestCognition(unittest.TestCase):
    def test_active_inference_agent(self):
        # 3 states, 3 observations, 2 actions
        agent = ActiveInferenceAgent(num_states=3, num_obs=3, num_actions=2)
        
        # Test default initial belief is uniform
        np.testing.assert_array_almost_equal(agent.q, np.array([1/3, 1/3, 1/3]))
        
        # Test infer_states updates belief stochastically
        q_updated = agent.infer_states(observation=1)
        self.assertEqual(len(q_updated), 3)
        self.assertAlmostEqual(np.sum(q_updated), 1.0)
        
        # Test free energy calculations
        vfe = agent.calculate_vfe(observation=1)
        self.assertTrue(vfe >= 0)
        
        # Test action selection
        action, action_probs = agent.select_action()
        self.assertIn(action, [0, 1])
        self.assertAlmostEqual(np.sum(action_probs), 1.0)

    def test_working_memory(self):
        mem = WorkingMemory(capacity=3, decay_rate=0.2)
        mem.add("event_1", importance=1.0)
        mem.add("event_2", importance=0.8)
        
        # Check they exist
        self.assertIn("event_1", mem.get_contents())
        self.assertIn("event_2", mem.get_contents())
        
        # Trigger decay by adding more elements
        mem.add("event_3", importance=0.9)
        mem.add("event_4", importance=1.0)
        
        # Capacity is 3, so first one (event_1 which decayed) or lowest importance should be dropped
        contents = mem.get_contents()
        self.assertEqual(len(contents), 3)
        self.assertIn("event_4", contents)

    def test_semantic_memory(self):
        mem = SemanticMemory(dimension=64)
        mem.store("Artificial General Intelligence research", {"topic": "AGI"})
        mem.store("High performance computing on Nvidia GPU CUDA", {"topic": "CUDA"})
        mem.store("Edge AI inference using TensorRT C++ runtimes", {"topic": "Edge"})
        
        # Query for GPU/CUDA computing
        results = mem.query("Nvidia graphics computing with CUDA", top_k=2)
        self.assertTrue(len(results) > 0)
        
        # Highest match should be the CUDA related entry
        self.assertEqual(results[0]["metadata"]["topic"], "CUDA")
        self.assertTrue(results[0]["score"] > 0)

if __name__ == "__main__":
    unittest.main()
