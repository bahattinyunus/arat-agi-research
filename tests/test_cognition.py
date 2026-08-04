import unittest
import numpy as np
from core_cognition.reasoning_engine import ActiveInferenceAgent
from core_cognition.episodic_memory import SemanticMemory, WorkingMemory, EpisodicMemory

class TestCognition(unittest.TestCase):
    def test_active_inference_agent(self):
        # 3 states, 3 observations, 2 actions
        agent = ActiveInferenceAgent(num_states=3, num_obs=3, num_actions=2, learning_rate=0.5)
        
        # Test default initial belief is uniform
        np.testing.assert_array_almost_equal(agent.q, np.array([1/3, 1/3, 1/3]))
        
        # Test infer_states updates belief
        q_updated = agent.infer_states(observation=1)
        self.assertEqual(len(q_updated), 3)
        self.assertAlmostEqual(np.sum(q_updated), 1.0)
        
        # Test free energy calculations
        vfe = agent.calculate_vfe(observation=1)
        self.assertTrue(vfe >= 0)
        
        # Test action selection stashes action stochastically
        action, action_probs = agent.select_action()
        self.assertEqual(agent.prev_action, action)
        self.assertIn(action, [0, 1])

        # Test model learning: update A matrix counts based on state beliefs
        initial_A = np.copy(agent.A)
        agent.update_model(actual_obs=2)
        # Likelihood distribution A should shift
        self.assertFalse(np.array_equal(initial_A, agent.A))

    def test_working_memory(self):
        mem = WorkingMemory(capacity=3, decay_rate=0.2)
        mem.add("event_1", importance=1.0)
        mem.add("event_2", importance=0.8)
        
        self.assertIn("event_1", mem.get_contents())
        self.assertIn("event_2", mem.get_contents())
        
        mem.add("event_3", importance=0.9)
        mem.add("event_4", importance=1.0)
        
        contents = mem.get_contents()
        self.assertEqual(len(contents), 3)
        self.assertIn("event_4", contents)

    def test_semantic_memory(self):
        mem = SemanticMemory(dimension=64)
        mem.store("Artificial General Intelligence research", {"topic": "AGI"})
        mem.store("High performance computing on Nvidia GPU CUDA", {"topic": "CUDA"})
        mem.store("Edge AI inference using TensorRT C++ runtimes", {"topic": "Edge"})
        
        results = mem.query("Nvidia graphics computing with CUDA", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["metadata"]["topic"], "CUDA")

    def test_memory_consolidation(self):
        ep_mem = EpisodicMemory()
        sem_mem = SemanticMemory(dimension=64)
        
        # Add duplicate episodes to create a clear trend: State="idle" + Action="search" -> Obs="data_found", Reward=1.0
        ep_mem.record_episode("idle", "search", "data_found", reward=1.0)
        ep_mem.record_episode("idle", "search", "data_found", reward=1.0)
        ep_mem.record_episode("idle", "search", "sensor_error", reward=0.0) # minor outlier
        
        consolidated = sem_mem.consolidate_episodes(ep_mem)
        self.assertEqual(consolidated, 1) # should consolidate exactly 1 fact
        
        # Query semantic memory to check if it has the generalized fact
        query_res = sem_mem.query("What happens when I search from idle state?")
        self.assertTrue(len(query_res) > 0)
        self.assertIn("leads to observation 'data_found'", query_res[0]["text"])

    def test_hierarchical_active_inference_agent(self):
        from core_cognition.reasoning_engine import HierarchicalActiveInferenceAgent
        
        high_params = {"num_states": 2, "num_obs": 2, "num_actions": 2, "learning_rate": 0.1}
        low_params = {"num_states": 3, "num_obs": 3, "num_actions": 3, "learning_rate": 0.2}
        goal_to_preference_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        h_agent = HierarchicalActiveInferenceAgent(high_params, low_params, goal_to_preference_matrix)
        
        # Run one step
        res = h_agent.run_step(high_obs=0, low_obs=1)
        
        self.assertIn("high_action", res)
        self.assertIn("low_action", res)
        self.assertEqual(len(res["high_belief"]), 2)
        self.assertEqual(len(res["low_belief"]), 3)
        
        # Test C matrix changes dynamically
        np.testing.assert_array_almost_equal(h_agent.low_agent.C, goal_to_preference_matrix[res["high_action"]], decimal=5)

if __name__ == "__main__":
    unittest.main()


