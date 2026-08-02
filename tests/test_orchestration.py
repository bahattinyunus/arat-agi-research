import unittest
import asyncio
import numpy as np
from core_cognition.agent_orchestration import EpistemicMessageBroker, EpistemicMessage, OrchestratedAgent

class DummyAgent(OrchestratedAgent):
    def __init__(self, agent_id, broker):
        super().__init__(agent_id, broker)
        self.received_messages = []

    async def process_message(self, message):
        self.received_messages.append(message)

class TestOrchestration(unittest.IsolatedAsyncioTestCase):
    async def test_pub_sub_broker(self):
        broker = EpistemicMessageBroker()
        agent1 = DummyAgent("agent_1", broker)
        agent2 = DummyAgent("agent_2", broker)
        
        # Subscribe agent1 to sensor_fusion topic
        agent1.subscribe_to("sensor_fusion")
        
        # Start agent1 listen loop task
        loop_task = asyncio.create_task(agent1.listen_loop())
        
        # Let agent2 publish a message
        belief_state = np.array([0.7, 0.2, 0.1])
        await agent2.send_message(
            topic="sensor_fusion",
            content="Detected obstacle at 2.4 meters",
            belief=belief_state,
            confidence=0.9
        )
        
        # Yield execution to allow broker processing
        await asyncio.sleep(0.1)
        
        # Verify agent1 received it
        self.assertEqual(len(agent1.received_messages), 1)
        received_msg = agent1.received_messages[0]
        self.assertEqual(received_msg.sender_id, "agent_2")
        self.assertEqual(received_msg.content, "Detected obstacle at 2.4 meters")
        np.testing.assert_array_equal(received_msg.belief, belief_state)
        self.assertEqual(received_msg.confidence, 0.9)
        
        # Cleanup
        agent1.stop()
        await loop_task

    def test_belief_fusion(self):
        broker = EpistemicMessageBroker()
        
        # 3 agents posting state probability distributions (3 states)
        belief_1 = np.array([0.8, 0.1, 0.1])
        belief_2 = np.array([0.6, 0.3, 0.1])
        belief_3 = np.array([0.1, 0.8, 0.1])  # Conflicting belief but lower confidence
        
        beliefs = [belief_1, belief_2, belief_3]
        confidences = [0.9, 0.8, 0.2]  # Agent 3 is not trusted/confident
        
        fused = broker.fuse_beliefs(beliefs, confidences)
        
        # Fused belief should weight agent 1 and 2 much more than agent 3
        # Expected = (0.9*b1 + 0.8*b2 + 0.2*b3) / 1.9
        expected = (0.9 * belief_1 + 0.8 * belief_2 + 0.2 * belief_3) / 1.9
        np.testing.assert_array_almost_equal(fused, expected)
        self.assertAlmostEqual(np.sum(fused), 1.0)

if __name__ == "__main__":
    unittest.main()
