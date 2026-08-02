import asyncio
import numpy as np
import logging

class EpistemicMessage:
    """
    Standard message wrapper for agent communication.
    Includes the sender ID, topic, content payload, and epistemic beliefs.
    """
    def __init__(self, sender_id, topic, content, belief=None, confidence=1.0):
        self.sender_id = sender_id
        self.topic = topic
        self.content = content
        self.belief = belief          # Optional numpy array representing state probabilities q(s)
        self.confidence = confidence  # Trust level / precision associated with the belief

class EpistemicMessageBroker:
    """
    Asynchronous Pub/Sub broker with support for epistemic belief fusion.
    """
    def __init__(self):
        self.subscribers = {}  # dict: {topic: list of callbacks/queues}
        self.logger = logging.getLogger("EpistemicBroker")
        logging.basicConfig(level=logging.INFO)

    def subscribe(self, topic, queue: asyncio.Queue):
        """Registers a queue to receive messages from a specific topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(queue)
        self.logger.info(f"Subscriber registered for topic: {topic}")

    async def publish(self, message: EpistemicMessage):
        """Publishes a message to all subscribers of the topic"""
        if message.topic in self.subscribers:
            tasks = []
            for queue in self.subscribers[message.topic]:
                tasks.append(queue.put(message))
            if tasks:
                await asyncio.gather(*tasks)

    def fuse_beliefs(self, beliefs, confidences):
        """
        Fuses multiple belief distributions q_i(s) into a consensus belief.
        Uses a Linear Opinion Pool (weighted arithmetic mean):
        q_fused(s) = sum( w_i * q_i(s) ) / sum(w_i)
        """
        if not beliefs:
            return None
        
        beliefs = [np.array(b) for b in beliefs]
        confidences = np.array(confidences)
        total_confidence = np.sum(confidences) + 1e-12
        
        # Normalize weights
        weights = confidences / total_confidence
        
        # Fused belief is the weighted sum
        fused_belief = np.zeros_like(beliefs[0])
        for w, b in zip(weights, beliefs):
            fused_belief += w * b
            
        return fused_belief

class OrchestratedAgent:
    """
    Base class for agents integrated into the OMEGA orchestrator.
    """
    def __init__(self, agent_id, broker: EpistemicMessageBroker):
        self.agent_id = agent_id
        self.broker = broker
        self.inbox = asyncio.Queue()
        self.is_running = False

    def subscribe_to(self, topic):
        self.broker.subscribe(topic, self.inbox)

    async def send_message(self, topic, content, belief=None, confidence=1.0):
        msg = EpistemicMessage(
            sender_id=self.agent_id,
            topic=topic,
            content=content,
            belief=belief,
            confidence=confidence
        )
        await self.broker.publish(msg)

    async def listen_loop(self):
        """Main listening loop for processing inbound epistemic messages"""
        self.is_running = True
        while self.is_running:
            try:
                # Wait for a message with timeout to allow clean shutdowns
                msg = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self.process_message(msg)
                self.inbox.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in agent {self.agent_id} listen loop: {e}")

    async def process_message(self, message: EpistemicMessage):
        """Override in subclasses to implement specific message handling"""
        pass

    def stop(self):
        self.is_running = False
