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
    Supports both Linear Opinion Pools and Log-Linear Opinion Pools (Product of Experts).
    """
    def __init__(self):
        self.subscribers = {}  # dict: {topic: list of callbacks/queues}
        self.logger = logging.getLogger("EpistemicBroker")
        logging.basicConfig(level=logging.INFO)

    def subscribe(self, topic, queue: asyncio.Queue):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(queue)
        self.logger.info(f"Subscriber registered for topic: {topic}")

    async def publish(self, message: EpistemicMessage):
        if message.topic in self.subscribers:
            tasks = []
            for queue in self.subscribers[message.topic]:
                tasks.append(queue.put(message))
            if tasks:
                await asyncio.gather(*tasks)

    def fuse_beliefs(self, beliefs, confidences):
        """
        Linear Opinion Pool:
        q_fused(s) = sum( w_i * q_i(s) ) / sum(w_i)
        """
        if not beliefs:
            return None
        
        beliefs = [np.array(b) for b in beliefs]
        confidences = np.array(confidences)
        total_confidence = np.sum(confidences) + 1e-12
        
        weights = confidences / total_confidence
        
        fused_belief = np.zeros_like(beliefs[0])
        for w, b in zip(weights, beliefs):
            fused_belief += w * b
            
        return fused_belief

    def fuse_beliefs_log_linear(self, beliefs, confidences):
        """
        Log-Linear Opinion Pool (Product of Experts):
        q_fused(s) = exp( sum( w_i * ln q_i(s) ) ) / Z
        Enforces consensus and intersections of beliefs.
        """
        if not beliefs:
            return None

        beliefs = [np.array(b) for b in beliefs]
        confidences = np.array(confidences)
        total_confidence = np.sum(confidences) + 1e-12
        weights = confidences / total_confidence

        # Compute weighted sum of log probabilities
        log_fused = np.zeros_like(beliefs[0])
        for w, b in zip(weights, beliefs):
            # Avoid log(0)
            log_fused += w * np.log(b + 1e-12)

        # Exponentiate and normalize
        fused_belief = np.exp(log_fused - np.max(log_fused)) # Stable exp
        fused_belief = fused_belief / (np.sum(fused_belief) + 1e-12)
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
        self.is_running = True
        while self.is_running:
            try:
                msg = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self.process_message(msg)
                self.inbox.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error in agent {self.agent_id} listen loop: {e}")

    async def process_message(self, message: EpistemicMessage):
        pass

    def stop(self):
        self.is_running = False
