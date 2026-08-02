import numpy as np
import time

class SimpleEmbedder:
    """
    A lightweight, deterministic string embedder.
    Converts strings into unit vectors of dimension 'd' using word and character hashing.
    Used for standalone vector search without external API dependencies.
    """
    def __init__(self, dimension=128):
        self.dimension = dimension

    def embed(self, text):
        vector = np.zeros(self.dimension)
        words = text.lower().split()
        
        # Word hashing
        for i, word in enumerate(words):
            hash_val = hash(word)
            idx = abs(hash_val) % self.dimension
            val = 1.0 / (i + 1.0)
            vector[idx] += val

        # Character n-gram hashing
        for i in range(len(text) - 2):
            ngram = text[i:i+3]
            hash_val = hash(ngram)
            idx = abs(hash_val) % self.dimension
            vector[idx] += 0.1

        vector += 1e-5
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector

class WorkingMemory:
    """
    Working memory model representing short-term context.
    Items have a decay factor and are cleared over time or capacity limits.
    """
    def __init__(self, capacity=5, decay_rate=0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.buffer = []

    def add(self, item, importance=1.0):
        for entry in self.buffer:
            entry["importance"] -= self.decay_rate
            
        self.buffer.append({
            "item": item,
            "importance": importance,
            "timestamp": time.time()
        })
        
        self.buffer = [entry for entry in self.buffer if entry["importance"] > 0]
        self.buffer = sorted(self.buffer, key=lambda x: x["importance"], reverse=True)
        self.buffer = self.buffer[:self.capacity]

    def get_contents(self):
        return [entry["item"] for entry in self.buffer]

class EpisodicMemory:
    """
    Episodic memory system storing sequence of states, actions, and observations.
    """
    def __init__(self):
        self.episodes = []

    def record_episode(self, state, action, observation, reward=0.0):
        episode = {
            "state": state,
            "action": action,
            "observation": observation,
            "reward": reward,
            "timestamp": time.time()
        }
        self.episodes.append(episode)

    def search_episodes(self, query_state=None, query_action=None):
        results = []
        for ep in self.episodes:
            match = True
            if query_state is not None and ep["state"] != query_state:
                match = False
            if query_action is not None and ep["action"] != query_action:
                match = False
            if match:
                results.append(ep)
        return results

class SemanticMemory:
    """
    Semantic memory database implementing vector storage, cosine similarity search,
    and consolidation of episodic logs.
    """
    def __init__(self, dimension=128):
        self.embedder = SimpleEmbedder(dimension=dimension)
        self.keys = []
        self.vectors = None
        self.metadata = []

    def store(self, text, metadata=None):
        vector = self.embedder.embed(text)
        self.keys.append(text)
        self.metadata.append(metadata or {})
        
        if self.vectors is None:
            self.vectors = np.expand_dims(vector, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vector])

    def query(self, query_text, top_k=3):
        if self.vectors is None:
            return []
            
        query_vector = self.embedder.embed(query_text)
        similarities = np.dot(self.vectors, query_vector)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.keys[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results

    def consolidate_episodes(self, episodic_memory: EpisodicMemory):
        """
        Consolidates raw episodic experiences into semantic generalized facts.
        Groups observations by state-action, structures fact texts, and writes to semantic storage.
        """
        episodes = episodic_memory.search_episodes()
        if not episodes:
            return 0
            
        # Group statistics
        state_action_groups = {}
        for ep in episodes:
            key = (ep["state"], ep["action"])
            if key not in state_action_groups:
                state_action_groups[key] = []
            state_action_groups[key].append(ep)
            
        consolidated_count = 0
        for (state, action), group in state_action_groups.items():
            avg_reward = np.mean([ep["reward"] for ep in group])
            # Find the most common observation
            observations = [ep["observation"] for ep in group]
            unique_obs, counts = np.unique(observations, return_counts=True)
            most_common_obs = unique_obs[np.argmax(counts)]
            
            fact_text = (
                f"Taking action '{action}' in state '{state}' frequently leads to "
                f"observation '{most_common_obs}' yielding an average reward of {avg_reward:.2f}."
            )
            
            # Avoid storing duplicate generalizations
            if fact_text not in self.keys:
                self.store(fact_text, {"type": "consolidated_fact", "state": state, "action": action})
                consolidated_count += 1
                
        return consolidated_count
