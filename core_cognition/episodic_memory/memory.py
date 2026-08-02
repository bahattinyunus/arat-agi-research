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
        # Deterministic vector generation using hashing
        vector = np.zeros(self.dimension)
        words = text.lower().split()
        
        # Word hashing
        for i, word in enumerate(words):
            hash_val = hash(word)
            idx = abs(hash_val) % self.dimension
            val = 1.0 / (i + 1.0)
            vector[idx] += val

        # Character n-gram hashing for sub-word features
        for i in range(len(text) - 2):
            ngram = text[i:i+3]
            hash_val = hash(ngram)
            idx = abs(hash_val) % self.dimension
            vector[idx] += 0.1

        # Add small constant noise to break zeros
        vector += 1e-5
        
        # L2 Normalize
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
        self.buffer = []  # List of dicts: {"item": x, "importance": float, "timestamp": float}

    def add(self, item, importance=1.0):
        # Decay current items
        for entry in self.buffer:
            entry["importance"] -= self.decay_rate
            
        # Add new item
        self.buffer.append({
            "item": item,
            "importance": importance,
            "timestamp": time.time()
        })
        
        # Sort by importance and prune
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
        self.episodes = []  # List of episodes

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
        """Filters episodes matching criteria"""
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
    Semantic memory database implementing vector storage and cosine similarity search.
    """
    def __init__(self, dimension=128):
        self.embedder = SimpleEmbedder(dimension=dimension)
        self.keys = []        # Store text keys
        self.vectors = None   # Numpy array of shape (num_records, dimension)
        self.metadata = []    # List of metadata dicts

    def store(self, text, metadata=None):
        vector = self.embedder.embed(text)
        self.keys.append(text)
        self.metadata.append(metadata or {})
        
        if self.vectors is None:
            self.vectors = np.expand_dims(vector, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vector])

    def query(self, query_text, top_k=3):
        """
        Queries semantic memory using cosine similarity.
        """
        if self.vectors is None:
            return []
            
        query_vector = self.embedder.embed(query_text)
        
        # Cosine similarity: (A . B) / (||A|| * ||B||)
        # Since vectors are already normalized, similarity is just the dot product
        similarities = np.dot(self.vectors, query_vector)
        
        # Get sorted indices in descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.keys[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        return results
