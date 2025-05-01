import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque

class FastKNNMemory:
    def __init__(self, model_name='all-MiniLM-L6-v2', metric='L2', 
    max_keys=1000, max_values=50, history_size=100,  anneal_rate=0.99, explore_phase=0):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim) if metric == 'L2' else faiss.IndexFlatIP(self.dim)
        self.keys = []  # Store key embeddings
        self.values = {}  # Store associated lists of (value string, value embedding) pairs
        self.max_keys = max_keys
        self.max_values = max_values
        self.history_size = history_size
        self.novelty_history = deque(maxlen=history_size)  # Running window for normalization
        self.anneal_factor = 1.0  # Initial annealing weight
        self.anneal_rate = anneal_rate  # Decay rate for annealing
        self.explore_phase = explore_phase
    
    def encode_text(self, text):
        return self.model.encode(text, convert_to_numpy=True).astype(np.float32)
    
    def insert_pair(self, str_k, str_v):
        key_embedding = self.encode_text(str_k)
        value_embedding = self.encode_text(str_v)
        
        if len(self.keys) == 0:
            self.index.add(np.array([key_embedding]))
            self.keys.append(key_embedding)
            self.values[len(self.keys) - 1] = [(str_v, value_embedding)]
        else:
            query_vector = key_embedding.reshape(1, -1)
            _, indices = self.index.search(query_vector, 1)
            nearest_index = indices[0][0] if indices[0][0] != -1 else None
            
            if nearest_index is not None and nearest_index < len(self.keys) and np.array_equal(self.keys[nearest_index], key_embedding):
                if len(self.values[nearest_index]) >= self.max_values:
                    self.values[nearest_index].pop(0)  # Keep the value list within limit
                self.values[nearest_index].append((str_v, value_embedding))
            else:
                if len(self.keys) >= self.max_keys:
                    self.keys.pop(0)
                    self.values.pop(0)
                    self.index.reset()
                    self.index.add(np.array(self.keys))
                self.index.add(np.array([key_embedding]))
                self.keys.append(key_embedding)
                self.values[len(self.keys) - 1] = [(str_v, value_embedding)]
    
    def retrieve(self, query_text, k=5):
        query_vector = self.encode_text(query_text).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        retrieved_keys = [self.keys[i] for i in indices[0] if i != -1]
        retrieved_values = [self.values[i] for i in indices[0] if i != -1]
        return retrieved_keys, retrieved_values, distances[0]

    def novelty_score_mean(self, str_k, str_v, k=5):
        if len(self.keys) <self.explore_phase:
            return None
        _, retrieved_values, _ = self.retrieve(str_k, k)
        value_embedding = self.encode_text(str_v)
        
        all_retrieved_embeddings = [v_emb for values in retrieved_values for _, v_emb in values]
        
        if not all_retrieved_embeddings:
            return 1.0  # Maximum novelty if no prior values exist
        
        all_retrieved_embeddings = np.array(all_retrieved_embeddings)
        centroid = np.mean(all_retrieved_embeddings, axis=0)
        
        # Compute distance from centroid
        novelty = np.linalg.norm(value_embedding - centroid)
        # Update history and normalize
        self.novelty_history.append(novelty)
        min_novelty = min(self.novelty_history) if self.novelty_history else novelty
        max_novelty = max(self.novelty_history) if self.novelty_history else novelty
        
        if max_novelty == min_novelty:
            return novelty  # Avoid division by zero
        return (novelty - min_novelty) / (max_novelty - min_novelty)

    def novelty_score_max(self, str_k, str_v, k=5):
        if len(self.keys) <self.explore_phase:
            return None
        _, retrieved_values, _ = self.retrieve(str_k, k)
        value_embedding = self.encode_text(str_v)
        
        # Flatten retrieved value embeddings
        all_retrieved_embeddings = [v_emb for values in retrieved_values for _, v_emb in values]
        
        if not all_retrieved_embeddings:
            return 1.0  # Maximum novelty if no prior values exist
        
        all_retrieved_embeddings = np.array(all_retrieved_embeddings)
        
        # Compute cosine similarity to nearest values
        value_embedding = value_embedding.reshape(1, -1)
        similarity = np.dot(all_retrieved_embeddings, value_embedding.T) / (
            np.linalg.norm(all_retrieved_embeddings, axis=1) * np.linalg.norm(value_embedding))
        
        novelty = 1 - np.max(similarity)  # Higher novelty if similarity is low
        # Update history and normalize
        self.novelty_history.append(novelty)
        min_novelty = min(self.novelty_history) if self.novelty_history else novelty
        max_novelty = max(self.novelty_history) if self.novelty_history else novelty
        
        if max_novelty == min_novelty:
            return 0.0  # Avoid division by zero
        normalized_novelty = (novelty - min_novelty) / (max_novelty - min_novelty)
        
        # Apply annealing factor
        novelty_score = self.anneal_factor * normalized_novelty
        
        # Update annealing factor
        self.anneal_factor *= self.anneal_rate
        return novelty_score
    
    def size(self):
        return self.index.ntotal
        
if __name__ == '__main__':
    # Example usage:
    knn_memory = FastKNNMemory()

    # Insert text-value pairs
    knn_memory.insert_pair("Example key 1", "Example value 1")
    knn_memory.insert_pair("Example key 1", "Example value 2")
    knn_memory.insert_pair("Example key 2", "Example value 3")
    knn_memory.insert_pair("Example key 2", "Example value 4")
    knn_memory.insert_pair("Example key 3", "Example value 5")
    knn_memory.insert_pair("Example key 4", "Example value 6")
    # Compute novelty score
    novelty = knn_memory.novelty_score_mean("Example key 1", "New value")
    print("Novelty Score:", novelty)
    novelty = knn_memory.novelty_score_max("Example key 1", "Example value 1")
    print("Novelty Score:", novelty)
    novelty = knn_memory.novelty_score_max("Example key 1", "Example value 7")
    print("Novelty Score:", novelty)