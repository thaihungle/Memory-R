import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque
import nltk
from nltk.tokenize import sent_tokenize

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
        sentences = sent_tokenize(str_v)
        key_embedding = self.encode_text(str_k)
        value_embeddings = [self.encode_text(sent) for sent in sentences]
        
        if len(self.keys) == 0:
            self.index.add(np.array([key_embedding]))
            self.keys.append(key_embedding)
            self.values[len(self.keys) - 1] = [(sent, emb) for sent, emb in zip(sentences, value_embeddings)]
        else:
            query_vector = key_embedding.reshape(1, -1)
            _, indices = self.index.search(query_vector, 1)
            nearest_index = indices[0][0] if indices[0][0] != -1 else None
            
            if nearest_index is not None and nearest_index < len(self.keys) and np.array_equal(self.keys[nearest_index], key_embedding):
                if len(self.values[nearest_index]) >= self.max_values:
                    self.values[nearest_index].pop(0)
                self.values[nearest_index].extend([(sent, emb) for sent, emb in zip(sentences, value_embeddings)])
            else:
                if len(self.keys) >= self.max_keys:
                    self.keys.pop(0)
                    self.values.pop(0)
                    self.index.reset()
                    self.index.add(np.array(self.keys))
                self.index.add(np.array([key_embedding]))
                self.keys.append(key_embedding)
                self.values[len(self.keys) - 1] = [(sent, emb) for sent, emb in zip(sentences, value_embeddings)]
    
    def retrieve(self, query_text, k=5):
        query_vector = self.encode_text(query_text).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        retrieved_keys = [self.keys[i] for i in indices[0] if i != -1]
        retrieved_values = [self.values[i] for i in indices[0] if i != -1]
        return retrieved_keys, retrieved_values, distances[0]

    def novelty_score_mean(self, str_k, str_v, k=5):
        if len(self.keys) < self.explore_phase:
            return 0

        # return self.novelty_score_max(str_k, str_v, k)

        sentences = sent_tokenize(str_v)
        _, retrieved_values, _ = self.retrieve(str_k, k)
        sentence_embeddings = [self.encode_text(sent) for sent in sentences]
        
        all_retrieved_embeddings = [v_emb for values in retrieved_values for _, v_emb in values]
        
        if not all_retrieved_embeddings:
            return 1.0  # Maximum novelty if no prior values exist
        
        all_retrieved_embeddings = np.array(all_retrieved_embeddings)
        centroid = np.mean(all_retrieved_embeddings, axis=0)
        
        novelty_scores = [np.linalg.norm(sent_emb - centroid) for sent_emb in sentence_embeddings]
        avg_novelty = np.mean(novelty_scores)
        
        self.novelty_history.append(avg_novelty)
        min_novelty = min(self.novelty_history) if self.novelty_history else avg_novelty
        max_novelty = max(self.novelty_history) if self.novelty_history else avg_novelty
        
        if max_novelty == min_novelty:
            return avg_novelty  # Avoid division by zero
        r = (avg_novelty - min_novelty) / (max_novelty - min_novelty)
        if np.isnan(r):
            return 0
        print(r)
        return r

    def novelty_score_max(self, str_k, str_v, k=5):
        if len(self.keys) < self.explore_phase:
            return 0
        sentences = sent_tokenize(str_v)
        _, retrieved_values, _ = self.retrieve(str_k, k)
        sentence_embeddings = [self.encode_text(sent) for sent in sentences]
        
        all_retrieved_embeddings = [v_emb for values in retrieved_values for _, v_emb in values]
        
        if not all_retrieved_embeddings:
            return 1.0  # Maximum novelty if no prior values exist
        
        all_retrieved_embeddings = np.array(all_retrieved_embeddings)
        
        novelty_scores = []
        for sent_emb in sentence_embeddings:
            sent_emb = sent_emb.reshape(1, -1)
            similarity = np.dot(all_retrieved_embeddings, sent_emb.T) / (
                np.linalg.norm(all_retrieved_embeddings, axis=1) * np.linalg.norm(sent_emb))
            novelty_scores.append(1 - np.max(similarity))
        
        avg_novelty = np.mean(novelty_scores)
        self.novelty_history.append(avg_novelty)
        min_novelty = min(self.novelty_history) if self.novelty_history else avg_novelty
        max_novelty = max(self.novelty_history) if self.novelty_history else avg_novelty
        
        if max_novelty == min_novelty:
            return 0.0  # Avoid division by zero
        normalized_novelty = (avg_novelty - min_novelty) / (max_novelty - min_novelty)
        
        novelty_score = self.anneal_factor * normalized_novelty
        self.anneal_factor *= self.anneal_rate
        return novelty_score
