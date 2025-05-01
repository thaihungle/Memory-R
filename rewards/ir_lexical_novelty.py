from zss import simple_distance, Node
import numpy as np
from collections import deque
import re

import hashlib
import numpy as np
from collections import deque
import numpy as np
from collections import deque
import random

class FastTreeNoveltyEstimator:
    def __init__(self, history_size=50):
        """
        Initializes a lexical-based novelty estimator with a limited history window.
        
        :param history_size: Number of past responses to store for novelty computation.
        """
        self.past_responses = deque(maxlen=history_size)
        self.min_sim = float("inf")  # Track min similarity for normalization
        self.max_sim = 0  # Track max similarity for normalization

    def _compute_jaccard(self, response1, response2):
        """
        Computes Jaccard similarity between two responses.
        
        :param response1: First response text.
        :param response2: Second response text.
        :return: Jaccard similarity (higher = more similar).
        """
        words1 = set(response1.lower().split())  # Convert to lowercase and tokenize
        words2 = set(response2.lower().split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union else 0  # Jaccard similarity

    def insert(self, response):
        """
        Inserts a new response into history.
        
        :param response: The LLM-generated response.
        """
        self.past_responses.append(response)

    def compute_novelty(self, response):
        """
        Computes the novelty score of a new response based on lexical similarity.
        
        :param response: The new LLM-generated response.
        :return: Normalized novelty score (0-1).
        """
        if not self.past_responses:
            return 1.0  # First response is maximally novel

        similarities = [self._compute_jaccard(response, past) for past in self.past_responses]
        avg_sim = np.mean(similarities)  # Average similarity to past responses

        # Update min/max similarity values
        self.min_sim = min(self.min_sim, avg_sim)
        self.max_sim = max(self.max_sim, avg_sim)

        # Normalize novelty to range [0,1]
        if self.max_sim > self.min_sim:
            novelty = (1 - avg_sim - self.min_sim) / (self.max_sim - self.min_sim)
        else:
            novelty = 0.5  # Default if range is too narrow

        return novelty

class TreeNoveltyEstimator:
    def __init__(self, history_size=50):
        """
        Initializes a tree-based novelty estimator with a limited history window.
        
        :param window_size: Number of past responses to store for novelty computation.
        """
        self.past_responses = deque(maxlen=history_size)
        self.min_dist = float("inf")
        self.max_dist = 0

    def _build_tree(self, response):
        """
        Converts a text response into a tree structure.
        Each step in reasoning becomes a node in the tree.
        
        :param response: The LLM-generated response (CoT reasoning + final answer).
        :return: Root node of the generated tree.
        """
        # steps = re.split(r'\n+|(?<=\.)\s+|(?<=\d:)\s+', response.strip())
        steps = response.strip().split("\n")
        steps = [step.strip() for step in steps if step.strip()]  # Remove empty lines
        k=8
        steps = random.sample(steps, min(k, len(steps)))  # Sample up to k steps

        root = Node("root")  # Root node (meta step)
        for line in steps:
            step_node = Node(line.strip())  # Each reasoning step is a child node
            root.addkid(step_node)
            

        return root

    def insert(self, response):
        """
        Inserts a new response into history, updating novelty normalization.
        
        :param response: The LLM-generated response.
        """
        new_tree = self._build_tree(response)

        if self.past_responses:
            # Compute tree edit distances to past responses
            distances = [simple_distance(new_tree, past_tree) for past_tree in self.past_responses]
            min_dist, max_dist = min(distances), max(distances)

            # Update global min/max for normalization
            self.min_dist = min(self.min_dist, min_dist)
            self.max_dist = max(self.max_dist, max_dist)

        self.past_responses.append(new_tree)

    def compute_novelty(self, response):
        """
        Computes the novelty score of a new response.
        
        :param response: The new LLM-generated response.
        :return: Normalized novelty score (0-1).
        """
        if not self.past_responses:
            return 1.0  # First response is maximally novel

        new_tree = self._build_tree(response)
        distances = [simple_distance(new_tree, past_tree) for past_tree in self.past_responses]

        # Compute min-max normalized novelty
        if self.max_dist > self.min_dist:
            novelty = (np.mean(distances) - self.min_dist) / (self.max_dist - self.min_dist)
        else:
            novelty = 0  # Default if normalization fails

        return novelty

if __name__ == '__main__':
    estimator = TreeNoveltyEstimator(window_size=50)

    # Example responses
    response1 = """Step 1: Assume x = 2
    Step 2: Compute f(x) = x^2 + 3x
    Step 3: Result is 10"""

    response2 = """Step 1: Let x = 2
    Step 2: Evaluate f(x) = x^2 + 3x + 1
    Step 3: Result is 11"""

    response3 = """Step 1: Define x = 3
    Step 2: Let do something different g(x) = x^3 - 2x
    Step 3: Result is 21"""

    # Insert first response
    estimator.insert(response1)

    # Compute novelty of a second response
    print("Novelty of response2:", estimator.compute_novelty(response2))  
    # Insert it into history
    estimator.insert(response2)

    # Compute novelty of a third response
    print("Novelty of response3:", estimator.compute_novelty(response3))  
    estimator.insert(response3)