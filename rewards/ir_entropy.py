import numpy as np
import collections
import math
from collections import deque

class EntropyNoveltyEstimator:
    def __init__(self, history_size=100):
        """
        Initializes the entropy-based novelty estimator.

        :param history_size: Number of past entropy values used for normalization.
        """
        self.past_texts = []  # Stores past generations
        self.novelty_history = deque(maxlen=history_size)  # Stores past entropy scores

    def _compute_entropy(self, text):
        """
        Computes Shannon entropy of the character distribution in the given text.
        :param text: Input string.
        :return: Entropy value.
        """
        if not text:
            return 0.0

        counter = collections.Counter(text)
        total = sum(counter.values())

        probabilities = np.array(list(counter.values())) / total
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def update_and_score(self, new_text):
        """
        Updates the memory with a new generation and computes its novelty score.

        :param new_text: The new generated text from the LLM.
        :return: Normalized novelty score (0 to 1).
        """
        # Combine new text with past generations
        combined_text = " ".join(self.past_texts + [new_text])
        entropy = self._compute_entropy(combined_text)
        if len(self.past_texts)>1000:
            self.past_texts.pop(0)

        # Store new text
        self.past_texts.append(new_text)
        
        # Normalize using running window
        self.novelty_history.append(entropy)
        min_entropy = min(self.novelty_history) if self.novelty_history else entropy
        max_entropy = max(self.novelty_history) if self.novelty_history else entropy

        if max_entropy == min_entropy:
            return 0.0  # Avoid division by zero
        
        return (entropy - min_entropy) / (max_entropy - min_entropy)

if __name__ == '__main__':
    novelty_estimator = EntropyNoveltyEstimator(history_size=50)

    gen1 = "Let x be a number. Then, we solve for x^2 = 4."
    gen2 = "The roots are x = ±2. Now, consider a transformation."
    gen3 = "Applying the Fourier transform to f(x) = e^(-x^2)..."

    score1 = novelty_estimator.update_and_score(gen1)
    score2 = novelty_estimator.update_and_score(gen2)
    score3 = novelty_estimator.update_and_score(gen3)

    print(f"Novelty Score 1: {score1:.4f}")
    print(f"Novelty Score 2: {score2:.4f}")
    print(f"Novelty Score 3: {score3:.4f}")