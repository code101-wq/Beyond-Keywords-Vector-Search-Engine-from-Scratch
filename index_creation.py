# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:13:26 2026

@author: user
"""

### Building the Index

def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2.-normalize each row vector. """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Prevent division by 0
    norms = np.where(norms == 0, 1e-10, norms)
    return vectors / norms

class VectorIndex:
    def __init__(self):
        self.vectors = None
        self.labels  = None
    
    def add(self, vectors: np.ndarray, labels: list):
        self.vectors = normalize(vectors)
        self.labels  = labels
        print(f"Indexed {len(labels)} items with {vectors.shape[1]}-dimensional embeddings.:")
        
    def search(self, query_vector: np.ndarray, top_k: int = 3):
        query_norm = normalize(query_vector.reshape(1, -1))
        # Perform Cosine Similarity below, the dot product of normalized vectors
        scores = self.vectors @ query_norm.T # shape: (n_items, 1)
        scores = scores.flatten()
        
        # Get the top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.labels[i], float(scores[i])) for i in top_indices]
    
index = VectorIndex()
index.add(embeddings, products)