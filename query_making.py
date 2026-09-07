# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:39:18 2026

@author: user
"""

### Running Querries

def make_query(center: np.ndarray, noise_scale: float= 0.05) -> np.ndarray:
    return center + np.random.randn(8) * noise_scale

# Here the querries are constructed from thesame cluster centers as their document

queries = {
    "audio equipment": make_query(electronics_center),
    "casual wear"    : make_query(clothing_center),
    "home furniture" : make_query(furniture_center)
    }
for query_name, q_vec in queries.items():
    print(f"\nQuery: '{query_name}'")
    results = index.search(q_vec, top_k=3)
    for rank, (label, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {label}")