# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:41:32 2026

@author: user
"""

### Visualizing the Embedding Space

import matplotlib.pyplot as plt
import matplotlib.patches as mpathces

# Writing the pca_2d function to reduce the dimensions from 8 to 2
import numpy as np
from sklearn.decomposition import PCA

def pca_2d(X: np.ndarray) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        X = np.array(X)
        
    if X.ndim != 2:
        raise ValueError(f"Input must be a 2D matrix of shape (samples, dimensions), got {X.ndim}D.")
        
    n_samples, n_features = X.shape
    
    if n_samples < 2 or n_features < 2:
        raise ValueError("Data must have at least 2 samples and 2 features to reduce to 2D.")

    # 2. Initialize the PCA model targeting 2 components
    pca = PCA(n_components=2)
    
    # 3. Fit the model (learn the directions of maximum variance) 
    #    and transform the data (project it onto those 2 directions)
    X_reduced = pca.fit_transform(X)
    
    return X_reduced

projected = pca_2d(embeddings)

cluster_colors = (
    ["#4A90D9"] * 5 +    # electronics -- blue
    ["#E8734A"] * 5 +    # clothing -- orange
    ["#5BAD72"] * 5      # furniture -- green
    )
cluster_labels = ["Electronicds"] * 5 + ["Clothing"] * 5 + ["Furniture"]

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(projected[:, 0], projected[:, 1],
           c=cluster_colors, s=100, edgecolor="white", linewidths=0.7, zorder=3)

# Plot query projections
q_projected = pca_2d(
    np.vstack(list(queries.values())) - embeddings.mean(axis=0)
)
for (qname, _), (qx, qy) in zip(queries.items(), q_projected):
    ax.scatter(qx, qy, marker="*", s=200, color="gold",
               edgecolors="#333", linewidths=0.6, zorder=4)
    ax.annotate(f"<- query: {qname}", (qx, qy),
                textcoords="offset points", xytext=(6, -8),
                fontsize=7, color="#555555", style="italic")
    
legend_patches = [
    mpatches.Patch(color="#4A90D9", label="Electronics"),
    mpatches.Patch(color="#E8734A", label="Clothing"),
    mpatches.Patch(color="#5BAD72", label="Furniture"),
    mpatches.Patch(color="gold",    label="Query vectors"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=6)
ax.set_title("Vector Search — Embedding Space (PCA projection)", fontsize=10, pad=10)
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("embedding_space_queries_only.png", dpi=150)
plt.show()
