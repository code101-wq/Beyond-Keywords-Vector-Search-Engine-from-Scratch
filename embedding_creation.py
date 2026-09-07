# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:11:52 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
This is a Vector Search(Semantic Search) program built from scratch using just
numpy.

Here we build our own embeddings but further we would use transformers to create 
embeddings.

The setup for this program is as given below
We aork with a set of short product description from an e-commerce catalog.
They are pre-embedded as 8-dimensional vectors, (sentence transformers will be 
used later).

"""

import numpy as np

np.random.seed(42)


### Setting up the dataset

# Product catalog - 3 semantic clusters: electronics, clothing, furniture
products = [
    "Wireless noise-cancelling headphones with 30-hour battery",
    "Bluetooth speaker with waterproof design",
    "USB-C hub with 7 ports and power delivery",
    "4K HDMI cable 6ft braided",
    "Mechanical keyboard with RGB backlight",
    "Men's slim-fit chino pants navy blue",
    "Women's merino wool turtleneck sweater",
    "Unisex running jacket lightweight windbreaker",
    "Leather chelsea boots for men",
    "Organic cotton crew neck t-shirt",
    "Solid oak dinning table seats 6",
    "Ergonomic mesh office chair lumbar support",
    "Linen sofa 3-seater natural beige",
    "Bamboo bookshelf 5-tier adjustable",
    "Memory foam mattress queen size medium firm"
    ]

# Simulate embeddings with a cluster structure
# Cluster centers in 8D space
electronics_center = np.array([0.9, 0.1, 0.2, 0.8, 0.1, 0.3, 0.7, 0.2])
clothing_center    = np.array([0.1, 0.8, 0.7, 0.1, 0.9, 0.2, 0.1, 0.8])
furniture_center   = np.array([0.2, 0.3, 0.9, 0.2, 0.1, 0.9, 0.3, 0.1])

n_per_cluster = 5
noise = 0.08

embeddings = np.vstack([
    electronics_center + np.random.randn(n_per_cluster, 8) * noise,
    clothing_center    + np.random.randn(n_per_cluster, 8) * noise,
    furniture_center   + np.random.randn(n_per_cluster, 8) * noise
    ])

print(f"Embeddings shape: {embeddings.shape}")
