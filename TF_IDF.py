# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:27:04 2026

@author: user
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [
    "Concepts like the machine learning pipeline for beginners",
    "Python Programming for beginners starting to learn",
    "Mastering coding syntax and software development fundamentals in python",
    "Snakes like the python reptiles learn to survive in warm climates and in water"
    ]
vectorizer = TfidfVectorizer()
vectorized_documents = vectorizer.fit_transform(documents)
print("Vocabulary: ", vectorizer.get_feature_names_out())
print("\n")
query = ["learn python programming"]
query_vector = vectorizer.transform(query)
print("Query vector: ")
print(query_vector.shape)
similarities_tfidf = cosine_similarity(query_vector, vectorized_documents)[0]
print("\n Results")
for i, score in enumerate(similarities_tfidf):
    print(f"Document {i} score: {score: .4f}")
