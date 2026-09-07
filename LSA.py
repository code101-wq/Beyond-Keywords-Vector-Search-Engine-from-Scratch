# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:24:03 2026

@author: user
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD
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


lsa_model = TruncatedSVD(n_components=2, random_state=42)
lsa_matrix = lsa_model.fit_transform(vectorized_documents)
print(f"Original TF_IDF shape: { vectorized_documents.shape} (Sparse words)")
print(f"Compressed LSA shape: {lsa_matrix.shape} (Dense hidden concepts)\n")
# Transforming the query into the compressed space
query_lsa = lsa_model.transform(query_vector)
similarities_lsa = cosine_similarity(query_lsa, lsa_matrix)[0]
for i, score in enumerate(similarities_lsa):
    print(f"Document {i} score: {score: .4f} Text: '{documents[i]}'")
