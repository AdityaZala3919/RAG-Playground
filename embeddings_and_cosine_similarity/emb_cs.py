import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# def cosine_similarity(a, b):
#     a = np.array(a)
#     b = np.array(b)
    
#     dot_product = np.dot(a, b)
#     magnitude_a = np.linalg.norm(a)
#     magnitude_b = np.linalg.norm(b)
    
#     if magnitude_a == 0 or magnitude_b == 0:
#         raise ValueError("Cosine similarity is not defined for zero-length vectors.")
    
#     return dot_product / (magnitude_a * magnitude_b)

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love driving my car",
    "Automobiles need fuel",
    "Bananas are yellow fruits"
]

embeddings = model.encode(sentences)

print(embeddings.shape)

similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(6, 6))

for i, sentence in enumerate(sentences):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, sentence, fontsize=9)

plt.axhline(0)
plt.axvline(0)
plt.title("Sentence Embeddings in 2D Space")
plt.show()

origin = np.zeros(2)

plt.figure(figsize=(6, 6))

for i, sentence in enumerate(sentences):
    vec = reduced_embeddings[i]
    plt.arrow(
        origin[0], origin[1],
        vec[0], vec[1],
        head_width=0.02,
        length_includes_head=True
    )
    plt.text(vec[0], vec[1], sentence, fontsize=9)

plt.axhline(0)
plt.axvline(0)
plt.title("Cosine Similarity = Angle Between Vectors")
plt.show()

query = "I like vehicles"
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]

for s, score in zip(sentences, similarities):
    print(f"{s} → {score:.3f}")