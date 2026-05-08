import faiss, json
import numpy as np
from embed import get_embeddings

INDEX_PATH = ".//data//faiss.index"
META_PATH = ".//data//metadata.json"

documents = [
    {"id": 1, "category": "tech", "text": "Artificial intelligence is transforming industry."},
    {"id": 2, "category": "sport", "text": "Football is the most popular sport in the world."},
    {"id": 3, "category": "tech", "text": "Neural networks are a subset of machine learning."},
    {"id": 4, "category": "cooking", "text": "Italian pasta recipes are easy to follow."},
    {"id": 5, "category": "sport", "text": "Olympics bring nations together through sportsmanship."}
]

# 1. Embed
texts = [d["text"] for d in documents]
embeddings = np.array(get_embeddings(texts)).astype("float32")
dimension = embeddings.shape[1]

query = "Deep learning in AI"
query_embedding = np.array(get_embeddings([query])).astype("float32")

# 2. Indexing
# Indexing: IndexFlatL2-Euclidean Distance, IndexFlatIP-Dot Product
index = faiss.IndexFlatL2(dimension)
# index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
# print(index.ntotal)

# 3. Storing Indexes & Metadata
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w") as f:
    json.dump(documents, f, indent=2)

# 4. Retrieval of Indexes & Metadata
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

# 5. Semantic Search
def search(query_vec, top_k=2):
    """Return top_k most similar docs"""
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        doc = metadata[idx]
        results.append({
            "rank": rank + 1,
            "id": doc["id"],
            "text": doc["text"],
            "category": doc["category"],
            "distance": float(distances[0][rank]),
        })
    return results

for r in search(query_embedding, top_k=5):
    print(r)

# 6. Metadata Filtering
def search_with_filter(query_vec, filter_key, filter_val, top_k=3):
    """Search then filter by metadata"""
    raw = search(query_vec, top_k=len(metadata))
    return [r for r in raw if r[filter_key] == filter_val][:top_k]

for r in search_with_filter(query_embedding, "category", "sport", top_k=2):
    print(r)