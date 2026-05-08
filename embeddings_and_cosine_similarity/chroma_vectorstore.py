import chromadb

chroma_client = chromadb.PersistentClient(path="./data")

collection = chroma_client.get_or_create_collection(name="my_collection", metadata={"hnsw:space": "cosine"})
# metadata={"hnsw:space": "ip"}
# Chroma uses Sentence Transformers all-MiniLM-L6-v2 by default

documents = [
    {"id": 1, "category": "tech", "text": "Artificial intelligence is transforming industry."},
    {"id": 2, "category": "sport", "text": "Football is the most popular sport in the world."},
    {"id": 3, "category": "tech", "text": "Neural networks are a subset of machine learning."},
    {"id": 4, "category": "cooking", "text": "Italian pasta recipes are easy to follow."},
    {"id": 5, "category": "sport", "text": "Olympics bring nations together through sportsmanship."},
    {"id": 6, "category": "cooking", "text": "I want to eat french "}
]

# 1. Vector Embedding Storage & Indexing
collection.add(
    ids=[str(d["id"]) for d in documents],
    documents=[d["text"] for d in documents],
    metadatas=[{"category": d["category"]} for d in documents]
)

# 2. Retrieval & Similarity Search
results = collection.query(
    query_texts=["Deep learning in AI"],
    n_results=2
)
print(results)

# 3. Metadata Filtering
results_mf = collection.query(
    query_texts=["olympic swimmer"],
    n_results=2,
    where={"category": "sport"}
)
print(results_mf)

"""
docker pull chromadb/chroma

docker run -v ./data:/app/data --name chroma chromadb/chroma
"""