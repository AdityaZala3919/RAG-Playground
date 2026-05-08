from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from embed import get_embeddings

client = QdrantClient(url="http://localhost:6333")

# 1. Embed
documents = [
    {"id": 1, "category": "tech", "text": "Artificial intelligence is transforming industry."},
    {"id": 2, "category": "sport", "text": "Football is the most popular sport in the world."},
    {"id": 3, "category": "tech", "text": "Neural networks are a subset of machine learning."},
    {"id": 4, "category": "cooking", "text": "Italian pasta recipes are easy to follow."},
    {"id": 5, "category": "sport", "text": "Olympics bring nations together through sportsmanship."}
]
embeddings = get_embeddings([d["text"] for d in documents])

query = "Deep learning in AI"
query_embedding = get_embeddings([query])[0]

client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)

# 2. Store Embeddings & Indexing
operation_info = client.upsert(
    collection_name="my_collection",
    wait=True,
    points=[
        PointStruct(id=d["id"], vector=embeddings[i], payload={"category": d["category"], "text": d["text"]})
        for i, d in enumerate(documents)
    ],
)
print(operation_info)

# 3. Retrieval & Semantic Search
search_result_1 = client.query_points(
    collection_name="my_collection",
    query=query_embedding,
    with_payload=False,
    limit=2
).points
print(search_result_1)

# 4. Metadata Filtering
search_result_2 = client.query_points(
    collection_name="my_collection",
    query=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
    ),
    with_payload=True,
    limit=2,
).points
print(search_result_2)

"""
docker pull qdrant/qdrant

docker run -p 6333:6333 -p 6334:6334     -v "$(pwd)/qdrant_storage:/qdrant/storage:z"     qdrant/qdrant
"""