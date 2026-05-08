from sentence_transformers import SentenceTransformer

sentences = [
    "Artificial intelligence is transforming every industry.",
    "Large language models are becoming increasingly powerful."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

print("S-BERT embedding shape:", embeddings.shape)
print(embeddings[0])

def sbert_embeddings(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings