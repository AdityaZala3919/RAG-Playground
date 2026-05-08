from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Artificial intelligence is transforming every industry.",
    "Large language models are becoming increasingly powerful."
]

# [
#     "Dog",
#     "Cat",
#     "Pizza"
# ]

vectorizer = TfidfVectorizer()
embedding = vectorizer.fit_transform(docs)
words = vectorizer.get_feature_names_out()

print(f"Word count: {len(words)} e.g.: {words[:10]}")
print("Embedding Shape:", embedding.shape)
print(embedding.toarray())

def tfidf_embeddings(docs):
    vectorizer = TfidfVectorizer()
    embedding = vectorizer.fit_transform(docs)
    words = vectorizer.get_feature_names_out()

    return embedding.toarray(), words