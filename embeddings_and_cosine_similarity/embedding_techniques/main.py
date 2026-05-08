from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
import numpy as np

def gemini_embeddings(texts):
    client = genai.Client(api_key="AIzaSyBSLdMUCR1ZGtUZQWbEhbz9Y6E4RFjKMJo")

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY", output_dimensionality=5)
    )

    vectors = [np.array(e.values) for e in response.embeddings]
    return np.array(vectors)

def sbert_embeddings(docs, model="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model)
    embeddings = model.encode(docs)
    return embeddings

def word2vec_embeddings(sentences, vector_size=10):
    tokens = [simple_preprocess(s) for s in sentences]
    model = Word2Vec(sentences=tokens, vector_size=vector_size, window=5, min_count=1)

    vectors = []
    for token_list in tokens:
        word_vecs = [model.wv[w] for w in token_list if w in model.wv]
        sent_vec = sum(word_vecs) / len(word_vecs)
        vectors.append(sent_vec)
    return vectors, model

def tfidf_embeddings(docs):
    vectorizer = TfidfVectorizer()
    embedding = vectorizer.fit_transform(docs)
    words = vectorizer.get_feature_names_out()

    return embedding.toarray(), words

def main():
    sentences = [
        "He drives car",
        "I use automobile"
    ]

    print("\n=== TF-IDF ===")
    tfidf, vocab = tfidf_embeddings(sentences)
    print(tfidf.shape)
    print(vocab)
    print(tfidf)
    print("\nCosine Similarity:")
    print(cosine_similarity(tfidf))

    print("\n=== Word2Vec (Avg sentence vectors) ===")
    w2v, w2v_model = word2vec_embeddings(sentences)
    print(len(w2v), len(w2v[0]))
    print(w2v)
    print("\nCosine Similarity:")
    print(cosine_similarity(w2v))

    print("\n=== Sentence-BERT ===")
    sb = sbert_embeddings(sentences)
    print(sb.shape)
    print(sb[0][:10])
    print("\nCosine Similarity:")
    print(cosine_similarity(sb))

    print("\n=== Gemini Embeddings ===")
    gemini = gemini_embeddings(sentences)
    print(gemini)
    print("\nGemini Cosine Matrix:")
    print(cosine_similarity(gemini))

    print("\n=== Gemini Pairwise Similarity ===")
    sim_matrix = cosine_similarity(gemini)
    for i, t1 in enumerate(sentences):
        for j in range(i + 1, len(sentences)):
            print(f"Similarity: '{t1}' <-> '{sentences[j]}' = {sim_matrix[i,j]:.4f}")

if __name__ == "__main__":
    main()