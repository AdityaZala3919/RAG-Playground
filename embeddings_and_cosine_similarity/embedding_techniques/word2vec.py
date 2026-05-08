from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [
    "Artificial intelligence is transforming every industry.",
    "Large language models are becoming increasingly powerful."
]

tokenized = [simple_preprocess(s) for s in sentences]
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)

print("Word2Vec for 'intelligence':", model.wv["intelligence"][:10])
print(model.wv.most_similar("models"))

def get_word2vec_embeddings(sentences, vector_size=100):
    tokens = [simple_preprocess(s) for s in sentences]
    model = Word2Vec(sentences=tokens, vector_size=vector_size, window=5, min_count=1)

    vectors = []
    for token_list in tokens:
        word_vecs = [model.wv[w] for w in token_list if w in model.wv]
        sent_vec = sum(word_vecs) / len(word_vecs)
        vectors.append(sent_vec)
    return vectors, model