import math
from collections import Counter

class BM25Retriever:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        Initializes the retriever with a corpus (list of tokenized documents).
        k1: Term frequency scaling.
        b: Document length normalization.
        """
        self.k1 = k1 
        self.b = b   
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_lengths = []
        self.term_frequencies = [] # List of Counter objects for each doc
        self.idf = {}
        
        self._initialize()

    def _initialize(self):
        """Calculates global stats: avgdl and IDF."""
        total_length = 0
        doc_occurrence_count = Counter() # How many docs contains a word

        for doc in self.corpus:
            # 1. Calculate length |D|
            d_length = len(doc)
            self.doc_lengths.append(d_length)
            total_length += d_length
            
            # 2. Count term frequencies f(qi, D)
            frequencies = Counter(doc)
            self.term_frequencies.append(frequencies)
            
            # 3. Track which terms appear in this document for IDF
            for term in frequencies:
                doc_occurrence_count[term] += 1

        # Calculate average document length (avgDL)
        self.avgdl = total_length / self.corpus_size

        # Calculate IDF for every unique term in the corpus
        for term, n_containing_docs in doc_occurrence_count.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log(
                (self.corpus_size - n_containing_docs + 0.5) / 
                (n_containing_docs + 0.5) + 1
            )

    def get_score(self, query, doc_index):
        """Calculates the BM25 score for a single document."""
        score = 0.0
        doc_freqs = self.term_frequencies[doc_index]
        d_length = self.doc_lengths[doc_index]

        for term in query:
            if term not in self.idf:
                continue
            
            # f(qi, D) - how many times term appears in this doc
            f_qi_D = doc_freqs[term]
            
            # The BM25 formula components
            numerator = f_qi_D * (self.k1 + 1)
            denominator = f_qi_D + self.k1 * (1 - self.b + self.b * (d_length / self.avgdl))
            
            score += self.idf[term] * (numerator / denominator)
        
        return score

    def retrieve(self, query):
        """Ranks all documents in the corpus for a given query."""
        scores = [self.get_score(query, i) for i in range(self.corpus_size)]
        # Return indices sorted by score descendingdescending
        return sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
    
# Simple Corpus (Tokenized)
corpus = [
    ["the", "blue", "apple", "is", "very", "blue"],
    ["red", "apples", "are", "sweet"],
    ["the", "sky", "is", "blue"]
]

# 1. Setup the search engine
search_engine = BM25Retriever(corpus)

# 2. Search for "blue apple"
query = ["blue", "apple"]
results = search_engine.retrieve(query)

print(f"Top Document Index: {results[0]}") # Should be Index 0