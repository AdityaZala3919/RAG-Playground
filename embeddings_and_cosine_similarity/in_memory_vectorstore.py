from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Embed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = InMemoryVectorStore(embedding=embeddings)  

doc1 = Document(
    page_content="LangChain lets you build applications powered by LLMs.",
    metadata={"source": "intro"}
)

doc2 = Document(
    page_content="Vector stores allow you to search and retrieve information based on embeddings.",
    metadata={"source": "vectors"}
)

# 2. Store

vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])

#

vector_store.delete(ids=["id1"])

# 3. Similarity search

query = "How do vector stores help with search?"

similar_docs = vector_store.similarity_search(query)

print(similar_docs)