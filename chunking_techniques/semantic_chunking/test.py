from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token="",
    repo_id="sentence-transformers/all-MiniLM-l6-v2",
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
query_result[:3]