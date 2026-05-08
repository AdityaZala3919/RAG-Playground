from fastapi import FastAPI

app = FastAPI()

@app.post("vectordb/faiss")
def faiss_v():
    pass

    # CPU-only Torch (no CUDA)
torch==2.3.1+cpu
--extra-index-url https://download.pytorch.org/whl/cpu