import torch
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F

MODEL_NAME = "facebook/wav2vec2-base"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_audio(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    return torch.tensor(wav, dtype=torch.float32), 16000


def embed_audio(path):
    wav, sr = load_audio(path)
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state
        embedding = out.mean(dim=1)
    return embedding  # [1, 768]

emb1 = embed_audio("falak.wav")
emb2 = embed_audio("me.wav")

print("embeddings of 1st audio :", emb1)
print("embeddings of 2nd audio :", emb2)

cosine_sim = F.cosine_similarity(emb1, emb2).item()

dot_sim = torch.matmul(emb1, emb2.T).item()

euclidean_dist = torch.norm(emb1 - emb2, p=2).item()

print("Embedding 1 shape:", emb1.shape)
print("Embedding 2 shape:", emb2.shape)

print("\nCosine Similarity:", round(cosine_sim, 4))
print("Dot Product Similarity:", round(dot_sim, 4))
print("Euclidean Distance:", round(euclidean_dist, 4))