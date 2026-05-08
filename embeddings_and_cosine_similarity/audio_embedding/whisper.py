import torch
import librosa
import numpy as np
from transformers import AutoProcessor, WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

MODEL = "openai/whisper-base"

processor = AutoProcessor.from_pretrained(MODEL)
model = WhisperModel.from_pretrained(MODEL)

def load_audio(path, sr=16000):
    audio, _ = librosa.load(path, sr=sr)
    return audio

def get_embedding(path):
    audio = load_audio(path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        # Forward to get encoder hidden states
        enc = model.encoder(inputs.input_features)
        hidden = enc.last_hidden_state.squeeze(0)   # [T, D]

    # Strategy: mean-pool across time (common embedding trick)
    emb = hidden.mean(dim=0)

    # Normalize (improves similarity metrics)
    emb = torch.nn.functional.normalize(emb, p=2, dim=0)

    return emb.cpu().numpy()

# ---- Paths ----
wav1 = "me.wav"
wav2 = "movies.wav"

emb1 = get_embedding(wav1)
emb2 = get_embedding(wav2)

cos_sim = cosine_similarity([emb1], [emb2])[0][0]
dot = np.dot(emb1, emb2)
euclid = euclidean(emb1, emb2)

print(f"Cosine similarity:  {cos_sim:.4f}")
print(f"Dot product:       {dot:.4f}")
print(f"Euclidean dist:    {euclid:.4f}")
