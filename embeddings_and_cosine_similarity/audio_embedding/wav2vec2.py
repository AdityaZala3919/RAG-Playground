import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor
import torch.nn.functional as F

# Pick one of these:
# MODEL_NAME = "microsoft/wav2vec2-base-sv"
MODEL_NAME = "microsoft/wavlm-base-plus-sv"  # WavLM model for speaker verification

# Load model + feature extractor
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_audio(path, target_sr=16000):
    audio, sr = librosa.load(path, sr=target_sr)
    return audio

def get_embedding(path):
    audio = load_audio(path)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Speaker embeddings
    emb = outputs.embeddings
    return F.normalize(emb, p=2, dim=1)  # L2 normalize improves cosine

def compare(audio1, audio2):
    emb1 = get_embedding(audio1)
    emb2 = get_embedding(audio2)

    # Cosine similarity
    cos_sim = F.cosine_similarity(emb1, emb2).item()

    # Dot product
    dot = torch.sum(emb1 * emb2).item()

    # Euclidean distance
    euclidean = torch.dist(emb1, emb2).item()
    euclid_similarity = 1 / (1 + euclidean)

    return cos_sim, dot, euclid_similarity


if __name__ == "__main__":
    file1 = "me.wav"
    file2 = "movies.wav"

    cos, dot, euc = compare(file1, file2)
    print(f"Cosine similarity: {cos:.4f}")
    print(f"Dot product:       {dot:.4f}")
    print(f"Euclid similarity: {euc:.4f}")

    # Decision rule (you can tune threshold)
    print("Same speaker?", cos > 0.5)
