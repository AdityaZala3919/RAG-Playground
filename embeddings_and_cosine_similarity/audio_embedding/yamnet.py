import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
from scipy.signal import resample
from sklearn.metrics.pairwise import cosine_similarity

# Load audio

def load_audio(path):
    wav, sr = sf.read(path)

    # stereo → mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    # resample to 16kHz
    target_sr = 16000
    wav = resample(wav, int(len(wav) * target_sr / sr))

    return wav.astype(np.float32)

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def embed(path):
    wav = load_audio(path)
    scores, embeddings, spectrogram = yamnet(wav)
    return embeddings.numpy().mean(axis=0)  # average to single vector

# Embeddings
e1 = embed("garden.wav")
e2 = embed("movies.wav")
print(e1)
print(e2)

# Similarity
sim = cosine_similarity([e1], [e2])[0][0]
print("similarity =", sim)
