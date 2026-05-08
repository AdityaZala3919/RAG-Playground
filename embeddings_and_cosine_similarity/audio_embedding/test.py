import torch
from speechbrain.pretrained import SpeakerRecognition
from torch.nn.functional import cosine_similarity

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa"
)

score = model.verify_files("test-2.wav", "movies.wav")
print(f"Similarity score: {score}")


# import torch
# import torchaudio
# from speechbrain.pretrained import SpeakerRecognition

# # Load pretrained ECAPA-TDNN
# model = SpeakerRecognition.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="ecapa"
# )

# def load_audio(path):
#     waveform, sr = torchaudio.load(path)
#     # Convert to 16 kHz mono if needed
#     if sr != 16000:
#         waveform = torchaudio.functional.resample(waveform, sr, 16000)
#     if waveform.shape[0] > 1:         # stereo -> mono
#         waveform = waveform.mean(dim=0, keepdim=True)
#     return waveform

# # ---- 1. Load wav files ----
# wav1 = load_audio("test-3.wav")
# wav2 = load_audio("test-2.wav")

# # ---- 2. Extract embeddings ----
# emb1 = model.encode_batch(wav1)
# emb2 = model.encode_batch(wav2)

# # ---- 3. Similarity metrics ----
# cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).mean().item()
# euclid = torch.norm(emb1 - emb2, p=2)
# dot = torch.sum(emb1 * emb2)

# print("Cosine similarity:", cos_sim)
# print("Euclidean distance:", float(euclid))
# print("Dot product:", float(dot))

# # ---- 4. Decision threshold ----
# same_speaker = cos_sim > 0.5
# print("Same speaker?:", bool(same_speaker))
