import torch
from transformers import BertTokenizer, BertModel

sentences = [
    "Artificial intelligence is transforming every industry.",
    "Large language models are becoming increasingly powerful."
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)

cls_embeddings = outputs.last_hidden_state[:, 0, :]
print("BERT base CLS shape:", cls_embeddings.shape)
print(cls_embeddings[0])