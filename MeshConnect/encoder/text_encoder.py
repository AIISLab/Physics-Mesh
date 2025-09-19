
import torch
from transformers import BertTokenizer, BertModel

input_embed = input("Enter in a CAD Model Description: ")

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
bert_inputs = tokenizer(input_embed, return_tensors="pt")

with torch.no_grad():
    outputs = model(**bert_inputs)

last_hidden_states = outputs.last_hidden_state

print(f"Shape of hidden state: {last_hidden_states.shape}")

# Embeddings
tokens = tokenizer.convert_ids_to_tokens(bert_inputs["input_ids"][0])
for token, embedding in zip(tokens, last_hidden_states[0]):
    print(f"Token: {token}, Embedding: {embedding.shape}, First 5 components of embedding: {embedding[:5]}")