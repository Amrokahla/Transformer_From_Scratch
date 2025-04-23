import streamlit as st
import torch
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import words
from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer
from model.attention_utils import plot_attention

nltk.download("words")

nltk_vocab = ["<pad>", "<sos>", "<eos>"] + list(set(words.words()))
vocab = {word: idx for idx, word in enumerate(nltk_vocab)}
tokenizer = SimpleTokenizer(vocab)

st.title("Transformer Attention Visualizer")

st.sidebar.header("Transformer Hyperparameters")
num_layers = st.sidebar.slider("Number of Layers", 1, 6, 2)
num_heads = st.sidebar.slider("Number of Heads", 1, 8, 2)
d_model = st.sidebar.slider("Model Dimension (d_model)", 64, 512, 128, step=64)

model = Transformer(vocab_size=len(vocab), num_layers=num_layers, num_heads=num_heads, d_model=d_model)
model.eval()

text = st.text_input("Enter a sentence to analyze attention:", "hello world")
if st.button("Run Transformer"):
    input_ids = tokenizer.encode(text)
    tokens = tokenizer.decode_to_tokens(input_ids)

    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        logits, attentions = model(input_tensor)

    pred_ids = logits.argmax(dim=-1)[0]
    predictions = tokenizer.decode_to_tokens(pred_ids.tolist())

    st.markdown("### Predicted Output")
    st.write(" ".join(predictions))

    st.subheader("Attention Visualization")
    selected_layer = st.slider("Select Layer", 0, num_layers - 1, 0)
    selected_head = st.slider("Select Head", 0, num_heads - 1, 0)

    selected_attention = attentions[selected_layer][0, selected_head].detach().numpy()

    st.markdown(f"#### Attention Map (Layer {selected_layer + 1}, Head {selected_head + 1})")
    plot_attention(selected_attention, tokens)
