import streamlit as st
import torch
import nltk
from nltk.corpus import words
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer

# Download words if needed
nltk.download("words", quiet=True)

# Build vocab
nltk_vocab = ["<pad>", "<sos>", "<eos>", "<unk>"] + list(set(words.words()))
vocab = {word: idx for idx, word in enumerate(nltk_vocab)}

# Attention heatmap plot
def plot_attention(attn, input_tokens):
    fig, ax = plt.subplots()
    sns.heatmap(attn, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis")
    st.pyplot(fig)

# Streamlit App
st.title("🔍 Transformer Attention Visualizer")

# Input
text = st.text_input("Enter sentence:", "hello world")

# Hyperparameter sliders
d_model = st.slider("Model dimension (d_model)", 32, 512, 128, step=32)
num_heads = st.slider("Number of Attention Heads", 1, 8, 4)
num_layers = st.slider("Number of Transformer Layers", 1, 6, 2)

# Run button
run = st.button("Run Transformer")

# Init session state
if "logits" not in st.session_state:
    st.session_state["logits"] = None
    st.session_state["attentions"] = None
    st.session_state["tokens"] = None

# Model execution
if run:
    tokenizer = SimpleTokenizer(vocab)
    model = Transformer(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=4*d_model
    )
    model.eval()

    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        logits, attentions = model(input_tensor)

    pred_ids = logits.argmax(dim=-1)[0]
    tokens = tokenizer.decode_to_tokens(input_ids)

    st.session_state["logits"] = logits
    st.session_state["attentions"] = attentions
    st.session_state["tokens"] = tokens

# Visualization UI
if st.session_state["attentions"]:
    tokens = st.session_state["tokens"]
    attentions = st.session_state["attentions"]

    st.markdown("### Attention Heatmap")
    layer = st.slider("Layer", 0, len(attentions) - 1, 0)
    head = st.slider("Head", 0, attentions[0].shape[1] - 1, 0)

    attn = attentions[layer][0, head].detach().numpy()
    plot_attention(attn, tokens)

    st.markdown("### Predicted Tokens")
    pred_ids = st.session_state["logits"].argmax(dim=-1)[0]
    pred_tokens = tokenizer.decode_to_tokens(pred_ids.tolist())
    st.write(pred_tokens)
