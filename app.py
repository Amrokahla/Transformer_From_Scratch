import streamlit as st
import torch
import nltk
from nltk.corpus import words
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer

nltk.download("words", quiet=True)

# Build vocab
nltk_vocab = ["<pad>", "<sos>", "<eos>", "<unk>"] + list(set(words.words()))
vocab = {word: idx for idx, word in enumerate(nltk_vocab)}

def plot_attention(attn, input_tokens):
    fig, ax = plt.subplots()
    sns.heatmap(attn, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis")
    st.pyplot(fig)

# Streamlit UI
st.title("Transformer Attention Visualizer")
st.subheader("What will the transformer pay attention to")

text = st.text_input("Enter sentence:", "hello world")

d_model = st.slider("Model dimension (d_model)", 64, 256, 128, step=64)
num_heads = st.slider("Number of Attention Heads", 4, 8, 4, step=4)
num_layers = st.slider("Number of Transformer Layers", 1, 6, 2)

run = st.button("Run Transformer")

# Session state initialization
if "logits" not in st.session_state:
    st.session_state["logits"] = None
    st.session_state["attentions"] = None
    st.session_state["tokens"] = None

if run:
    tokenizer = SimpleTokenizer(vocab)
    model = Transformer(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=4 * d_model
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

if st.session_state["attentions"] is not None:
    tokens = st.session_state["tokens"]
    attentions = st.session_state["attentions"]

    st.markdown("### Attention Heatmap")
    num_layers_available = len(attentions)
    layer = st.selectbox("Layer", list(range(num_layers_available)), index=0, format_func=lambda x: f"Layer {x}")

    num_heads_available = attentions[0].shape[1]
    head = st.selectbox("Head", list(range(num_heads_available)), index=0, format_func=lambda x: f"Head {x}")

    attn = attentions[layer][0, head].detach().numpy()
    plot_attention(attn, tokens)
