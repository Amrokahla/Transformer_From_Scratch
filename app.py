import streamlit as st
import torch
from model import MiniTransformerBlock
from sample_data import get_dummy_input
from attention_utils import plot_attention

st.set_page_config(page_title="Mini Transformer Attention")

st.title("🧠 Mini Transformer Attention Visualizer")
st.markdown("Visualizing attention from a single encoder block.")

seq_len = st.slider("Sequence Length", 4, 20, 6)
num_heads = st.slider("Number of Heads", 1, 8, 4)

x = get_dummy_input(seq_len=seq_len)
block = MiniTransformerBlock(d_model=64, num_heads=num_heads)
output, attn_weights = block(x)

head = st.selectbox("Head to visualize", list(range(num_heads)))
fig = plot_attention(attn_weights, head)
st.plotly_chart(fig)
