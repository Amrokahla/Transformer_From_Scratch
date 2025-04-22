import streamlit as st
from attention_utils import generate_dummy_attention, plot_attention
from sample_data import get_dummy_inputs

st.set_page_config(page_title="Transformer Attention Visualization")

st.title("🎥 Transformer Attention Visualizer")
st.markdown("Explore attention maps from a dummy Transformer setup.")

num_heads = st.slider("Select number of heads", 1, 8, 4)
seq_len = st.slider("Sequence length", 4, 20, 6)

dummy_attention = generate_dummy_attention(num_heads=num_heads, seq_len=seq_len)
head_to_display = st.selectbox("Head to visualize", list(range(num_heads)))

fig = plot_attention(dummy_attention, head=head_to_display)
st.plotly_chart(fig)
