import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def plot_attention(attn, input_tokens):
    fig, ax = plt.subplots()
    sns.heatmap(attn, xticklabels=input_tokens, yticklabels=input_tokens, ax=ax)
    st.pyplot(fig)
