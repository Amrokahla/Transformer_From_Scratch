import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def plot_attention(attn, input_tokens):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attn, xticklabels=input_tokens, yticklabels=input_tokens, cmap="viridis", cbar=True, ax=ax)
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)
