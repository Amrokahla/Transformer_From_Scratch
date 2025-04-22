import torch
import plotly.express as px

def generate_dummy_attention(num_heads=2, seq_len=6):

    return torch.rand(1, num_heads, seq_len, seq_len)

def plot_attention(attention_weights, head=0):
    attn = attention_weights[0, head].detach().numpy()
    fig = px.imshow(attn,
                    labels=dict(x="Key", y="Query", color="Attention"),
                    x=list(range(attn.shape[1])),
                    y=list(range(attn.shape[0])),
                    color_continuous_scale='Viridis',
                    title=f"Head {head} Attention")
    return fig
