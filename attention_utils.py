import plotly.express as px

def plot_attention(attn_weights, head=0):
    heatmap = attn_weights[0, head].cpu().numpy()
    fig = px.imshow(heatmap,
                    labels=dict(x="Key", y="Query", color="Attention"),
                    x=list(range(heatmap.shape[1])),
                    y=list(range(heatmap.shape[0])),
                    color_continuous_scale='Viridis',
                    title=f"Head {head} Attention")
    return fig
