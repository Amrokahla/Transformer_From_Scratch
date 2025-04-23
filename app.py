import streamlit as st
import torch
from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer
from model.attention_utils import plot_attention
import nltk
from nltk.corpus import words

# Dummy vocab
<<<<<<< HEAD
nltk.download("words")

# Build vocab from NLTK words
nltk_vocab = ["<pad>", "<sos>", "<eos>"] + list(set(words.words()))
vocab = {word: idx for idx, word in enumerate(nltk_vocab)}
=======
vocab = vocab = {
    "<pad>": 0, "<sos>": 1, "<eos>": 2, "hello": 3, "world": 4,
    "how": 5, "are": 6, "you": 7, "i": 8, "am": 9, "fine": 10,
    "happy": 11, "the": 12, "sun": 13, "is": 14, "bright": 15,
    "it": 16, "raining": 17, "weather": 18, "nice": 19
}
>>>>>>> 026554b8cac52e4e11eea2860a7c9f86bb48f9b1

tokenizer = SimpleTokenizer(vocab)

# Load model
model = Transformer(vocab_size=len(vocab))
model.eval()

st.title("Dummy Transformer Visualization")
text = st.text_input("Enter a sentence:", "hello world")

if st.button("Run Transformer"):
    input_ids = tokenizer.encode(text)
    tokens = tokenizer.decode(input_ids)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        logits, attentions = model(input_tensor)

    pred_ids = logits.argmax(dim=-1)[0]
    predictions = tokenizer.decode(pred_ids.tolist())

    st.markdown("### Output Tokens")
    st.write(predictions)

    st.markdown("### Attention Map (Layer 1, Head 1)")
    plot_attention(attentions[0][0][0].detach().numpy(), tokens)
