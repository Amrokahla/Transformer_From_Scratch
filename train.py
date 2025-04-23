import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer
import nltk
from nltk.corpus import words
from tqdm import tqdm

# Download NLTK word list
nltk.download("words")

# Build vocab from NLTK words
nltk_vocab = ["<pad>", "<sos>", "<eos>", "<unk>"] + list(set(words.words()))
vocab = {word: idx for idx, word in enumerate(nltk_vocab)}

# Initialize tokenizer and model
tokenizer = SimpleTokenizer(vocab)
model = Transformer(vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load training data
with open("data/train.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
max_len = 100  # +2 for <sos> and <eos>

# Training loop
for epoch in tqdm(range(20)):
    total_loss = 0
    for line in lines:
        input_ids = tokenizer.encode(line, max_len=max_len)
        input_tensor = torch.tensor([input_ids[:-1]], dtype=torch.long).to(device)  # input sequence
        target_tensor = torch.tensor([input_ids[1:]], dtype=torch.long).to(device)  # target shifted by 1

        logits, _ = model(input_tensor)
        loss = criterion(logits.view(-1, logits.shape[-1]), target_tensor.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")

# Save model weights
torch.save(model.state_dict(), "model_weights.pth")
