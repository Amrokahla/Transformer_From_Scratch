import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer
from model.tokenizer import SimpleTokenizer


vocab = {
    "<pad>": 0, "<sos>": 1, "<eos>": 2, "hello": 3, "world": 4,
    "how": 5, "are": 6, "you": 7, "i": 8, "am": 9, "fine": 10,
    "happy": 11, "the": 12, "sun": 13, "is": 14, "bright": 15,
    "it": 16, "raining": 17, "weather": 18, "nice": 19
}
tokenizer = SimpleTokenizer(vocab)
model = Transformer(vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with open("data/train.txt") as f:
    lines = [line.strip() for line in f if line.strip()]
max_len = max(len(line.split()) for line in lines) + 2

for epoch in range(100):
    total_loss = 0
    for line in lines:
        input_ids = tokenizer.encode(line, max_len=max_len)
        input_tensor = torch.tensor([input_ids[:-1]], dtype=torch.long).to(device)
        target_tensor = torch.tensor([input_ids[1:]], dtype=torch.long).to(device)

        logits, _ = model(input_tensor)
        loss = criterion(logits.view(-1, logits.shape[-1]), target_tensor.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model_weights.pth")
