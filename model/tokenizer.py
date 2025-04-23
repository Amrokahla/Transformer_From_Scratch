class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["<pad>"]
        self.sos_id = vocab.get("<sos>", None)
        self.eos_id = vocab.get("<eos>", None)

    def encode(self, text, max_len=None):
        tokens = text.lower().strip().split()
        token_ids = [self.sos_id] + [self.vocab.get(t, self.pad_id) for t in tokens] + [self.eos_id]
        if max_len:
            token_ids = token_ids[:max_len]
            token_ids += [self.pad_id] * (max_len - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        tokens = []
        for i in token_ids:
            token = self.inv_vocab.get(i, "<unk>")
            if token == "<eos>":
                break
            if token != "<pad>" and token != "<sos>":
                tokens.append(token)
        return tokens
