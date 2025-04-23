class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return [self.vocab.get(token, 0) for token in text.lower().split()]

    def decode(self, ids):
        return [self.inv_vocab.get(idx, "<unk>") for idx in ids]