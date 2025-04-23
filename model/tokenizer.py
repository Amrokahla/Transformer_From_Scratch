class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["<pad>"]
        self.sos_id = vocab.get("<sos>", None)
        self.eos_id = vocab.get("<eos>", None)
        self.unk_id = vocab.get("<unk>")
        if self.unk_id is None:
            self.unk_id = len(self.vocab)
            self.vocab["<unk>"] = self.unk_id
            self.inv_vocab[self.unk_id] = "<unk>"

    def encode(self, text, max_len=None):
        tokens = text.lower().strip().split()
        token_ids = [self.sos_id] + [self.vocab.get(t, self.unk_id) for t in tokens] + [self.eos_id]

        if max_len:
            token_ids = token_ids[:max_len - 1] + [self.eos_id]
            token_ids = [self.sos_id] + token_ids[1:]
            token_ids += [self.pad_id] * (max_len - len(token_ids))

        return token_ids

    def decode_to_tokens(self, token_ids):
        tokens = []
        for i in token_ids:
            token = self.inv_vocab.get(i, "<unk>")
            if token == "<eos>":
                break
            if token not in {"<pad>", "<sos>"}:
                tokens.append(token)
        return tokens

    def decode_to_text(self, token_ids):
        return " ".join(self.decode_to_tokens(token_ids))
