class SimpleTokenizer:
    def __init__(self, chars=None):
        if chars:
            self.chars = sorted(list(set(chars)))
            self.vocab_size = len(self.chars) + 2 # +2 for <PAD> and <UNK>
            self.stoi = { ch:i+2 for i,ch in enumerate(self.chars) }
            self.stoi['<PAD>'] = 0
            self.stoi['<UNK>'] = 1
            self.itos = { i:s for s,i in self.stoi.items() }
        else:
            self.stoi = {}
            self.itos = {}

    def encode(self, s):
        return [self.stoi.get(c, self.stoi['<UNK>']) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, '<UNK>') for i in l])

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        return cls(chars)
