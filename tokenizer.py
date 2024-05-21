class Tokenizer():
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.base_size = len(vocab)
        self.stoi = {char: i for i, char in enumerate(vocab)}
        self.itos = {i: char for i, char in enumerate(vocab)}
        
    def encode(self, text: list[str]) -> list[int]:
        text = [self.stoi[char] for char in text]
        for i in range(self.base_size, len(self.vocab)):
            text = self.merge(text, self.vocab[i], i)
        return text

    def decode(self, text: list[int]) -> list[str]:
        return [self.itos[idx] for idx in text]
    
    def get_pair_frequencies(self, text: list[int]) -> dict:
        frequencies = {}
        for i in range(len(text)-1):
            if text[i] in {self.stoi[char] for char in (' ', '\n', '?', ':', ';', ',', '.')}:
                continue
            if (text[i], text[i+1]) not in frequencies:
                frequencies[(text[i], text[i+1])] = 0
            frequencies[(text[i], text[i+1])] += 1
        return frequencies
    
    def merge(self, text, token_pair, token_idx):
        i = 0
        while i + 1 < len(text):
            if (text[i], text[i+1]) == token_pair:
                text[i] = token_idx
                text.pop(i + 1)
            i += 1
        return text

    def train(self, text, num_additions):
        for i in range(num_additions):
            print(i)
            
            stats = self.get_pair_frequencies(text)
            pair = max(stats, key=stats.get)
            text = self.merge(text, pair, len(self.vocab))
            pair_str = ''.join(self.decode(pair))
            self.stoi[pair_str] = len(self.vocab)
            self.itos[len(self.vocab)] = pair_str 
            self.vocab.append(pair_str)