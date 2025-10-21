import re
from collections import Counter

TOKEN = re.compile(r"\S+")

def tokenize(text: str):
    return TOKEN.findall(text)

def build_vocab(texts, vocab_size=30000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= vocab_size:
            break
    return vocab

def tokens_to_ids(tokens, vocab, max_len=256):
    ids = [vocab.get(tok, 1) for tok in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return ids
