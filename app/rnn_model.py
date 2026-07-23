import random
import re
from collections import Counter

import torch
import torch.nn as nn


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


class RNNTextGenerator:
    def __init__(self, corpus, device=None):
        self.corpus = corpus
        self.device = device or torch.device("cpu")
        self.tokens = self.simple_tokenizer(" ".join(corpus))
        self.word_to_index, self.index_to_word = self.build_vocab(self.tokens)
        self.model = WordRNN(len(self.word_to_index)).to(self.device)
        self.is_trained = False

    def simple_tokenizer(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, tokens):
        counts = Counter(tokens)
        vocabulary = ["<unk>"] + sorted(counts)
        word_to_index = {
            word: index for index, word in enumerate(vocabulary)
        }
        index_to_word = {
            index: word for word, index in word_to_index.items()
        }
        return word_to_index, index_to_word

    def train_model(self, epochs=300, learning_rate=0.01):
        if len(self.tokens) < 2:
            return

        inputs = torch.tensor(
            [self.word_to_index[word] for word in self.tokens[:-1]],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
        targets = torch.tensor(
            [self.word_to_index[word] for word in self.tokens[1:]],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.is_trained = True

    def generate_text(self, start_word, num_words=20, temperature=0.8):
        if not self.is_trained or num_words <= 0:
            return ""

        current_word = start_word.lower()
        generated_words = [current_word]
        hidden = None

        for _ in range(num_words - 1):
            current_index = self.word_to_index.get(
                current_word,
                self.word_to_index["<unk>"]
            )
            input_tensor = torch.tensor(
                [[current_index]],
                dtype=torch.long,
                device=self.device
            )

            with torch.no_grad():
                logits, hidden = self.model(input_tensor, hidden)
                logits = logits[0, -1] / temperature
                probabilities = torch.softmax(logits, dim=0)
                next_index = torch.multinomial(probabilities, 1).item()

            next_word = self.index_to_word.get(next_index, "<unk>")
            if next_word == "<unk>":
                next_word = random.choice(self.tokens)

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)
