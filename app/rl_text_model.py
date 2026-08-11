import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


FORMAT_PREFIX = "that is a great question"
FORMAT_SUFFIX = "let me know if you have any other questions"


class RLWordPolicy(nn.Module):
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


class RLTextGenerator:
    def __init__(self, corpus, device=None):
        self.corpus = corpus
        self.device = device or torch.device("cpu")
        self.tokens = self.simple_tokenizer(" ".join(corpus))
        self.word_to_index, self.index_to_word = self.build_vocab(self.tokens)
        self.model = RLWordPolicy(len(self.word_to_index)).to(self.device)
        self.is_trained = False

    def simple_tokenizer(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, tokens):
        counts = Counter(tokens)
        vocabulary = ["<unk>"] + sorted(counts)
        word_to_index = {word: index for index, word in enumerate(vocabulary)}
        index_to_word = {index: word for word, index in word_to_index.items()}
        return word_to_index, index_to_word

    def encode(self, text):
        return [
            self.word_to_index.get(token, self.word_to_index["<unk>"])
            for token in self.simple_tokenizer(text)
        ]

    def decode(self, indices):
        words = [
            self.index_to_word.get(index, "<unk>")
            for index in indices
            if self.index_to_word.get(index, "<unk>") != "<unk>"
        ]
        return " ".join(words)

    def format_for_display(self, text):
        prefix = FORMAT_PREFIX
        suffix = FORMAT_SUFFIX
        clean_text = text.strip().rstrip(".")

        suffix_position = clean_text.find(suffix)
        if suffix_position >= 0:
            clean_text = clean_text[:suffix_position + len(suffix)]
        else:
            clean_text = f"{clean_text} {suffix}"

        if clean_text.startswith(prefix):
            clean_text = "That is a great question" + clean_text[len(prefix):]
        else:
            clean_text = clean_text[:1].upper() + clean_text[1:]

        clean_text = clean_text[: -len(suffix)].rstrip()
        clean_text = f"{clean_text}. Let me know if you have any other questions"

        return clean_text.strip() + "."

    def reward_text(self, text):
        clean_text = text.lower().strip()
        clean_text = re.sub(r"[^a-z0-9\s]", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        reward = 0.0

        if clean_text.startswith(FORMAT_PREFIX):
            reward += 1.0
        else:
            prefix_words = FORMAT_PREFIX.split()
            generated_words = clean_text.split()
            matched = sum(
                1 for expected, actual in zip(prefix_words, generated_words)
                if expected == actual
            )
            reward += 0.15 * matched

        if FORMAT_SUFFIX in clean_text:
            reward += 1.0
        else:
            suffix_words = FORMAT_SUFFIX.split()
            generated_words = clean_text.split()
            matched = sum(
                1 for expected, actual in zip(reversed(suffix_words), reversed(generated_words))
                if expected == actual
            )
            reward += 0.10 * matched

        if len(clean_text.split()) >= len(FORMAT_PREFIX.split()) + len(FORMAT_SUFFIX.split()):
            reward += 0.25

        return reward

    def train_supervised(self, examples, epochs=200, learning_rate=0.01):
        pairs = []
        for example in examples:
            indices = self.encode(example)
            if len(indices) >= 2:
                pairs.append(indices)

        if not pairs:
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(epochs):
            random.shuffle(pairs)
            for indices in pairs:
                inputs = torch.tensor(
                    indices[:-1],
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                targets = torch.tensor(
                    indices[1:],
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)

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

    def sample_indices(self, start_word="that", max_words=24, temperature=0.9):
        current_index = self.word_to_index.get(
            start_word.lower(),
            self.word_to_index["<unk>"]
        )
        generated = [current_index]
        log_probs = []
        hidden = None

        for _ in range(max_words - 1):
            input_tensor = torch.tensor(
                [[current_index]],
                dtype=torch.long,
                device=self.device
            )
            logits, hidden = self.model(input_tensor, hidden)
            logits = logits[0, -1] / max(temperature, 1e-6)
            distribution = torch.distributions.Categorical(logits=logits)
            next_index = distribution.sample()

            generated.append(int(next_index.item()))
            log_probs.append(distribution.log_prob(next_index))
            current_index = int(next_index.item())

        return generated, log_probs

    def train_with_reinforce(
        self,
        episodes=400,
        max_words=24,
        learning_rate=0.002,
        temperature=0.9
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        reward_history = []
        baseline = 0.0

        self.model.train()
        for episode in range(episodes):
            indices, log_probs = self.sample_indices(
                start_word="that",
                max_words=max_words,
                temperature=temperature
            )
            generated_text = self.decode(indices)
            reward = self.reward_text(generated_text)
            reward_history.append(reward)

            baseline = 0.9 * baseline + 0.1 * reward
            advantage = reward - baseline
            policy_loss = -torch.stack(log_probs).sum() * advantage

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if (episode + 1) % 50 == 0:
                recent_reward = sum(reward_history[-50:]) / 50
                print(
                    f"Episode {episode + 1}/{episodes} "
                    f"Average reward: {recent_reward:.3f}"
                )

        self.model.eval()
        self.is_trained = True
        return reward_history

    def generate_text(self, question, max_words=24, temperature=0.7):
        if not self.is_trained:
            return ""

        with torch.no_grad():
            indices, _ = self.sample_indices(
                start_word="that",
                max_words=max_words,
                temperature=temperature
            )

        generated = self.decode(indices)
        if FORMAT_SUFFIX not in generated:
            generated = f"{generated} {FORMAT_SUFFIX}"
        if not generated.startswith(FORMAT_PREFIX):
            generated = f"{FORMAT_PREFIX} {generated}"

        return self.format_for_display(generated)

    def save(self, path):
        torch.save(
            {
                "corpus": self.corpus,
                "word_to_index": self.word_to_index,
                "index_to_word": self.index_to_word,
                "state_dict": self.model.state_dict(),
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.corpus = checkpoint["corpus"]
        self.word_to_index = checkpoint["word_to_index"]
        self.index_to_word = checkpoint["index_to_word"]
        self.model = RLWordPolicy(len(self.word_to_index)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.is_trained = True


def build_rl_training_corpus():
    return [
        "That is a great question diffusion models generate images by gradually denoising random noise let me know if you have any other questions",
        "That is a great question reinforcement learning improves a model by giving rewards for preferred outputs let me know if you have any other questions",
        "That is a great question an energy model assigns low energy to more likely samples let me know if you have any other questions",
        "That is a great question policy gradient updates the model toward actions that receive higher reward let me know if you have any other questions",
        "That is a great question text generation can be post trained by rewarding outputs that follow the requested format let me know if you have any other questions",
        "The Count of Monte Cristo is a novel written by Alexandre Dumas",
        "this is another example sentence",
        "we are generating text based on bigram probabilities",
        "bigram models are simple but effective",
    ]
