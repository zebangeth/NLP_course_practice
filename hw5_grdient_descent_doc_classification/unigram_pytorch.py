"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
import random

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp

def generate_optimal_text(token_probs, length):
    """Generate text based on optimal unigram probabilities."""
    tokens, probs = zip(*token_probs.items())
    return [random.choices(tokens, weights=probs)[0] for _ in range(length)]

def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    print(len(tokens))

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 100
    learning_rate = 0.1

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # Compute optimal probabilities
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    optimal_probs = {token: count / total_tokens for token, count in token_counts.items()}

    # Generate the "optimal" text using optimal unigram model
    optimal_text = generate_optimal_text(optimal_probs, total_tokens)
    optimal_encodings = np.hstack([onehot(vocabulary, token) for token in optimal_text])
    optimal_x = torch.tensor(optimal_encodings.astype(float))

    # Compute the loss for the "optimal" text as the minimum possible loss
    with torch.no_grad():
        optimal_logp = model(optimal_x)
        min_loss = loss_fn(optimal_logp).item()

    # Visualize final token probabilities
    learned_probs = torch.nn.Softmax(0)(model.s).detach().numpy().flatten()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(vocabulary)), learned_probs, alpha=0.5, label='Learned')
    plt.bar(range(len(vocabulary)), [optimal_probs.get(token, 0) for token in vocabulary], 
            alpha=0.5, label='Optimal')
    plt.xticks(range(len(vocabulary)), vocabulary, rotation='vertical')
    plt.xlabel('Tokens')
    plt.ylabel('Probability')
    plt.title('Learned vs Optimal Token Probabilities')
    plt.legend()
    plt.tight_layout()
    plt.savefig('token_probabilities.png')

    # Visualize loss over time
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.axhline(y=min_loss, color='r', linestyle='--', label='Minimum Possible Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_over_time.png')


if __name__ == "__main__":
    gradient_descent_example()
