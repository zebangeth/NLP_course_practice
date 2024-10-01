"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional, Dict
from torch import nn
import matplotlib.pyplot as plt
from collections import Counter
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
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logp = torch.nn.LogSoftmax(0)(self.s)
        return torch.sum(input, 1, keepdim=True).T @ logp

def generate_optimal_text(token_probs: Dict[str, float], length: int) -> List[str]:
    """Generate text based on optimal unigram probabilities."""
    tokens, probs = zip(*token_probs.items())
    return [random.choices(tokens, weights=probs)[0] for _ in range(length)]

def calculate_optimal_probabilities(tokens: List[str]) -> Dict[str, float]:
    """Calculate optimal probabilities based on token frequencies."""
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    return {token: count / total_tokens for token, count in token_counts.items()}

def calculate_min_loss(model: nn.Module, optimal_x: torch.Tensor) -> float:
    """Calculate the minimum possible loss using the optimal text."""
    with torch.no_grad():
        optimal_logp = model(optimal_x)
        return loss_fn(optimal_logp).item()

def visualize_token_probabilities(vocabulary: List[str], learned_probs: np.ndarray, optimal_probs: Dict[str, float]):
    """Visualize learned vs optimal token probabilities."""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(vocabulary))
    width = 0.35
    
    ax.bar(x - width/2, learned_probs, width, label='Learned', alpha=0.8)
    ax.bar(x + width/2, [optimal_probs.get(token, 0) for token in vocabulary], width, label='Optimal', alpha=0.8)

    ax.set_xlabel('Tokens')
    ax.set_ylabel('Probability')
    ax.set_title('Learned vs Optimal Token Probabilities')
    ax.set_xticks(x)
    
    display_vocab = [token if token not in [" ", None] else ("SPACE" if token == " " else "UNK") for token in vocabulary]
    ax.set_xticklabels(display_vocab, rotation='vertical')
    
    ax.legend()

    plt.tight_layout()
    plt.savefig('token_probabilities.png')

def visualize_loss_over_time(losses: List[float], min_loss: float):
    """Visualize loss over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.axhline(y=min_loss, color='r', linestyle='--', label='Minimum Possible Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_over_time.png')

def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 100
    learning_rate = 0.1

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    optimal_probs = calculate_optimal_probabilities(tokens)
    optimal_text = generate_optimal_text(optimal_probs, len(tokens))
    optimal_encodings = np.hstack([onehot(vocabulary, token) for token in optimal_text])
    optimal_x = torch.tensor(optimal_encodings.astype(float))

    min_loss = calculate_min_loss(model, optimal_x)

    learned_probs = torch.nn.Softmax(0)(model.s).detach().numpy().flatten()
    
    visualize_token_probabilities(vocabulary, learned_probs, optimal_probs)
    visualize_loss_over_time(losses, min_loss)

if __name__ == "__main__":
    gradient_descent_example()
