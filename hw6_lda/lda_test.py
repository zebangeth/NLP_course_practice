"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np
from scipy.optimize import linear_sum_assignment


def lda_gen(vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    """Generate a document using the LDA model."""
    
    # document length
    doc_len = np.random.poisson(xi)

    # topic distribution
    theta = np.random.dirichlet(alpha)

    # number of topics
    num_topics = beta.shape[0]

    words = []
    for _ in range(doc_len):
        # sample a topic: topic from theta
        topic = np.random.choice(num_topics, p=theta)

        # sample a word from the topic: word from beta[topic]
        word_idx = np.random.choice(len(vocabulary), p=beta[topic])
        words.append(vocabulary[word_idx])

    return words


def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(100)
    ]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )

    # Show the inferred alpha (topic distribution per document)
    print("Inferred alpha: ", model.alpha)

    # Show the inferred beta vectors (word distribution per topic)
    for idx, topic in model.show_topics(formatted=False):
        print("----------------")
        print(f"Inferred Topic {idx}:")
        for word, prob in topic:
            print(f"{word}: {prob}")

    # Print true topics for comparison
    print("\nTrue Topics:")
    for i, topic in enumerate(beta):
        print("----------------")
        print(f"True Topic {chr(ord('A') + i)}:")
        words_with_prob = sorted(list(zip(vocabulary, topic)), key=lambda x: x[1], reverse=True)
        for word, prob in words_with_prob:
            print(f"{word}: {prob}")

    # Reorders the columns of the beta matrix to match the order of the original vocabulary
    inferred_beta = model.get_topics()  # shape (num_topics, vocab_size)
    vocab_word_ids = [dictionary.token2id[word] for word in vocabulary]
    inferred_beta_reordered = inferred_beta[:, vocab_word_ids]

    # Compute cosine similarity between each pair of inferred and true topics
    num_topics = beta.shape[0]
    similarity_matrix = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            inferred_topic = inferred_beta_reordered[i]
            true_topic = beta[j]
            # Compute cosine similarity
            cosine_sim = np.dot(inferred_topic, true_topic) / (np.linalg.norm(inferred_topic) * np.linalg.norm(true_topic))
            similarity_matrix[i, j] = cosine_sim

    # print("\nSimilarity Matrix (rows: inferred topics, columns: true topics):")
    # print(similarity_matrix)

    # Find the best matching between inferred and true topics
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    print("\nTopic Mapping:")
    for inferred_i, true_j in zip(row_ind, col_ind):
        print(f"Inferred Topic {inferred_i} maps to True Topic {chr(ord('A') + true_j)} with similarity {similarity_matrix[inferred_i, true_j]:.4f}")

if __name__ == "__main__":
    test()
