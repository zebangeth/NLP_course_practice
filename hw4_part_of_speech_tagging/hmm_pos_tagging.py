import nltk
from viterbi import viterbi
from collections import defaultdict
import numpy as np

# nltk.download('brown')
# nltk.download('universal_tagset')

# tagged sentences from the Brown corpus as training data
tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

# the set of tags and words
tags = set()
words = set()
for sent in tagged_sents:
    for word, tag in sent:
        tags.add(tag)
        words.add(word.lower())

# Add the unknown token
words.add('UNK')

# dictionaries for mapping words and tags to indices
tag2idx = {tag: i for i, tag in enumerate(tags)}
word2idx = {word: i for i, word in enumerate(words)}

# dictionaries for mapping indices to words and tags
idx2tag = {i: tag for tag, i in tag2idx.items()}
idx2word = {i: word for word, i in word2idx.items()}

# the number of states and observations
num_states = len(tags)
num_observations = len(words) + 1 # +1 for add-1 smoothing

# Initialize the transition, emission, and initial state matrices with zeros
A = np.zeros((num_states, num_states))  # Transition matrix
B = np.zeros((num_states, num_observations))  # Emission matrix
pi = np.zeros(num_states)  # Initial state distribution 

# count occurrences of transitions, emissions, and initial states
for sent in tagged_sents:
    for i, (word, tag) in enumerate(sent):
        if i == 0:
            pi[tag2idx[tag]] += 1
        else:
            prev_tag = sent[i-1][1]
            A[tag2idx[prev_tag]][tag2idx[tag]] += 1
        
        word_idx = word2idx.get(word.lower(), len(words))
        B[tag2idx[tag]][word_idx] += 1

# apply add-1 smoothing
A += 1
B += 1
pi += 1

# normalize the matrices
A = A / A.sum(axis=1, keepdims=True)
B = B / B.sum(axis=1, keepdims=True)
pi = pi / pi.sum()

test_sents = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]

for sent in test_sents:
    # print(sent)
    obs = [word2idx.get(word.lower(), word2idx['UNK']) for word, _ in sent]
    states, prob = viterbi(obs, pi, A, B)
    
    print("Original sentence:")
    print(" ".join([word for word, _ in sent]))
    print("Actual POS sequence:")
    print(" ".join([tag for _, tag in sent]))
    print("Predicted POS sequence:")
    print(" ".join([idx2tag[state] for state in states]))
    print("Probability of predicted sequence:", prob)
    print()
