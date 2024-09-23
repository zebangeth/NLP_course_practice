import nltk
from viterbi import viterbi
from collections import defaultdict
import numpy as np

nltk.download('brown')
nltk.download('universal_tagset')

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
