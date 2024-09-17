import csv
from collections import defaultdict

def load_word_frequencies(file_path: str) -> dict:
    """
    Load word frequencies from a CSV file and convert counts to probabilities.
    """
    word_freq = {}
    total_count = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for word, count in reader:
            count = int(count)
            word_freq[word] = count
            total_count += count
    
    for word in word_freq:
        word_freq[word] = word_freq[word] / total_count
    
    return word_freq

word_freq = load_word_frequencies('word_frequency.csv')

# Test
print('Unigram language model loaded.')
print(list(word_freq.keys())[:10])
print('Total number of words:', len(word_freq))
print('Word frequency of "the":', word_freq['the'])
print('Word frequency of "google":', word_freq['google'])
print('---')

def load_error_model(additions_path: str, deletions_path: str, substitutions_path: str) -> dict:
    """
    Load the error/channel model from CSV files and normalize counts to probabilities.
    """
    error_model = defaultdict(lambda: defaultdict(float))
    
    # Load additions
    with open(additions_path, 'r') as f:
        reader = csv.reader(f)
        for prefix, added, count in reader:
            if prefix == 'prefix':
                continue
            error_model['add'][(prefix, added)] += int(count)
    
    # Load deletions
    with open(deletions_path, 'r') as f:
        reader = csv.reader(f)
        for prefix, deleted, count in reader:
            if prefix == 'prefix':
                continue
            error_model['del'][(prefix, deleted)] += int(count)
    
    # Load substitutions
    with open(substitutions_path, 'r') as f:
        reader = csv.reader(f)
        for original, substituted, count in reader:
            if original == 'original':
                continue
            error_model['sub'][(original, substituted)] += int(count)
    
    # Normalize counts to probabilities
    for error_type in error_model:
        total = sum(error_model[error_type].values())
        for key in error_model[error_type]:
            error_model[error_type][key] /= total
    
    return error_model

error_model = load_error_model('additions.csv', 'deletions.csv', 'substitutions.csv')

# Test
print('Error model loaded.')
print('Additions:', list(error_model['add'].items())[:5])
print('Deletions:', list(error_model['del'].items())[:5])
print('Substitutions:', list(error_model['sub'].items())[:5])
print('---')

def generate_candidates(word: str) -> set:
    """
    Generate a set of candidate words by applying all possible single edits to the input word.
    """
    candidates = set([word])
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Insertions
    for i in range(len(word) + 1):
        for c in alphabet:
            candidates.add(word[:i] + c + word[i:])

    # Deletions
    for i in range(len(word)):
        candidates.add(word[:i] + word[i+1:])
    
    # Substitutions
    for i in range(len(word)):
        for c in alphabet:
            candidates.add(word[:i] + c + word[i+1:])
    
    return candidates

# Test
print('Test generate_candidates:')
print('Candidates for "hi":', generate_candidates('hi'), len(generate_candidates('hi')))
print('Candidates for "world":', generate_candidates('world'), len(generate_candidates('world')))
print('Candidates for "xxx":', generate_candidates('xxx'), len(generate_candidates('xxx')))
print('---')

