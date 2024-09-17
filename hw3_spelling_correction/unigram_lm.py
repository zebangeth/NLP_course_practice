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
print(list(word_freq.keys())[:10])
print('Total number of words:', len(word_freq))
print('Word frequency of "the":', word_freq['the'])
print('Word frequency of "google":', word_freq['google'])
