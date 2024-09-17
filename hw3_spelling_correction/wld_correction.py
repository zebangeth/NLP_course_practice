import csv
import math
import heapq
from collections import defaultdict

# Load word frequencies
word_counts = {}
total_word_count = 0
V = 26

with open('word_frequency.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        word, count = row
        count = int(count)
        word_counts[word] = count
        total_word_count += count

# Compute word probabilities
word_probabilities = {}

for word, count in word_counts.items():
    word_probabilities[word] = count / total_word_count

# Load additions
insertion_counts = defaultdict(int)
insertion_prefix_counts = defaultdict(int)

with open('additions.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'prefix':
            continue
        prefix, added, count = row
        count = int(count)
        insertion_counts[(prefix, added)] = count
        insertion_prefix_counts[prefix] += count

# Load deletions
deletion_counts = defaultdict(int)
deletion_prefix_counts = defaultdict(int)

with open('deletions.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'prefix':
            continue
        prefix, deleted, count = row
        count = int(count)
        deletion_counts[(prefix, deleted)] = count
        deletion_prefix_counts[prefix] += count

# Load substitutions
substitution_counts = defaultdict(int)
substitution_original_counts = defaultdict(int)

with open('substitutions.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'original':
            continue
        original, substituted, count = row
        count = int(count)
        substitution_counts[(original, substituted)] = count
        substitution_original_counts[original] += count

# Error probabilities
def insertion_probability(prefix, added):
    count = insertion_counts.get((prefix, added), 0)
    total = insertion_prefix_counts.get(prefix, 0)
    prob = (count + 1) / (total + V)
    return prob

def deletion_probability(prefix, deleted):
    count = deletion_counts.get((prefix, deleted), 0)
    total = deletion_prefix_counts.get(prefix, 0)
    prob = (count + 1) / (total + V)
    return prob

def substitution_probability(original, substituted):
    count = substitution_counts.get((original, substituted), 0)
    total = substitution_original_counts.get(original, 0)
    prob = (count + 1) / (total + V)
    return prob

# Weighted Levenshtein Distance
def weighted_levenshtein(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    dp[0][0] = 0
    for i in range(1, m+1):
        c = s1[i-1]
        prefix = s1[i-2] if i > 1 else '#'
        cost = -math.log(deletion_probability(prefix, c))
        dp[i][0] = dp[i-1][0] + cost
    for j in range(1, n+1):
        c = s2[j-1]
        prefix = s2[j-2] if j > 1 else '#'
        cost = -math.log(insertion_probability(prefix, c))
        dp[0][j] = dp[0][j-1] + cost
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            c1 = s1[i-1]
            c2 = s2[j-1]
            prefix1 = s1[i-2] if i > 1 else '#'
            prefix2 = s2[j-2] if j > 1 else '#'
            
            if c1 == c2:
                substitution_cost = 0
            else:
                sub_prob = substitution_probability(c1, c2)
                substitution_cost = -math.log(sub_prob)
            
            del_prob = deletion_probability(prefix1, c1)
            ins_prob = insertion_probability(prefix2, c2)
            
            deletion_cost = -math.log(del_prob)
            insertion_cost = -math.log(ins_prob)
            
            dp[i][j] = min(
                dp[i-1][j] + deletion_cost,
                dp[i][j-1] + insertion_cost,
                dp[i-1][j-1] + substitution_cost
            )
    return dp[m][n]

# Generate candidates
def generate_candidates(misspelled_word, max_candidates=5):
    candidates = []
    for word in word_counts:
        if abs(len(word) - len(misspelled_word)) > 2:
            continue
        distance = weighted_levenshtein(word, misspelled_word)
        heapq.heappush(candidates, (distance, word))
    top_candidates = heapq.nsmallest(max_candidates, candidates)
    return top_candidates

# Correct word
def correct_word(misspelled_word):
    candidates = generate_candidates(misspelled_word)
    best_candidate = misspelled_word
    max_posterior = float('-inf')
    
    for distance, candidate in candidates:
        error_model_prob = math.exp(-distance)
        language_model_prob = word_probabilities.get(candidate, 1e-10)
        posterior = math.log(error_model_prob) + math.log(language_model_prob)
        
        if posterior > max_posterior:
            max_posterior = posterior
            best_candidate = candidate
    
    return best_candidate

# Spell corrector function
def spell_corrector(misspelled_word):
    corrected_word = correct_word(misspelled_word)
    return corrected_word

if __name__ == "__main__":
    while True:
        misspelled_word = input("Enter a word to correct (or 'exit' to quit): ")
        if misspelled_word.lower() == 'exit':
            break
        corrected = spell_corrector(misspelled_word)
        print(f"Suggested correction: {corrected}")
