# Markov Text Generator Example Applications and Test Observations 

## Key Observations

1. Deterministic vs. Stochastic: The deterministic mode consistently produced the same output for each configuration, while the stochastic mode generally produced varied results, demonstrating the randomness in word selection.

2. n-gram size impact: 
   - Larger n-gram sizes (n=4) tended to produce more coherent phrases compared to smaller n-gram sizes (n=3). 
   - Larger n-gram sizes produced less stochastic results. This is because there are fewer occurrences of matching text in the corpus for larger n-grams, which limits the number of possible continuations. 
   - The n=4 model sometimes produced complete, sensible phrases (e.g., "she was not a thing to be thought of ;"). 

## Test Configurations

```
# Example applications of the finish_sentence function in both deterministic and stochastic modes.

# Seed words: 'she was not'
# Test case 1: Deterministic
sentence = ['she', 'was', 'not']
n = 3
randomize = False
result = finish_sentence(sentence, n, corpus, randomize)
print(f"Deterministic (n={n}): {' '.join(result)}")

# Test case 2: Stochastic
sentence = ['she', 'was', 'not']
n = 3
randomize = True
for _ in range(3):
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Stochastic (n={n}): {' '.join(result)}")

# Test case 3: Different n-gram size (n = 4) and deterministic mode
sentence = ['she', 'was', 'not']
n = 4
randomize = False
result = finish_sentence(sentence, n, corpus, randomize)
print(f"Deterministic (n={n}): {' '.join(result)}")

# Test case 4: Different n-gram size (n = 4) and stochastic mode
sentence = ['she', 'was', 'not']
n = 4
randomize = True
for _ in range(3):
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Stochastic (n={n}): {' '.join(result)}")

# Seed words: 'could not even'
# Test case 5: Deterministic
sentence = ['could', 'not', 'even']
n = 3
randomize = False
result = finish_sentence(sentence, n, corpus, randomize)
print(f"Deterministic (n={n}): {' '.join(result)}")

# Test case 6: Stochastic
sentence = ['could', 'not', 'even']
n = 3
randomize = True
for _ in range(3):
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Stochastic (n={n}): {' '.join(result)}")

# Test case 7: Different n-gram size (n = 4) and deterministic mode
sentence = ['could', 'not', 'even']
n = 4
randomize = False
result = finish_sentence(sentence, n, corpus, randomize)
print(f"Deterministic (n={n}): {' '.join(result)}")

# Test case 8: Different n-gram size (n = 4) and stochastic mode
sentence = ['could', 'not', 'even']
n = 4
randomize = True
for _ in range(3):
    result = finish_sentence(sentence, n, corpus, randomize)
    print(f"Stochastic (n={n}): {' '.join(result)}")
```

1. Seed phrase: "she was not"
   - n-gram sizes: 3 and 4
   - Modes: Deterministic and Stochastic (3 runs each for stochastic)

2. Seed phrase: "could not even"
   - n-gram sizes: 3 and 4
   - Modes: Deterministic and Stochastic (3 runs each for stochastic)

## Results Analysis

### Seed phrase: "she was not"

```
Deterministic (n=3): she was not in the world , and the two
Stochastic (n=3): she was not quite understand the parsonage at delaford ,
Stochastic (n=3): she was not in every change of sentiments which mrs.
Stochastic (n=3): she was not easily overcome .
Deterministic (n=4): she was not a thing to be thought of ;
Stochastic (n=4): she was not only ready to worship him as a
Stochastic (n=4): she was not doomed , however , i am afraid
Stochastic (n=4): she was not suspected of any extraordinary interest in it
```

1. n-gram size: 3
   - Deterministic: The model consistently produced "she was not in the world , and the two"
   - Stochastic: Three different outputs were generated, demonstrating variability:
     1. "she was not quite understand the parsonage at delaford ,"
     2. "she was not in every change of sentiments which mrs."
     3. "she was not easily overcome ."

2. n-gram size: 4
   - Deterministic: The model produced "she was not a thing to be thought of ;"
   - Stochastic: Three different outputs were generated:
     1. "she was not only ready to worship him as a"
     2. "she was not doomed , however , i am afraid"
     3. "she was not suspected of any extraordinary interest in it"

### Seed phrase: "could not even"

```
Deterministic (n=3): could not even the boisterous mirth of sir john ,
Stochastic (n=3): could not even himself .
Stochastic (n=3): could not even genteel , she fell into a calmer
Stochastic (n=3): could not even kiss them .
Deterministic (n=4): could not even kiss them .
Stochastic (n=4): could not even kiss them .
Stochastic (n=4): could not even wish him successful , she heartily wished
Stochastic (n=4): could not even kiss them .
```

1. n-gram size: 3
   - Deterministic: The model consistently produced "could not even the boisterous mirth of sir john ,"
   - Stochastic: Three different outputs were generated:
     1. "could not even himself ."
     2. "could not even genteel , she fell into a calmer"
     3. "could not even kiss them ."

2. n-gram size: 4
   - Deterministic: The model produced "could not even kiss them ."
   - Stochastic: 
     1. "could not even kiss them ."
     2. "could not even wish him successful , she heartily wished"
     3. "could not even kiss them ."
