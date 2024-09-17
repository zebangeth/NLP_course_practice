
## Key Assumptions in This Implementation


### 1. Only Single-Edit Corrections

- **Assumption:**
  - The correct spelling of a misspelled word is at most **one edit distance** away, as 80% of are within edit distance of one (Daniel Jurafsky & James H. Martin, 2024). 
  - The candidate generation function produces words that are exactly one insertion, deletion, or substitution away from the input word.
- **Limitation:**
  - Words requiring multiple edits for correction are not considered, potentially missing the correct word.

### 2. No Transposition Errors

- **Assumption:**
  - The model does not account for **transposition errors**, where two adjacent characters are swapped (e.g., 'hte' instead of 'the').
- **Limitation:**
  - Misspelled words resulting from transpositions may not be corrected.

### 3. Candidate Filtering Based on Vocabulary

- **Assumption:**
  - Only candidates that exist in the vocabulary (`word_freq` dictionary generated from count_1w.txt) are considered.
  - Words not present in the vocabulary, even if they are valid words, will not be suggested.
- **Limitation:**
  - The spell checker cannot suggest corrections to words outside the known vocabulary.

### 4. Independence of Errors

- **Assumption:**
  - Errors are considered independently, without accounting for the sequence of characters or contextual information.
  - The model only considers the prior character in the context of surrounding characters.
- **Limitation:**
  - May not accurately reflect real-world error patterns, where certain errors are more likely in specific contexts.

---

## Scenarios Where This Spelling Corrector Works Well

**Example 1: Single-Character Substitution**

- **Misspelled Word:** `helo`
- **Correct Word:** `help`
- **Explanation:**
  - The misspelling `helo` is one substitution away from `help`.
  - The correct word `help` has a higher posterior probability due to its frequency and the error model.
  
**Example 2: Single-Character Deletion**

- **Misspelled Word:** `worl`
- **Correct Word:** `world`
- **Explanation:**
  - The misspelling `worl` is one deletion away from `world`.
  - The correct word `world` is selected based on the combined probability.

**Example 3: Single-Character Insertion**

- **Misspelled Word:** `googlee`
- **Correct Word:** `google`
- **Explanation:**
  - The misspelling `googlee` is one insertion away from `google`.
  - The high frequency of `google` helps in selecting it as the correction.

## Scenarios Where It Could Do Better

**Example 1: Transposition Errors**

- **Misspelled Word:** `chekc`
- **Intended Word:** `check`
- **Generated Word:** `cheks`
- **Explanation:**
  - The misspelling involves transposition of 'c' and 'k'.
  - Candidate generation does not consider words more than one edit away.
  - `check` is not included in the candidate set, and thus not correctly corrected.
- **Improvement Suggestion:**
  - Include transposition as a possible edit in candidate generation.
  - Generate candidates by swapping adjacent characters.

**Example 2: Errors Requiring Multiple Edits**

- **Misspelled Word:** `acomodate`
- **Intended Word:** `accommodate`
- **Generated Word:** `accomodate`
- **Explanation:**
  - The misspelling lacks one 'c' and one 'm'.
  - Requires two insertions to correct.
- **Improvement Suggestion:**
  - Allow candidate generation to consider edits up to two or more.
  - Use frequency data to limit the candidate set for efficiency.

**Example 3: Uncommon Words Not in Vocabulary**

- **Misspelled Word:** `zebangg`
- **Intended Word:** `zebang`
- **Generated Word:** `zebangg`
- **Explanation:**
  - If `zebang` (my first name) is not in the vocabulary, the spell checker cannot confirm it's correct.
  - May not suggest any correction or suggest incorrect ones.
- **Improvement Suggestion:**
  - Expand the vocabulary with more words.
  - Use a larger corpus to build the word frequency dictionary.

### Reasons for the Poor Behaviors

1. **Single-Edit Limitation:**
   - Missing correct candidates that are more than one edit away.
2. **Lack of Transposition Handling:**
   - Cannot correct common typing errors involving swapped letters.
3. **Vocabulary Limitations:**
   - Cannot suggest or confirm words not present in the vocabulary.

## Potential Improvement

**1. Extend Edit Distance**

- **Allow Multiple Edits:** Increase the maximum edit distance to consider candidates that are two or more edits away.
- **Implementation:** Modify the `generate_candidates` function to apply edits recursively or iterate over possible combinations.

**2. Include Transposition Errors**

- **Add Transpositions to Edits:** Generate candidates by swapping adjacent characters in the misspelled word.
- **Implementation:**
  - Add error stats for transposition edits. 
  - In the `generate_candidates` function, add:
    ```python
    # Transpositions
    for i in range(len(word) - 1):
        candidates.add(word[:i] + word[i+1] + word[i] + word[i+2:])
    ```

**3. Use Weighted Error Probabilities in Candidate Generation**

- **Prioritize Likely Edits:** Use error probabilities to guide candidate generation, focusing on more probable errors.
- **Implementation:** Instead of generating all possible edits, generate those with higher error probabilities.
- **Benefits:**
  - Reduces the candidate set to more likely corrections.
  - Improves efficiency and accuracy.

**4. Expand the Vocabulary**

- **Include More Words:** Use a larger corpus to build the vocabulary, capturing less common words.

**5. Incorporate N-gram or Other More Advanced Language Models**

- **Use Contextual Information:** If correcting words within sentences, use n-grams to consider surrounding words.
- **Implementation:** Adjust `P(w)` to consider the context, improving the selection of the correct word among candidates.
- **Benefits:**
  - Improves accuracy, especially when multiple corrections are plausible.

