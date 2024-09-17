import numpy as np
import matplotlib.pyplot as plt

# Define the observations
observations = ["apple", "apple", "apple", "apple", "apple", "apple", "banana", "banana", "banana", "banana"]
n_apple, n_banana, n_total = 6, 4, 10

# Create an array of p_apple values
p_apple_range = np.linspace(0, 1, 1000)

# Calculate the probability for each p_apple value
probabilities = []
for p_apple in p_apple_range:
    p_banana = 1 - p_apple
    prob = (p_apple ** n_apple) * (p_banana ** n_banana)
    probabilities.append(prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(p_apple_range, probabilities)
plt.title("Probability of Observations under Unigram Model")
plt.xlabel("p_apple\n(p_banana = 1 - p_apple)")
plt.ylabel("Probability")
plt.grid(True)

# Show the maximum probability point
max_prob_index = np.argmax(probabilities)
max_p_apple = p_apple_range[max_prob_index]
plt.axvline(x=max_p_apple, color='r', linestyle='--', label=f'Max at p_apple = {max_p_apple:.2f}')

plt.legend()
plt.show()
