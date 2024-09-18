import numpy as np
import pandas as pd

# WARNING: Most of this code was optimized with ChatGPT
# with initial exploration on this blog: https://adeveloperdiary.com/data-science/machine-learning/forward-and-backward-algorithm-in-hidden-markov-model/

# Given transition and emission matrices
a = np.array([[0.54, 0.46], [0.49, 0.51]])  # Transition probabilities
b = np.array([[0.16, 0.26, 0.58], [0.25, 0.28, 0.47]])  # Emission probabilities
initial_distribution = np.array([0.5, 0.5])  # Initial distribution

# Number of states (hidden) and observations (visible)
states = [0, 1]  # Hidden states
observations = [0, 1, 2]  # Visible states

# Generate a sequence of hidden states and observations
def generate_sequence(length):
    hidden_states = []
    visible_sequence = []

    # Initial state
    current_state = np.random.choice(states, p=initial_distribution)

    # Generate hidden states and visible states
    for _ in range(length):
        hidden_states.append(current_state)  # Append the current hidden state

        # Generate visible observation based on emission probabilities
        visible_observation = np.random.choice(observations, p=b[current_state])
        visible_sequence.append(visible_observation)

        # Generate next hidden state based on transition probabilities
        current_state = np.random.choice(states, p=a[current_state])

    return hidden_states, visible_sequence

# Generate data
sequence_length = 100  # Set the length of the sequence
hidden_states, visible_sequence = generate_sequence(sequence_length)

print(len(hidden_states))
print(len(visible_sequence))

# Save to CSV
# df = pd.DataFrame({'Hidden': hidden_states, 'Visible': visible_sequence})
# df.to_csv('data_python.csv', index=False)

# print("Sample data generated and saved to 'data_python.csv'")


def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    
    return alpha

# Assuming the forward function is defined as above

# Generate the sequence of hidden and visible states
hidden_states, visible_sequence = generate_sequence(sequence_length)

# Convert visible_sequence to a numpy array (V) if it isn't already
V = np.array(visible_sequence)

# Pass the generated data into the forward algorithm
alpha = forward(V, a, b, initial_distribution)

# Print the result
print("Alpha (forward probabilities):")
print(alpha)

def log_forward(V, a, b, initial_distribution):
    # Convert to log space to prevent underflow
    log_a = np.log(a)
    log_b = np.log(b)
    log_initial_distribution = np.log(initial_distribution)

    # Initialize alpha in log space
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = log_initial_distribution + log_b[:, V[0]]

    # Compute the forward log probabilities recursively
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Instead of multiplying, we sum the log probabilities
            alpha[t, j] = np.logaddexp.reduce(alpha[t - 1] + log_a[:, j]) + log_b[j, V[t]]

    return alpha

# Example usage
alpha_log = log_forward(V, a, b, initial_distribution)
print("Log-Alpha (log-forward probabilities):")
print(alpha_log)
