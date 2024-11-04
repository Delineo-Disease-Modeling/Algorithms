import numpy as np

# Parameters
alpha, beta, gamma = 0.1, 0.1, 0.1  # Initial guesses for alpha, beta, gamma
learning_rate = 0.001
num_iterations = 1000  # Number of gradient descent steps

# Decay factor and smoothing parameter
delta = 0.1  # Decay factor for beta
lambda_smoothing = 0.01  # Smoothing factor

# Observed data (replace with actual observed values from SafeGraph or other sources)
observed_movement = np.array([...])  # Observed movement probabilities

# Sample input data (example)
C = np.array([10, 15, 20])  # Capacity for each POI
S_prev = np.array([5, 10, 15])  # Current population size for each POI at previous timestep
F_prev = np.array([0.3, 0.5, 0.7])  # Previous visit counts
A_prev = np.array([0.4, 0.6, 0.8])  # After-visit tendency

# Helper function to compute the movement probability distribution M
def compute_movement_probability(alpha, beta, gamma, C, S_prev, F_prev, A_prev, delta, lambda_smoothing):
    # Decay previous visits and apply smoothing
    F_prev_adjusted = delta * F_prev
    A_prev_adjusted = lambda_smoothing * A_prev
    
    # Calculate movement probability M based on current parameters
    M = np.abs(alpha * (C - S_prev) + beta * F_prev_adjusted + gamma * A_prev_adjusted)
    return M / np.sum(M)  # Normalize to make it a probability distribution

# Loss function (Mean Squared Error)
def compute_loss(predicted, observed):
    return np.mean((predicted - observed) ** 2)

# Gradient descent optimization
for i in range(num_iterations):
    # Compute current movement probabilities
    M = compute_movement_probability(alpha, beta, gamma, C, S_prev, F_prev, A_prev, delta, lambda_smoothing)
    
    # Compute the loss
    loss = compute_loss(M, observed_movement)
    
    # Compute gradients
    grad_alpha = np.sum(2 * (M - observed_movement) * (C - S_prev))
    grad_beta = np.sum(2 * (M - observed_movement) * F_prev * delta)
    grad_gamma = np.sum(2 * (M - observed_movement) * A_prev * lambda_smoothing)
    
    # Update parameters
    alpha -= learning_rate * grad_alpha
    beta -= learning_rate * grad_beta
    gamma -= learning_rate * grad_gamma
    
    # Optionally, print progress
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss}, alpha = {alpha}, beta = {beta}, gamma = {gamma}")

print(f"Optimized parameters: alpha = {alpha}, beta = {beta}, gamma = {gamma}")