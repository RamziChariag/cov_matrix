# Functions that will be needed in all simulations
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import pandas as pd
import os

def generate_cov_matrix(mu, sigma, k, non_zero_prob, seed):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Step 1: Generate a random lower triangular matrix from a normal distribution
    lower_triangular = np.random.normal(mu, sigma, (k, k))
    lower_triangular[np.triu_indices(k, k=1)] = 0.0  # Set upper triangle to zero

    # Step 2: Set elements to zero with a specified probability
    zero_probs = np.random.random((k,k))
    np.fill_diagonal(zero_probs, 0) # keep the diagonal elements
    lower_triangular[zero_probs > non_zero_prob] = 0

    # Step 3: generate a PD matrix
    variance_covariance_matrix = np.dot(lower_triangular, lower_triangular.T)
    
    diagonal_vector = np.diagonal(variance_covariance_matrix)

    return diagonal_vector, variance_covariance_matrix

def is_positive_semidefinite(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.all(eigenvalues >= 0)

def generate_multivariate_data(cov_matrix, n, seed):
    # Get the dimension of the covariance matrix (number of variables)
    k = cov_matrix.shape[0]

    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate means from uniform distribution between -50 and 50
    means = np.random.uniform(low=-50, high=50, size=k)

    # Generate data from multivariate normal distribution with the generated means and covariance matrix
    data = np.random.multivariate_normal(mean=means, cov=cov_matrix, size=n)

    return data

def scaled_f_norm(matrix):
    # Get the dimension of the matrix
    k = matrix.shape[0]
    norm = np.linalg.norm(matrix, 'fro')/np.sqrt(k)
    return norm

def sample_covariance_matrix(data):
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    return cov_matrix

def braket_operator_identity(matrix):
    # Get the dimension of the matrix
    k = matrix.shape[0]
    # Generate identity matrix of proper size
    identity_matrix = np.eye(k)
    # Calculate the bracket operator
    bracket = np.trace(np.dot(matrix.T, identity_matrix))/k

    return bracket

def f_norm_variance(X):
    num_rows = X.shape[0]
    sum_scaled_norms = 0.0

    for i in range(num_rows):
        x = X[i, :]  # Get the current row x
        diff_matrix = np.outer(x, x) - X
        scaled_norm = scaled_f_norm(diff_matrix)**2
        sum_scaled_norms += scaled_norm

    average_scaled_norm = sum_scaled_norms / num_rows
    return average_scaled_norm