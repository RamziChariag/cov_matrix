# Functions that will be needed in all simulations
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import pandas as pd
from scipy.stats import multivariate_t
from scipy.linalg import eigh
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

    # Step 3: generate matrix
    variance_covariance_matrix = np.dot(lower_triangular, lower_triangular.T)

    # Step 4: ensure diagonal dominance
    diagonal_vector = np.diagonal(variance_covariance_matrix)
    row_sums = np.sum(np.abs(variance_covariance_matrix), axis = 1)
    column_sums = np.sum(np.abs(variance_covariance_matrix), axis = 0)
    new_diagonal_vector = np.maximum(diagonal_vector, row_sums, column_sums)
    np.fill_diagonal(variance_covariance_matrix, new_diagonal_vector)

    return new_diagonal_vector, variance_covariance_matrix

def make_positive_definite(variance_covariance_matrix):
    diagonal_vector = np.diagonal(variance_covariance_matrix)
    row_sums = np.sum(np.abs(variance_covariance_matrix), axis = 1)
    column_sums = np.sum(np.abs(variance_covariance_matrix), axis = 0)
    new_diagonal_vector = np.maximum(diagonal_vector, row_sums, column_sums)
    np.fill_diagonal(variance_covariance_matrix, new_diagonal_vector)
    return variance_covariance_matrix

def make_positive_definite_2(variance_covariance_matrix, eta):
    variance_covariance_matrix = variance_covariance_matrix + eta * np.eye(variance_covariance_matrix.shape[0])
    return variance_covariance_matrix

def is_positive_semidefinite(matrix):
    eigenvalues, _ = eigh(matrix)
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
    k = matrix.shape[1]
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

def f_norm_variance(X,S):
    num_rows = X.shape[0]
    sum_scaled_norms = 0.0

    for i in range(num_rows):
        x = X[i, :]  # Get the current row x
        diff_matrix = np.outer(x, x) - S
        scaled_norm = scaled_f_norm(diff_matrix)**2
        sum_scaled_norms += scaled_norm

    average_scaled_norm = sum_scaled_norms / num_rows
    return average_scaled_norm

def generate_serially_correlated_disturbances(mu_e, sigma_e, N1, N2, T, dim_to_correlate):
    
    n = N1 * N2 * T

    disturbances = np.random.normal(mu_e, sigma_e, n)
    
    if dim_to_correlate == "N1":
        disturbances = disturbances.reshape(N1, N2, T)
        for t in range(T):
            for n2 in range(N2):
                disturbances[:, n2, t] += disturbances[:, n2, t-1] if t > 0 else 0
        disturbances = disturbances.flatten()
    elif dim_to_correlate == "N2":
        disturbances = disturbances.reshape(N1, N2, T)
        for t in range(T):
            for n1 in range(N1):
                disturbances[n1, :, t] += disturbances[n1, :, t-1] if t > 0 else 0
        disturbances = disturbances.flatten()
    elif dim_to_correlate == "T":
        disturbances = disturbances.reshape(N1, N2, T)
        for n1 in range(N1):
            for n2 in range(N2):
                disturbances[n1, n2, :] += disturbances[n1, n2, :-1] if n2 > 0 else 0
        disturbances = disturbances.flatten()
    
    return disturbances

def transform_xy(X,y, number_of_variables):
    # Convert to a pandas DataFrame
    column_names = [f"x_{i+1}" for i in range(number_of_variables)]
    new_column_names_i = [f"x_{i+1}_bar_i" for i in range(number_of_variables)]
    new_column_names_j = [f"x_{i+1}_bar_j" for i in range(number_of_variables)]
    new_column_names_t = [f"x_{i+1}_bar_t" for i in range(number_of_variables)]
    fixed_effect_column_names = ["fe_1", "fe_2", "fe_3"]
    all_column_names = column_names + fixed_effect_column_names
    df = pd.DataFrame(data=X, columns=all_column_names)
    df['y'] = y

    # Generate the x_bar_i columns for each fixed effect
    for col, new_col_i, new_col_j, new_col_t in zip(column_names, new_column_names_i, new_column_names_j, new_column_names_t):
        grouped_i = df.groupby('fe_1')[col].transform('mean')
        df[new_col_i] = grouped_i
        
        grouped_j = df.groupby('fe_2')[col].transform('mean')
        df[new_col_j] = grouped_j
        
        grouped_t = df.groupby('fe_3')[col].transform('mean')
        df[new_col_t] = grouped_t

    # Generate the y_bar columns for each fixed effect
    for fe_col in fixed_effect_column_names:
        grouped_y = df.groupby(fe_col)['y'].transform('mean')
        df[f'y_bar_{fe_col}'] = grouped_y
        df = df.reset_index(drop=True)

    df['x_tilde'] = df['x_1'] + 2* df['x_1'].mean() - df['x_1_bar_i'] - df['x_1_bar_j'] - df['x_1_bar_t']
    df['y_tilde'] = df['y'] + 2* df['y'].mean() - df['y_bar_fe_1'] - df['y_bar_fe_2'] - df['y_bar_fe_3']

    # Convert calculated columns back to arrays
    x_tilde = df['x_tilde'].values
    x_tilde = x_tilde.reshape(-1, 1)
    y_tilde = df['y_tilde'].values
    return x_tilde, y_tilde

def generate_omega(case, mu_e, sigma_e, non_zero_prob, N1, N2, T, seed):
    n = N1 * N2 * T
    if case == 1:
        omega = np.eye(n) * sigma_e 
    elif case == 2:
        omega = np.kron(np.eye(N2 * T) , generate_cov_matrix(mu_e, sigma_e, N1, non_zero_prob, seed)[1])
    elif case == 3:
        omega = np.kron(np.eye(T) , generate_cov_matrix(mu_e, sigma_e, N1* N2, non_zero_prob, seed)[1])

    return omega 


def generate_disturbances(mu_e_vec, omega, n, t_dist_degree, lambda_parameter, mu_U, specification, seed):
    np.random.seed(seed)
    if(specification == "normal"):
        disturbances = np.random.multivariate_normal(mu_e_vec, omega)
    elif(specification == "t"):
        multivariate_t_dist = multivariate_t(mu_e_vec, shape=omega, df=t_dist_degree)
        disturbances = multivariate_t_dist.rvs()
    elif(specification == "sn"):
        tau = np.abs(np.random.multivariate_normal(mu_e_vec, np.eye(n))-mu_e_vec)+mu_e_vec # Draw from folded normal distribution at any mean, half normal in case mean = 0 
        lambda_skew = -(lambda_parameter**2)*np.eye(n) # The larger the lambda, the more skewed the distribution
        lambda_mat = np.sqrt(np.abs(lambda_skew))
        sigma_mat = omega - lambda_skew
        U = np.random.multivariate_normal(np.full(n, mu_U), sigma_mat)
        disturbances = np.dot(lambda_mat,tau) + U

    return disturbances

def generate_penalty_matrix(max_pen,size, ksi_1, ksi_2):
    # Create an empty matrix filled with zeros
    matrix = np.zeros((size, size))

    # Fill the upper and lower triangles with values between 0 and 1 using the sigmoid function
    for i in range(size):
        for j in range(i+1, size):
            # Calculate the value based on the distance from the diagonal
            distance = j - i
            value = max_pen / (1 + np.exp(-(distance-ksi_1)/ksi_2))
            
            # Set the symmetric values in the matrix
            matrix[i][j] = value
            matrix[j][i] = value

    return matrix

def zero_elements_below_tolerance(matrix, tolerance):
    """
    Set matrix elements below a specified tolerance to zero.

    Args:
    matrix (numpy.ndarray): The input matrix.
    tolerance (float): The tolerance value. Elements smaller than this will be set to zero.

    Returns:
    numpy.ndarray: The matrix with elements below the tolerance set to zero.
    """
    # Create a copy of the input matrix
    modified_matrix = matrix.copy()
    
    # Set elements below the tolerance to zero
    modified_matrix[np.abs(modified_matrix) < tolerance] = 0.0
    
    return modified_matrix