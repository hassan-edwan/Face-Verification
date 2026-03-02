import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between batches of vectors.
    Input: a, b of shape (N, D)
    Output: array of length N
    """
    # 1. Compute row-wise dot product: (N, D) * (N, D) -> sum across D
    dot_product = np.sum(a * b, axis=1)
    
    # 2. Compute row-wise L2 norms
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    
    # 3. Return similarity (add small epsilon to avoid division by zero)
    return dot_product / (norm_a * norm_b + 1e-9)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance between batches of vectors.
    Input: a, b of shape (N, D)
    Output: array of length N
    """
    # 1. Compute (a - b) squared, sum across D, then sqrt
    # NumPy handles the (N, D) subtraction element-wise automatically
    return np.linalg.norm(a - b, axis=1)