import numpy as np

def _euclidean_distance(a, b):
    """
    Euclidean distance between vector/matrix a and vector b.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def _hamming_distance(a, b):
    """
    Hamming distance, a simple matching dissimilarity function,
    distance between vector/matrix a and vector b.
    """
    return np.sum(a != b, axis=1)

def generalized_distance(X_num_1, X_num_2, X_cat_1, X_cat_2,
                         continuous_scaling, alpha, beta):
    """ Gower distance.

    X_num_1 and X_cat_2 should have multiple rows.
    X_num_2 and X_cat_2 should only have one row.
    """

    if continuous_scaling is not 'discretize':
        num_dist = _euclidean_distance(X_num_1, X_num_2)
    else:
        num_dist = _hamming_distance(X_num_1, X_num_2)

    cat_dist = _hamming_distance(X_cat_1, X_cat_2)

    total_dist = (alpha * num_dist) + (beta * cat_dist)
    return(total_dist)
