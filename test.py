import numpy as np
from kprototypes import kPrototypes
from knn import kNN

def generate_test_data():
    """ Generates data that can be used for testing SpaceJam.

    Returns
    -------
    X_num : ndarray
        (n x d) array of float data

    X_cat : ndarray
        (n x d) array of catergorical data

    y : ndarray
        (n x 1) array of target labels

    Usage
    -----

    >>> X_num, X_cat, bins = generate_test_data()

    """
    X_num = np.array([
        # duration, bytes
        [100, 300],
        [0, 30],
        [0, 23],
        [400,30],
        [1000,903490],
        [400,40],
        [0,30],
        [0,13],
        [900,3],
        [1000,3]])

    X_cat = np.array([
        ['apple',  'bravo', 'blue', 'short'],
        ['orange', 'delta', 'yellow', 'tiny'],
        ['banana', 'alpha', 'red', 'long'],
        ['pear',   'gamma', 'purple', 'fat'],
        ['apple',  'alpha', 'red', 'skinny'],
        ['apple',  'gamma', 'red', 'average'],
        ['pear',   'gamma', 'purple', 'fat'],
        ['apple',  'alpha', 'red', 'skinny'],
        ['apple',  'gamma', 'red', 'average'],
        ['banana', 'blue', 'blue', 'short']])

    y = np.array([['good'],
                  ['bad'],
                  ['good'],
                  ['bad'],
                  ['good'],
                  ['bad']])

    return X_num, X_cat, y


X_num, X_cat, y = generate_test_data()

# test k-Prototypes

model = kPrototypes(n_prototypes=3, chain_length=3)
model.fit(X_num, X_cat, verbose=True, return_assignments=False)

print(model.prototypes.num)
print(model.prototypes.cat)

# test k-NN

nearest_neighbors = kNN(n_neighbors=3).predict(X_num[:6], X_cat[:6], y,
                                               X_num[6:], X_cat[6:])

print(nearest_neighbors)
