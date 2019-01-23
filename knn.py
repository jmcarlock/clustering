import numpy as np
from distances import generalized_distance

class kNN:
    """
    k nearest neighbor clustering for networking data with a hierarchical distance for IP address.

    A lazy algorithm. No training is necessary.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors to consider in the output

    include_continous : bool
        if False, continous values are discritized

    alpha : float
        Numeric distance scaling parameter

    beta : float
        Categorical distance scaling parameter

    Usage
    -----

    >>> import numpy as np
    >>> from kprototypes import kNN

    >>> model = kNN(n_neighbors=3)

    >>> y_pred = model.predict(X_num, X_cat, y,
                               X_test_num, X_test_cat)

    """
    def __init__(self, n_neighbors=3, include_continous=True, alpha=1, beta=1):

        self.n_neighbors = n_neighbors
        self.include_continous = include_continous

        self.alpha = alpha
        self.beta = beta

    def _point_distances(self, X_num, X_cat, X_num_test, X_cat_test):
        """
        Takes points and assigns them to their closest neighbors using the generalized distance.
        """
        distances = np.empty(shape=(X_num_test.shape[0], X_num.shape[0]))
        for index in range(X_num_test.shape[0]):
            total_distance = generalized_distance(X_num, X_num_test[index,:],
                                                  X_cat, X_cat_test[index,:],
                                                  self.include_continous, self.alpha, self.beta)
            distances[index,:] = total_distance

        return(distances)


    def predict(self, X_num, X_cat, y, X_num_test, X_cat_test):
        """
        kNN algorithm.

        Parameters
        ----------

        Training data:
            X_num : ndarray
                (n x d) array of float data

            X_cat : ndarray
                (n x d) array of catergorical data

            y : ndarray
                (n x 1) array of class labels

        Testing data:
            X_num_test : ndarray
                (n x d) array of float data

            X_cat_test : ndarray
                (n x d) array of catergorical data

        Returns
        -------

        y_pred : ndarray
            (n x 1) array of predictions from k-NN

        """
        n_test = X_num_test.shape[0]

        distances = self._point_distances(X_num, X_cat, X_num_test, X_cat_test)

        # get the distance, each row gives indices sorted ascending
        distances = distances.argsort(axis=1)

        # we only care about the n_neighbors amount of columns
        distances = distances[:,:self.n_neighbors]

        min_dists = y[distances]

        prediction = []

        for index in range(min_dists.shape[0]):
            unique_inidices, unique_counts = np.unique(min_dists[index], return_counts=True)
            prediction.append(unique_inidices[unique_counts.argmax()])

        return(prediction)
