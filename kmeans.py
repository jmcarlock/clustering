import numpy as np

class kMeans:
    """
    k-Means clustering. Assumes dense data.

    Includes AFKMC2 initialization.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form and clusters to generate

    initialization : list
        list of index numbers to choose as first clusters, bypasses AFKMC2 init

    chain_length : int
        if initialization_indices are not passed, length of the markov chain for AFKMC2 initialization

    seeds : int
        if initialization_indices are not passed, random seed for AFKMC2 initialization

    max_iters : int, default: 100
        Maximum number of times the fitting process will iterate

    Usage
    -----

    >>> import numpy as np
    >>> from kmeans import kMeans

    >>> model = kMeans(n_clusters=100)
    >>> point_assignments, distances = model.fit(X, verbose=True,
                                                 return_assignments=True)

    >>> y = model.predict(X_test)

    >>> clusters = model.clusters

    >>> loss = model.reconstruction_error

    """
    def __init__(self, n_clusters, initialization=None, max_iters=100,
                 chain_length=2000, seed=None):

        self.n_clusters = n_clusters
        self.initialization = initialization

        self.chain_length = chain_length
        self.random_state = np.random.RandomState(seed)

        self.max_iters = max_iters

        self.clusters = None
        self.reconstruction_error = None

    def _AFKMC2_init(self, X):
        """
        Assumption Free kMC^2 initialization from Bachem's
        "Fast and Provably Good Seedings for k-Means".

        Takes samples using a Monte Carlo distribution built on a distance
        distribution from the first cluster center choice.

        Parameters
        ----------
        X : ndarray
            (n x d) array of float data


        Returns
        -------

        cluster_indices : list
            list of indices in the data to use as initial clusters. (length n_clusters)
        """

        # the candidate_index is a random choice of a single index from X
        candidate_index = self.random_state.choice(X.shape[0])
        cluster_indices = [candidate_index]

        # generate the proposal distribution q on the first candidate_index
        q = stance(X, X[candidate_index], self.beta)

        # proposal distribution has a regularization term
        q = (0.5*(q/q.sum()))+(1/(2*X.shape[0]))

        for i in range(self.n_clusters-1):
            candidates_indices = self.random_state.choice(X.shape[0], size=self.chain_length, p=q, replace=False)
            q_cand = q[candidates_indices]

            distances = np.empty(shape=(len(cluster_indices), self.chain_length))
            for dist_index, data_index in enumerate(cluster_indices):
                total_distance = distance(X[candidates_indices], X[data_index])
                distances[dist_index,:] = total_distance

            distances = np.min(distances, axis=0)

            acceptance_probs = self.random_state.random_sample(size=self.chain_length)

            for j in range(self.chain_length-1):
                cand_prob = distances[j]/q_cand[j]

                if j == 0:
                    curr_ind = j
                    curr_prob = cand_prob

                if (curr_prob == 0.0 or cand_prob/curr_prob > acceptance_probs[j]) and j not in cluster_indices:
                    curr_ind = j
                    curr_prob = cand_prob

            cluster_indices.append(candidates_indices[curr_ind])

        return(cluster_indices)

    def _log_clusters(self):
        """
        Logs the clusters.
        """
        print('numeric clusters:')
        print(self.clusters)

    def _expectation(self, X, n_samples):
        """
        Expectation step.

        Takes points and assigns them to their closest clusters using the generalized distance.
        """
        distances = np.empty(shape=(self.n_clusters, n_samples))
        for cluster_index in range(self.n_clusters):
            total_distance = distance(X, self.clusters[cluster_index])
            distances[cluster_index,:] = total_distance

        point_assignments = distances.argmin(axis=0)

        return(point_assignments, distances)


    def _maximization(self, X, point_assignments):
        """
        Maximization step.

        Recomputes new clusters.
        """
        for cluster_index in range(self.n_clusters):

            subset = (point_assignments == cluster_index)

            self.clusters[cluster_index] = X[subset].mean(axis=0)


    def fit(self, X, verbose=False, return_assignments=False):
        """
        Fits a k-clusters model with a custom distance function.

        Parameters
        ----------
        X : ndarray
            (n x d) array of float data

        verbose : bool, default: False

        return_assignments : bool, default: False
            return ndarrays of point_assignments and distances
        """

        self.reconstruction_error = []

        n_samples = X.shape[0]

        if not self.initialization:
            # Initialize the cluster centers.
            cluster_centers = self._AFKMC2_init(X)
        else:
            cluster_centers = self.initialization_indices

        # Populate the clusters
        self.clusters = X[cluster_centers,:]

        if verbose is True:
            print('initialization:')
            self._log_clusters()

        # TODO broadcast this per iteration
        old_point_assignments = np.empty(shape=(n_samples))

        for iteration in range(self.max_iters):

            # Expectation Step
            point_assignments, distances = self._expectation(X, n_samples)

            # Maximization Step
            self._maximization(X, point_assignments)

            # Stop if converged
            if (point_assignments != old_point_assignments).sum() == 0:
                if verbose is True:
                    print('\nDone!')
                break

            iter_reconsruction_error = np.sum(distances.T[np.arange(len(point_assignments)), point_assignments])
            self.reconstruction_error.append(iter_reconsruction_error)

            if verbose is True:
                print('\niteration', iteration, 'reconsruction_error', iter_reconsruction_error)
                self._log_clusters()

            old_point_assignments = point_assignments.copy()

        if return_assignments is True:
            return(point_assignments, distances)

    def predict(self, X, return_reconstruction_error=False):
        """
        Parameters
        ----------
        X : ndarray
            (n x d) array of float data

        return_reconstruction_error : bool
            return reconstruction error on predict


        Returns
        -------
        ndarrays of point_assignments and distances

        """
        n_samples = X.shape[0]
        point_assignments, distances = self._expectation(X, n_samples)

        if return_reconstruction_error:
            reconsruction_error = np.sum(distances.T[np.arange(len(point_assignments)), point_assignments])
            return(point_assignments, distances, reconsruction_error)

        else:
            return(point_assignments, distances)
