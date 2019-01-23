import logging
import numpy as np
from distances import generalized_distance

logger = logging.getLogger(__name__)

class Prototypes:
    def __init__(self):
        """
        Object for storing prototypes.
        """
        self.num = None
        self.cat = None

class kPrototypes:
    """
    k-Prototypes clustering. Assumes dense data.

    Includes AFKMC2 initialization.

    Parameters
    ----------
    n_prototypes : int
        Number of clusters to form and prototypes to generate

    initialization : list
        list of index numbers to choose as first prototypes, bypasses AFKMC2 init

    include_continous : bool
        if False, continous values are discritized and the algorithm becomes k-Modes

    chain_length : int
        if initialization_indices are not passed, length of the markov chain for AFKMC2 initialization

    seeds : int
        if initialization_indices are not passed, random seed for AFKMC2 initialization

    max_iters : int, default: 100
        Maximum number of times the fitting process will iterate

    alpha : float
        Numeric distance scaling parameter

    beta : float
        Categorical distance scaling parameter

    Usage
    -----

    >>> import numpy as np
    >>> from kprototypes import kPrototypes

    >>> model = kPrototypes(n_prototypes=100)
    >>> point_assignments, distances = model.fit(X_num, X_cat, verbose=True,
                                                 return_assignments=True)

    >>> y = model.predict(X_test_num, X_test_cat)

    >>> prototypes_num = model.prototypes.num
    >>> prototypes_cat = model.prototypes.cat

    >>> loss = model.reconstruction_error

    """
    def __init__(self, n_prototypes, initialization=None, include_continous=True, max_iters=100,
                 chain_length=2000, seed=None, alpha=1, beta=1):

        self.n_prototypes = n_prototypes
        self.initialization = initialization
        self.include_continous = include_continous

        self.chain_length = chain_length
        self.random_state = np.random.RandomState(seed)

        self.max_iters = max_iters
        self.alpha = alpha
        self.beta = beta

        self.prototypes = Prototypes()
        self.reconstruction_error = None

    def _AFKMC2_init(self, X_num, X_cat):
        """
        Assumption Free kMC^2 initialization from Bachem's
        "Fast and Provably Good Seedings for k-Means".

        Takes samples using a Monte Carlo distribution built on a distance
        distribution from the first cluster center choice.

        Parameters
        ----------
        X_num : ndarray
            (n x d) array of float data

        X_cat : ndarray
            (n x d) array of catergorical data


        Returns
        -------

        prototype_indices : list
            list of indices in the data to use as initial prototypes. (length n_prototypes)
        """

        # the candidate_index is a random choice of a single index from X
        candidate_index = self.random_state.choice(X_num.shape[0])
        prototype_indices = [candidate_index]

        # generate the proposal distribution q on the first candidate_index
        q = generalized_distance(X_num, X_num[candidate_index],
                                 X_cat, X_cat[candidate_index],
                                 self.include_continous, self.alpha, self.beta)

        # proposal distribution has a regularization term
        q = (0.5*(q/q.sum()))+(1/(2*X_num.shape[0]))

        for i in range(self.n_prototypes-1):
            candidates_indices = self.random_state.choice(X_num.shape[0], size=self.chain_length, p=q, replace=False)
            q_cand = q[candidates_indices]

            distances = np.empty(shape=(len(prototype_indices), self.chain_length))
            for dist_index, data_index in enumerate(prototype_indices):
                total_distance = generalized_distance(X_num[candidates_indices], X_num[data_index],
                                                      X_cat[candidates_indices], X_cat[data_index],
                                                      self.include_continous, self.alpha, self.beta)
                distances[dist_index,:] = total_distance

            distances = np.min(distances, axis=0)

            acceptance_probs = self.random_state.random_sample(size=self.chain_length)

            for j in range(self.chain_length-1):
                cand_prob = distances[j]/q_cand[j]

                if j == 0:
                    curr_ind = j
                    curr_prob = cand_prob

                if (curr_prob == 0.0 or cand_prob/curr_prob > acceptance_probs[j]) and j not in prototype_indices:
                    curr_ind = j
                    curr_prob = cand_prob

            prototype_indices.append(candidates_indices[curr_ind])

        return(prototype_indices)

    def _log_prototypes(self):
        """
        Logs the prototypes.
        """
        logger.info('numeric prototypes:')
        logger.info(self.prototypes.num)
        logger.info('\ncategorical prototypes:')
        logger.info(self.prototypes.cat)

    def _expectation(self, X_num, X_cat, n_samples):
        """
        Expectation step.

        Takes points and assigns them to their closest prototypes using the generalized distance.
        """
        distances = np.empty(shape=(self.n_prototypes, n_samples))
        for prototype_index in range(self.n_prototypes):
            total_distance = generalized_distance(X_num, self.prototypes.num[prototype_index],
                                                  X_cat, self.prototypes.cat[prototype_index],
                                                  self.include_continous, self.alpha, self.beta)
            distances[prototype_index,:] = total_distance

        point_assignments = distances.argmin(axis=0)

        return(point_assignments, distances)


    def _maximization(self, X_num, X_cat, point_assignments):
        """
        Maximization step.

        Recomputes new prototypes.
        """
        for prototype_index in range(self.n_prototypes):

            subset = (point_assignments == prototype_index)

            self.prototypes.num[prototype_index] = X_num[subset].mean(axis=0)

            for column_index in range(X_cat.shape[1]):
                keys, counts = np.unique(X_cat[subset, column_index], return_counts=True)
                self.prototypes.cat[prototype_index, column_index] = keys[counts.argmax()]


    def fit(self, X_num, X_cat, verbose=False, return_assignments=False):
        """
        Fits a k-prototypes model with a custom distance function.

        Parameters
        ----------
        X_num : ndarray
            (n x d) array of float data

        X_cat : ndarray
            (n x d) array of catergorical data

        verbose : bool, default: False

        return_assignments : bool, default: False
            return ndarrays of point_assignments and distances
        """

        self.reconstruction_error = []

        n_samples = X_num.shape[0]

        if not self.initialization:
            # Initialize the cluster centers.
            cluster_centers = self._AFKMC2_init(X_num, X_cat)
        else:
            cluster_centers = self.initialization_indices

        # Populate the prototypes
        self.prototypes.num = X_num[cluster_centers,:]
        self.prototypes.cat = X_cat[cluster_centers,:]

        if verbose is True:
            logger.info('initialization:')
            self._log_prototypes()

        # TODO broadcast this per iteration
        old_point_assignments = np.empty(shape=(n_samples))

        for iteration in range(self.max_iters):

            # Expectation Step
            point_assignments, distances = self._expectation(X_num, X_cat, n_samples)

            # Maximization Step
            self._maximization(X_num, X_cat, point_assignments)

            # Stop if converged
            if (point_assignments != old_point_assignments).sum() == 0:
                if verbose is True:
                    logger.info('\nDone!')
                break

            iter_reconsruction_error = np.sum(distances.T[np.arange(len(point_assignments)), point_assignments])
            self.reconstruction_error.append(iter_reconsruction_error)

            if verbose is True:
                logger.info('\niteration', iteration, 'reconsruction_error', iter_reconsruction_error)
                self._log_prototypes()

            old_point_assignments = point_assignments.copy()

        if return_assignments is True:
            return(point_assignments, distances)

    def predict(self, X_num, X_cat, return_reconstruction_error=False):
        """
        Parameters
        ----------
        X_num : ndarray
            (n x d) array of float data

        X_cat : ndarray
            (n x d) array of catergorical data

        return_reconstruction_error : bool
            return reconstruction error on predict


        Returns
        -------
        ndarrays of point_assignments and distances

        """
        n_samples = X_num.shape[0]
        point_assignments, distances = self._expectation(X_num, X_cat, n_samples)

        if return_reconstruction_error:
            reconsruction_error = np.sum(distances.T[np.arange(len(point_assignments)), point_assignments])
            return(point_assignments, distances, reconsruction_error)

        else:
            return(point_assignments, distances)
