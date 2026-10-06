import numpy as np
from lingam import HighDimDirectLiNGAM
from sklearn.utils import check_array


class PriorConstrainedHighDimLiNGAM(HighDimDirectLiNGAM):
    """An extension of the high-dimensional LiNGAM algorithm [1]_ with prior constraints.

    The candidate parent set of each variable is restricted using the possible
    ancestors matrix obtained from the PC algorithm.

    References
    ----------
    .. [1] Wang, Y. Samuel, and Mathias Drton. "High-dimensional causal discovery
       under non-Gaussianity." Biometrika 107.1 (2020): 41-59.
    """

    def __init__(
        self,
        J=3,
        K=4,
        alpha=0.5,
        estimate_adj_mat=True,
        possible_ancestors=None,
        random_state=None,
    ):
        """Construct a PriorConstrainedHighDimLiNGAM model.

        Parameters
        ----------
        J : int, optional (default=3)
            Assumed largest in-degree.
        K : int, optional (default=4)
            The degree of the moment which is non-Gaussianity.
        alpha : float, optional (default=0.5)
            The value for pruning away false parents.
        estimate_adj_mat : bool, optional (default=True)
            If False, skip the estimation of the adjacency matrix.
        possible_ancestors : array-like, shape (n_features, n_features), optional (default=None)
            Boolean matrix in child-by-parent orientation. The matrix obtained from
            the PC algorithm must be transposed before being passed here.
            ``possible_ancestors[u, v] == True`` means that ``v`` is a possible
            ancestor of ``u``. If provided, candidate
            parents outside this set are excluded entirely.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        super().__init__(
            J=J,
            K=K,
            alpha=alpha,
            estimate_adj_mat=estimate_adj_mat,
            random_state=random_state,
        )

        self._possible_ancestors = (
            np.asarray(possible_ancestors, dtype=bool) if possible_ancestors is not None else None
        )

    def _validate_possible_ancestors(self, p):
        """Validate the shape of the possible ancestors matrix and clear its diagonal.

        Parameters
        ----------
        p : int
            Number of variables.

        Raises
        ------
        ValueError
            If the shape of ``possible_ancestors_`` does not match ``(p, p)``.
        """
        if self._possible_ancestors is None:
            return

        if self._possible_ancestors.shape != (p, p):
            raise ValueError(
                f"possible_ancestors shape {self._possible_ancestors.shape} does not "
                f"match number of variables ({p}, {p})."
            )

        np.fill_diagonal(self._possible_ancestors, False)

    def _get_candidate_parents(self, v, ordered):
        """Compute the candidate parent set ``C_v = Ordered ∩ PossibleAncestors(v)``.

        Parameters
        ----------
        v : int
            Index of the target variable.
        ordered : list of int
            Variables that have already been placed in the causal order.

        Returns
        -------
        candidates : set of int
            Candidate parent indices for ``v``.
        """
        if self._possible_ancestors is None:
            return set(ordered)

        pa_v = set(np.where(self._possible_ancestors[v, :])[0])
        return set(ordered).intersection(pa_v)

    def fit(self, X):
        """Fit the model to X using possible ancestors constraints.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Fitted model.
        """
        X_arr = check_array(X)
        p = X_arr.shape[1]

        self._validate_possible_ancestors(p)

        self._Y = X_arr
        self._yty = self._Y.T @ self._Y

        cut_off = 0
        theta = []
        psi = list(range(p))

        prune_stats = np.full((p, p), 1e5)
        np.fill_diagonal(prune_stats, 0)

        while len(psi) > 1:
            new_stats = []
            for v in psi:
                cond_set = set.intersection(
                    set(theta), set(np.argwhere(prune_stats[v] > cut_off).flatten())
                )
                cond_set = set.union(cond_set, set(theta[-1:] if len(theta) > 0 else set()))

                if self._possible_ancestors is not None:
                    cond_set = cond_set & self._get_candidate_parents(v, theta)

                last_root = theta[-1] if len(theta) > 0 else None
                if last_root is not None and last_root not in cond_set:
                    # Do not condition on the previous root if it is not a possible ancestor.
                    last_root = None
                    cond_set = set()

                stats = self._get_prune_stats(v, psi, self._K, last_root, cond_set, self._J)

                new_stats.append(stats)
            new_stats = np.array(new_stats)

            prune_stats[psi, :] = np.min([prune_stats[psi, :], new_stats], axis=0)
            np.fill_diagonal(prune_stats, 0)

            max_taus = np.max(prune_stats[np.ix_(psi, psi)], axis=1)

            r = psi[np.argmin(max_taus)]

            cut_off = max(cut_off, min(max_taus) * self._alpha)

            theta.append(r)
            psi.remove(r)

        self._causal_order = [*theta, *psi]

        if not self._estimate_adj_mat:
            return self

        if self._Y.shape[0] <= self._Y.shape[1]:
            # The base class's n<=p (LassoLarsCV) estimator has no prior_knowledge
            # parameter, so a constraint-aware override is used instead.
            return self._estimate_adjacency_matrix2_with_constraints(self._Y)

        return self._estimate_adjacency_matrix(self._Y, prior_knowledge=self._possible_ancestors)

    def _estimate_adjacency_matrix2_with_constraints(self, X):
        """Estimate adjacency matrix by causal order with possible-ancestor constraints."""
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]

            if self._possible_ancestors is not None:
                # Mirrors _estimate_adjacency_matrix's prior_knowledge filtering, since
                # this n<=p path does not go through that base-class method.
                predictors = [p for p in predictors if self._possible_ancestors[target, p]]

            if len(predictors) == 0:
                continue

            B[target, predictors] = self._predict_adaptive_lasso(X, predictors, target)

        self._adjacency_matrix = B
        return self

    @property
    def possible_ancestors_(self):
        """Return the possible-ancestor constraints supplied to the model."""
        return self._possible_ancestors


