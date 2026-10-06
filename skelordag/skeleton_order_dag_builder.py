import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class SkeletonOrderDAGBuilder:
    """Orient a PC skeleton using a LiNGAM causal order to build the final DAG.

    Given an undirected skeleton and a causal order, this class orients each
    skeleton edge according to the causal order, and estimates linear causal
    effects and network metrics from the resulting DAG.
    """

    def __init__(
        self,
        skeleton,
        causal_order,
    ):
        """Construct a SkeletonOrderDAGBuilder.

        Parameters
        ----------
        skeleton : array-like, shape (n_features, n_features)
            Undirected skeleton, where ``skeleton[i, j] == skeleton[j, i] == 1``
            means there is an edge between ``Xi`` and ``Xj``.
        causal_order : list of int
            Causal order estimated by a LiNGAM-type algorithm, ordered from
            the most upstream to the most downstream variable.
        """
        self._validate_inputs(skeleton, causal_order)
        self._skeleton = np.asarray(skeleton, dtype=int)
        self._causal_order = list(causal_order)
        self._dag = None
        self._adjacency_matrix = None

    def _validate_inputs(
        self,
        skeleton,
        causal_order,
    ):
        """Validate the skeleton and causal order.

        Parameters
        ----------
        skeleton : array-like, shape (n_features, n_features)
            Undirected skeleton matrix.
        causal_order : list of int
            Causal order over the variable indices.
        Raises
        ------
        ValueError
            If the skeleton is not square/symmetric or ``causal_order`` is
            inconsistent with the number of variables.
        """
        skeleton_arr = np.asarray(skeleton)

        if skeleton_arr.ndim != 2 or skeleton_arr.shape[0] != skeleton_arr.shape[1]:
            raise ValueError(f"skeleton must be a square matrix, got shape {skeleton_arr.shape}.")

        if not np.array_equal(skeleton_arr != 0, skeleton_arr.T != 0):
            raise ValueError("skeleton must be symmetric.")

        p = skeleton_arr.shape[0]

        if len(causal_order) != p:
            raise ValueError(
                f"causal_order length ({len(causal_order)}) does not match "
                f"number of variables ({p})."
            )

        if len(set(causal_order)) != len(causal_order):
            raise ValueError("causal_order must not contain duplicate indices.")

        if any(idx < 0 or idx >= p for idx in causal_order):
            raise ValueError("causal_order contains an index out of range.")

    def build(self, X=None):
        """Orient each undirected skeleton edge according to the causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Training data. If given, causal effects are also estimated and
            stored in ``adjacency_matrix_`` right after the DAG is built.

        Returns
        -------
        self : object
            Fitted builder, with ``dag_`` populated (and ``adjacency_matrix_``
            populated if ``X`` was given).
        """
        p = self._skeleton.shape[0]
        dag = np.zeros((p, p), dtype=int)
        order_position = {node: pos for pos, node in enumerate(self._causal_order)}

        for i in range(p):
            for j in range(i + 1, p):
                has_edge = self._skeleton[i, j] != 0 or self._skeleton[j, i] != 0

                if not has_edge:
                    continue

                if order_position[i] < order_position[j]:
                    dag[j, i] = 1  # i -> j
                else:
                    dag[i, j] = 1  # j -> i

        self._dag = dag

        if X is not None:
            self._adjacency_matrix = self._estimate_adjacency_matrix(X)

        return self

    def _estimate_adjacency_matrix(
        self,
        X,
    ):
        """Estimate linear causal effects from the generated DAG.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        causal_effects : array-like, shape (n_features, n_features)
            Estimated causal effects, where ``causal_effects[v, parent]``
            is the regression coefficient of ``parent`` on ``v``.
        """
        if self._dag is None:
            raise ValueError("build() must be called before _estimate_adjacency_matrix().")

        dag = self._dag
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)

        if X_arr.shape[1] != dag.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[1]} columns but DAG has {dag.shape[0]} variables."
            )

        p = dag.shape[0]
        causal_effects = np.zeros((p, p), dtype=float)

        for v in range(p):
            parents_v = np.where(dag[v, :] != 0)[0]

            if len(parents_v) == 0:
                continue

            lr = LinearRegression()
            lr.fit(X_arr[:, parents_v], X_arr[:, v])
            causal_effects[v, parents_v] = lr.coef_

        return causal_effects

    @property
    def skeleton_(self):
        """Return the skeleton used by the builder."""
        return self._skeleton

    @property
    def causal_order_(self):
        """Return the causal order used by the builder."""
        return self._causal_order

    @property
    def dag_(self):
        """Return the generated DAG, if built."""
        return self._dag

    @property
    def adjacency_matrix_(self):
        """Return the weighted adjacency matrix from build(X), or None if not estimated."""
        return self._adjacency_matrix

