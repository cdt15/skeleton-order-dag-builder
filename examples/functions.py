"""Functions for causal graph estimation, 2D visualization and evaluation."""

import os
import platform
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from skelordag import PC, PriorConstrainedHighDimLiNGAM, SkeletonOrderDAGBuilder
from tqdm import tqdm


GRAPH_TYPES = ("random", "chain", "hub")
COEF_RANGE = (0.5, 1.5)  # |coefficient| ~ Uniform(0.5, 1.5) with a random sign
NOISE_RANGE = (-1.0, 1.0)  # Uniform (non-Gaussian) errors, as LiNGAM requires
GRID_SCALE = 3  # coordinate grid side = GRID_SCALE * ceil(sqrt(n_vars))
R_PACKAGES = ("pcalg", "jsonlite", "igraph")


def check_environment(r_packages=R_PACKAGES):
    """Print the Python, R and R package versions used by the notebooks.

    Parameters
    ----------
    r_packages : sequence of str
        R packages whose versions are printed.

    Raises
    ------
    RuntimeError
        If ``Rscript`` is not found on ``PATH``.
    subprocess.CalledProcessError
        If an R package is not installed.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript was not found on PATH. Install R and add its bin directory to PATH.")

    r_version = subprocess.run([rscript, "--version"], capture_output=True, text=True)
    r_names = ", ".join(f'"{name}"' for name in r_packages)
    r_result = subprocess.run(
        [rscript, "-e", f"for (p in c({r_names})) cat(p, as.character(packageVersion(p)), \"\\n\")"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(f"Python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform : {platform.platform()}")
    print(f"CPU      : {platform.processor()} ({os.cpu_count()} logical cores)")
    print(f"Rscript  : {rscript}")
    print((r_version.stdout + r_version.stderr).strip().splitlines()[0])
    print("R packages:")
    print(r_result.stdout.strip())


def simulate(n_vars, n_samples, graph_type, edge_prob, seed):
    """Generate data from a linear non-Gaussian acyclic model with a known DAG.

    Edges are drawn along a random causal order. Each variable is standardized right
    after it is generated and its incoming coefficients are rescaled accordingly, so
    ``B_true`` describes the returned data exactly and long chains do not explode.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    n_samples : int
        Number of samples.
    graph_type : {"random", "chain", "hub"}
        "random": each forward pair gets an edge with probability ``edge_prob``;
        "chain": a single path; "hub": the first variable causes all others.
    edge_prob : float
        Edge probability, used only for "random".
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``B_true`` (pandas.DataFrame, ``B[parent, child]``), ``causal_order``
        (list of str), ``X`` (pandas.DataFrame) and ``coords`` (pandas.DataFrame
        with columns ``variable``, ``x``, ``y``; distinct integer grid points).

    Raises
    ------
    ValueError
        If ``graph_type`` is unknown.
    """
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"Unknown graph_type {graph_type!r}; expected one of {GRAPH_TYPES}.")
    rng = np.random.default_rng(seed)
    names = [f"x{i}" for i in range(n_vars)]
    order = rng.permutation(n_vars)

    # Edges and data are built in causal-order positions, then mapped back to variables.
    mask = np.zeros((n_vars, n_vars), dtype=bool)
    if graph_type == "random":
        mask = np.triu(rng.random((n_vars, n_vars)) < edge_prob, k=1)
    elif graph_type == "chain":
        mask[np.arange(n_vars - 1), np.arange(1, n_vars)] = True
    else:
        mask[0, 1:] = True
    signs = rng.choice([-1.0, 1.0], size=mask.shape)
    W = np.where(mask, rng.uniform(*COEF_RANGE, size=mask.shape) * signs, 0.0)

    errors = rng.uniform(*NOISE_RANGE, size=(n_samples, n_vars))
    X_pos = np.zeros((n_samples, n_vars))
    for k in range(n_vars):
        x = X_pos[:, :k] @ W[:k, k] + errors[:, k]
        scale = x.std()
        X_pos[:, k] = x / scale
        W[:k, k] /= scale

    B = np.zeros_like(W)
    B[np.ix_(order, order)] = W
    X = np.empty_like(X_pos)
    X[:, order] = X_pos
    # Distinct integer grid points, so no two nodes overlap.
    side = GRID_SCALE * int(np.ceil(np.sqrt(n_vars)))
    cells = rng.choice(side * side, size=n_vars, replace=False)
    coords = pd.DataFrame({"variable": names, "x": cells % side, "y": cells // side})
    return {
        "B_true": pd.DataFrame(B, index=names, columns=names),
        "causal_order": [names[i] for i in order],
        "X": pd.DataFrame(X, columns=names),
        "coords": coords,
    }


def estimate_dag(X, alpha, seed, lingam_params=None):
    """Estimate a DAG with a PC skeleton and a prior-constrained LiNGAM causal order.

    Parameters
    ----------
    X : pandas.DataFrame
        Observed data, shape (n_samples, n_variables).
    alpha : float
        Significance level of the PC algorithm.
    seed : int
        Random seed passed to LiNGAM.
    lingam_params : dict, optional
        Extra keyword arguments for ``PriorConstrainedHighDimLiNGAM``
        (``J``, ``K``, ``alpha``).

    Returns
    -------
    B : pandas.DataFrame
        Weighted adjacency matrix with ``B[parent, child]`` = coefficient.
    causal_order : list of str
        Estimated causal order, most upstream first.
    """
    pc = PC(alpha=alpha).fit(X)
    lingam = PriorConstrainedHighDimLiNGAM(
        possible_ancestors=pc.possible_ancestors_.T,
        random_state=seed,
        estimate_adj_mat=False,  # coefficients are estimated by the builder instead
        **(lingam_params or {}),
    ).fit(X)
    builder = SkeletonOrderDAGBuilder(pc.skeleton_, lingam.causal_order_).build(X)
    names = list(X.columns)
    # skelordag returns B[child, parent]; transpose once to the internal B[parent, child].
    B = pd.DataFrame(builder.adjacency_matrix_.T, index=names, columns=names)
    return B, [names[i] for i in lingam.causal_order_]


def plot_dag(
    B,
    coords,
    edge_labels=None,
    threshold=0.0,
    top_k=None,
    ax=None,
    title=None,
    max_labeled_nodes=50,
    **kwargs,
):
    """Draw a DAG with nodes placed at the given 2D coordinates.

    Parameters
    ----------
    B : pandas.DataFrame
        Matrix with ``B[parent, child] != 0`` for each edge (coefficients or
        bootstrap probabilities).
    coords : pandas.DataFrame
        Coordinate table with columns ``variable``, ``x`` and ``y``.
    edge_labels : pandas.DataFrame, optional
        Matrix in the same layout as ``B`` whose values are printed on the edges.
    threshold : float, default 0.0
        Keep only edges with ``|B| >= threshold``.
    top_k : int, optional
        Keep only the ``top_k`` edges with the largest ``|B|``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str, optional
        Plot title.
    max_labeled_nodes : int, default 50
        Node labels are hidden when there are more nodes than this.
    **kwargs
        Passed to ``networkx.draw_networkx`` (``node_color``, ``node_size``,
        ``width``, ...). By default node size, arrow size and font size shrink
        as the number of nodes grows beyond 10.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the graph was drawn on.
    G : networkx.DiGraph
        The drawn graph after filtering.
    """
    edges = B.stack()
    edges = edges[(edges != 0) & (edges.abs() >= threshold)]
    if top_k is not None:
        edges = edges.loc[edges.abs().sort_values(ascending=False).index[:top_k]]

    G = nx.DiGraph()
    G.add_nodes_from(coords["variable"])
    G.add_edges_from(edges.index)
    pos = {v: (x, y) for v, x, y in coords[["variable", "x", "y"]].itertuples(index=False)}

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    # Shrink nodes, arrows and labels as the graph grows so crowded graphs stay readable.
    scale = min(1.0, 10 / max(G.number_of_nodes(), 1))
    style = {
        "node_size": max(20, 300 * scale),
        "node_color": "#9ecae1",
        "edge_color": "#555555",
        "arrowsize": max(5, 12 * scale**0.5),
        "font_size": max(5, 8 * scale**0.5),
        "with_labels": G.number_of_nodes() <= max_labeled_nodes,
    }
    style.update(kwargs)
    nx.draw_networkx(G, pos, ax=ax, arrows=True, **style)
    if edge_labels is not None:
        labels = {(u, v): f"{edge_labels.loc[u, v]:.2f}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=style["font_size"])
    # networkx hides the ticks; show them so the coordinates can be read off the plot.
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Causal graph ({G.number_of_edges()} edges)")
    return ax, G


def network_metrics(B, causal_order):
    """Compute per-variable network metrics of a DAG.

    Parameters
    ----------
    B : pandas.DataFrame
        Weighted adjacency matrix with ``B[parent, child] != 0`` for each edge.
    causal_order : list of str
        Causal order, most upstream first.

    Returns
    -------
    pandas.DataFrame
        One row per variable with ``causal_order`` (position in the order,
        0 = most upstream), in / out degrees, weighted in / out degrees (sums of
        absolute coefficients) and ancestor / descendant counts.

    Notes
    -----
    Source / sink variables are those with ``in_degree == 0`` / ``out_degree == 0``.
    """
    G = nx.from_pandas_adjacency(B.abs(), create_using=nx.DiGraph)
    position = {v: i for i, v in enumerate(causal_order)}
    rows = [
        {
            "variable": v,
            "causal_order": position[v],
            "in_degree": G.in_degree(v),
            "out_degree": G.out_degree(v),
            "weighted_in_degree": G.in_degree(v, weight="weight"),
            "weighted_out_degree": G.out_degree(v, weight="weight"),
            "ancestor_count": len(nx.ancestors(G, v)),
            "descendant_count": len(nx.descendants(G, v)),
        }
        for v in B.index
    ]
    return pd.DataFrame(rows)


def bootstrap(X, n_boot, alpha, seed, ci_level=0.95, lingam_params=None):
    """Re-estimate the DAG on row-resampled data and summarize edge stability.

    Iteration ``i`` uses seed ``seed + i``. Failed iterations are recorded and
    skipped; probabilities use the number of successful iterations as the denominator.

    Parameters
    ----------
    X : pandas.DataFrame
        Observed data, shape (n_samples, n_variables).
    n_boot : int
        Number of bootstrap iterations.
    alpha : float
        Significance level of the PC algorithm.
    seed : int
        Base random seed.
    ci_level : float, default 0.95
        Confidence level of the percentile intervals.
    lingam_params : dict, optional
        Extra keyword arguments for ``PriorConstrainedHighDimLiNGAM``.

    Returns
    -------
    dict
        ``prob`` (pandas.DataFrame, ``prob[i, j]`` = share of iterations with
        ``i -> j``), ``metrics_ci`` (pandas.DataFrame, per-variable mean and
        percentile CI of each network metric), ``n_success`` (int), ``n_failed``
        (int) and ``failures`` (pandas.DataFrame of iteration and error message).

    Raises
    ------
    RuntimeError
        If every iteration fails.
    """
    names = list(X.columns)
    counts = np.zeros((len(names), len(names)))
    metric_frames, failures = [], []
    for i in tqdm(range(n_boot), desc="Bootstrap"):
        rng = np.random.default_rng(seed + i)
        X_boot = X.iloc[rng.integers(0, len(X), len(X))].reset_index(drop=True)
        try:
            B, order = estimate_dag(X_boot, alpha, seed + i, lingam_params)
        except Exception as exc:
            failures.append({"iteration": i, "error": repr(exc)})
            continue
        counts += B.to_numpy() != 0
        metric_frames.append(network_metrics(B, order).assign(iteration=i))

    n_success = n_boot - len(failures)
    if n_success == 0:
        raise RuntimeError("All bootstrap iterations failed.")
    q = (1 - ci_level) / 2
    long = pd.concat(metric_frames).melt(id_vars=["variable", "iteration"], var_name="metric")
    metrics_ci = (
        long.groupby(["variable", "metric"], sort=False)["value"]
        .agg(mean="mean", lower=lambda s: s.quantile(q), upper=lambda s: s.quantile(1 - q), n="count")
        .reset_index()
    )
    return {
        "prob": pd.DataFrame(counts / n_success, index=names, columns=names),
        "metrics_ci": metrics_ci,
        "n_success": n_success,
        "n_failed": len(failures),
        "failures": pd.DataFrame(failures, columns=["iteration", "error"]),
    }


def select_bootstrap_edges(prob, threshold=0.3):
    """Build a graph from bootstrap probabilities, one direction per variable pair.

    For each pair, the direction with the larger bootstrap probability is kept, and
    it is dropped if that probability is below ``threshold``. Pairs whose two
    directions have equal probability get no edge.

    Parameters
    ----------
    prob : pandas.DataFrame
        Bootstrap edge probabilities, ``prob[i, j]`` = share of iterations with ``i -> j``.
    threshold : float, default 0.3
        Minimum probability for an edge to be kept.

    Returns
    -------
    pandas.DataFrame
        Matrix in the same layout as ``prob``, holding the probability of each kept
        edge and 0 elsewhere.

    Notes
    -----
    Directions are chosen pair by pair, so the result is not guaranteed to be acyclic.
    """
    P = prob.to_numpy()
    keep = (P > P.T) & (P >= threshold)
    return pd.DataFrame(np.where(keep, P, 0.0), index=prob.index, columns=prob.columns)


def precision_recall(B_true, B_est):
    """Compute the precision and recall of the estimated directed edges.

    An estimated edge counts as correct only if it exists in the true DAG with the
    same direction.

    Parameters
    ----------
    B_true : pandas.DataFrame
        True adjacency matrix, ``B[parent, child] != 0`` for each edge.
    B_est : pandas.DataFrame
        Estimated adjacency matrix in the same layout.

    Returns
    -------
    precision : float
        Correct edges / estimated edges; NaN if no edge was estimated.
    recall : float
        Correct edges / true edges; NaN if the true DAG has no edge.
    """
    T = B_true.to_numpy() != 0
    E = B_est.loc[B_true.index, B_true.columns].to_numpy() != 0
    tp = (T & E).sum()
    precision = tp / E.sum() if E.sum() else np.nan
    recall = tp / T.sum() if T.sum() else np.nan
    return float(precision), float(recall)
