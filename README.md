# Skeleton Order DAG Builder
SkeletonOrderDAGBuilder builds a DAG by combining an undirected skeleton (PC algorithm) with a causal order (LiNGAM-type algorithm).

## Requirements

- Python 3.10 or later
- R (`Rscript` must be on the `PATH`) with the R packages `pcalg`, `jsonlite`, and `igraph`.
  They are used by `skelordag.PC`, which runs `pcalg::pc()` via `Rscript`.

```r
install.packages(c("pcalg", "jsonlite", "igraph"))
```

## Installation

```bash
pip install git+https://github.com/cdt15/skeleton-order-dag-builder.git
```

The Python package is imported as `skelordag`.

## Usage

```python
import pandas as pd
from skelordag import PC, PriorConstrainedHighDimLiNGAM, SkeletonOrderDAGBuilder

# X: shape (n_samples, n_features), numeric columns only
X = pd.read_csv("sample.csv")

# 1. Estimate the skeleton and possible ancestors with the PC algorithm
pc = PC(alpha=0.05).fit(X)

# 2. Estimate the causal order with LiNGAM restricted by the possible ancestors
#    (possible_ancestors_ must be transposed to child-by-parent orientation)
lingam = PriorConstrainedHighDimLiNGAM(
    possible_ancestors=possible_ancestors.T,
    random_state=42,
).fit(X)

# 3. Orient the skeleton by the causal order and estimate causal effects
builder = SkeletonOrderDAGBuilder(pc.skeleton_, lingam.causal_order_).build(X)

print(builder.dag_)
print(builder.adjacency_matrix_)
```

## Classes

- `PC`: runs the PC algorithm (`pcalg::pc()`) via `Rscript`.
- `PriorConstrainedHighDimLiNGAM`: high-dimensional LiNGAM with candidate parents restricted by prior knowledge.
- `SkeletonOrderDAGBuilder`: orients the skeleton according to the causal order.
- `RScriptRunner`: helper for running R scripts.
