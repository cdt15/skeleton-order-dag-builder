import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.utils import check_array

from .r_script_runner import RScriptRunner


class PC:
    """A wrapper for ``pcalg::pc()`` that runs the PC algorithm via Rscript.

    This class saves the input data as CSV, generates an R script that calls
    ``pcalg::pc()`` with Fisher's Z test (``gaussCItest``), and reads back the
    estimated CPDAG, skeleton, and possible ancestors.
    """

    def __init__(
        self,
        alpha=0.01,
        background_knowledge=None,
        rscript_path="Rscript",
    ):
        """Construct a PC model.

        Parameters
        ----------
        alpha : float, optional (default=0.01)
            Significance level for the conditional independence test used
            by the PC algorithm.
        background_knowledge : dict, optional (default=None)
            Background knowledge dictionary. Only ``"forbidden_edges"`` is
            used in the current implementation, e.g.
            ``{"forbidden_edges": [(0, 1), (2, 3)]}``.
        rscript_path : str, optional (default="Rscript")
            Path to the ``Rscript`` executable.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

        self._alpha = alpha
        self._background_knowledge = background_knowledge
        self._rscript_path = rscript_path

        self._runner = RScriptRunner(
            rscript_path=rscript_path,
        )

        self._cpdag = None
        self._skeleton = None
        self._possible_ancestors = None

        self._work_dir = None
        self._metadata = None

    def fit(self, X):
        """Fit the model to X and estimate the CPDAG, skeleton, and possible ancestors.

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
        if self._background_knowledge is not None:
            forbidden_edges = self._background_knowledge.get("forbidden_edges", [])
            for i, j in forbidden_edges:
                if not (0 <= i < p and 0 <= j < p):
                    raise ValueError(
                        f"forbidden_edges contains an out-of-range index: ({i}, {j})."
                    )

        work_dir = Path(tempfile.mkdtemp(prefix="highdim_causal_pc_"))
        self._work_dir = work_dir

        try:
            output_dir = work_dir / "results"
            output_dir.mkdir(parents=True, exist_ok=True)

            input_path = self._save_input_csv(X_arr)
            bk_path = self._save_background_knowledge()

            script_content = self._build_r_script()

            args = [
                "--input", str(input_path),
                "--output_dir", str(output_dir),
                "--alpha", str(self._alpha),
            ]
            if bk_path is not None:
                args += ["--background_knowledge", str(bk_path)]

            self._runner.run(script_content, args, work_dir=work_dir)

            self._load_results(output_dir)
        finally:
            # PC creates work_dir itself, so RScriptRunner's own cleanup never runs for it.
            shutil.rmtree(work_dir, ignore_errors=True)

        return self

    def _save_input_csv(self, X):
        """Save the input data as ``input.csv``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        input_path : Path
            Path to the saved CSV file.
        """
        input_path = self._work_dir / "input.csv"
        pd.DataFrame(X).to_csv(input_path, index=False)
        return input_path

    def _save_background_knowledge(self):
        """Save background knowledge as ``background_knowledge.json``.

        Returns
        -------
        bk_path : Path or None
            Path to the saved JSON file, or ``None`` if there is no
            background knowledge.
        """
        if self._background_knowledge is None:
            return None

        bk_path = self._work_dir / "background_knowledge.json"
        with open(bk_path, "w", encoding="utf-8") as f:
            json.dump(self._background_knowledge, f)
        return bk_path

    def _build_r_script(self):
        """Return the R script content that runs the PC algorithm.

        Returns
        -------
        script_content : str
            R script that reads ``input.csv``, runs ``pcalg::pc()``, and
            writes the CPDAG, skeleton, possible ancestors, and metadata.
        """
        return r"""
suppressPackageStartupMessages({
  library(pcalg)
  library(jsonlite)
})

parse_args <- function(args) {
  opts <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    value <- args[i + 1]
    opts[[key]] <- value
    i <- i + 2
  }
  opts
}

args <- commandArgs(trailingOnly = TRUE)
opts <- parse_args(args)

input_path <- opts[["input"]]
output_dir <- opts[["output_dir"]]
alpha <- as.numeric(opts[["alpha"]])
bk_path <- opts[["background_knowledge"]]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

X <- read.csv(input_path, check.names = FALSE)

if (any(is.na(X))) {
  stop("Input data contains missing values.")
}

p <- ncol(X)
n <- nrow(X)

suffStat <- list(
  C = cor(X),
  n = n
)

fixedGaps <- matrix(FALSE, nrow = p, ncol = p)

if (!is.null(bk_path) && file.exists(bk_path)) {
  bk <- fromJSON(bk_path)
  if (!is.null(bk$forbidden_edges) && length(bk$forbidden_edges) > 0) {
    forbidden <- bk$forbidden_edges
    for (k in seq_len(nrow(forbidden))) {
      i <- forbidden[k, 1] + 1
      j <- forbidden[k, 2] + 1
      fixedGaps[i, j] <- TRUE
      fixedGaps[j, i] <- TRUE
    }
  }
}

pc_fit <- pc(
  suffStat = suffStat,
  indepTest = gaussCItest,
  alpha = alpha,
  labels = colnames(X),
  fixedGaps = fixedGaps
)

cpdag <- as(pc_fit@graph, "matrix")

skeleton <- ifelse(
  cpdag != 0 | t(cpdag) != 0,
  1,
  0
)

possible_ancestors <- matrix(
  FALSE,
  nrow = p,
  ncol = p
)

for (v in seq_len(p)) {
  pa <- possAn(cpdag, v, type = "cpdag")

  if (length(pa) > 0) {
    possible_ancestors[v, pa] <- TRUE
  }
}

diag(possible_ancestors) <- FALSE

write.csv(
  cpdag,
  file.path(output_dir, "cpdag.csv"),
  row.names = FALSE
)

write.csv(
  skeleton,
  file.path(output_dir, "skeleton.csv"),
  row.names = FALSE
)

write.csv(
  possible_ancestors,
  file.path(output_dir, "possible_ancestors.csv"),
  row.names = FALSE
)

metadata <- list(
  n_samples = n,
  n_features = p,
  alpha = alpha
)

write_json(
  metadata,
  file.path(output_dir, "metadata.json"),
  auto_unbox = TRUE
)
"""

    def _load_results(self, output_dir):
        """Load the output files produced by the R script.

        Parameters
        ----------
        output_dir : Path
            Directory containing ``cpdag.csv``, ``skeleton.csv``,
            ``possible_ancestors.csv``, and ``metadata.json``.

        Raises
        ------
        RuntimeError
            If an expected output file was not generated.
        """
        cpdag_path = output_dir / "cpdag.csv"
        skeleton_path = output_dir / "skeleton.csv"
        possible_ancestors_path = output_dir / "possible_ancestors.csv"
        metadata_path = output_dir / "metadata.json"

        for path in (cpdag_path, skeleton_path, possible_ancestors_path, metadata_path):
            if not path.exists():
                raise RuntimeError(f"Expected output file was not generated: {path}")

        self._cpdag = pd.read_csv(cpdag_path).to_numpy()
        self._skeleton = pd.read_csv(skeleton_path).to_numpy().astype(int)
        self._possible_ancestors = pd.read_csv(possible_ancestors_path).to_numpy().astype(bool)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    @property
    def cpdag_(self):
        """Return the estimated CPDAG.

        Returns
        -------
        cpdag : array-like, shape (n_features, n_features)
            Estimated CPDAG adjacency matrix.
        """
        return self._cpdag

    @property
    def skeleton_(self):
        """Return the estimated skeleton.

        Returns
        -------
        skeleton : array-like, shape (n_features, n_features)
            Estimated undirected skeleton matrix.
        """
        return self._skeleton

    @property
    def possible_ancestors_(self):
        """Return the possible ancestors matrix.

        Returns
        -------
        possible_ancestors : array-like, shape (n_features, n_features)
            Boolean matrix where ``possible_ancestors[u, v] == True`` means
            ``u`` is a possible ancestor of ``v``. Rows represent possible
            ancestors and columns represent their possible descendants, using
            the same direction as the CPDAG adjacency matrix.
        """
        return self._possible_ancestors

