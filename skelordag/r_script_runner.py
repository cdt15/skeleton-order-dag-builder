import shutil
import subprocess
import tempfile
from pathlib import Path


class RScriptRunner:
    """A common utility to generate and run R scripts from Python via ``Rscript``.

    ``rpy2`` is intentionally not used. Instead, data is exchanged between
    Python and R through CSV/JSON files in a temporary working directory,
    and the R script itself is executed as a subprocess.
    """

    def __init__(
        self,
        rscript_path="Rscript",
    ):
        """Construct an RScriptRunner.

        Parameters
        ----------
        rscript_path : str, optional (default="Rscript")
            Path to the ``Rscript`` executable.
        """
        self._rscript_path = rscript_path

        self.last_work_dir_ = None
        self.last_stdout_ = None
        self.last_stderr_ = None

    def run(
        self,
        script_content,
        args,
        work_dir=None,
    ):
        """Save an R script to a file and run it with ``Rscript``.

        Parameters
        ----------
        script_content : str
            Content of the R script to execute.
        args : list of str
            Command-line arguments passed to the R script.
        work_dir : Path, optional (default=None)
            Working directory to use. If ``None``, a new temporary
            directory is created.

        Returns
        -------
        completed : subprocess.CompletedProcess
            Result of the executed subprocess.
        """
        self._check_rscript_available()

        created_work_dir = work_dir is None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="highdim_causal_"))
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        self.last_work_dir_ = work_dir

        try:
            script_path = self._write_script(script_content, work_dir)
            completed = self._run_subprocess(script_path, args)
            self.last_stdout_ = completed.stdout
            self.last_stderr_ = completed.stderr
            return completed
        finally:
            # Only clean up directories created by this call, not caller-supplied ones.
            if created_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _check_rscript_available(self):
        """Check whether the ``Rscript`` executable is available.

        Raises
        ------
        RuntimeError
            If ``Rscript --version`` cannot be executed.
        """
        try:
            subprocess.run(
                [self._rscript_path, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Rscript executable not found or not runnable: {self._rscript_path!r}"
            ) from exc

    def _write_script(self, script_content, work_dir):
        """Write the R script content to ``script.R`` in the working directory.

        Parameters
        ----------
        script_content : str
            Content of the R script to write.
        work_dir : Path
            Directory in which to save the script.

        Returns
        -------
        script_path : Path
            Path to the saved script file.
        """
        script_path = work_dir / "script.R"
        script_path.write_text(script_content, encoding="utf-8")
        return script_path

    def _run_subprocess(
        self,
        script_path,
        args,
    ):
        """Run the R script using ``subprocess.run()``.

        Parameters
        ----------
        script_path : Path
            Path to the R script to execute.
        args : list of str
            Command-line arguments passed to the R script.

        Returns
        -------
        completed : subprocess.CompletedProcess
            Result of the executed subprocess.

        Raises
        ------
        RuntimeError
            If the R script exits with a non-zero return code.
        """
        completed = subprocess.run(
            [self._rscript_path, str(script_path), *args],
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Rscript execution failed (returncode={completed.returncode}).\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        return completed
