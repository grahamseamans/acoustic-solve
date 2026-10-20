"""acoustic_solve.blab_venv -- the venv's acceptance check: the lock is read (a direct-URL wheel's version from its
file name), the live venv passes the fast check, and the deep check's tree comparison names a changed, a missing and
an extra file (byte comparison, not the stat shortcut). In the blab venv:

    blabrun tests/test_blab_venv.py
"""
import tempfile
import time
from pathlib import Path

from acoustic_solve import blab_venv, pins


def test_the_lock_is_read_with_the_engine_wheels_version_from_its_file_name():
    locked = blab_venv.locked_versions()
    assert locked["beat-engine"] == pins.ENGINE and locked["gmsh"] and "boundary-lab" not in locked


def test_the_venv_blabrun_runs_in_passes_the_fast_check():
    assert blab_venv.fast_problems() == []


def test_the_tree_comparison_names_a_changed_a_missing_and_an_extra_file():
    root = Path(tempfile.mkdtemp(prefix="blab-tree-"))
    for side in ("expected", "installed"):
        (root / side / "solvers").mkdir(parents=True)
        (root / side / "solvers" / "same.py").write_text("x = 1\n")
        (root / side / "changed.py").write_text("y = 1\n")
        (root / side / "__pycache__").mkdir()
        (root / side / "__pycache__" / "z.pyc").write_text(side)
    (root / "installed" / "changed.py").write_text("y = 2\n")          # same size: only a byte comparison sees it
    (root / "expected" / "solvers" / "missing.py").write_text("")
    (root / "installed" / "extra.py").write_text("")
    differences = blab_venv.tree_differences(root / "expected", root / "installed")
    assert sorted(differences) == ["blab/changed.py: differs", "blab/extra.py: only installed",
                                   "blab/solvers/missing.py: only in pin + patches"], differences


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
