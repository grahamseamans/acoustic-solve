"""THE BLAB VENV'S ACCEPTANCE CHECK, run by bin/blabrun before anything runs in the venv.

    python -m acoustic_solve.blab_venv            # fast (every blabrun): the venv IS venv/requirements.lock
    python -m acoustic_solve.blab_venv --deep     # + the installed blab IS boundary_lab_commit + patches/boundary-lab

FAST: Python 3.12; every distribution installed is either locked at exactly its locked version or one of the three
installed from source (boundary-lab, podd, pip), and every locked one is installed; beat-engine is pins.ENGINE; gmsh
is the venv's own pip wheel (not a system python3-gmsh seen through site-packages). DEEP: Boundary Lab at the pinned
commit, read from a clone's git objects (never its working tree), patched with patches/boundary-lab/*.patch, has
exactly the files of the installed `blab` package, byte for byte. Anything else RAISES, naming every difference.
"""
import filecmp
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from acoustic_solve import pins

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = REPOSITORY_ROOT / "venv" / "requirements.lock"
PATCHES_DIRECTORY = REPOSITORY_ROOT / "patches" / "boundary-lab"
INSTALLED_FROM_SOURCE = {"boundary-lab", "podd", "pip"}
PYTHON = (3, 12)
BOUNDARY_LAB_CLONE = Path(os.environ.get("BOUNDARY_LAB_CLONE", Path.home() / "boundary-lab"))
LOCK_LINE = re.compile(r"^([a-z0-9][a-z0-9._-]*)(?:==(\S+)| @ (\S+))")


class VenvDrifted(SystemExit):
    """The venv is not the lock (or the installed blab is not the pin + patches): every difference in the message."""


def normalised(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions(lock_file=LOCK_FILE):
    """{name: version} of the lock; a direct-URL line's version is read from its wheel's file name."""
    versions = {}
    for line in Path(lock_file).read_text().splitlines():
        match = LOCK_LINE.match(line)
        if not match:
            continue
        name, version, url = match.groups()
        if url is not None:
            wheel = url.rsplit("/", 1)[-1].replace("%2B", "+")
            version = wheel.split("-")[1]
        versions[name] = version
    if not versions:
        raise VenvDrifted(f"(╯°□°)╯ {lock_file} locks nothing")
    return versions


def fast_problems():
    problems = []
    if sys.version_info[:2] != PYTHON:
        problems.append(f"python {sys.version.split()[0]}, the lock is for {'.'.join(map(str, PYTHON))}")
    locked = locked_versions()
    installed = {normalised(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
    for name, version in sorted(locked.items()):
        if installed.get(name) != version:
            problems.append(f"{name}: installed {installed.get(name)}, locked {version}")
    for name in sorted(set(installed) - set(locked) - INSTALLED_FROM_SOURCE):
        problems.append(f"{name} {installed[name]} is installed and not locked")
    for name in sorted(INSTALLED_FROM_SOURCE - set(installed)):
        problems.append(f"{name} is not installed (venv/build-venv.sh installs it from source)")
    if locked.get("beat-engine") != pins.ENGINE:
        problems.append(f"the lock's beat-engine {locked.get('beat-engine')} is not pins.toml's engine {pins.ENGINE}")
    gmsh = importlib.util.find_spec("gmsh")
    if gmsh is None or not Path(gmsh.origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
        problems.append(f"gmsh is not the venv's own wheel ({gmsh.origin if gmsh else 'not importable'})")
    return problems


def installed_blab_directory():
    spec = importlib.util.find_spec("blab")
    if spec is None:
        raise VenvDrifted("(╯°□°)╯ blab is not importable in this venv")
    return Path(spec.origin).parent


def tree_differences(expected, installed):
    """Every file that differs between two trees (missing, extra, or other bytes), __pycache__ ignored."""
    differences = []

    def walk(comparison, relative):
        differences.extend(f"{relative / name}: only in pin + patches" for name in comparison.left_only)
        differences.extend(f"{relative / name}: only installed" for name in comparison.right_only)
        differences.extend(f"{relative / name}: differs" for name in comparison.diff_files)
        differences.extend(f"{relative / name}: unreadable" for name in comparison.funny_files)
        for name, child in comparison.subdirs.items():
            walk(child, relative / name)
    comparison = filecmp.dircmp(expected, installed, ignore=["__pycache__"])
    filecmp.clear_cache()
    walk(comparison, Path("blab"))
    # dircmp's diff_files is a shallow (stat) comparison; recheck every common file by content
    for path in Path(expected).rglob("*"):
        twin = Path(installed) / path.relative_to(expected)
        if path.is_file() and "__pycache__" not in path.parts and twin.is_file() and \
                path.read_bytes() != twin.read_bytes():
            entry = f"blab/{path.relative_to(expected)}: differs"
            if entry not in differences:
                differences.append(entry)
    return differences


def deep_problems():
    if not (BOUNDARY_LAB_CLONE / ".git").exists():
        return [f"no Boundary Lab clone at {BOUNDARY_LAB_CLONE} (BOUNDARY_LAB_CLONE) to read the pinned commit from"]
    with tempfile.TemporaryDirectory(prefix="blab-pin-") as scratch:
        archive = subprocess.run(["git", "-C", str(BOUNDARY_LAB_CLONE), "archive", pins.BOUNDARY_LAB_COMMIT],
                                 capture_output=True, check=False)
        if archive.returncode != 0:
            return [f"{BOUNDARY_LAB_CLONE} has no commit {pins.BOUNDARY_LAB_COMMIT}: "
                    f"{archive.stderr.decode().strip()}"]
        subprocess.run(["tar", "-x", "-C", scratch], input=archive.stdout, check=True)
        patches = sorted(str(p) for p in PATCHES_DIRECTORY.glob("*.patch"))
        applied = subprocess.run(["git", "apply", *patches], cwd=scratch, capture_output=True, text=True)
        if applied.returncode != 0:
            return [f"patches/boundary-lab/ does not apply to {pins.BOUNDARY_LAB_COMMIT}: {applied.stderr.strip()}"]
        return tree_differences(Path(scratch) / "src" / "blab", installed_blab_directory())


def main(deep):
    problems = fast_problems() + (deep_problems() if deep else [])
    if problems:
        raise VenvDrifted(f"(╯°□°)╯ blabrun: {sys.prefix} is not venv/requirements.lock"
                          + (" + the pinned Boundary Lab" if deep else "") + ":\n  - " + "\n  - ".join(problems)
                          + "\n  rebuild it: venv/build-venv.sh <new venv dir>")
    if deep:
        print(f"blabrun: {sys.prefix} is the lock and blab is {pins.BOUNDARY_LAB_COMMIT[:8]} + "
              f"{len(list(PATCHES_DIRECTORY.glob('*.patch')))} patches (＾_＾)")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--deep"]):
        sys.exit(__doc__)
    main(deep=sys.argv[1:] == ["--deep"])
