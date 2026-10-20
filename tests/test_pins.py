"""THE PINS ARE WRITTEN ONCE (pins.toml) and everything else reads them: this fails loudly on any drift.

- every pin the image is built from is an ARG of image/Dockerfile with NO default (a build without it fails), and the
  workflow passes exactly `python3 -m acoustic_solve.pins docker-build-args` (+ IMAGE_COMMIT) as the build arguments;
- no pinned value is written by hand in any code line of the repository (Dockerfile, shell, Python, workflow, bin/):
  comments may cite a commit (a source line read at it), code may not hold one;
- the lock, the patch that pins Boundary Lab's BEAT Engine and pins.toml name one engine, one wheel;
- the image's venv is venv/requirements.lock, --require-hashes --no-deps, then Boundary Lab --no-deps + pip check;
- the image declares `engine` (asked of its venv) and `solver_version` (the build argument) as its facts.
Standard library only:

    python3 tests/test_pins.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from acoustic_solve import pins  # noqa: E402

DOCKERFILE = REPOSITORY_ROOT / "image" / "Dockerfile"
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "image.yml"
LOCK = REPOSITORY_ROOT / "venv" / "requirements.lock"
BEAT_PIN_PATCH = REPOSITORY_ROOT / "patches" / "boundary-lab" / "0001-pin-beat-engine-ti2.patch"
CODE_SUFFIXES = {".py", ".sh", ".yml", ".toml", ""}       # "" = Dockerfile, bin/blabrun
COMMENT = {".py": "#", ".sh": "#", ".yml": "#", ".toml": "#", "": "#"}


def tracked_code_files():
    listed = subprocess.run(["git", "-C", str(REPOSITORY_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
                            capture_output=True, text=True, check=True).stdout.split()
    files = []
    for name in listed:
        path = REPOSITORY_ROOT / name
        if name in ("pins.toml", "venv/requirements.lock") or name.startswith("patches/") or not path.is_file():
            continue
        if path.suffix in CODE_SUFFIXES and path != Path(__file__).resolve():
            files.append(path)
    return files


def code_lines(path):
    """The lines of a file that are not comments (a Python docstring is not code either: its module's docstring)."""
    lines = path.read_text().splitlines()
    if path.suffix == ".py":
        source = path.read_text()
        docstring = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', source, re.S)
        skip = docstring.group(0).count("\n") + 1 if docstring else 0
        lines = lines[skip:]
    return [line for line in lines if not line.lstrip().startswith(COMMENT[path.suffix])]


def test_every_image_pin_is_a_dockerfile_arg_without_a_default():
    declared = dict(re.findall(r"^ARG\s+(\w+)(=.*)?$", DOCKERFILE.read_text(), re.M))
    for argument in [*pins.DOCKER_BUILD_ARGUMENT_BY_PIN.values(), "IMAGE_COMMIT"]:
        assert argument in declared, f"image/Dockerfile declares no ARG {argument}"
        assert declared[argument] == "", f"image/Dockerfile ARG {argument} has a default {declared[argument]!r}"
    assert set(pins.docker_build_arguments()) == {f"{a}={pins.PINS[k]}" for k, a in
                                                  pins.DOCKER_BUILD_ARGUMENT_BY_PIN.items()}


def test_the_workflow_passes_the_pins_and_the_commit_as_build_arguments():
    workflow = WORKFLOW.read_text()
    assert "python3 -m acoustic_solve.pins docker-build-args" in workflow
    assert 'echo "IMAGE_COMMIT=${GITHUB_SHA}"' in workflow
    assert "build-args: ${{ steps.pins.outputs.args }}" in workflow
    assert "file: image/Dockerfile" in workflow and "context: ." in workflow
    printed = subprocess.run([sys.executable, "-m", "acoustic_solve.pins", "docker-build-args"], capture_output=True,
                             text=True, check=True, env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")}).stdout
    assert printed.split("\n")[:-1] == pins.docker_build_arguments(), printed


def test_no_pinned_value_is_written_by_hand_in_any_code_line():
    values = {key: str(value) for key, value in pins.PINS.items() if key != "solver_version"}
    values["pod_image tag"] = pins.POD_IMAGE.rsplit(":", 1)[1]
    copies = [f"{path.relative_to(REPOSITORY_ROOT)}: {key} ({line.strip()[:100]})"
              for path in tracked_code_files() for line in code_lines(path)
              for key, value in values.items() if value in line]
    assert not copies, "pins written by hand (read them from acoustic_solve.pins):\n  " + "\n  ".join(copies)


def test_the_lock_the_beat_pin_patch_and_the_pins_name_one_engine_and_one_wheel():
    lock_line = next(line for line in LOCK.read_text().splitlines() if line.startswith("beat-engine @ "))
    url = lock_line.split(" @ ", 1)[1].split()[0]
    assert f"beat_engine-{pins.ENGINE.replace('+', '%2B')}-" in url, (url, pins.ENGINE)
    lock_hash = re.search(r"beat-engine @ \S+ \\\n\s+--hash=sha256:([0-9a-f]{64})", LOCK.read_text()).group(1)
    patch = BEAT_PIN_PATCH.read_text()
    assert f"+  \"beat-engine @ {url}#sha256={lock_hash}\"," in patch, "the patch pins another wheel than the lock"
    assert f'"{pins.ENGINE}"' in patch, "the patch's version check names another engine"


def test_the_image_declares_its_engine_and_solver_version_as_facts():
    facts = DOCKERFILE.read_text().split("# ---- the image's facts", 1)[1]
    assert '"engine": beat_engine.__version__' in facts and 'int(os.environ["SOLVER_VERSION"])' in facts
    assert 'test "$(/opt/blab-venv/bin/python -c \'import beat_engine; print(beat_engine.__version__)\')" = ' \
           '"${BEAT_ENGINE_VERSION}"' in DOCKERFILE.read_text()


def test_the_image_venv_is_the_lock_installed_without_a_resolver():
    dockerfile = DOCKERFILE.read_text()
    assert "COPY venv/requirements.lock /opt/blab-requirements.lock" in dockerfile
    assert "pip install --no-cache-dir --no-deps --require-hashes -r /opt/blab-requirements.lock" in dockerfile
    assert "pip install --no-cache-dir --no-deps /opt/boundary-lab" in dockerfile
    assert "/opt/blab-venv/bin/pip check" in dockerfile
    pip_installs = re.findall(r"pip install [^\n]*", dockerfile)
    assert all("--no-deps" in line for line in pip_installs), f"a pip install that resolves: {pip_installs}"
    ignored = (REPOSITORY_ROOT / ".dockerignore").read_text().splitlines()
    assert "!venv/requirements.lock" in ignored, ".dockerignore keeps the lock out of the build context"


def test_a_pins_file_with_a_key_missing_or_a_floating_image_is_refused():
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix="pins-"))
    text = (REPOSITORY_ROOT / "pins.toml").read_text()
    for broken, words in ((text.replace("solver_version = ", "solver_versio = "), "exactly"),
                          (re.sub(r'pod_image = ".*"', 'pod_image = "ghcr.io/x/y:latest"', text), ":latest"),
                          (re.sub(r"solver_version = \d+", 'solver_version = "1"', text), "not an integer")):
        (scratch / "pins.toml").write_text(broken)
        try:
            pins.read_pins(scratch / "pins.toml")
        except pins.PinsBroken as error:
            assert words in str(error), error
        else:
            raise AssertionError(f"accepted a broken pins file ({words})")


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
