"""THE PINS (pins.toml at the repository root), read once: what a solve runs on and what the image is built from.

    from acoustic_solve import pins
    pins.POD_IMAGE, pins.ENGINE, pins.SOLVER_VERSION, pins.BOUNDARY_LAB_COMMIT, pins.JULIA_VERSION, pins.CUDA_BASE_IMAGE

    python3 -m acoustic_solve.pins docker-build-args     # KEY=VALUE per line: the image's build arguments

Nothing else in this repository or a client writes one of these values; tests/test_pins.py checks it. Standard
library only (tomllib, Python >= 3.11), so the image workflow's runner reads it with its own python3.
"""
import sys
import tomllib
from pathlib import Path

PINS_FILE = Path(__file__).resolve().parents[2] / "pins.toml"

# pins.toml key -> the image/Dockerfile ARG it is passed as (every one; the Dockerfile has no default for any)
DOCKER_BUILD_ARGUMENT_BY_PIN = {
    "solver_version": "SOLVER_VERSION",
    "engine": "BEAT_ENGINE_VERSION",
    "boundary_lab_commit": "BLAB_COMMIT",
    "julia_version": "JULIA_VERSION",
    "cuda_base_image": "BASE_IMAGE",
}
CLIENT_ONLY_PINS = ("pod_image",)          # which built image to run: an output of the build, never an input to it


class PinsBroken(RuntimeError):
    """pins.toml missing, unreadable, or not exactly the keys this module knows."""


def read_pins(path=PINS_FILE):
    try:
        with open(path, "rb") as stream:
            values = tomllib.load(stream)
    except FileNotFoundError:
        raise PinsBroken(f"(╯°□°)╯ no {path}: acoustic_solve is imported from somewhere other than its repository "
                         f"(it reads its pins from the repo root; use the checkout, PYTHONPATH=<repo>/src)") from None
    expected = set(DOCKER_BUILD_ARGUMENT_BY_PIN) | set(CLIENT_ONLY_PINS)
    if set(values) != expected:
        raise PinsBroken(f"(╯°□°)╯ {path} has keys {sorted(values)}; exactly {sorted(expected)} expected")
    if not isinstance(values["solver_version"], int) or isinstance(values["solver_version"], bool):
        raise PinsBroken(f"(╯°□°)╯ {path}: solver_version {values['solver_version']!r} is not an integer")
    if values["pod_image"].endswith(":latest") or ":" not in values["pod_image"].rsplit("/", 1)[-1]:
        raise PinsBroken(f"(╯°□°)╯ {path}: pod_image {values['pod_image']!r} must name one pushed tag, never :latest")
    return values


PINS = read_pins()
POD_IMAGE = PINS["pod_image"]
ENGINE = PINS["engine"]
SOLVER_VERSION = PINS["solver_version"]
BOUNDARY_LAB_COMMIT = PINS["boundary_lab_commit"]
JULIA_VERSION = PINS["julia_version"]
CUDA_BASE_IMAGE = PINS["cuda_base_image"]


def docker_build_arguments(pins=PINS):
    """[KEY=VALUE] of the image's build arguments, in DOCKER_BUILD_ARGUMENT_BY_PIN's order."""
    return [f"{argument}={pins[key]}" for key, argument in DOCKER_BUILD_ARGUMENT_BY_PIN.items()]


if __name__ == "__main__":
    if sys.argv[1:] == ["docker-build-args"]:
        print("\n".join(docker_build_arguments()))
    else:
        sys.exit(__doc__)
