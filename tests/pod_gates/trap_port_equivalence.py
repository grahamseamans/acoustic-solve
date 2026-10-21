"""POD GATE 2 of PLANS/2026-10-20-trap-velocity-port.md: the trap face as a velocity port, on a real solve. Not a test
(the tests never solve): it STAGES three CPU solves of one tiny room for podd, and, once they are DONE, CHECKS them.

    blabrun tests/pod_gates/trap_port_equivalence.py stage <podd queue dir>     # build + JOB, three stages
    blabrun tests/pod_gates/trap_port_equivalence.py check <podd queue dir>     # after podd brought raw/ home

THE ROOM: 2.4 x 2 x 1.8 m, rigid walls, a 0.3 m cube source with one port (its whole front face), a seat grid, one
corner trap 0.5 m wide x 1 m tall with one mic 0.1 m in front of its face centre; 20-200 Hz every 2.5 Hz. Solved:
  port      the trap face a velocity_port, bulk loss ETA_HIGH
  rho-c     the same room, the trap face a rho c termination, ETA_HIGH        (the same mesh parameters)
  port-low  as `port`, bulk loss ETA_LOW

a. EQUIVALENCE: `port` closed with G = 1 / (rho c) on the face average against `rho-c`, at the seats. Not identical:
   the termination is pointwise, the port uniform, so the relative difference should fall like (ka)^2 toward low
   frequency (a = the face's equivalent radius sqrt(A / pi)). Asserted: the error under EQUIVALENCE_LOW_HZ is under
   EQUIVALENCE_LOW_ERROR, and its log-log slope against frequency over the band is at least EQUIVALENCE_MIN_SLOPE.
b. ENERGY BALANCE: rigid walls, so the source's input power = the trap's absorbed power + the bulk loss's share
   (positive, ~ eta: the trap's damping dominates the field, so the field barely moves with eta). Asserted: the share
   is never negative beyond round-off, and `port`'s share over `port-low`'s is ETA_HIGH / ETA_LOW within
   ENERGY_SHARE_RATIO_SLACK (median over frequency).
The thresholds are the plan's expectations written down before the solve; the printed table is what to judge.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from acoustic_solve.room import analysis, build, traps
from acoustic_solve.room.spec import SOUND_SPEED_M_PER_S

STAGES = {"port": ("velocity_port", "high"), "rho-c": ("rho_c", "high"), "port-low": ("velocity_port", "low")}
STAGE_PREFIX = "trap-port-gate-"
ETA = {"high": 1e-2, "low": 1e-3}
FREQUENCIES_HZ = [20.0 + 2.5 * k for k in range(73)]          # 20 .. 200 Hz
TRAP = {"name": "corner", "mount": "corner", "corner": [1, 1], "level": "floor", "width_m": 0.5, "height_m": 1.0,
        "mic_points_m": {"front": [0.1, 0, 0]}}
ROOM = {"dimensions_m": [2.4, 2.0, 1.8],
        "sources": [{"name": "source", "size_m": [0.3, 0.3, 0.3], "centre_m": [0.5, 1.0, 0.6], "aim_m": [2.0, 1.0],
                     "ports": [{"name": "front", "face": "front", "shape": "face"}]}],
        "seats": {"centre": {"centre_m": [1.5, 0.9, 1.0], "step_m": [0.1, 0.1, 0.1], "half_count": [1, 1, 0]}},
        "h_far_m": 0.2, "h_near_m": 0.08, "frequencies_hz": FREQUENCIES_HZ}
SOURCE_PORT = "excitation:front"
EXPECTED_S, RAM_GB, CPU = 300, 4, 4                          # a tiny room's CPU solve: measured by the first run
EQUIVALENCE_LOW_HZ, EQUIVALENCE_LOW_ERROR, EQUIVALENCE_MIN_SLOPE = 40.0, 1e-2, 1.5
ENERGY_SHARE_ROUND_OFF, ENERGY_SHARE_RATIO_SLACK = 1e-9, 0.3


def gate_spec(name, face, eta):
    return {**ROOM, "name": f"{STAGE_PREFIX}{name}", "traps": [{**TRAP, "face": face}],
            "absorption": {"bulk_loss_factor": eta, "rho_c_patch_fraction": 0.0}}


def stage(queue):
    """Build the three stages in the queue directory and write their JOBs (acoustic_solve.room.build.submit)."""
    queue = Path(queue)
    specs = Path(tempfile.mkdtemp(prefix="trap-port-gate-specs-"))
    for name, (face, eta) in STAGES.items():
        spec_path = specs / f"{name}.json"
        spec_path.write_text(json.dumps(gate_spec(name, face, ETA[eta]), indent=1))
        staged = build.main(queue / f"{STAGE_PREFIX}{name}", spec_path)
        build.submit(staged, EXPECTED_S, RAM_GB, CPU)
        print(f"queued {staged}")


def relative_seat_error(port_seats, reference_seats):
    return np.linalg.norm(port_seats - reference_seats, axis=1) / np.linalg.norm(reference_seats, axis=1)


def closed_rho_c(trap_room):
    gains = traps.law(trap_room, {TRAP["name"]: {f"{TRAP['name']}/face": 1 / traps.RHO_C}})
    return traps.closed_loop(trap_room, gains, {SOURCE_PORT: 1.0})


def check(queue):
    """Read the three solved stages, print the gate's table, assert the docstring's a and b."""
    queue = Path(queue)
    port, port_low = (traps.load_trap_room(queue / f"{STAGE_PREFIX}{name}") for name in ("port", "port-low"))
    reference = analysis.load_room_solve(queue / f"{STAGE_PREFIX}rho-c")
    frequencies = port.room.frequencies_hz
    if not (np.array_equal(frequencies, reference.frequencies_hz) and port.room.point_names == reference.point_names
            and np.array_equal(frequencies, port_low.room.frequencies_hz)):
        raise RuntimeError("(╯°□°)╯ the gate's stages differ in frequencies or seat points")
    loop, loop_low = closed_rho_c(port), closed_rho_c(port_low)
    error = relative_seat_error(loop.seat_pressure, reference.port(SOURCE_PORT))
    radius = math.sqrt(TRAP["width_m"] * TRAP["height_m"] / math.pi)
    ka = 2 * math.pi * frequencies / SOUND_SPEED_M_PER_S * radius
    shares = []
    for closed in (loop, loop_low):
        source_in, trapped = closed.source_input_w.sum(axis=1), closed.trap_absorbed_w.sum(axis=1)
        shares.append((source_in - trapped) / source_in)
    print(f"{'f Hz':>7} {'ka':>7} {'seat err':>10} {'err/(ka)^2':>11} {'P_in W':>10} {'trap W':>10} "
          f"{'bulk share':>11} {'share low eta':>14}")
    for i, f in enumerate(frequencies):
        print(f"{f:7.1f} {ka[i]:7.3f} {error[i]:10.2e} {error[i] / ka[i] ** 2:11.3f} "
              f"{loop.source_input_w[i].sum():10.3e} {loop.trap_absorbed_w[i].sum():10.3e} {shares[0][i]:11.2e} "
              f"{shares[1][i]:14.2e}")
    slope = np.polyfit(np.log(frequencies), np.log(error), 1)[0]
    low_error = float(error[frequencies <= EQUIVALENCE_LOW_HZ].max())
    ratio = float(np.median(shares[0] / shares[1]))
    print(f"equivalence: max error under {EQUIVALENCE_LOW_HZ:g} Hz {low_error:.2e}, log-log slope {slope:.2f}; "
          f"energy: median bulk-share ratio {ratio:.2f} (eta ratio {ETA['high'] / ETA['low']:g}), "
          f"least share {min(s.min() for s in shares):.2e}")
    failures = []
    if low_error > EQUIVALENCE_LOW_ERROR:
        failures.append(f"equivalence error {low_error:.2e} under {EQUIVALENCE_LOW_HZ:g} Hz > {EQUIVALENCE_LOW_ERROR}")
    if slope < EQUIVALENCE_MIN_SLOPE:
        failures.append(f"equivalence error's slope {slope:.2f} < {EQUIVALENCE_MIN_SLOPE} (expected ~2: (ka)^2)")
    if min(s.min() for s in shares) < -ENERGY_SHARE_ROUND_OFF:
        failures.append("the bulk loss's share is negative: the balance creates energy")
    if abs(ratio / (ETA["high"] / ETA["low"]) - 1) > ENERGY_SHARE_RATIO_SLACK:
        failures.append(f"the bulk loss's share does not scale with eta (ratio {ratio:.2f})")
    if failures:
        raise RuntimeError("(╯°□°)╯ trap port gate FAILED: " + "; ".join(failures))
    print("trap port gate PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("stage", "check"):
        {"stage": stage, "check": check}[sys.argv[1]](sys.argv[2])
    else:
        sys.exit(__doc__)
