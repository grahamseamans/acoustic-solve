"""acoustic_solve.room.analysis -- what every room study measures with, checked without a solve:

- the decay (Schroeder T20 / EDT) on single modes of known T60 (the docstring's 2026-10-15 check, kept): T20 within
  2 % and EDT within 10 % wherever bt60 >= 5.5;
- the modal sum is reciprocal (source and seat swapped give one pressure) and its peaks, at a light bulk loss, are
  the shoebox's analytic modes;
- a room solve is read at its seat points (each one a mesh node, or refused) and chunks join in frequency order;
- the free field is read per probe, at the room's frequencies, in the room's point order (or refused);
- a flat room gain has no spread.
In the blab venv (numpy):

    blabrun tests/test_room_analysis.py
"""
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_boundary_lab_results import write_result  # noqa: E402
from acoustic_solve.room import analysis  # noqa: E402
from acoustic_solve.room.spec import FREE_FIELD_SPECS_FILE, POINTS_FILE, seat_point_indices  # noqa: E402

GRID_HZ = np.arange(15.0, 300.0 + 1e-9, 0.5)
DELAY_S = 0.009


def single_mode(f0, t60):
    """One room mode as the solver would give it (exp(+i omega t), s = i omega): 1 / ((s + delta)^2 + omega0^2), its
    energy decaying by 60 dB in t60 (delta = ln(1000) / t60), delayed by DELAY_S."""
    s = 2j * np.pi * GRID_HZ
    delta = math.log(1000.0) / t60
    return np.exp(-s * DELAY_S) / ((s + delta) ** 2 + (2 * np.pi * f0) ** 2)


def test_t20_and_edt_recover_a_single_modes_t60_where_bt60_is_at_least_5_5():
    checked = 0
    for t60 in (0.3, 0.6):
        for centre in (50.0, 80.0, 125.0, 200.0):
            measured = analysis.decay(GRID_HZ, single_mode(centre, t60))[f"{centre:g}"]
            if measured["bt60"] < 5.5:
                continue
            checked += 1
            assert abs(measured["t20_s"] / t60 - 1) <= 0.02, (centre, t60, measured)
            assert abs(measured["edt_s"] / t60 - 1) <= 0.10, (centre, t60, measured)
    assert checked >= 6, checked


def test_the_modal_sum_is_reciprocal_and_peaks_at_the_analytic_modes():
    dimensions = (4.0, 3.0, 2.5)
    source, seat = [0.3, 0.4, 0.5], [2.9, 1.7, 1.1]
    frequencies = np.arange(20.0, 120.0, 0.25)
    forward, modes = analysis.modal_sum(frequencies, dimensions, 0.01, source, 1e-3, [seat])
    backward, _ = analysis.modal_sum(frequencies, dimensions, 0.01, seat, 1e-3, [source])
    assert modes > 100 and np.allclose(forward, backward, rtol=1e-12, atol=0)
    level = 20 * np.log10(np.abs(forward[:, 0]))
    analytic = [f for f, _ in analysis.analytic_modes(dimensions, 120.0)]
    for peak in analysis.peaks(frequencies, level):
        assert min(abs(peak["hz"] - f) for f in analytic) <= 0.5, (peak, analytic)


def a_room_stage(frequencies, nodes, names_and_points, pressure_at_nodes):
    stage = Path(tempfile.mkdtemp(prefix="room-stage-"))
    write_result(stage / "raw", frequencies, ["excitation:front", "excitation:rear"],
                 [{"id": "acoustic:fem-nodal-pressure", "quantity": "fem_nodal_pressure",
                   "axes": ["excitation", "fem_node"], "values": pressure_at_nodes}],
                 domains=[{"id": "fem:room", "kind": "fem_volume", "coordinates": {"points_m": nodes}}])
    (stage / POINTS_FILE).write_text(json.dumps({"seat_points_m": names_and_points}))
    return stage


def test_a_room_solve_is_read_at_its_seat_points_and_chunks_join_in_frequency_order():
    nodes = np.array([[0, 0, 0], [1, 1, 1], [2, 1, 1], [3, 0, 0.0]])
    points = {"centre/x+0y+0": [2, 1, 1], "centre/x+1y+0": [1, 1, 1]}
    pressure = np.arange(2 * 2 * 4, dtype=complex).reshape(2, 2, 4)
    high = a_room_stage([60.0, 70.0], nodes, points, pressure)
    low = a_room_stage([40.0, 50.0], nodes, points, pressure + 100)
    solve = analysis.load_room_solve(high)
    assert solve.point_names == list(points) and np.array_equal(solve.pressure, pressure[:, :, [2, 1]])
    assert np.array_equal(solve.port("excitation:rear"), pressure[:, 1][:, [2, 1]])
    joined = analysis.join_frequency_chunks([high, low])
    assert list(joined.frequencies_hz) == [40.0, 50.0, 60.0, 70.0]
    assert np.array_equal(joined.pressure[:2], (pressure + 100)[:, :, [2, 1]])
    assert seat_point_indices(joined.point_names, "centre") == (0, [0, 1])
    off = a_room_stage([40.0], nodes, {"centre/x+0y+0": [2, 1, 1.5]}, pressure[:1])
    try:
        analysis.load_room_solve(off)
    except RuntimeError as error:
        assert "not embedded" in str(error), error
    else:
        raise AssertionError("a seat point off every node was read")


def test_the_free_field_is_read_per_probe_at_the_rooms_frequencies_in_the_rooms_point_order():
    stage = Path(tempfile.mkdtemp(prefix="free-field-"))
    values = np.arange(3 * 2 * 2, dtype=complex).reshape(3, 2, 2)
    write_result(stage / "raw", [40.0, 50.0, 60.0], ["excitation:front", "excitation:rear"],
                 [{"id": "acoustic:pressure:probe:medium", "quantity": "pressure", "axes": ["excitation", "observation"],
                   "values": values}])
    (stage / FREE_FIELD_SPECS_FILE).write_text(json.dumps({"probes": {"medium": ["s/x+0y+0", "s/x+1y+0"]}}))
    alone = analysis.free_field(stage, "medium", np.array([60.0, 40.0]), ["s/x+0y+0", "s/x+1y+0"])
    assert np.array_equal(alone["excitation:rear"], values[[2, 0], 1])
    try:
        analysis.free_field(stage, "medium", np.array([40.0]), ["s/x+1y+0", "s/x+0y+0"])
    except RuntimeError as error:
        assert "has points" in str(error)
    else:
        raise AssertionError("another point order was read")


def test_a_flat_room_gain_has_no_spread():
    frequencies = np.arange(20.0, 300.0, 1.0)
    bands = analysis.room_gain_band_metrics(frequencies, np.full((len(frequencies), 3), 2.0), 0, [0, 1, 2],
                                            {"20-100": (20.0, 100.0)})
    assert bands["20-100"] == {"sd_db": 0.0, "sd_third_db": 0.0, "spread_db": 0.0, "max_minus_min_db": 0.0,
                               "top_peaks": []}, bands


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
