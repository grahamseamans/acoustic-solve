"""acoustic_solve.boundary_lab_result -- the one reader: quantities stacked per frequency, weighted sums of ports bit
for bit for a raw sum and by the weights otherwise, and every refusal loud (incomplete, ports or convention that differ
per frequency, a quantity at some frequencies only, other axes, a weighted sum of a negative-time result). In the blab
venv (numpy):

    blabrun tests/test_boundary_lab_result.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_boundary_lab_results import write_result  # noqa: E402
from acoustic_solve import weighted_sums  # noqa: E402
from acoustic_solve.boundary_lab_result import BoundaryLabResult, BoundaryLabResultError  # noqa: E402

FREQUENCIES = [50.0, 100.0, 200.0]
PORTS = ["excitation:woofer", "excitation:rear_woofer"]
RNG = np.random.default_rng(7)
POLAR = RNG.normal(size=(3, 2, 5)) + 1j * RNG.normal(size=(3, 2, 5))


def a_result(**changes):
    directory = Path(tempfile.mkdtemp(prefix="blab-result-")) / "raw"
    arguments = {"frequencies_hz": FREQUENCIES, "port_ids": PORTS,
                 "quantities": [{"id": "acoustic:pressure:horizontal-polar", "quantity": "pressure",
                                 "axes": ["excitation", "observation"], "values": POLAR}],
                 "domains": [{"id": "polar:h", "kind": "polar_observation", "metadata": {"plane": "horizontal"},
                              "coordinates": {"angle_deg": np.linspace(-180, 180, 5)}}]}
    return write_result(directory, **{**arguments, **changes})


def raises(call, *words):
    try:
        call()
    except BoundaryLabResultError as error:
        assert all(word in str(error) for word in words), (words, str(error))
        return
    raise AssertionError(f"no refusal ({words})")


def test_a_quantity_is_stacked_per_frequency_and_a_raw_sum_is_the_ports_added_bit_for_bit():
    result = BoundaryLabResult(a_result())
    assert list(result.frequencies_hz) == FREQUENCIES and result.port_names() == ["woofer", "rear_woofer"]
    assert np.array_equal(result.quantity("acoustic:pressure:horizontal-polar"), POLAR)
    assert result.quantity("acoustic:no-such") is None
    raw = result.weighted_terms([{"port": "woofer", "chain": []}, {"port": "rear_woofer", "chain": []}])
    assert all(weights is None for _, weights in raw)
    assert np.array_equal(result.weighted_sum("acoustic:pressure:horizontal-polar", raw), POLAR[:, 0] + POLAR[:, 1])
    assert list(result.polar_angles_by_plane()["horizontal"]) == [-180, -90, 0, 90, 180]


def test_a_weighted_sum_multiplies_each_port_by_its_weight_per_frequency():
    result = BoundaryLabResult(a_result())
    chain = [{"polarity": -1, "delay_ms": 1.0}]
    terms = result.weighted_terms([{"port": "woofer", "chain": []}, {"port": "rear_woofer", "chain": chain}])
    summed = result.weighted_sum("acoustic:pressure:horizontal-polar", terms)
    for index, frequency in enumerate(FREQUENCIES):
        weight = weighted_sums.chain_weight(chain, frequency)
        assert np.allclose(summed[index], POLAR[index, 0] + weight * POLAR[index, 1], rtol=0, atol=1e-12)
    raises(lambda: result.weighted_terms([{"port": "tweeter", "chain": []}]), "no port 'tweeter'")


def test_a_weighted_sum_of_a_negative_time_result_is_refused_and_a_raw_one_is_not():
    result = BoundaryLabResult(a_result(phasor_convention="exp(-i omega t)"))
    result.weighted_terms([{"port": "woofer", "chain": []}])
    raises(lambda: result.weighted_terms([{"port": "woofer", "chain": [{"delay_ms": 1.0}]}]), "exp(-i omega t)")


def test_an_incomplete_result_is_refused_unless_only_its_manifest_is_asked_for():
    directory = a_result(status="interrupted")
    raises(lambda: BoundaryLabResult(directory), "not a complete result", "1/3")
    stopped = BoundaryLabResult(directory, require_complete=False)
    assert stopped.status == "interrupted" and not stopped.is_complete
    raises(lambda: stopped.frequencies_hz, "without requiring it complete")
    raises(lambda: BoundaryLabResult(directory.parent / "nothing"), "no manifest")


def test_a_frequency_whose_ports_or_convention_differ_from_the_manifest_is_refused():
    directory = a_result()
    meta_path = directory / "frequencies" / "000001.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "excitation_port_ids": PORTS[::-1]}))
    raises(lambda: BoundaryLabResult(directory).quantity("acoustic:pressure:horizontal-polar"), "ports")
    meta_path.write_text(json.dumps({**meta, "diagnostics": {"phasor_convention": "exp(-i omega t)"}}))
    raises(lambda: BoundaryLabResult(directory).quantity("acoustic:pressure:horizontal-polar"), "phasor convention")


def test_a_quantity_at_some_frequencies_only_or_on_other_axes_is_refused():
    directory = a_result()
    meta_path = directory / "frequencies" / "000002.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "quantities": []}))
    raises(lambda: BoundaryLabResult(directory).quantity("acoustic:pressure:horizontal-polar"), "at 2 of 3")
    raises(lambda: BoundaryLabResult(a_result()).quantity("acoustic:pressure:horizontal-polar",
                                                          axes=["excitation", "fem_node"]), "axes")
    raises(lambda: BoundaryLabResult(a_result()).quantity_of_kind("fem_nodal_pressure"), "retain")


def test_frequency_indices_find_each_asked_frequency_and_refuse_one_off_the_grid():
    result = BoundaryLabResult(a_result())
    assert result.frequency_indices([200.0, 50.0]) == [2, 0]
    raises(lambda: result.frequency_indices([75.0]), "75.0 Hz is not on its grid")


def test_a_transducer_port_gives_its_velocity_current_and_reference_volts_and_a_velocity_port_none():
    velocity = np.zeros((3, 2, 1), dtype=complex)
    velocity[:, 0, 0] = [1 + 1j, 2, 3j]
    current = np.ones((3, 2, 1), dtype=complex)
    result = BoundaryLabResult(a_result(
        quantities=[{"id": "mechanical:diaphragm-velocity", "quantity": "v", "axes": ["excitation", "t"],
                     "values": velocity},
                    {"id": "electrical:voice-coil-current", "quantity": "i", "axes": ["excitation", "t"],
                     "values": current}],
        diagnostics={"transducer_reference_voltage_v": 2.83}))
    v, i, volts = result.transducer_of_port(0)
    assert list(v) == [1 + 1j, 2, 3j] and volts == 2.83
    assert result.transducer_of_port(1) is None


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
