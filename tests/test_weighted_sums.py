"""acoustic_solve.weighted_sums -- THE WEIGHTED SUM OF PORTS: its weights (filters, delay, polarity, gain) against
answers known in closed form, its declarations checked, resolved down to ports, in words. Moved from monitoring's
tools/tests/test_sounds.py on 2026-10-19 (the speaker half -- default sums, roles -- stayed there). Standard library:

    python3 tests/test_weighted_sums.py
"""
import cmath
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acoustic_solve import weighted_sums  # noqa: E402


def db(value):
    return 20 * math.log10(abs(value))


def degrees(value):
    return math.degrees(cmath.phase(value))


def close(actual, expected, tolerance, what):
    if not abs(actual - expected) <= tolerance:
        raise AssertionError(f"{what}: {actual!r}, expected {expected!r} (+- {tolerance})")


def refused(function, says):
    try:
        function()
    except weighted_sums.WeightedSumRefused as refusal:
        if says not in str(refusal):
            raise AssertionError(f"refused, but not saying {says!r}: {refusal}")
        return
    raise AssertionError(f"not refused (expected {says!r})")


def lr(order, response, fc=100.0):
    return {"kind": "linkwitz_riley", "order": order, "response": response, "frequency_hz": fc}


def bw(order, response, fc=100.0):
    return {"kind": "butterworth", "order": order, "response": response, "frequency_hz": fc}


FREQUENCIES = [20.0, 50.0, 70.7, 100.0, 141.4, 200.0, 500.0, 2000.0]


def test_linkwitz_riley_4_is_minus_6_at_its_corner_and_sums_to_an_all_pass():
    for f in FREQUENCIES:
        low, high = weighted_sums.stage_response(lr(4, "low_pass"), f), weighted_sums.stage_response(lr(4, "high_pass"), f)
        close(abs(low + high), 1.0, 1e-12, f"|LR4 LP + HP| at {f} Hz")
        close(abs(degrees(low) - degrees(high)) % 360, 0.0, 1e-9, f"LR4 LP and HP in phase at {f} Hz")
    close(db(weighted_sums.stage_response(lr(4, "low_pass"), 100.0)), 20 * math.log10(0.5), 1e-12, "LR4 LP at fc")
    close(db(weighted_sums.stage_response(lr(4, "high_pass"), 100.0)), 20 * math.log10(0.5), 1e-12, "LR4 HP at fc")
    close(abs(weighted_sums.stage_response(lr(4, "low_pass"), 100.0) - (-0.5)), 0.0, 1e-12, "LR4 LP at fc: 1/(i sqrt 2)^2")
    close(weighted_sums.stage_response(lr(4, "low_pass"), 1e-3).real, 1.0, 1e-9, "LR4 LP at DC")
    # 24 dB/octave far from the corner
    close(db(weighted_sums.stage_response(lr(4, "low_pass"), 1600.0)) - db(weighted_sums.stage_response(lr(4, "low_pass"), 3200.0)),
          4 * 20 * math.log10(2), 0.01, "LR4 LP slope")


def test_linkwitz_riley_2_needs_one_way_inverted():
    """LR2: LP and HP are 180 deg apart at every frequency, so LP - HP is the all-pass (the textbook polarity)."""
    for f in FREQUENCIES:
        low, high = weighted_sums.stage_response(lr(2, "low_pass"), f), weighted_sums.stage_response(lr(2, "high_pass"), f)
        close(abs(low - high), 1.0, 1e-12, f"|LR2 LP - HP| at {f} Hz")


def test_butterworth_is_minus_3_at_its_corner_and_6n_per_octave():
    for order in (1, 2, 3, 4):
        for response, far in (("low_pass", (3200.0, 6400.0)), ("high_pass", (3.125, 1.5625))):
            close(db(weighted_sums.stage_response(bw(order, response), 100.0)), -10 * math.log10(2), 1e-12,
                  f"BW{order} {response} at fc")
            close(db(weighted_sums.stage_response(bw(order, response), far[0]))
                  - db(weighted_sums.stage_response(bw(order, response), far[1])), 6.0206 * order, 0.01,
                  f"BW{order} {response} slope")
            for f in FREQUENCIES:   # maximally flat: |H|^2 = 1 / (1 + (f/fc)^(2n)) (low), (fc/f)^(2n) for high
                ratio = f / 100.0 if response == "low_pass" else 100.0 / f
                close(abs(weighted_sums.stage_response(bw(order, response), f)) ** 2, 1 / (1 + ratio ** (2 * order)), 1e-12,
                      f"BW{order} {response} |H|^2 at {f}")


def test_a_positive_delay_is_a_phase_lag_of_360_f_tau():
    """exp(+i omega t): a positive delay tau is exp(-i omega tau), a LAG (Boundary Lab's Phasor Convention table)."""
    for f in FREQUENCIES:
        weight = weighted_sums.term_weight({"sound": "x", "delay_ms": 1.0}, f)
        close(abs(weight), 1.0, 1e-12, "a delay's magnitude")
        close(cmath.phase(weight / cmath.exp(-2j * math.pi * f * 1e-3)), 0.0, 1e-9, f"delay phase at {f} Hz")
    close(degrees(weighted_sums.term_weight({"sound": "x", "delay_ms": 0.25}, 100.0)), -9.0, 1e-9, "0.25 ms at 100 Hz")


def test_polarity_and_gain():
    close(weighted_sums.term_weight({"sound": "x", "polarity": -1}, 100.0).real, -1.0, 1e-15, "inverted")
    close(db(weighted_sums.term_weight({"sound": "x", "gain_db": -6.0}, 100.0)), -6.0, 1e-12, "-6 dB")
    combined = weighted_sums.term_weight({"sound": "x", "gain_db": 3.0, "polarity": -1, "delay_ms": 0.5,
                                   "filters": [lr(4, "high_pass")]}, 150.0)
    by_hand = (10 ** (3 / 20) * -1 * cmath.exp(-2j * math.pi * 150.0 * 0.5e-3)
               * weighted_sums.stage_response(lr(4, "high_pass"), 150.0))
    close(abs(combined - by_hand), 0.0, 1e-12, "gain x polarity x delay x filter")


def test_a_biquad_is_its_z_transform_at_z_equals_exp_i_omega_over_fs():
    fs = 48000.0
    one_sample = {"kind": "biquad", "b": [0.0, 1.0, 0.0], "a": [1.0, 0.0, 0.0], "sample_rate_hz": fs}
    for f in FREQUENCIES:
        close(abs(weighted_sums.stage_response(one_sample, f) - weighted_sums.term_weight({"sound": "x", "delay_ms": 1000 / fs}, f)),
              0.0, 1e-12, f"z^-1 is one sample's delay at {f} Hz")
    # RBJ cookbook low-pass at f0 = 100 Hz, Q = 1/sqrt(2): -3.01 dB at f0 and close to the analog BW2 well below fs
    w0 = 2 * math.pi * 100.0 / fs
    alpha = math.sin(w0) / (2 / math.sqrt(2))
    rbj = {"kind": "biquad", "sample_rate_hz": fs, "b": [(1 - math.cos(w0)) / 2, 1 - math.cos(w0), (1 - math.cos(w0)) / 2],
           "a": [1 + alpha, -2 * math.cos(w0), 1 - alpha]}
    close(db(weighted_sums.stage_response(rbj, 100.0)), -10 * math.log10(2), 1e-9, "RBJ LP at f0")
    close(abs(weighted_sums.stage_response(rbj, 50.0) - weighted_sums.stage_response(bw(2, "low_pass"), 50.0)), 0.0, 1e-3,
          "RBJ LP ~ analog BW2 at 50 Hz")
    refused(lambda: weighted_sums.stage_response(rbj, 30000.0), "no response at 30000.0 Hz")


def test_zpk_is_the_butterworth_given_as_its_poles():
    wc = 2 * math.pi * 100.0
    poles = [[wc * math.cos(angle), wc * math.sin(angle)] for angle in (3 * math.pi / 4, 5 * math.pi / 4)]
    zpk = {"kind": "zpk", "zeros_rad_s": [], "poles_rad_s": poles, "gain": wc * wc}
    for f in FREQUENCIES:
        close(abs(weighted_sums.stage_response(zpk, f) - weighted_sums.stage_response(bw(2, "low_pass"), f)), 0.0, 1e-12,
              f"zpk = BW2 at {f} Hz")


def test_a_table_is_exact_at_the_solved_frequencies_and_refuses_any_other():
    table = {"kind": "table", "frequencies_hz": [100.0, 103.5], "real": [0.5, 0.0], "imag": [0.0, -1.0]}
    close(abs(weighted_sums.stage_response(table, 103.5) - (-1j)), 0.0, 0.0, "the tabled value")
    refused(lambda: weighted_sums.stage_response(table, 101.0), "0 values at 101.0 Hz")


def test_declarations_are_checked_loudly():
    good = {"xo": {"terms": [{"sound": "woofer", "filters": [lr(4, "high_pass")]}]}}
    weighted_sums.check_declaration(good)
    for bad, says in (({"x/y": {"terms": [{"sound": "a"}]}}, "a folder name"),
                      ({"xo": {"terms": []}}, "non-empty"),
                      ({"xo": {"terms": [{"sound": "a", "gian_db": 1}]}}, "no keys ['gian_db']"),
                      ({"xo": {"terms": [{"sound": "a", "polarity": 0}]}}, "polarity 0"),
                      ({"xo": {"terms": [{"sound": "a", "filters": [lr(3, "low_pass")]}]}}, "even integer"),
                      ({"xo": {"terms": [{"sound": "a", "filters": [{**lr(4, "low_pass"), "q": 1}]}]}}, "exactly the keys"),
                      ({"xo": {"terms": [{"sound": "a", "filters": [lr(4, "band_pass")]}]}}, "response 'band_pass'"),
                      ({"xo": {"terms": [{"sound": "a", "filters": [{"kind": "fir"}]}]}}, "a filter stage is one of")):
        refused(lambda: weighted_sums.check_declaration(bad), says)


def test_nested_sums_multiply_their_weights_down_to_the_ports():
    ports = ["woofer", "rear_woofer"]
    declared = {"head_hp": {"terms": [{"sound": "woofer", "filters": [lr(4, "high_pass")], "delay_ms": 0.5}]},
                "system": {"terms": [{"sound": "head_hp", "gain_db": -2.0},
                                     {"sound": "rear_woofer", "polarity": -1, "filters": [lr(4, "low_pass")]}]}}
    weighted_sums.check_declaration(declared)
    system = weighted_sums.resolve("system", declared, ports)
    assert [port for port, _ in system] == ["woofer", "rear_woofer"]
    f = 157.0
    by_hand = (10 ** (-2 / 20) * cmath.exp(-2j * math.pi * f * 0.5e-3)
               * weighted_sums.stage_response(lr(4, "high_pass"), f))
    close(abs(weighted_sums.chain_weight(system[0][1], f) - by_hand), 0.0, 1e-12, "nested weight = product")
    close(abs(weighted_sums.chain_weight(system[1][1], f) + weighted_sums.stage_response(lr(4, "low_pass"), f)),
          0.0, 1e-12, "the inverted rear woofer")
    assert weighted_sums.resolve("woofer", declared, ports) == [("woofer", [])]
    assert weighted_sums.chain_weight([], f) is None, "a raw term is added as it is"
    refused(lambda: weighted_sums.resolve("a", {"a": {"terms": [{"sound": "b"}]}, "b": {"terms": [{"sound": "a"}]}},
                                          ports), "contains itself")
    refused(lambda: weighted_sums.resolve("a", {"a": {"terms": [{"sound": "tweeter"}]}}, ports), "names 'tweeter'")


def test_a_resolved_sum_in_words():
    terms = [{"port": "woofer", "chain": [{"filters": [lr(4, "high_pass")], "delay_ms": 0.75}]},
             {"port": "rear_woofer", "chain": [{"filters": [lr(4, "low_pass")], "polarity": -1}]}]
    words = weighted_sums.resolved_terms_words(terms, "A2 crossed")
    assert words == "A2 crossed: woofer (LR4 HP 100 Hz, 0.75 ms) + rear_woofer (LR4 LP 100 Hz, inverted)", words
    assert weighted_sums.resolved_terms_words([{"port": "woofer", "chain": []}]) == "woofer"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed (^_^)")
