"""A WEIGHTED SUM OF PORTS: the algebra every summed sound is made of.

    sum = sum over terms k of  w_k(f) * q_k(f)

q_k is any linear quantity of excitation port k driven ALONE (every other port present at zero drive) -- a pressure
at a field point, a velocity, a volume velocity -- at a solved frequency f; w_k is a complex weight per frequency. A
port alone is the one-term sum, w = 1. The solve is linear, so the sum is exact: the system as if the ports were
driven together, each through its weight. Every port is driven from the same t = 0, so path differences stay in it.

THE CONVENTION. Boundary Lab's results are exp(+i omega t) (~/boundary-lab/docs/advanced/Phasor Convention.md at the
pinned commit; each result's manifest carries `phasor_convention`): s = +i omega, an outgoing wave exp(-i k r), a
positive delay exp(-i omega tau). So a weight is the analog transfer function H(s) at s = +i omega, as a crossover is
written. A weighted (non-raw) sum of a result in any other convention RAISES (boundary_lab_result); a raw sum (every
weight 1) is the same in either.

THE DECLARATION (a client keeps it wherever it likes; monitoring: a row's inputs.json "sums"):

    {"<name>": {"note": "<words, optional>",
                "terms": [{"sound": "<a port, or another sum>",          (required)
                           "gain_db": 0, "polarity": 1 | -1, "delay_ms": 0,  (optional, these defaults)
                           "filters": [<stage>, ...]}, ...]}}            (optional, applied in cascade)

    a stage, one of:
      {"kind": "linkwitz_riley", "order": 2|4|6|8, "response": "low_pass"|"high_pass", "frequency_hz": f}
      {"kind": "butterworth", "order": n, "response": "low_pass"|"high_pass", "frequency_hz": f}
      {"kind": "biquad", "b": [b0, b1, b2], "a": [a0, a1, a2], "sample_rate_hz": fs}      H(z), z = exp(+i omega/fs)
      {"kind": "zpk", "zeros_rad_s": [[re, im], ...], "poles_rad_s": [[re, im], ...], "gain": k}  k prod(s-z)/prod(s-p)
      {"kind": "table", "frequencies_hz": [...], "real": [...], "imag": [...]}   one complex value per SOLVED frequency
                                                                                 (no interpolation: a missing one raises)

    A term's "sound" names a port (its id without "excitation:") or another sum; a sum of sums multiplies the weights
    down to the ports (cycles raise). resolve() turns a name into [(port, chain)], the chain being the term dicts
    (without "sound") from the outermost sum in; chain_weight() is that chain's weight at one frequency.

Standard library only: a client's plain-python readers import it.
"""
import cmath
import math

POSITIVE_TIME = "exp(+i omega t)"
TERM_KEYS = {"sound", "gain_db", "polarity", "delay_ms", "filters"}
SUM_KEYS = {"terms", "note"}
RESPONSES = ("low_pass", "high_pass")
STAGE_KEYS = {"linkwitz_riley": {"kind", "order", "response", "frequency_hz"},
              "butterworth": {"kind", "order", "response", "frequency_hz"},
              "biquad": {"kind", "b", "a", "sample_rate_hz"},
              "zpk": {"kind", "zeros_rad_s", "poles_rad_s", "gain"},
              "table": {"kind", "frequencies_hz", "real", "imag"}}
TABLE_FREQUENCY_TOLERANCE = 1e-9    # relative: a table's frequency IS a solved one (the request's list), not near it


class WeightedSumRefused(SystemExit):
    """A declaration, a weight or a resolution that cannot be used as asked: a SystemExit, so a command line exits
    with the message."""


def refuse(message):
    raise WeightedSumRefused(f"(╯°□°)╯ weighted sums: {message}")


# ---------------------------------------------------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------------------------------------------------

def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_stage(stage, where):
    if not isinstance(stage, dict) or stage.get("kind") not in STAGE_KEYS:
        refuse(f"{where}: a filter stage is one of {sorted(STAGE_KEYS)} as {{'kind': ...}}, not {stage!r}")
    kind = stage["kind"]
    if set(stage) != STAGE_KEYS[kind]:
        refuse(f"{where}: a {kind} stage has exactly the keys {sorted(STAGE_KEYS[kind])}, not {sorted(stage)}")
    if kind in ("linkwitz_riley", "butterworth"):
        order = stage["order"]
        even = kind == "linkwitz_riley"
        if not isinstance(order, int) or isinstance(order, bool) or order < (2 if even else 1) or (even and order % 2):
            refuse(f"{where}: {kind} order {order!r} ({'an even integer >= 2' if even else 'an integer >= 1'})")
        if stage["response"] not in RESPONSES:
            refuse(f"{where}: {kind} response {stage['response']!r} is not one of {RESPONSES}")
        if not is_number(stage["frequency_hz"]) or stage["frequency_hz"] <= 0:
            refuse(f"{where}: {kind} frequency_hz {stage['frequency_hz']!r} (a positive number)")
    elif kind == "biquad":
        for key in ("b", "a"):
            if not (isinstance(stage[key], list) and len(stage[key]) == 3 and all(map(is_number, stage[key]))):
                refuse(f"{where}: biquad {key} {stage[key]!r} (three numbers)")
        if stage["a"][0] == 0:
            refuse(f"{where}: biquad a0 is 0")
        if not is_number(stage["sample_rate_hz"]) or stage["sample_rate_hz"] <= 0:
            refuse(f"{where}: biquad sample_rate_hz {stage['sample_rate_hz']!r}")
    elif kind == "zpk":
        for key in ("zeros_rad_s", "poles_rad_s"):
            if not (isinstance(stage[key], list) and all(isinstance(root, list) and len(root) == 2
                                                         and all(map(is_number, root)) for root in stage[key])):
                refuse(f"{where}: zpk {key} {stage[key]!r} (a list of [re, im] in rad/s)")
        if not is_number(stage["gain"]):
            refuse(f"{where}: zpk gain {stage['gain']!r}")
    else:
        lengths = {len(stage[key]) if isinstance(stage[key], list) else -1 for key in ("frequencies_hz", "real", "imag")}
        if len(lengths) != 1 or -1 in lengths or not all(is_number(value) for key in ("frequencies_hz", "real", "imag")
                                                        for value in stage[key]):
            refuse(f"{where}: a table has three equal-length lists of numbers frequencies_hz, real, imag")


def check_term(term, where):
    if not isinstance(term, dict) or not isinstance(term.get("sound"), str) or not term["sound"]:
        refuse(f"{where}: a term is {{'sound': <a port or a sum>, ...}}, not {term!r}")
    unknown = set(term) - TERM_KEYS
    if unknown:
        refuse(f"{where}: a term has no keys {sorted(unknown)} (it has {sorted(TERM_KEYS)})")
    if "gain_db" in term and not is_number(term["gain_db"]):
        refuse(f"{where}: gain_db {term['gain_db']!r}")
    if "polarity" in term and term["polarity"] not in (1, -1):
        refuse(f"{where}: polarity {term['polarity']!r} (1 or -1)")
    if "delay_ms" in term and not is_number(term["delay_ms"]):
        refuse(f"{where}: delay_ms {term['delay_ms']!r}")
    if not isinstance(term.get("filters", []), list):
        refuse(f"{where}: filters is a list of stages")
    for index, stage in enumerate(term.get("filters", [])):
        check_stage(stage, f"{where} filter {index}")


def check_declaration(sums, where="the declared sums"):
    """The shape of a declaration (names resolve later, against the solved ports). RAISES naming the key."""
    if not isinstance(sums, dict):
        refuse(f"{where} is {{<name>: {{'terms': [...]}}}}, not {type(sums).__name__}")
    for name, declaration in sums.items():
        if not isinstance(name, str) or not name or any(c in name for c in "/:+ "):
            refuse(f"{where}: sum name {name!r} (a folder name: no '/', ':', '+' or space)")
        if not isinstance(declaration, dict) or set(declaration) - SUM_KEYS or "terms" not in declaration:
            refuse(f"{where}.{name}: a sum has 'terms' (and optionally 'note'), not {declaration!r}")
        if not isinstance(declaration["terms"], list) or not declaration["terms"]:
            refuse(f"{where}.{name}: terms is a non-empty list")
        for index, term in enumerate(declaration["terms"]):
            check_term(term, f"{where}.{name} term {index}")


# ---------------------------------------------------------------------------------------------------------------------
# the weights
# ---------------------------------------------------------------------------------------------------------------------

def butterworth_low_pass(order, normalised_s):
    """THE Butterworth low-pass of this order at s / omega_c: prod over its poles p_k (unit circle, left half plane)
    of -p_k / (s - p_k), which is 1 at DC."""
    response = 1 + 0j
    for k in range(1, order + 1):
        pole = cmath.exp(1j * math.pi * (2 * k + order - 1) / (2 * order))
        response *= -pole / (normalised_s - pole)
    return response


def stage_response(stage, frequency_hz):
    """One filter stage's H at s = +i 2 pi f (positive time)."""
    if frequency_hz <= 0:
        refuse(f"a weight at {frequency_hz} Hz: frequencies are positive")
    s = 2j * math.pi * frequency_hz
    kind = stage["kind"]
    if kind in ("butterworth", "linkwitz_riley"):
        butterworth_order = stage["order"] // 2 if kind == "linkwitz_riley" else stage["order"]
        normalised_s = s / (2 * math.pi * stage["frequency_hz"])
        argument = normalised_s if stage["response"] == "low_pass" else 1 / normalised_s   # HP(s) = LP(1/s)
        response = butterworth_low_pass(butterworth_order, argument)
        return response * response if kind == "linkwitz_riley" else response
    if kind == "biquad":
        if frequency_hz >= stage["sample_rate_hz"] / 2:
            refuse(f"a biquad at {stage['sample_rate_hz']} Hz sampling has no response at {frequency_hz} Hz")
        inverse_z = cmath.exp(-s / stage["sample_rate_hz"])        # z^-1: one sample's delay, exp(-i omega T)
        b, a = stage["b"], stage["a"]
        return (b[0] + b[1] * inverse_z + b[2] * inverse_z ** 2) / (a[0] + a[1] * inverse_z + a[2] * inverse_z ** 2)
    if kind == "zpk":
        response = complex(stage["gain"])
        for re, im in stage["zeros_rad_s"]:
            response *= s - complex(re, im)
        for re, im in stage["poles_rad_s"]:
            response /= s - complex(re, im)
        return response
    matches = [index for index, tabled in enumerate(stage["frequencies_hz"])
               if abs(tabled - frequency_hz) <= TABLE_FREQUENCY_TOLERANCE * frequency_hz]
    if len(matches) != 1:
        refuse(f"a table weight has {len(matches)} values at {frequency_hz} Hz (one per solved frequency, exactly)")
    return complex(stage["real"][matches[0]], stage["imag"][matches[0]])


def term_weight(term, frequency_hz):
    """One term's own weight: 10^(gain_db/20) x polarity x exp(-s tau) x its filters in cascade."""
    s = 2j * math.pi * frequency_hz
    weight = 10 ** (term.get("gain_db", 0) / 20) * term.get("polarity", 1) * cmath.exp(-s * term.get("delay_ms", 0) / 1000)
    for stage in term.get("filters", []):
        weight *= stage_response(stage, frequency_hz)
    return complex(weight)


def chain_is_raw(chain):
    return all(term.get("gain_db", 0) == 0 and term.get("polarity", 1) == 1 and term.get("delay_ms", 0) == 0
               and not term.get("filters") for term in chain)


def chain_weight(chain, frequency_hz):
    """A resolved term's weight: the product of its chain's term weights. None for a raw chain (exactly 1: the sum
    adds the quantities themselves, bit for bit as a port alone does)."""
    if chain_is_raw(chain):
        return None
    weight = 1 + 0j
    for term in chain:
        weight *= term_weight(term, frequency_hz)
    return weight


# ---------------------------------------------------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------------------------------------------------

def resolve(name, declarations, port_names, path=()):
    """[(port, chain)] of a sound: a port is [(port, [])]; a sum's terms resolved down to ports, each chain the term
    dicts (without "sound") from the outermost sum in."""
    if name in path:
        refuse(f"sum {name!r} contains itself: {' -> '.join(path + (name,))}")
    if name in port_names:
        return [(name, [])]
    if name not in declarations:
        refuse(f"{' -> '.join(path) or 'a sum'} names {name!r}, which is no port ({sorted(port_names)}) and no sum "
               f"({sorted(declarations)})")
    resolved = []
    for term in declarations[name]["terms"]:
        own = {key: value for key, value in term.items() if key != "sound"}
        resolved += [(port, [own] + chain) for port, chain in
                     resolve(term["sound"], declarations, port_names, path + (name,))]
    return resolved


# ---------------------------------------------------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------------------------------------------------

STAGE_WORDS = {"low_pass": "LP", "high_pass": "HP"}


def stage_words(stage):
    kind = stage["kind"]
    if kind == "linkwitz_riley":
        return f"LR{stage['order']} {STAGE_WORDS[stage['response']]} {stage['frequency_hz']:g} Hz"
    if kind == "butterworth":
        return f"BW{stage['order']} {STAGE_WORDS[stage['response']]} {stage['frequency_hz']:g} Hz"
    if kind == "biquad":
        return f"biquad @{stage['sample_rate_hz']:g} Hz"
    if kind == "zpk":
        return f"zpk {len(stage['zeros_rad_s'])}z/{len(stage['poles_rad_s'])}p"
    return f"table ({len(stage['frequencies_hz'])} f)"


def chain_words(chain):
    words = []
    for term in chain:
        words += [stage_words(stage) for stage in term.get("filters", [])]
        if term.get("gain_db", 0):
            words.append(f"{term['gain_db']:+g} dB")
        if term.get("polarity", 1) == -1:
            words.append("inverted")
        if term.get("delay_ms", 0):
            words.append(f"{term['delay_ms']:g} ms")
    return ", ".join(words)


def resolved_terms_words(resolved_terms, note=None):
    """A resolved sum in words ([{"port", "chain"}]): its note if it has one, then each port with its weight's words."""
    terms = " + ".join(term["port"] + (f" ({chain_words(term['chain'])})" if chain_words(term["chain"]) else "")
                       for term in resolved_terms)
    return f"{note}: {terms}" if note else terms
