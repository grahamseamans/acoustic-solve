"""acoustic_solve.canonical_digest -- one value, one id: key order and 40 vs 40.0 do not change it, tuples are lists,
a value with no JSON form is refused. Standard library:

    python3 tests/test_canonical_digest.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acoustic_solve import canonical_digest  # noqa: E402


def test_key_order_integral_floats_and_tuples_are_one_id():
    a = canonical_digest.digest_identifier({"b": 40, "a": (1.0, 2.5)})
    b = canonical_digest.digest_identifier({"a": [1, 2.5], "b": 40.0})
    assert a == b and len(a) == canonical_digest.IDENTIFIER_HEX_DIGITS
    assert canonical_digest.canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert canonical_digest.digest_identifier({"a": 1}) != canonical_digest.digest_identifier({"a": 1.5})


def test_a_value_with_no_json_form_is_refused():
    for value in (float("nan"), float("inf"), {"a": object()}):
        try:
            canonical_digest.canonical_json(value)
        except canonical_digest.NotCanonical:
            continue
        raise AssertionError(f"{value!r} was given an id")


def test_file_sha256_is_the_files_bytes():
    path = Path(tempfile.mkdtemp()) / "x"
    path.write_bytes(b"abc")
    assert canonical_digest.file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
