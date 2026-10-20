"""THE DIGEST THAT NAMES THINGS: the first 12 hex digits of sha256 over the canonical JSON of a value.

Canonical JSON: keys sorted, integral floats written as integers, tuples as lists, no whitespace, UTF-8 as is -- so
`40` and `40.0` and any key order are one id. A value with no JSON form (NaN, infinity, an object) RAISES NotCanonical:
a name that silently stringified it would be a name of nothing. Standard library only, so every environment a client
runs in (a build123d venv, the Boundary Lab venv, a plain python3) imports the same functions.

What a client hashes, and which of its keys are not hashed, is the client's (monitoring: model/identity.py for exports,
tools/runstore.py for rows); this is only the digest.
"""
import hashlib
import json

IDENTIFIER_HEX_DIGITS = 12


class NotCanonical(ValueError):
    """A value with no JSON form (NaN, infinity, an object)."""


def canonical_value(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise NotCanonical(f"{value!r} has no JSON form")
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    raise NotCanonical(f"{type(value).__name__} {value!r} has no JSON form")


def canonical_json(value):
    return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_identifier(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:IDENTIFIER_HEX_DIGITS]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
