"""A `blab project solve` result directory written by hand, in Boundary Lab's own layout (manifest.json, domains.json +
domains.npz, frequencies/<index>.json + .npz: ~/boundary-lab/src/blab/headless.py at the pinned commit), for the tests
of the reader and the room analysis. No solver.
"""
import json
from pathlib import Path

import numpy as np


def write_result(directory, frequencies_hz, port_ids, quantities, domains=(), status="complete",
                 phasor_convention="exp(+i omega t)", diagnostics=None):
    """quantities: [{"id", "quantity", "axes", "values": array (frequency, port, ...), "metadata"?}];
    domains: [{"id", "kind", "metadata"?, "coordinates": {name: array}}]."""
    directory = Path(directory)
    (directory / "frequencies").mkdir(parents=True)
    domain_arrays, domain_records = {}, []
    for number, domain in enumerate(domains):
        keys = {}
        for name, array in domain["coordinates"].items():
            keys[name] = f"domain{number}_{name}"
            domain_arrays[keys[name]] = np.asarray(array)
        domain_records.append({"id": domain["id"], "kind": domain["kind"], "metadata": domain.get("metadata", {}),
                               "coordinates": keys})
    np.savez(directory / "domains.npz", **domain_arrays)
    (directory / "domains.json").write_text(json.dumps({"domains": domain_records}))
    results = []
    for index, frequency in enumerate(frequencies_hz):
        stem = f"{index:06d}"
        arrays = {f"q{n}": np.asarray(q["values"][index]) for n, q in enumerate(quantities)}
        np.savez(directory / "frequencies" / f"{stem}.npz", **arrays)
        meta = {"freq_hz": float(frequency), "excitation_port_ids": list(port_ids), "arrays_file": f"{stem}.npz",
                "quantities": [{"id": q["id"], "quantity": q["quantity"], "key": f"q{n}", "axes": q["axes"],
                                "metadata": q.get("metadata", {})} for n, q in enumerate(quantities)],
                "diagnostics": {"phasor_convention": phasor_convention, **(diagnostics or {})}}
        (directory / "frequencies" / f"{stem}.json").write_text(json.dumps(meta))
        results.append({"freq_hz": float(frequency), "metadata_file": f"frequencies/{stem}.json",
                        "arrays_file": f"frequencies/{stem}.npz"})
    manifest = {"schema": "boundary-lab-headless-result", "status": status,
                "completion_mask": [True] * len(frequencies_hz) if status == "complete" else
                [True] + [False] * (len(frequencies_hz) - 1),
                "frequencies_hz": [float(f) for f in frequencies_hz], "excitation_port_ids": list(port_ids),
                "phasor_convention": phasor_convention, "project_sha256": "p" * 64, "results": results,
                "domains_metadata_file": "domains.json", "domains_file": "domains.npz"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory
