"""A `blab project solve` result directory written by hand, in Boundary Lab's own layout (manifest.json, domains.json +
domains.npz, frequencies/<index>.json + .npz: ~/boundary-lab/src/blab/headless.py at the pinned commit), for the tests
of the reader and the room analysis (a room's: write_room_result). No solver.
"""
import json
from pathlib import Path

import numpy as np


def write_result(directory, frequencies_hz, port_ids, quantities, domains=(), status="complete",
                 phasor_convention="exp(+i omega t)", diagnostics=None):
    """quantities: [{"id", "quantity", "axes", "values": array (frequency, port, ...), "metadata"?}];
    domains: [{"id", "kind", "metadata"?, "coordinates": {name: array}, "topology"?: {name: array}}]."""
    directory = Path(directory)
    (directory / "frequencies").mkdir(parents=True)
    domain_arrays, domain_records = {}, []
    for number, domain in enumerate(domains):
        keys = {}
        for name, array in domain["coordinates"].items():
            keys[name] = f"domain{number}_{name}"
            domain_arrays[keys[name]] = np.asarray(array)
        topology = {}
        for name, array in domain.get("topology", {}).items():
            topology[name] = f"domain{number}_topology_{name}"
            domain_arrays[topology[name]] = np.asarray(array)
        domain_records.append({"id": domain["id"], "kind": domain["kind"], "metadata": domain.get("metadata", {}),
                               "coordinates": keys, "topology": topology})
    np.savez(directory / "domains.npz", **domain_arrays)
    (directory / "domains.json").write_text(json.dumps({"domains": domain_records}))
    results = []
    for index, frequency in enumerate(frequencies_hz):
        stem = f"{index:06d}"
        arrays = {f"q{n}": np.asarray(q["values"][index]) for n, q in enumerate(quantities)}
        np.savez(directory / "frequencies" / f"{stem}.npz", **arrays)
        meta = {"freq_hz": float(frequency), "excitation_port_ids": list(port_ids), "arrays_file": f"{stem}.npz",
                "quantities": [{"id": q["id"], "quantity": q["quantity"], "key": f"q{n}", "axes": q["axes"],
                                "metadata": q.get("metadata", {}), "target_id": q.get("target_id"),
                                "shape": list(np.asarray(q["values"][index]).shape)}
                               for n, q in enumerate(quantities)],
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


def write_room_result(directory, frequencies_hz, port_ids, nodes, pressure_at_nodes, tetrahedra10=None):
    """A room solve's result: one fem_volume domain (its nodes, and its ten-node tetrahedra when given, with Boundary
    Lab's per-node region_index, corner tetrahedra and per-cell region_index beside them) and the retained
    fem_nodal_pressure (frequency, port, node)."""
    domain = {"id": "fem:room", "kind": "fem_volume", "coordinates": {"points_m": nodes},
              "metadata": {"node_counts": [len(nodes)], "region_ids": ["region:room"]}}
    if tetrahedra10 is not None:
        domain["coordinates"]["region_index"] = np.zeros(len(nodes), dtype=np.int32)
        domain["topology"] = {"tetrahedra": np.asarray(tetrahedra10)[:, :4], "tetrahedra10": tetrahedra10,
                              "region_index": np.zeros(len(tetrahedra10), dtype=np.int32)}
        domain["metadata"]["tetra_counts"] = [len(tetrahedra10)]
    return write_result(directory, frequencies_hz, port_ids,
                        [{"id": "acoustic:fem-nodal-pressure", "quantity": "fem_nodal_pressure",
                          "axes": ["excitation", "fem_node"], "values": pressure_at_nodes,
                          "metadata": {"node_counts": [len(nodes)]}, "target_id": "fem:room"}],
                        domains=[domain])
