"""REDUCE A ROOM RESULT TO THE NODES ITS READERS NEED, on the pod, before it comes home (acoustic_solve.room.build
ships this file into build/ beside build/keep_nodes.npy when the room has field planes; run-job.sh runs it with the
Python that runs blab, after a complete solve). Standalone: numpy and the standard library only, so it runs wherever
blab does.

    <python> reduce_nodes.py <raw dir> <keep_nodes.npy>

keep_nodes.npy holds the COORDINATES (node, 3) of the nodes to keep (the builder read them off the same mesh the
solve used, build/room.msh). In raw/'s one fem_volume domain this keeps those nodes and every tetrahedron whose ten
nodes are all kept (renumbered), every per-node coordinate array and per-cell topology array cut to match, and every
frequency's quantities on that domain's fem_node axis cut to the kept nodes. raw/ stays one valid Boundary Lab result
of the sub-mesh (acoustic_solve.boundary_lab_result reads it unchanged), marked "reduced_to_kept_nodes" in its
manifest; a result already marked is left alone. A kept coordinate that is no node of the result, an incomplete
result, or a domain of several regions RAISES (exit non-zero: the job fails, and the full raw/ comes home).
"""
import json
import os
import sys

import numpy as np

COORDINATE_ROUNDING_M = 1e-7          # the same mesh file on both sides: coordinates agree far closer than this


def write_json(path, value):
    with open(path + ".tmp", "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def write_npz(path, arrays):
    with open(path + ".tmp", "wb") as stream:
        np.savez(stream, **arrays)
    os.replace(path + ".tmp", path)


def keys_of(points):
    return [tuple(row) for row in np.rint(np.asarray(points) / COORDINATE_ROUNDING_M).astype(np.int64)]


def reduce(raw, keep_file):
    manifest_path = os.path.join(raw, "manifest.json")
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest.get("reduced_to_kept_nodes"):
        print(f"reduce_nodes: {raw} is already reduced ({manifest['reduced_to_kept_nodes']})")
        return manifest["reduced_to_kept_nodes"]
    if manifest.get("status") != "complete" or not all(manifest["completion_mask"]):
        sys.exit(f"reduce_nodes (╯°□°)╯ {raw} is not a complete result (status {manifest.get('status')})")
    domains_json = os.path.join(raw, manifest.get("domains_metadata_file", "domains.json"))
    domains_npz = os.path.join(raw, manifest.get("domains_file", "domains.npz"))
    with open(domains_json) as stream:
        domains = json.load(stream)
    arrays = dict(np.load(domains_npz))
    volumes = [domain for domain in domains["domains"] if domain.get("kind") == "fem_volume"]
    if len(volumes) != 1:
        sys.exit(f"reduce_nodes (╯°□°)╯ {raw}: {len(volumes)} fem_volume domains, not one")
    domain = volumes[0]
    if len(domain["metadata"].get("node_counts", [])) != 1:
        sys.exit(f"reduce_nodes (╯°□°)╯ {raw}: a domain of {domain['metadata'].get('node_counts')} regions")
    points = arrays[domain["coordinates"]["points_m"]]
    node_count = len(points)
    index_of = {key: index for index, key in enumerate(keys_of(points))}
    wanted = np.load(keep_file)
    missing = [i for i, key in enumerate(keys_of(wanted)) if key not in index_of]
    if missing:
        sys.exit(f"reduce_nodes (╯°□°)╯ {len(missing)} kept coordinates are no node of the result (the first "
                 f"{wanted[missing[0]].tolist()}): another mesh?")
    kept = np.zeros(node_count, dtype=bool)
    kept[[index_of[key] for key in keys_of(wanted)]] = True
    renumber = np.cumsum(kept) - 1
    tetrahedra10 = arrays[domain["topology"]["tetrahedra10"]]
    cell_count = len(tetrahedra10)
    kept_cells = kept[tetrahedra10].all(axis=1)
    for key in domain["coordinates"].values():
        if arrays[key].shape[0] != node_count:
            sys.exit(f"reduce_nodes (╯°□°)╯ coordinate {key} has {arrays[key].shape[0]} rows, not {node_count}")
        arrays[key] = arrays[key][kept]
    for name, key in domain["topology"].items():
        if arrays[key].shape[0] != cell_count:
            sys.exit(f"reduce_nodes (╯°□°)╯ topology {name} has {arrays[key].shape[0]} rows, not {cell_count}")
        array = arrays[key][kept_cells]
        arrays[key] = renumber[array] if array.ndim == 2 else array          # node indices vs a per-cell label
    domain["metadata"]["node_counts"] = [int(kept.sum())]
    domain["metadata"]["tetra_counts"] = [int(kept_cells.sum())]
    for result in manifest["results"]:
        meta_path = os.path.join(raw, result["metadata_file"])
        with open(meta_path) as stream:
            meta = json.load(stream)
        on_nodes = [q for q in meta["quantities"] if q.get("target_id") == domain["id"] and q["axes"][-1] == "fem_node"]
        if not on_nodes:
            continue
        npz_path = os.path.join(raw, result["arrays_file"])
        values = dict(np.load(npz_path))
        for quantity in on_nodes:
            if values[quantity["key"]].shape[-1] != node_count:
                sys.exit(f"reduce_nodes (╯°□°)╯ {npz_path} {quantity['key']}: {values[quantity['key']].shape}, not "
                         f"{node_count} nodes")
            values[quantity["key"]] = values[quantity["key"]][..., kept]
            quantity["shape"] = list(values[quantity["key"]].shape)
            if "node_counts" in quantity.get("metadata", {}):
                quantity["metadata"]["node_counts"] = [int(kept.sum())]
        write_npz(npz_path, values)
        write_json(meta_path, meta)
    write_npz(domains_npz, arrays)
    write_json(domains_json, domains)
    record = {"kept_nodes": int(kept.sum()), "of_nodes": node_count, "kept_tetrahedra": int(kept_cells.sum()),
              "of_tetrahedra": cell_count}
    manifest["reduced_to_kept_nodes"] = record
    write_json(manifest_path, manifest)
    print(f"reduce_nodes: {raw} reduced: {record}")
    return record


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    reduce(sys.argv[1], sys.argv[2])
