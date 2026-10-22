"""THE POD SIDE'S RESULT: every finished frequency of a running blab solve, moved into raw/ as it lands (run-job.sh
runs this every second and once at the end, with the Python that runs blab: its shebang). acoustic_solve.solve_job
ships it with run-job.sh into every solve's stage. Standalone: the standard library, and numpy only to reduce.

    <python> collect_result.py plan <request.json> <raw> <solving request>     prints how many frequencies are left
    <python> collect_result.py collect <solving> <raw> [<keep_nodes.npy>]      prints raw/'s frequencies done
    <python> collect_result.py finish <solving> <raw> [<keep_nodes.npy>]       the last collect: raw/'s status

WHY (podd.job_contract `resume_from`, 2026-10-22): podd never ships what earlier attempts brought home back up to a
pod; a resumed attempt gets raw/manifest.json only, and podd lays the attempt's raw/ over the stage's at home. So
blab solves into <solving> (.solving/, never an output: blab refuses an existing directory, and its leftovers die
with the pod), and every file that leaves the pod in raw/ is final:

- raw/ is made from blab's first manifest (its frequencies = the request's, none done; every non-frequency file blab
  wrote: project snapshot, request, compiled system, domains) -- on a resumed attempt it is already home, and the
  pod's raw/ holds the manifest only;
- each frequency blab finished goes to raw/frequencies/<raw's index> (renumbered to raw's frequency list, blab's own
  tolerance), REDUCED to the kept nodes when the project keeps nodes (acoustic_solve.room.build's keep_nodes.npy: a
  room with field planes), then raw/manifest.json is rewritten (after the files: a manifest never names a file the
  pod does not have; a file the manifest does not name is solved again);
- finish: status "complete" when every frequency is in, else "interrupted"; a resumed attempt adds a `resumed`
  record.

REDUCTION (a room with field planes comes home small): keep_nodes.npy holds the COORDINATES (node, 3) of the nodes to
keep (the builder read them off the same mesh the solve used, build/room.msh). In the one fem_volume domain this
keeps those nodes and every tetrahedron whose ten nodes are all kept (renumbered), every per-node coordinate array
and per-cell topology array cut to match (raw/domains.*, once), and each frequency's quantities on that domain's
fem_node axis cut to the kept nodes. raw/ stays one valid Boundary Lab result of the sub-mesh
(acoustic_solve.boundary_lab_result reads it unchanged), marked "reduced_to_kept_nodes" in its manifest. A kept
coordinate that is no node of the result, a domain of several regions, another project, or a raw/ reduced
differently RAISES (exit non-zero: the job fails).
"""
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

COORDINATE_ROUNDING_M = 1e-7          # the same mesh file on both sides: coordinates agree far closer than this
MANIFEST = "manifest.json"
FREQUENCIES = "frequencies"


def refuse(message):
    sys.exit(f"collect_result (╯°□°)╯ {message}")


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def write_json(path, value):
    with open(path + ".tmp", "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def write_npz(path, arrays):
    import numpy as np
    with open(path + ".tmp", "wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(path + ".tmp", path)


def same_frequency(a, b):
    return abs(a - b) <= max(1e-5, abs(a) * 2e-6)                 # blab's own tolerance (headless.py)


def plan(request_path, raw, solving_request_path):
    """The request with only the frequencies raw/ lacks (all of them when no raw/ came: a first attempt)."""
    request = read_json(request_path)
    have = []
    if os.path.isfile(os.path.join(raw, MANIFEST)):
        manifest = read_json(os.path.join(raw, MANIFEST))
        have = [f for f, done in zip(manifest["frequencies_hz"], manifest["completion_mask"]) if done]
    elif os.path.exists(raw):
        refuse(f"{raw} without {MANIFEST}: podd ships raw/{MANIFEST} alone, nothing else of raw/")
    left = [f for f in request["frequencies_hz"] if not any(math.isclose(f, h, rel_tol=2e-6, abs_tol=1e-5)
                                                            for h in have)]
    write_json(solving_request_path, {**request, "frequencies_hz": left})
    return len(left)


class Reduction:
    """The kept nodes of the solving result's one fem_volume domain (from its FULL domains and keep_nodes.npy)."""

    def __init__(self, solving, keep_file):
        import numpy as np
        self.np = np
        manifest = read_json(os.path.join(solving, MANIFEST))
        self.domains_json = read_json(os.path.join(solving, manifest.get("domains_metadata_file", "domains.json")))
        self.arrays = dict(np.load(os.path.join(solving, manifest.get("domains_file", "domains.npz"))))
        volumes = [domain for domain in self.domains_json["domains"] if domain.get("kind") == "fem_volume"]
        if len(volumes) != 1:
            refuse(f"{solving}: {len(volumes)} fem_volume domains, not one")
        self.domain = volumes[0]
        if len(self.domain["metadata"].get("node_counts", [])) != 1:
            refuse(f"{solving}: a domain of {self.domain['metadata'].get('node_counts')} regions")
        points = self.arrays[self.domain["coordinates"]["points_m"]]
        self.node_count = len(points)
        index_of = {key: index for index, key in enumerate(self.keys_of(points))}
        wanted = np.load(keep_file)
        missing = [i for i, key in enumerate(self.keys_of(wanted)) if key not in index_of]
        if missing:
            refuse(f"{len(missing)} kept coordinates are no node of the result (the first "
                   f"{wanted[missing[0]].tolist()}): another mesh?")
        self.kept = np.zeros(self.node_count, dtype=bool)
        self.kept[[index_of[key] for key in self.keys_of(wanted)]] = True
        tetrahedra10 = self.arrays[self.domain["topology"]["tetrahedra10"]]
        self.cell_count = len(tetrahedra10)
        self.kept_cells = self.kept[tetrahedra10].all(axis=1)
        self.record = {"kept_nodes": int(self.kept.sum()), "of_nodes": self.node_count,
                       "kept_tetrahedra": int(self.kept_cells.sum()), "of_tetrahedra": self.cell_count}

    def keys_of(self, points):
        return [tuple(row) for row in self.np.rint(self.np.asarray(points) / COORDINATE_ROUNDING_M).astype(self.np.int64)]

    def domains(self):
        """(domains.json, domains.npz arrays) cut to the kept nodes and cells."""
        arrays, domains_json = dict(self.arrays), json.loads(json.dumps(self.domains_json))
        domain = next(d for d in domains_json["domains"] if d["id"] == self.domain["id"])
        renumber = self.np.cumsum(self.kept) - 1
        for key in domain["coordinates"].values():
            if arrays[key].shape[0] != self.node_count:
                refuse(f"coordinate {key} has {arrays[key].shape[0]} rows, not {self.node_count}")
            arrays[key] = arrays[key][self.kept]
        for name, key in domain["topology"].items():
            if arrays[key].shape[0] != self.cell_count:
                refuse(f"topology {name} has {arrays[key].shape[0]} rows, not {self.cell_count}")
            array = arrays[key][self.kept_cells]
            arrays[key] = renumber[array] if array.ndim == 2 else array          # node indices vs a per-cell label
        domain["metadata"]["node_counts"] = [self.record["kept_nodes"]]
        domain["metadata"]["tetra_counts"] = [self.record["kept_tetrahedra"]]
        return domains_json, arrays

    def frequency(self, meta, npz_path):
        """One frequency's (metadata, arrays) with its quantities on the domain's nodes cut to the kept ones."""
        values = dict(self.np.load(npz_path))
        for quantity in meta["quantities"]:
            if quantity.get("target_id") != self.domain["id"] or quantity["axes"][-1] != "fem_node":
                continue
            if values[quantity["key"]].shape[-1] != self.node_count:
                refuse(f"{npz_path} {quantity['key']}: {values[quantity['key']].shape}, not {self.node_count} nodes")
            values[quantity["key"]] = values[quantity["key"]][..., self.kept]
            quantity["shape"] = list(values[quantity["key"]].shape)
            if "node_counts" in quantity.get("metadata", {}):
                quantity["metadata"]["node_counts"] = [self.record["kept_nodes"]]
        return meta, values


def start_raw(solving, raw, solved, reduction):
    """raw/ from blab's first manifest: its frequencies, none done, every non-frequency file (domains reduced)."""
    os.makedirs(os.path.join(raw, FREQUENCIES))
    for name in os.listdir(solving):
        if name in (MANIFEST, FREQUENCIES) or name.startswith("."):
            continue
        shutil.copyfile(os.path.join(solving, name), os.path.join(raw, name))
    manifest = {**solved, "status": "running", "completion_mask": [False] * len(solved["frequencies_hz"]),
                "results": [None] * len(solved["frequencies_hz"]), "engine_runs": []}
    if reduction:
        domains_json, arrays = reduction.domains()
        write_json(os.path.join(raw, solved.get("domains_metadata_file", "domains.json")), domains_json)
        write_npz(os.path.join(raw, solved.get("domains_file", "domains.npz")), arrays)
        manifest["reduced_to_kept_nodes"] = reduction.record
    return manifest


def collect(solving, raw, keep_file=None):
    """Move every frequency blab finished into raw/ (module docstring); raw/'s manifest, or None before blab wrote
    its first."""
    if not os.path.isfile(os.path.join(solving, MANIFEST)):
        return read_json(os.path.join(raw, MANIFEST)) if os.path.isfile(os.path.join(raw, MANIFEST)) else None
    solved = read_json(os.path.join(solving, MANIFEST))
    started = os.path.isfile(os.path.join(raw, MANIFEST))
    manifest = read_json(os.path.join(raw, MANIFEST)) if started else solved
    if manifest["project_sha256"] != solved["project_sha256"]:
        refuse(f"{solving} solved another project ({solved['project_sha256']} against raw/'s "
               f"{manifest['project_sha256']})")
    finished = []                                                      # (raw's index, blab's result), not yet in raw/
    for done, result in zip(solved["completion_mask"], solved["results"]):
        if not done or result is None:
            continue
        matches = [i for i, f in enumerate(manifest["frequencies_hz"]) if same_frequency(result["freq_hz"], f)]
        if not matches:
            refuse(f"{solving} solved {result['freq_hz']} Hz, which raw/ never asked for")
        if not (started and manifest["completion_mask"][matches[0]]):
            finished.append((matches[0], result))
    if started and not finished:                                       # most passes: nothing new
        return manifest
    # numpy and the kept nodes only now: this runs every second
    reduction = Reduction(solving, keep_file) if keep_file and os.path.isfile(keep_file) else None
    if not started:
        manifest = start_raw(solving, raw, solved, reduction)
    elif manifest.get("reduced_to_kept_nodes") != (reduction.record if reduction else None):
        refuse(f"raw/ is reduced as {manifest.get('reduced_to_kept_nodes')}, this attempt as "
               f"{reduction.record if reduction else None}: never one result of two meshes")
    os.makedirs(os.path.join(raw, FREQUENCIES), exist_ok=True)
    for index, result in finished:
        stem = f"{index:06d}"
        meta = read_json(os.path.join(solving, result["metadata_file"]))
        npz_path = os.path.join(solving, result["arrays_file"])
        if reduction:
            meta, values = reduction.frequency(meta, npz_path)
            write_npz(os.path.join(raw, FREQUENCIES, f"{stem}.npz"), values)
            os.remove(npz_path)
        else:
            os.replace(npz_path, os.path.join(raw, FREQUENCIES, f"{stem}.npz"))
        meta["arrays_file"] = f"{stem}.npz"
        write_json(os.path.join(raw, FREQUENCIES, f"{stem}.json"), meta)
        manifest["completion_mask"][index] = True
        manifest["results"][index] = {"freq_hz": result["freq_hz"], "metadata_file": f"{FREQUENCIES}/{stem}.json",
                                      "arrays_file": f"{FREQUENCIES}/{stem}.npz"}
    for run in solved.get("engine_runs") or []:
        if run not in manifest["engine_runs"]:
            manifest["engine_runs"].append(run)
    write_json(os.path.join(raw, MANIFEST), manifest)
    return manifest


def finish(solving, raw, keep_file=None, resumed=False):
    """The last collect, then raw/'s status (and a resumed attempt's record)."""
    manifest = collect(solving, raw, keep_file)
    if manifest is None:
        return None
    complete = all(manifest["completion_mask"])
    manifest["status"] = "complete" if complete else "interrupted"
    if complete:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if resumed:
        solved = read_json(os.path.join(solving, MANIFEST)) if os.path.isfile(os.path.join(solving, MANIFEST)) else {}
        manifest.setdefault("resumed", []).append({"started_at_utc": solved.get("started_at_utc"),
                                                   "status": solved.get("status"),
                                                   "frequencies_folded": sum(solved.get("completion_mask") or [])})
    write_json(os.path.join(raw, MANIFEST), manifest)
    return manifest


def frequencies_done(manifest):
    return sum(manifest["completion_mask"]) if manifest else 0


if __name__ == "__main__":
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "plan":
        print(plan(*arguments))
    elif command == "collect":
        print(frequencies_done(collect(*arguments)))
    elif command == "finish":
        resumed = "--resumed" in arguments
        print(frequencies_done(finish(*[a for a in arguments if a != "--resumed"], resumed=resumed)))
    else:
        refuse(f"no command {command!r}: plan | collect | finish")
