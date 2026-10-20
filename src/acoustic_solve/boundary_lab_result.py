"""THE ONE READER of a `blab project solve` result directory (what run-job.sh brings home as raw/).

    from acoustic_solve.boundary_lab_result import BoundaryLabResult
    result = BoundaryLabResult(raw_directory)                    # complete, or it raises
    result.frequencies_hz, result.excitation_port_ids, result.phasor_convention
    pressure = result.quantity("acoustic:pressure:horizontal-polar")          # (frequency, port, ...)
    nodal = result.quantity_of_kind("fem_nodal_pressure", axes=["excitation", "fem_node"])
    summed = result.weighted_sum("acoustic:pressure:horizontal-polar", result.weighted_terms(resolved_terms))

THE FORMAT (Boundary Lab's HeadlessResultWriter, ~/boundary-lab/src/blab/headless.py at the pinned commit; schema
`boundary-lab-headless-result`): manifest.json -- status, completion_mask, frequencies_hz, excitation_port_ids,
phasor_convention, project_sha256, engine_runs, results [{freq_hz, metadata_file, arrays_file}] --; domains.json +
domains.npz (the observation and mesh domains, each {id, kind, coordinates: {name: npz key}, metadata}); per frequency
frequencies/<index>.json ({freq_hz, excitation_port_ids, quantities [{id, quantity, key, axes, metadata}],
diagnostics}) + its .npz. Every quantity's first axis is the excitation port.

What it refuses, loudly: no manifest; a result that is not complete (unless asked for the manifest only); frequencies
out of order; a frequency whose ports or phasor convention differ from the manifest's; a quantity present at some
frequencies and not others, or on other axes than asked; a WEIGHTED sum of a result that is not exp(+i omega t).
Nothing is read before it is asked for: the manifest at construction, the rest on first use (an index over two
thousand rows reads manifests only).
"""
import json
from functools import cached_property
from pathlib import Path

import numpy as np

from acoustic_solve import weighted_sums

MANIFEST_FILE = "manifest.json"
COMPLETE = "complete"
PORT_ID_PREFIX = "excitation:"
FREQUENCY_MATCH_TOLERANCE_HZ = 1e-6


class BoundaryLabResultError(RuntimeError):
    """A result directory that is not what a reader asked of it."""


def read_manifest(directory):
    path = Path(directory) / MANIFEST_FILE
    if not path.is_file():
        raise BoundaryLabResultError(f"(╯°□°)╯ {path}: no manifest -- not a `blab project solve` result directory")
    return json.loads(path.read_text())


class BoundaryLabResult:
    """One result directory. require_complete=False reads a stopped one's manifest (its status says how far it got)."""

    def __init__(self, directory, require_complete=True):
        self.directory = Path(directory)
        self.manifest = read_manifest(self.directory)
        if require_complete and not self.is_complete:
            raise BoundaryLabResultError(
                f"(╯°□°)╯ {self.directory}: status {self.manifest.get('status')!r}, "
                f"{sum(self.manifest.get('completion_mask') or [])}/{len(self.manifest.get('completion_mask') or [])} "
                f"frequencies: not a complete result")
        self.complete = require_complete

    # -- the manifest ----------------------------------------------------------------------------------------------

    @property
    def status(self):
        return self.manifest.get("status")

    @property
    def is_complete(self):
        mask = self.manifest.get("completion_mask") or []
        return self.status == COMPLETE and bool(mask) and all(mask)

    @property
    def project_sha256(self):
        return self.manifest.get("project_sha256")

    @property
    def phasor_convention(self):
        return self.manifest["phasor_convention"]

    @property
    def excitation_port_ids(self):
        return list(self.manifest["excitation_port_ids"])

    def port_names(self):
        """The ports by the names a weighted sum uses: each id without "excitation:"."""
        names = []
        for port_id in self.excitation_port_ids:
            if not port_id.startswith(PORT_ID_PREFIX):
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: port id {port_id!r} lacks {PORT_ID_PREFIX!r}")
            names.append(port_id[len(PORT_ID_PREFIX):])
        return names

    @cached_property
    def frequencies_hz(self):
        self._require_complete_for("the frequencies")
        frequencies = np.array(self.manifest["frequencies_hz"], dtype=float)
        if len(frequencies) > 1 and np.any(np.diff(frequencies) <= 0):
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: frequencies not strictly ascending")
        return frequencies

    def frequency_indices(self, frequencies_hz):
        """The index of each asked frequency on this result's grid; one that is not on it RAISES."""
        indices = []
        for frequency in np.atleast_1d(frequencies_hz):
            match = np.flatnonzero(np.isclose(self.frequencies_hz, frequency, rtol=0,
                                              atol=FREQUENCY_MATCH_TOLERANCE_HZ))
            if len(match) != 1:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: {frequency} Hz is not on its grid")
            indices.append(int(match[0]))
        return indices

    def _require_complete_for(self, what):
        if not self.complete:
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: {what} of a result read without requiring it "
                                         f"complete (status {self.status!r})")

    # -- the domains -------------------------------------------------------------------------------------------------

    @cached_property
    def domains(self):
        return json.loads((self.directory / self.manifest.get("domains_metadata_file", "domains.json")).read_text())[
            "domains"]

    @cached_property
    def _domain_arrays(self):
        return np.load(self.directory / self.manifest.get("domains_file", "domains.npz"))

    def domain(self, domain_id):
        found = [domain for domain in self.domains if domain["id"] == domain_id]
        if len(found) != 1:
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: {len(found)} result domains {domain_id!r}")
        return found[0]

    def domains_of_kind(self, kind):
        return [domain for domain in self.domains if domain.get("kind") == kind]

    def domain_coordinate(self, domain, coordinate):
        """One coordinate array of one domain (a domain dict, or its id)."""
        domain = self.domain(domain) if isinstance(domain, str) else domain
        if coordinate not in domain["coordinates"]:
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: domain {domain['id']} has no coordinate "
                                         f"{coordinate!r} ({sorted(domain['coordinates'])})")
        return self._domain_arrays[domain["coordinates"][coordinate]]

    def polar_angles_by_plane(self):
        """{plane: angles in degrees} of every polar_observation domain (Boundary Lab: "horizontal", "vertical")."""
        return {domain["metadata"]["plane"]: np.asarray(self.domain_coordinate(domain, "angle_deg"), dtype=float)
                for domain in self.domains_of_kind("polar_observation")}

    def radiator_effective_area_by_component(self):
        """{component id: effective area m^2} of the "components:radiators" domain; {} when the result has none."""
        if not any(domain["id"] == "components:radiators" for domain in self.domains):
            return {}
        ids = [str(value) for value in self.domain_coordinate("components:radiators", "component_id")]
        return dict(zip(ids, map(float, self.domain_coordinate("components:radiators", "effective_area_m2"))))

    # -- the frequencies ---------------------------------------------------------------------------------------------

    @cached_property
    def frequency_metadata(self):
        """Every frequency's metadata, in the manifest's order, each checked against the manifest."""
        self._require_complete_for("the per-frequency metadata")
        metadata = []
        for index, entry in enumerate(self.manifest["results"]):
            meta = json.loads((self.directory / entry["metadata_file"]).read_text())
            if meta.get("excitation_port_ids") != self.excitation_port_ids:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}/{entry['metadata_file']}: ports "
                                             f"{meta.get('excitation_port_ids')}, the manifest's "
                                             f"{self.excitation_port_ids}")
            if abs(float(meta["freq_hz"]) - self.frequencies_hz[index]) > FREQUENCY_MATCH_TOLERANCE_HZ:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}/{entry['metadata_file']}: {meta['freq_hz']} "
                                             f"Hz where the manifest has {self.frequencies_hz[index]} Hz")
            convention = (meta.get("diagnostics") or {}).get("phasor_convention", self.phasor_convention)
            if convention != self.phasor_convention:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}/{entry['metadata_file']}: phasor convention "
                                             f"{convention!r}, the manifest's {self.phasor_convention!r}")
            metadata.append(meta)
        return metadata

    def diagnostics(self):
        return [meta.get("diagnostics") or {} for meta in self.frequency_metadata]

    def _frequency_arrays(self, index):
        return np.load(self.directory / self.manifest["results"][index]["arrays_file"])

    def _quantity_records(self, select, describe):
        records = [next((q for q in meta["quantities"] if select(q)), None) for meta in self.frequency_metadata]
        present = [record is not None for record in records]
        if not any(present):
            return None
        if not all(present):
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: {describe} at {sum(present)} of "
                                         f"{len(present)} frequencies")
        return records

    def _stacked(self, records, describe, axes, last_axis_indices=None):
        """Every frequency's array stacked; with last_axis_indices, only those entries of the last axis are kept, one
        frequency at a time (a room's seat nodes out of its every FEM node: never the whole field in memory)."""
        if axes is not None and any(record["axes"] != list(axes) for record in records):
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: {describe} on axes "
                                         f"{sorted({tuple(r['axes']) for r in records})}, not {list(axes)}")
        arrays = []
        for index, record in enumerate(records):
            array = self._frequency_arrays(index)[record["key"]]
            arrays.append(array if last_axis_indices is None else array[..., last_axis_indices])
        return np.stack(arrays)

    def has_quantity(self, quantity_id):
        return self._quantity_records(lambda q: q["id"] == quantity_id, quantity_id) is not None

    def quantity_metadata(self, quantity_id):
        """The first frequency's record of a quantity (its metadata: represented areas, ...); None when absent."""
        records = self._quantity_records(lambda q: q["id"] == quantity_id, quantity_id)
        return None if records is None else records[0]

    def quantity(self, quantity_id, axes=None, last_axis_indices=None):
        """(frequency, port, ...) values of one quantity id; None when the result has none. last_axis_indices keeps
        only those entries of the last axis (picked per frequency)."""
        records = self._quantity_records(lambda q: q["id"] == quantity_id, quantity_id)
        return None if records is None else self._stacked(records, quantity_id, axes, last_axis_indices)

    def quantity_of_kind(self, kind, axes=None, last_axis_indices=None):
        """(frequency, port, ...) values of the one quantity of this kind (its `quantity` field); RAISES if absent.
        last_axis_indices as for quantity()."""
        records = self._quantity_records(lambda q: q.get("quantity") == kind, f"a {kind} quantity")
        if records is None:
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: no {kind} quantity (was it retained? the request's "
                                         f"`retain`)")
        return self._stacked(records, kind, axes, last_axis_indices)

    # -- weighted sums of ports -------------------------------------------------------------------------------------

    def weighted_terms(self, resolved_terms):
        """[(port index, weight per frequency or None for a raw term)] of a resolved sum ([{"port", "chain"}],
        weighted_sums). A weighted term on a result that is not exp(+i omega t) RAISES: its weights are written in
        that convention."""
        names = self.port_names()
        terms = []
        for term in resolved_terms:
            if term["port"] not in names:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: no port {term['port']!r} ({names})")
            weights = [weighted_sums.chain_weight(term["chain"], float(f)) for f in self.frequencies_hz]
            if weights[0] is not None and self.phasor_convention != weighted_sums.POSITIVE_TIME:
                raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: a weighted sum is written in "
                                             f"{weighted_sums.POSITIVE_TIME} and this result is in "
                                             f"{self.phasor_convention}")
            terms.append((names.index(term["port"]), None if weights[0] is None else np.array(weights)))
        return terms

    def weighted_sum(self, quantity_id, weighted_terms):
        """(frequency, ...) of one quantity summed over a sum's ports, each times its weight (a raw term added as it
        is, bit for bit); None when the result lacks the quantity."""
        values = self.quantity(quantity_id)
        if values is None:
            return None
        total = None
        for port_index, weights in weighted_terms:
            term = values[:, port_index]
            if weights is not None:
                term = weights.reshape((-1,) + (1,) * (term.ndim - 1)) * term
            total = term if total is None else total + term
        return total

    # -- a transducer ------------------------------------------------------------------------------------------------

    def transducer_of_port(self, port_index):
        """(velocity, current, reference volts) per frequency of the electrodynamic transducer this port drives, or
        None when it drives none (an ideal-velocity port)."""
        velocity = self.quantity("mechanical:diaphragm-velocity")
        current = self.quantity("electrical:voice-coil-current")
        if velocity is None or current is None or not np.abs(velocity[:, port_index]).max() > 0.0:
            return None
        column = int(np.argmax(np.abs(velocity[0, port_index])))
        volts = {diagnostics.get("transducer_reference_voltage_v") for diagnostics in self.diagnostics()}
        if len(volts) != 1 or None in volts:
            raise BoundaryLabResultError(f"(╯°□°)╯ {self.directory}: transducer reference voltage {volts}")
        return velocity[:, port_index, column], current[:, port_index, column], volts.pop()
