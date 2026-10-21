"""A SOLVED ROOM, read and measured: the seat pressures of every port, the free field of the same box, the rigid-room
modal sum that validates a solve, and the metrics of a room gain at a seat. What a study mixes from the ports (its
patterns, crossovers, control laws) is the study's own, a weighted sum of ports (acoustic_solve.weighted_sums); this
module is what every room study measures with.

    solve = load_room_solve(stage)                         # <stage>/raw (complete), seat_points.json
    solve = join_frequency_chunks([stage, ...])            # one case solved in frequency chunks
    alone = free_field(free_stage, probe, solve.frequencies_hz, solve.point_names)   # {port id: (f, point)}

Sign: a port's +1 m/s is along the FEM domain's outward normal, INTO the box (monitoring's
PLANS/2026-10-14-room-cardioid-sub.md: the pilot's phase is the modal sum's + 180 deg); in the free-field BEM it is
along the body's outward normal. Each solve's ports share their sign, so a weighted mix stands and |room / free field|
does not see it. Convention: exp(+i omega t) (the result's manifest; a delay tau is exp(-i omega tau), the free-field
Green function exp(-i k r) / (4 pi r)).

THE ROOM GAIN of a mix at a point: its room pressure over its own free-field pressure at the same point (the same box
alone in free space, one exterior BEM, acoustic_solve.room.build freefield; the same weights): what the room does to
the mix, with what the mix does by itself divided out, i.e. the response an ideal EQ of the source's anechoic curve
there would leave.

METRICS of a room gain per band (room_gain_band_metrics):
  sd_db            standard deviation of the room gain in dB at the seat, unsmoothed
  sd_third_db      the same after 1/3-octave power smoothing (smoothed over the whole grid, then cut to the band)
  spread_db        mean over frequency of the standard deviation across the seat grid (seat-to-seat variation)
  max_minus_min_db at the seat
  top_peaks        the TOP_PEAKS most prominent local maxima of the seat gain with >= 3 dB prominence: frequency,
                   height, half-power Q, T60 = 2.2 Q / f
DECAY per 1/3-octave band (DECAY_BANDS_HZ), of a seat PRESSURE (the room's, nothing divided): the impulse response by
inverse FFT of the complex seat response (zero outside the solved grid), band-passed by a zero-phase order-3
Butterworth band magnitude, Schroeder's backward integral from the energy peak; EDT = 6 x the 0..-10 dB fit, T20 = 3 x
the -5..-25 dB fit. bt60 = bandwidth x T20 says how far the band filter's own response shapes the decay. Checked
2026-10-15 on single modes of known T60 (0.3 / 0.6 s, 15-300 Hz at 0.5 Hz, 9 ms delay): T20 within 2 % at
bt60 >= 5.5 (50, 80, 125, 200 Hz), 15-25 % LOW at bt60 <= 3.7 (20, 25, 31.5 Hz at 0.6 s); EDT within 10 % at
bt60 >= 5.5 (tests/test_room_analysis.py keeps the check). Bands under bt60 5 are reported and not trusted.

VALIDATION (modal_sum): a solve of a rigid-walled room (bulk loss only) against the modal sum of a rigid shoebox with
the same bulk loss eta, a point monopole of the given volume velocity at the given point (Kuttruff, Room Acoustics,
ch. 3; the solver's operator K - k^2 M + i eta k^2 M):
    p(r) = i omega rho Q sum_n psi_n(r) psi_n(r0) / (V Lambda_n (k_n^2 - k^2 (1 - i eta)))
    psi_n = prod cos(n_i pi x_i / L_i), Lambda_n = prod (1 if n_i == 0 else 1/2).
The source box is a rigid obstacle the modal sum does not have, so a small offset is expected.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from acoustic_solve.boundary_lab_result import BoundaryLabResult
from acoustic_solve.room.spec import DENSITY_KG_PER_M3, FREE_FIELD_SPECS_FILE, POINTS_FILE, SOUND_SPEED_M_PER_S

MODAL_SUM_CUTOFF_HZ = 1500.0
TOP_PEAKS = 3
PEAK_PROMINENCE_DB = 3.0
DECAY_BANDS_HZ = (20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0)
DECAY_SAMPLE_RATE_HZ = 2000.0
NODE_ON_POINT_M = 1e-6               # a seat point is an embedded mesh node: this close, or it was not embedded


@dataclass
class RoomSolve:
    """One room's solve at its seat points: pressure (frequency, port, point), the ports by id."""
    frequencies_hz: np.ndarray
    port_ids: list
    point_names: list
    points_m: np.ndarray
    pressure: np.ndarray
    diagnostics: list

    def port(self, port_id):
        """(frequency, point) of one port driven alone."""
        if port_id not in self.port_ids:
            raise RuntimeError(f"(╯°□°)╯ no port {port_id!r} in this solve ({self.port_ids})")
        return self.pressure[:, self.port_ids.index(port_id)]


def room_result(stage):
    """(<stage>/raw complete, its FEM nodes (node, 3)): a room solve has one fem_volume domain."""
    stage = Path(stage)
    result = BoundaryLabResult(stage / "raw")
    fem = result.domains_of_kind("fem_volume")
    if len(fem) != 1:
        raise RuntimeError(f"(╯°□°)╯ {stage}/raw has {len(fem)} fem_volume domains, not one: not a room solve")
    return result, result.domain_coordinate(fem[0], "points_m")


def node_indices(nodes, points, what):
    """The FEM node at each point ((point, 3)): each one must BE a node (embedded, or a mesh node itself)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    distance, index = cKDTree(nodes).query(points)
    far = np.flatnonzero(distance > NODE_ON_POINT_M)
    if far.size:
        raise RuntimeError(f"(╯°□°)╯ {len(far)} {what}s are off every node (the first, {points[far[0]].tolist()}, by "
                           f"{distance[far[0]]:.3g} m): not embedded")
    return [int(i) for i in index]


def nodal_pressure(result, indices):
    """(frequency, port, node) of the retained fem_nodal_pressure at those nodes, complex128 (mixes are computed in
    complex128)."""
    return result.quantity_of_kind("fem_nodal_pressure", axes=["excitation", "fem_node"],
                                   last_axis_indices=indices).astype(complex)


def load_room_solve(stage):
    """<stage>/raw (complete) at the seat points of <stage>/seat_points.json: each one must be a mesh node."""
    result, nodes = room_result(stage)
    points = json.loads((Path(stage) / POINTS_FILE).read_text())["seat_points_m"]
    names = list(points)
    at_seats = nodal_pressure(result, node_indices(nodes, [points[n] for n in names], "seat point"))
    return RoomSolve(result.frequencies_hz, result.excitation_port_ids, names, np.array([points[n] for n in names]),
                     at_seats, result.diagnostics())


def join_frequency_chunks(stages):
    """One case solved as frequency chunks (one stage each) -> one RoomSolve, sorted by frequency. Chunks that differ
    in ports or seat points, or overlap in frequency, RAISE."""
    parts = [load_room_solve(stage) for stage in stages]
    first = parts[0]
    for part in parts:
        if part.port_ids != first.port_ids or part.point_names != first.point_names:
            raise RuntimeError("(╯°□°)╯ the chunks of one case differ in ports or seat points")
    frequencies = np.concatenate([part.frequencies_hz for part in parts])
    order = np.argsort(frequencies)
    if np.any(np.diff(frequencies[order]) <= 0):
        raise RuntimeError("(╯°□°)╯ the chunks of one case overlap in frequency")
    return RoomSolve(frequencies[order], first.port_ids, first.point_names, first.points_m,
                     np.concatenate([part.pressure for part in parts])[order],
                     [d for part in parts for d in part.diagnostics])


def free_field(free_stage, probe, frequencies, names=None):
    """The box alone in free space: {port id: pressure (f, point)} of one probe of the free-field stage
    (acoustic_solve.room.build free_field), at the given frequencies (each one of the stage's grid; a room solved in
    frequency chunks takes its own). names, when given, must be the probe's points in order (the room's seat points)."""
    free_stage = Path(free_stage)
    order = json.loads((free_stage / FREE_FIELD_SPECS_FILE).read_text())["probes"].get(probe)
    if order is None or (names is not None and order != names):
        raise RuntimeError(f"(╯°□°)╯ {free_stage}: probe {probe!r} has points {order}, the room has {names}")
    result = BoundaryLabResult(free_stage / "raw")
    pressure = result.quantity(f"acoustic:pressure:probe:{probe}", axes=["excitation", "observation"])
    if pressure is None:
        raise RuntimeError(f"(╯°□°)╯ {free_stage}/raw: no probe {probe!r}")
    pressure = pressure[result.frequency_indices(frequencies)].astype(complex)   # mixes are computed in complex128
    return {port: pressure[:, index] for index, port in enumerate(result.excitation_port_ids)}


def modal_sum(frequencies, dimensions_m, bulk_loss_factor, source_point_m, volume_velocity_m3_per_s, points_m):
    """(pressure (f, point), modes in the sum) of a point monopole in a rigid shoebox with bulk loss (the docstring)."""
    lx, ly, lz = dimensions_m
    eta = bulk_loss_factor
    n_max = [int(2 * MODAL_SUM_CUTOFF_HZ * length / SOUND_SPEED_M_PER_S) + 1 for length in (lx, ly, lz)]
    nx, ny, nz = np.meshgrid(*[np.arange(n) for n in n_max], indexing="ij")
    nx, ny, nz = nx.ravel(), ny.ravel(), nz.ravel()
    kn2 = (np.pi * nx / lx) ** 2 + (np.pi * ny / ly) ** 2 + (np.pi * nz / lz) ** 2
    keep = kn2 <= (2 * np.pi * MODAL_SUM_CUTOFF_HZ / SOUND_SPEED_M_PER_S) ** 2
    nx, ny, nz, kn2 = nx[keep], ny[keep], nz[keep], kn2[keep]
    norm = np.where(nx > 0, 0.5, 1.0) * np.where(ny > 0, 0.5, 1.0) * np.where(nz > 0, 0.5, 1.0) * lx * ly * lz

    def psi(p):
        return (np.cos(nx * np.pi * p[..., 0:1] / lx) * np.cos(ny * np.pi * p[..., 1:2] / ly)
                * np.cos(nz * np.pi * p[..., 2:3] / lz))

    source = psi(np.array(source_point_m, dtype=float)[None, :])[0]
    seat = psi(np.asarray(points_m, dtype=float))                   # (point, mode)
    frequencies = np.asarray(frequencies, dtype=float)
    k = 2 * np.pi * frequencies / SOUND_SPEED_M_PER_S
    denominator = kn2[None, :] - (k ** 2)[:, None] * (1 - 1j * eta)   # (f, mode)
    p = 1j * 2 * np.pi * frequencies[:, None] * DENSITY_KG_PER_M3 * volume_velocity_m3_per_s * \
        ((source / norm)[None, :] / denominator) @ seat.T
    return p, int(keep.sum())


def analytic_modes(dimensions_m, below_hz, highest_order=11):
    """[(frequency rounded to 0.1 Hz, (a, b, c))] of a rigid shoebox up to below_hz, sorted."""
    lx, ly, lz = dimensions_m
    modes = []
    for a in range(highest_order + 1):
        for b in range(highest_order + 1):
            for c in range(highest_order + 1):
                f = SOUND_SPEED_M_PER_S / 2 * math.sqrt((a / lx) ** 2 + (b / ly) ** 2 + (c / lz) ** 2)
                if 0 < f <= below_hz:
                    modes.append((round(f, 1), (a, b, c)))
    return sorted(modes)


T60_DECAY_NEPERS = 3 * math.log(10)   # 60 dB of energy = 6.908 nepers of amplitude: T60 = this / delta


def modal_decay_rate(dimensions_m, mode, mean_admittance):
    """delta (1/s, the amplitude's e-folding rate) of a rigid shoebox's mode (a, b, c) when every surface has the same
    small real normalised admittance `mean_admittance` (beta: rho c / Z_wall). Calafa et al. 2026, Table 1 group 1
    (references/README.md): along an axis of length l, both walls beta, gamma = beta k l, the root
    q = (n + sqrt(n^2 + 8 i gamma / pi^2)) / 2 gives, to first order, k_axis = (pi / l) q with
    Im(k_axis^2) = 4 beta k / l for n > 0 and 2 beta k / l for n = 0 (k = omega / c); summed over the axes
    (k^2 = sum k_axis^2, eq. 3) and divided by 2 k: Im k = beta sum_axis (2 if n > 0 else 1) / l, delta = c Im k.
    T60 = T60_DECAY_NEPERS / delta. The rho c tiles (spec.rho_c_tiles) of fraction f are beta = f for every mode
    with fewer half-waves along each surface than it has cells."""
    return SOUND_SPEED_M_PER_S * mean_admittance * sum((2.0 if n > 0 else 1.0) / length
                                                       for n, length in zip(mode, dimensions_m))


def modal_t60_s(dimensions_m, mode, mean_admittance):
    return T60_DECAY_NEPERS / modal_decay_rate(dimensions_m, mode, mean_admittance)


def admittance_for_modal_t60(dimensions_m, mode, t60_s):
    """The uniform beta that gives `mode` a T60 of t60_s (modal_decay_rate is linear in beta)."""
    return T60_DECAY_NEPERS / (t60_s * modal_decay_rate(dimensions_m, mode, 1.0))


def third_octave_smoothed(frequencies, gain_db):
    """1/3-octave power smoothing along the first axis (frequency); any further axes (points) each on its own."""
    power = 10 ** (gain_db / 10)
    out = np.empty_like(gain_db)
    for i, f in enumerate(frequencies):
        band = (frequencies >= f * 2 ** (-1 / 6)) & (frequencies <= f * 2 ** (1 / 6))
        out[i] = 10 * np.log10(power[band].mean(axis=0))
    return out


def peaks(frequencies, gain_db):
    """Local maxima with >= PEAK_PROMINENCE_DB prominence; Q from the half-power width on a 0.05 Hz linear
    interpolation."""
    fine = np.arange(frequencies[0], frequencies[-1], 0.05)
    level = np.interp(fine, frequencies, gain_db)
    found = []
    for i in range(1, len(frequencies) - 1):
        if not (gain_db[i] > gain_db[i - 1] and gain_db[i] >= gain_db[i + 1]):
            continue
        left, right = gain_db[:i][::-1], gain_db[i + 1:]
        prominence = gain_db[i] - max(left.min() if left.size else -np.inf, right.min() if right.size else -np.inf)
        if prominence < PEAK_PROMINENCE_DB:
            continue
        top = int(np.argmin(abs(fine - frequencies[i])))
        lo = top
        while lo > 0 and level[lo] > level[top] - 3:
            lo -= 1
        hi = top
        while hi < len(fine) - 1 and level[hi] > level[top] - 3:
            hi += 1
        width = fine[hi] - fine[lo]
        q = frequencies[i] / width if width > 0 else float("nan")
        found.append({"hz": float(frequencies[i]), "db": round(float(gain_db[i]), 1),
                      "prominence_db": round(float(prominence), 1), "q": round(float(q), 1),
                      "t60_s": round(float(2.2 * q / frequencies[i]), 2)})
    return found


def room_gain_band_metrics(frequencies, gain_db, centre_index, grid_indices, bands_hz):
    """{band: {sd_db, sd_third_db, spread_db, max_minus_min_db, top_peaks}} of a room gain (f, point) in dB, the seat
    at centre_index and its ear-height grid at grid_indices (the docstring's METRICS)."""
    seat = gain_db[:, centre_index]
    smoothed = third_octave_smoothed(frequencies, seat)
    bands = {}
    for band, (lo, hi) in bands_hz.items():
        inside = (frequencies >= lo) & (frequencies <= hi)
        found = sorted(peaks(frequencies[inside], seat[inside]), key=lambda peak: -peak["prominence_db"])
        bands[band] = {"sd_db": round(float(seat[inside].std()), 2),
                       "sd_third_db": round(float(smoothed[inside].std()), 2),
                       "spread_db": round(float(gain_db[inside][:, grid_indices].std(axis=1).mean()), 2),
                       "max_minus_min_db": round(float(seat[inside].max() - seat[inside].min()), 1),
                       "top_peaks": found[:TOP_PEAKS]}
    return bands


def decay(frequencies, pressure):
    """{band centre: {edt_s, t20_s, bt60}} of one seat's complex response on a uniform grid (the docstring)."""
    step = float(frequencies[1] - frequencies[0])
    if not np.allclose(np.diff(frequencies), step):
        raise RuntimeError("(╯°□°)╯ the decay needs a uniform frequency grid")
    bins = np.arange(0.0, DECAY_SAMPLE_RATE_HZ / 2 + step / 2, step)
    spectrum = np.zeros(len(bins), dtype=complex)
    at = np.rint(frequencies / step).astype(int)
    if not np.allclose(bins[at], frequencies):
        raise RuntimeError("(╯°□°)╯ the frequency grid is not on multiples of its step")
    spectrum[at] = pressure
    dt = 1.0 / DECAY_SAMPLE_RATE_HZ
    out = {}
    for centre in DECAY_BANDS_HZ:
        lo, hi = centre * 2 ** (-1 / 6), centre * 2 ** (1 / 6)
        f0 = math.sqrt(lo * hi)
        with np.errstate(divide="ignore"):
            omega = (bins ** 2 - f0 ** 2) / (bins * (hi - lo))
        gain = np.where(bins > 0, 1 / np.sqrt(1 + omega ** 6), 0.0)
        # the period is 1 / step; the zero-phase band-pass rings before t = 0 too, which wraps to the period's end:
        # rolled by a quarter period the pre-ringing sits before the peak, and the integral runs from the peak to the
        # period's end (three quarters of it, 1.5 s on a 0.5 Hz grid)
        quarter = len(bins) // 2
        response = np.roll(np.fft.irfft(spectrum * gain), quarter)
        energy = response ** 2
        start = int(np.argmax(energy))
        if not quarter // 2 <= start < 2 * quarter:     # zero-phase: the peak may lead t = 0 by a few ms
            raise RuntimeError(f"(╯°□°)╯ the {centre:g} Hz band's energy peak is at {(start - quarter) * dt:.3f} s")
        end = len(response)
        curve = np.cumsum(energy[start:end][::-1])[::-1]
        curve_db = 10 * np.log10(curve / curve[0])
        t = np.arange(len(curve_db)) * dt

        def fit(top, bottom):
            span = (curve_db <= top) & (curve_db >= bottom)
            if span.sum() < 4:
                return float("nan")
            slope = np.polyfit(t[span], curve_db[span], 1)[0]
            return -60.0 / slope
        t20 = fit(-5.0, -25.0)
        out[f"{centre:g}"] = {"edt_s": round(float(fit(0.0, -10.0)), 3), "t20_s": round(float(t20), 3),
                              "bt60": round(float((hi - lo) * t20), 1)}
    return out
