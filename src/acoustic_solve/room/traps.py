"""TRAPS AS PORTS, CLOSED BY A LAW (PLANS/2026-10-20-trap-velocity-port.md): a room solved with its trap faces as
velocity ports (acoustic_solve.room.build, "face": "velocity_port") is every trap behaviour at once; a law picks one in
post-processing, and the watts each trap eats come out of the same numbers.

    trap_room = load_trap_room(stage)          # the room solve + every port's face-averaged pressure + the trap mics
    gains = law(trap_room, {"corner": {"corner/face": 1 / RHO_C}})      # (frequency, trap, sensed)
    loop = closed_loop(trap_room, gains, {"excitation:front": 1.0})
    loop.seat_pressure, loop.trap_velocity, loop.trap_absorbed_w, loop.source_input_w

THE ROOM IS LINEAR. Per frequency, over the source ports s (weights w) and the trap ports t (velocities v):
    p(x) = sum_s H_s(x) w_s + sum_t H_t(x) v_t
H_k(x) is port k driven alone at 1 m/s (along the FEM domain's outward normal: INTO the box, INTO the trap). What the
traps can SENSE (spec.sensed_names: each velocity_port trap's face-averaged pressure "<trap>/face", each trap mic
"<trap>/<mic>") is linear too, q = Q_s w + Q_t v, and A LAW is v = G q with G complex (trap, sensed) per frequency,
so per frequency
    (I - G Q_t) v = G Q_s w
one small solve. Its CONDITION is max(1, sigma_max) / sigma_min of I - G Q_t: the ordinary condition number, anchored
to the identity's scale so that a 1 x 1 closure near 0 (a loop gain near 1: the response near unbounded) counts as
ill-conditioned too; beyond CLOSURE_CONDITION_LIMIT (or singular) it RAISES rather than return its garbage.
Convention exp(+i omega t) (acoustic_solve.room.analysis): a delay tau in a law is a factor
exp(-i omega tau), and any weighted_sums term's weight (weighted_sums.term_weight: gain, polarity, delay, filters) is a
law gain as it stands. G = 0 is the trap off (v = 0). The steady state of an UNSTABLE law means nothing; stability is
not checked here.

THE FACE AVERAGE of the FEM field over a port's physical group is EXACT: the field is quadratic on each 6-node
surface triangle, and on a straight-sided flat triangle the mean of a quadratic is the mean of its three midside
values (the vertices weigh nothing), so the face integral is sum over triangles of area / 3 x (its three midside
pressures). A curved triangle (a disc port's rim: gmsh puts the midside node on the arc) is integrated
isoparametrically with Dunavant's 7-point degree-5 rule (Int. J. Numer. Meth. Eng. 21, 1985), which is that same sum
on a straight triangle and exact on a curved one in a plane (a quadratic times a linear Jacobian). Divided by the
group's area (checked against the builder's). The triangles are read from build/room.msh, matched to the result's FEM
nodes by position.

POWER, at the solve's unit scale times the weights (1 m/s ports): through a port's face along its velocity's
direction (into the box, into the trap) 0.5 Re(p_face conj(u)) A. A trap's absorbed power is that (positive = it eats
watts: G = 1 / (rho c) on its face average absorbs, -1 / (rho c) emits); a source port's input power is its negative
(into the air).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from acoustic_solve.boundary_lab_result import PORT_ID_PREFIX
from acoustic_solve.room import analysis
from acoustic_solve.room.spec import (DENSITY_KG_PER_M3, MESH_FILE, POINTS_FILE, SOUND_SPEED_M_PER_S, SPEC_FILE,
                                      TRAP_GROUP_PREFIX, checked_traps, sensed_names, trap_group, trap_points)

RHO_C = DENSITY_KG_PER_M3 * SOUND_SPEED_M_PER_S      # Pa / (m/s): the plane-wave impedance a rho c face terminates in
CLOSURE_CONDITION_LIMIT = 1e8                        # (I - G Q_t) beyond this: the closed loop's v would be garbage
FACE_AREA_RELATIVE_TOLERANCE = 1e-2                  # the mesh's P2 area against the builder's (CAD) area of the
                                                     #   group (a check that it is the group): a flat face meets it
                                                     #   to round-off, a coarse disc's P2 rim falls ~1e-3 short.
                                                     #   Powers use the MESH's area: the one the solve integrates
TRIANGLE6 = "triangle6"                              # gmsh order: corners 0 1 2, midsides 01 12 20
# Dunavant degree 5: (barycentric point, weight of the reference triangle's area), the three orbits
DUNAVANT_7 = ([((1 / 3, 1 / 3, 1 / 3), 0.225)]
              + [(point, 0.132394152788506) for point in
                 ((0.059715871789770, 0.470142064105115, 0.470142064105115),
                  (0.470142064105115, 0.059715871789770, 0.470142064105115),
                  (0.470142064105115, 0.470142064105115, 0.059715871789770))]
              + [(point, 0.125939180544827) for point in
                 ((0.797426985353087, 0.101286507323456, 0.101286507323456),
                  (0.101286507323456, 0.797426985353087, 0.101286507323456),
                  (0.101286507323456, 0.101286507323456, 0.797426985353087))])


def triangle6_shapes(l1, l2, l3):
    """(N (6,), dN/dxi (6,), dN/deta (6,)) at barycentric (l1, l2, l3), xi = l2, eta = l3, gmsh's node order."""
    shapes = np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1])
    d_xi = np.array([1 - 4 * l1, 4 * l2 - 1, 0.0, 4 * (l1 - l2), 4 * l3, -4 * l3])
    d_eta = np.array([1 - 4 * l1, 0.0, 4 * l3 - 1, -4 * l2, 4 * l2, 4 * (l1 - l3)])
    return shapes, d_xi, d_eta


@dataclass
class TrapRoom:
    """One room solved with its traps as ports. Every array is (frequency, driven port, ...), ports in room.port_ids
    order; face_pressure's last axis is every port's own face (the same order), sensed's the sensed names."""
    room: analysis.RoomSolve
    source_port_ids: list
    trap_names: list
    trap_port_ids: list
    face_area_m2: np.ndarray
    face_pressure: np.ndarray
    sensed_names: list
    sensed: np.ndarray

    def port_indices(self, port_ids):
        return [self.room.port_ids.index(port_id) for port_id in port_ids]


@dataclass
class ClosedLoop:
    """One law and one source mix, closed: the velocity of every port (the sources' weights, the traps' v), and what
    they make. (frequency, ...) each; power in W at the solve's unit scale times the weights."""
    frequencies_hz: np.ndarray
    port_ids: list
    port_velocity: np.ndarray            # (f, port)
    trap_velocity: np.ndarray            # (f, trap)
    seat_pressure: np.ndarray            # (f, seat point), the room's point order
    face_pressure: np.ndarray            # (f, port): each port's face-averaged pressure
    sensed: np.ndarray                   # (f, sensed)
    trap_absorbed_w: np.ndarray          # (f, trap): positive = eating watts
    source_input_w: np.ndarray           # (f, source port): into the air
    closure_condition: np.ndarray        # (f,): max(1, sigma_max) / sigma_min of I - G Q_t (the docstring)


def face_node_weights(mesh, group):
    """(the mesh's node indices, their weights in m^2 (sum: the area), the area) of one physical group's 6-node
    triangles: the docstring's exact P2 face integral as a weight per node (on a straight triangle, area / 3 on each
    midside node and 0 on the corners)."""
    cell_sets = mesh.cell_sets_dict.get(group)
    if not cell_sets:
        raise RuntimeError(f"(╯°□°)╯ the mesh has no physical group {group!r}")
    if set(cell_sets) != {TRIANGLE6}:
        raise RuntimeError(f"(╯°□°)╯ group {group!r} is {sorted(cell_sets)}, not only {TRIANGLE6} (a P2 surface)")
    triangles = mesh.cells_dict[TRIANGLE6][cell_sets[TRIANGLE6]]
    where = mesh.points[triangles]                                           # (triangle, 6, 3)
    per_triangle_node = np.zeros(triangles.shape)
    for (l1, l2, l3), weight in DUNAVANT_7:
        shapes, d_xi, d_eta = triangle6_shapes(l1, l2, l3)
        jacobian = np.linalg.norm(np.cross(d_xi @ where, d_eta @ where), axis=1)   # dS / (dxi deta), (triangle,)
        per_triangle_node += (weight / 2) * jacobian[:, None] * shapes[None, :]
    nodes, inverse = np.unique(triangles, return_inverse=True)
    weights = np.zeros(len(nodes))
    np.add.at(weights, inverse.reshape(triangles.shape), per_triangle_node)
    return nodes, weights, float(per_triangle_node.sum())


def load_trap_room(stage):
    """A solved room stage (<stage>/raw complete, room.json, seat_points.json, build/room.msh) with velocity_port traps:
    the seats, every port's face-averaged pressure, the trap mics, in one read of the retained nodal pressure."""
    stage = Path(stage)
    spec = json.loads((stage / SPEC_FILE).read_text())
    facts = json.loads((stage / POINTS_FILE).read_text())
    port_traps = [trap for trap in checked_traps(spec) if trap["face"] == "velocity_port"]
    if not port_traps:
        raise RuntimeError(f"(╯°□°)╯ {stage}: no velocity_port trap, nothing to close")
    result, nodes = analysis.room_result(stage)
    port_ids = result.excitation_port_ids
    trap_port_ids = [PORT_ID_PREFIX + trap_group(trap) for trap in port_traps]
    trap_like = [p for p in port_ids if p.startswith(PORT_ID_PREFIX + TRAP_GROUP_PREFIX)]
    if trap_like != trap_port_ids:
        raise RuntimeError(f"(╯°□°)╯ {stage}: the solve's trap ports {trap_like}, the spec's {trap_port_ids}")
    source_port_ids = [p for p in port_ids if p not in trap_port_ids]
    seat_names, mic_names = list(facts["seat_points_m"]), list(facts["trap_points_m"])
    if mic_names != list(trap_points(spec)):
        raise RuntimeError(f"(╯°□°)╯ {stage}: {POINTS_FILE}'s trap points {mic_names} are not the spec's")
    mesh = meshio.read(stage / "build" / MESH_FILE)
    faces = []
    for group in result.port_names():
        face_nodes, weights, area = face_node_weights(mesh, group)
        built = facts["area_m2"][group]
        if abs(area - built) > FACE_AREA_RELATIVE_TOLERANCE * built:
            raise RuntimeError(f"(╯°□°)╯ {group}: its triangles' area {area} m^2, the builder's {built} m^2")
        faces.append((weights, area, analysis.node_indices(nodes, mesh.points[face_nodes], f"{group} face node")))
    seat_index = analysis.node_indices(nodes, [facts["seat_points_m"][n] for n in seat_names], "seat point")
    mic_index = analysis.node_indices(nodes, [facts["trap_points_m"][n] for n in mic_names], "trap point")
    pressure = analysis.nodal_pressure(result, seat_index + mic_index + [i for _, _, index in faces for i in index])
    at_seats = pressure[..., :len(seat_index)]
    at_mics = pressure[..., len(seat_index):len(seat_index) + len(mic_index)]
    face_pressure, start = [], len(seat_index) + len(mic_index)
    for weights, area, index in faces:
        face_pressure.append(pressure[..., start:start + len(index)] @ weights / area)
        start += len(index)
    face_pressure = np.stack(face_pressure, axis=-1)
    names = sensed_names(spec)
    trap_faces = [port_ids.index(p) for p in trap_port_ids]
    sensed = np.concatenate([face_pressure[..., trap_faces], at_mics], axis=-1)
    if sensed.shape[-1] != len(names):
        raise RuntimeError(f"(╯°□°)╯ {stage}: {sensed.shape[-1]} sensed quantities, the spec names {names}")
    room = analysis.RoomSolve(result.frequencies_hz, port_ids, seat_names,
                              np.array([facts["seat_points_m"][n] for n in seat_names]), at_seats,
                              result.diagnostics())
    return TrapRoom(room, source_port_ids, [trap["name"] for trap in port_traps], trap_port_ids,
                    np.array([area for _, area, _ in faces]), face_pressure, names, sensed)


def join_trap_room_chunks(stages):
    """One room solved as frequency chunks (one stage each, the same spec but its frequencies) -> one TrapRoom, sorted
    by frequency. Chunks that differ in ports, traps, sensed names, seat points or face areas, or overlap in
    frequency, RAISE."""
    parts = [load_trap_room(stage) for stage in stages]
    first = parts[0]
    for part in parts:
        if (part.room.port_ids != first.room.port_ids or part.room.point_names != first.room.point_names
                or part.trap_names != first.trap_names or part.sensed_names != first.sensed_names
                or not np.allclose(part.face_area_m2, first.face_area_m2, rtol=1e-9)):
            raise RuntimeError("(╯°□°)╯ the chunks of one room differ in ports, traps, sensed names, seats or faces")
    frequencies = np.concatenate([part.room.frequencies_hz for part in parts])
    order = np.argsort(frequencies)
    if np.any(np.diff(frequencies[order]) <= 0):
        raise RuntimeError("(╯°□°)╯ the chunks of one room overlap in frequency")

    def joined(name):
        return np.concatenate([getattr(part, name) for part in parts])[order]
    room = analysis.RoomSolve(frequencies[order], first.room.port_ids, first.room.point_names, first.room.points_m,
                              np.concatenate([part.room.pressure for part in parts])[order],
                              [d for part in parts for d in part.room.diagnostics])
    return TrapRoom(room, first.source_port_ids, first.trap_names, first.trap_port_ids, first.face_area_m2,
                    joined("face_pressure"), first.sensed_names, joined("sensed"))


def per_frequency(value, count, what):
    """A number, or one per frequency -> (count,) complex."""
    array = np.asarray(value, dtype=complex)
    if array.ndim == 0:
        return np.full(count, complex(array))
    if array.shape != (count,):
        raise RuntimeError(f"(╯°□°)╯ {what}: shape {array.shape}, not one number or {count} (one per frequency)")
    return array


def law(trap_room, gains):
    """{trap: {sensed name: gain (a number, or one per frequency; m/s per Pa for a pressure)}} -> G (frequency, trap,
    sensed); what is not named is 0. A trap or sensed name the room does not have RAISES."""
    frequencies = len(trap_room.room.frequencies_hz)
    matrix = np.zeros((frequencies, len(trap_room.trap_names), len(trap_room.sensed_names)), dtype=complex)
    for trap, row in gains.items():
        if trap not in trap_room.trap_names:
            raise RuntimeError(f"(╯°□°)╯ law: no velocity_port trap {trap!r} (the room's: {trap_room.trap_names})")
        for name, gain in row.items():
            if name not in trap_room.sensed_names:
                raise RuntimeError(f"(╯°□°)╯ law: trap {trap} senses {name!r}, not one of {trap_room.sensed_names}")
            matrix[:, trap_room.trap_names.index(trap), trap_room.sensed_names.index(name)] = per_frequency(
                gain, frequencies, f"law {trap} <- {name}")
    return matrix


def closed_loop(trap_room, law_matrix, source_weights):
    """The law G (frequency, trap, sensed) closed on the source mix {source port id: weight (a number, or one per
    frequency)}, EVERY source port named (0 for one not driven) -> ClosedLoop (the docstring's closure)."""
    room = trap_room.room
    frequencies = len(room.frequencies_hz)
    traps, sensed = len(trap_room.trap_names), len(trap_room.sensed_names)
    gain = np.asarray(law_matrix, dtype=complex)
    if gain.shape != (frequencies, traps, sensed):
        raise RuntimeError(f"(╯°□°)╯ the law is {gain.shape}, not (frequency, trap, sensed) = "
                           f"{(frequencies, traps, sensed)}")
    if set(source_weights) != set(trap_room.source_port_ids):
        raise RuntimeError(f"(╯°□°)╯ source weights for {sorted(source_weights)}, the room's source ports are "
                           f"{sorted(trap_room.source_port_ids)} (every one, 0 for one not driven)")
    source_index, trap_index = trap_room.port_indices(trap_room.source_port_ids), trap_room.port_indices(
        trap_room.trap_port_ids)
    weights = np.stack([per_frequency(source_weights[p], frequencies, p) for p in trap_room.source_port_ids], axis=1)
    sensed_of_sources = np.swapaxes(trap_room.sensed[:, source_index], 1, 2)      # Q_s (f, sensed, source)
    sensed_of_traps = np.swapaxes(trap_room.sensed[:, trap_index], 1, 2)          # Q_t (f, sensed, trap)
    closure = np.eye(traps) - gain @ sensed_of_traps
    singular_values = np.linalg.svd(closure, compute_uv=False)                       # (f, trap), descending
    with np.errstate(divide="ignore"):
        condition = np.maximum(1.0, singular_values[:, 0]) / singular_values[:, -1]
    bad = np.flatnonzero(~np.isfinite(condition) | (condition > CLOSURE_CONDITION_LIMIT))
    if bad.size:
        raise RuntimeError(f"(╯°□°)╯ the closed loop is singular or ill-conditioned (cond > "
                           f"{CLOSURE_CONDITION_LIMIT:g}) at {room.frequencies_hz[bad].tolist()} Hz: the law is at or "
                           f"near an unbounded response there")
    trap_velocity = np.linalg.solve(closure, (gain @ (sensed_of_sources @ weights[..., None])))[..., 0]
    velocity = np.zeros((frequencies, len(room.port_ids)), dtype=complex)
    velocity[:, source_index], velocity[:, trap_index] = weights, trap_velocity
    face_pressure = np.einsum("fp,fpq->fq", velocity, trap_room.face_pressure)
    into_faces_w = 0.5 * np.real(face_pressure * np.conj(velocity)) * trap_room.face_area_m2
    return ClosedLoop(room.frequencies_hz, room.port_ids, velocity, trap_velocity,
                      np.einsum("fp,fpx->fx", velocity, room.pressure), face_pressure,
                      np.einsum("fp,fps->fs", velocity, trap_room.sensed), into_faces_w[:, trap_index],
                      -into_faces_w[:, source_index], condition)
