"""acoustic_solve.room.traps -- traps as ports closed by a law, checked without a solve:

- the P2 face integral is exact on a quadratic field over a real (gmsh, seconds) trap face and a disc port, and is the
  midside rule (area / 3 on each midside node, 0 on the corners) on straight triangles;
- a solved room stage (that mesh, a fake result of known quadratic fields, its FEM nodes shuffled) is read: every
  port's face average, the trap mics, the seats;
- the closure against a scalar case by hand and a 2-trap case by direct substitution; a singular closure refused;
- a 1-D duct (source piston at one end, trap piston at the other, analytic): G = 1 / (rho c) on the face average
  makes the duct anechoic and absorbs what the source puts in, -1 / (rho c) emits, and the source's input power
  equals the trap's absorbed power for any law (a lossless duct);
- a law naming a trap or a sensed quantity the room does not have is refused.
In the blab venv (gmsh, meshio, numpy, scipy):

    blabrun tests/test_room_traps.py
"""
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import meshio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_boundary_lab_results import write_room_result  # noqa: E402
from test_room_build import ROOM  # noqa: E402
from acoustic_solve.room import analysis, build, spec, traps  # noqa: E402

FREQUENCIES_HZ = np.array([40.0, 41.0])
BUILT = {}


def built_stage():
    """The test room (tests/test_room_build.py's: a velocity_port trap "far" with two mics, a rho_c trap "near"),
    built once."""
    if "stage" not in BUILT:
        root = Path(tempfile.mkdtemp(prefix="room-traps-"))
        (root / "room.json").write_text(json.dumps(ROOM))
        BUILT["stage"] = build.main(root / "stage", root / "room.json")
    return BUILT["stage"]


def quadratic(points, scale=1.0):
    """A quadratic field with every kind of term, so a rule that is not exact on quadratics shows."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return scale * (3.0 + 2 * x - y + 0.5 * z + x * x - 1.5 * x * y + 0.7 * y * z + 2.2 * z * z - 0.3 * y * y)


def far_face_mean(field):
    """The exact mean of a field over the far trap's face: a rectangle, so 3 x 3 Gauss-Legendre (exact to degree 5)."""
    frame = spec.trap_face_frame(ROOM["traps"][0], ROOM["dimensions_m"])
    nodes, weights = np.polynomial.legendre.leggauss(3)
    centre, along, up = (np.array(frame[key]) for key in ("face_centre_m", "along", "up"))
    width, height = ROOM["traps"][0]["width_m"], ROOM["traps"][0]["height_m"]
    points = np.array([centre + a * width / 2 * along + u * height / 2 * up for a in nodes for u in nodes])
    return float(np.dot(np.outer(weights, weights).ravel(), field(points)) / 4)


def side_disc_mean(field):
    """The exact mean of a quadratic over the side port's disc (left face, centre (0, 0.15, 0) in the box frame, r
    0.05; the box at (0.5, 1, 0.6) unyawed, so the disc's plane is x-z): f(c) + (r^2 / 8) (f_xx + f_zz), the second
    derivatives by central differences (exact on a quadratic)."""
    centre, radius, step = np.array([0.5, 1.15, 0.6]), 0.05, 1e-3
    second = sum((field(centre + step * e) - 2 * field(centre) + field(centre - step * e)) / step ** 2
                 for e in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])))
    return float(field(centre) + radius ** 2 / 8 * second)


def test_the_p2_face_integral_is_exact_on_a_quadratic_and_the_midside_rule_on_straight_triangles():
    mesh = meshio.read(built_stage() / "build" / spec.MESH_FILE)
    nodes, weights, area = traps.face_node_weights(mesh, "trap:far")
    assert abs(area - 0.5) < 1e-12, area
    assert abs(np.dot(weights, quadratic(mesh.points[nodes])) / area - far_face_mean(quadratic)) < 1e-12
    triangles = mesh.cells_dict["triangle6"][mesh.cell_sets_dict["trap:far"]["triangle6"]]
    corners = np.isin(nodes, triangles[:, :3])
    assert np.abs(weights[corners]).max() < 1e-15, "a straight triangle's corners weigh something"
    disc_nodes, disc_weights, disc_area = traps.face_node_weights(mesh, "side")
    assert abs(disc_area / (math.pi * 0.05 ** 2) - 1) < 2e-3, disc_area      # a coarse rim: P2 arcs, not the circle
    assert abs(np.dot(disc_weights, quadratic(mesh.points[disc_nodes])) / disc_area / side_disc_mean(quadratic)
               - 1) < 1e-4
    try:
        traps.face_node_weights(mesh, "trap:nowhere")
    except RuntimeError as error:
        assert "no physical group" in str(error)
    else:
        raise AssertionError("a group the mesh does not have was read")


def test_a_solved_stage_is_read_face_averages_mics_and_seats_with_its_nodes_shuffled():
    stage = built_stage()
    mesh = meshio.read(stage / "build" / spec.MESH_FILE)
    shuffled = np.random.default_rng(1).permutation(len(mesh.points))
    nodes = mesh.points[shuffled]
    ports = ["excitation:front", "excitation:rear", "excitation:side", "excitation:trap:far"]
    scale = np.array([[1.0, -2.0, 0.5j, 3.0 + 1j], [2.0, 1.0, -1j, 0.25]])          # (frequency, port)
    pressure = scale[:, :, None] * quadratic(nodes)[None, None, :]
    write_room_result(stage / "raw", FREQUENCIES_HZ, ports, nodes, pressure)
    trap_room = traps.load_trap_room(stage)
    assert trap_room.trap_names == ["far"] and trap_room.trap_port_ids == ["excitation:trap:far"]
    assert trap_room.source_port_ids == ports[:3]
    assert trap_room.sensed_names == ["far/face", "far/front", "far/high"]
    facts = json.loads((stage / spec.POINTS_FILE).read_text())
    far = far_face_mean(quadratic)
    assert np.allclose(trap_room.face_pressure[:, :, 3], scale * far, rtol=1e-12, atol=0)
    assert np.allclose(trap_room.face_pressure[:, :, 2], scale * side_disc_mean(quadratic), rtol=1e-4, atol=0)
    mics = quadratic(np.array([facts["trap_points_m"][name] for name in ("far/front", "far/high")]))
    assert np.allclose(trap_room.sensed, scale[:, :, None] * np.array([far, *mics])[None, None, :], rtol=1e-12)
    seats = quadratic(np.array(list(facts["seat_points_m"].values())))
    assert np.allclose(trap_room.room.pressure, scale[:, :, None] * seats[None, None, :], rtol=1e-12)
    assert np.allclose(trap_room.face_area_m2[[0, 3]], [0.09, 0.5], rtol=1e-12)


def toy_trap_room(seat_pressure, face_pressure, sensed, sensed_names, trap_count):
    """A TrapRoom from arrays (frequency, port, ...), the sources first, the traps last, every face 1 m^2."""
    frequencies, ports = seat_pressure.shape[:2]
    port_ids = [f"excitation:s{k}" for k in range(ports - trap_count)] + [f"excitation:trap:t{k}"
                                                                         for k in range(trap_count)]
    room = analysis.RoomSolve(np.arange(1.0, frequencies + 1), port_ids,
                              [f"seat/{k}" for k in range(seat_pressure.shape[2])],
                              np.zeros((seat_pressure.shape[2], 3)), seat_pressure, [])
    return traps.TrapRoom(room, port_ids[:ports - trap_count], [f"t{k}" for k in range(trap_count)],
                          port_ids[ports - trap_count:], np.ones(ports), face_pressure, sensed_names, sensed)


def test_the_closure_is_the_scalar_case_by_hand_and_the_two_trap_case_by_substitution_and_singular_is_refused():
    h_source, h_trap, q_source, q_trap, g = 2.0 + 1j, -0.5 + 0.25j, 1.5 - 1j, 0.3 + 0.4j, 0.7 - 0.2j
    scalar = toy_trap_room(np.array([[[h_source], [h_trap]]]), np.array([[[0, q_source], [0, q_trap]]]),
                           np.array([[[q_source], [q_trap]]]), ["t0/face"], 1)
    loop = traps.closed_loop(scalar, traps.law(scalar, {"t0": {"t0/face": g}}), {"excitation:s0": 2.0})
    v = g * q_source * 2.0 / (1 - g * q_trap)
    assert np.isclose(loop.trap_velocity[0, 0], v) and np.isclose(loop.seat_pressure[0, 0], h_source * 2.0 + h_trap * v)
    try:
        traps.closed_loop(scalar, traps.law(scalar, {"t0": {"t0/face": 1 / q_trap}}), {"excitation:s0": 1.0})
    except RuntimeError as error:
        assert "singular or ill-conditioned" in str(error), error
    else:
        raise AssertionError("a singular closure was solved")
    rng = np.random.default_rng(7)

    def complex_random(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    frequencies, sources, trap_count, sensed, seats = 3, 2, 2, 3, 4
    ports = sources + trap_count
    sensed_values = complex_random(frequencies, ports, sensed)
    two = toy_trap_room(complex_random(frequencies, ports, seats), complex_random(frequencies, ports, ports),
                        sensed_values, ["t0/face", "t1/face", "t0/mic"], trap_count)
    gain = 0.3 * complex_random(frequencies, trap_count, sensed)
    weights = {"excitation:s0": complex_random(frequencies), "excitation:s1": 0.5}
    loop = traps.closed_loop(two, gain, weights)
    w = np.stack([weights["excitation:s0"], np.full(frequencies, 0.5)], axis=1)
    for f in range(frequencies):
        q = w[f] @ sensed_values[f, :sources] + loop.trap_velocity[f] @ sensed_values[f, sources:]
        assert np.allclose(loop.trap_velocity[f], gain[f] @ q, rtol=1e-12, atol=1e-12)   # v = G q, substituted
        assert np.allclose(loop.sensed[f], q, rtol=1e-12, atol=1e-12)
        drive = np.concatenate([w[f], loop.trap_velocity[f]])
        assert np.allclose(loop.seat_pressure[f], drive @ two.room.pressure[f], rtol=1e-12, atol=1e-12)
    try:
        traps.closed_loop(two, gain, {"excitation:s0": 1.0})
    except RuntimeError as error:
        assert "every one" in str(error)
    else:
        raise AssertionError("a source port left out of the mix was taken as 0")


def duct(frequency_hz, length_m=1.7, seats_m=(0.2, 0.9, 1.5)):
    """A 1-D duct, 1 m^2: the source piston at x = 0, the trap piston at x = L, each 1 m/s along the air's outward
    normal (the source's into its box: -x; the trap's into the trap: +x); exp(+i omega t), i omega rho u = -dp/dx:
        source alone  p = i rho c cos k(L - x) / sin kL       trap alone  p = i rho c cos kx / sin kL."""
    k = 2 * math.pi * frequency_hz / spec.SOUND_SPEED_M_PER_S
    amplitude = 1j * traps.RHO_C / math.sin(k * length_m)

    def source(x):
        return amplitude * np.cos(k * (length_m - np.asarray(x)))

    def trap(x):
        return amplitude * np.cos(k * np.asarray(x))
    seats = np.array(seats_m)
    return (np.array([[source(seats), trap(seats)]]),
            np.array([[[source(0.0), source(length_m)], [trap(0.0), trap(length_m)]]]),
            np.array([[[source(length_m)], [trap(length_m)]]]))


def test_a_rho_c_law_makes_a_duct_anechoic_and_absorbs_its_negative_emits_and_the_power_balances():
    for frequency in (37.0, 140.0):
        seats, faces, sensed = duct(frequency)
        trap_room = toy_trap_room(seats, faces, sensed, ["t0/face"], 1)
        absorbing = traps.closed_loop(trap_room, traps.law(trap_room, {"t0": {"t0/face": 1 / traps.RHO_C}}),
                                      {"excitation:s0": 1.0})
        assert np.allclose(np.abs(absorbing.seat_pressure), traps.RHO_C, rtol=1e-9), "not anechoic"
        assert np.isclose(absorbing.trap_absorbed_w[0, 0], 0.5 * traps.RHO_C, rtol=1e-9), absorbing.trap_absorbed_w
        assert np.isclose(absorbing.source_input_w[0, 0], absorbing.trap_absorbed_w[0, 0], rtol=1e-9)
        emitting = traps.closed_loop(trap_room, traps.law(trap_room, {"t0": {"t0/face": -0.5 / traps.RHO_C}}),
                                     {"excitation:s0": 1.0})
        assert emitting.trap_absorbed_w[0, 0] < 0, emitting.trap_absorbed_w
        assert np.isclose(emitting.source_input_w[0, 0], emitting.trap_absorbed_w[0, 0], rtol=1e-9)
        delayed = traps.closed_loop(trap_room, traps.law(trap_room, {"t0": {"t0/face": np.exp(
            -2j * math.pi * frequency * 1e-3) / (2 * traps.RHO_C)}}), {"excitation:s0": 1.0})
        assert np.isclose(delayed.source_input_w[0, 0], delayed.trap_absorbed_w[0, 0], rtol=1e-9)
        off = traps.closed_loop(trap_room, traps.law(trap_room, {}), {"excitation:s0": 1.0})
        assert off.trap_velocity[0, 0] == 0 and abs(off.source_input_w[0, 0]) < 1e-9 * traps.RHO_C


def test_a_law_naming_what_the_room_does_not_have_is_refused():
    trap_room = toy_trap_room(*duct(50.0), ["t0/face"], 1)
    for gains, words in (({"t9": {"t0/face": 1.0}}, "no velocity_port trap"), ({"t0": {"t0/mic": 1.0}}, "senses"),
                         ({"t0": {"t0/face": [1.0, 2.0]}}, "one per frequency")):
        try:
            traps.law(trap_room, gains)
        except RuntimeError as error:
            assert words in str(error), error
        else:
            raise AssertionError(f"law {gains} was accepted")


if __name__ == "__main__":
    started = time.time()
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
    print(f"all passed in {time.time() - started:.1f} s")
