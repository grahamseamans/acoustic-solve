# 2026-10-20 — the trap face as a velocity port

For the active bass trap project's shared-sim-stack plan (step 4). The user's aim (2026-10-19): "see the bass trap
eating watts out of the sim", in an energized room, and then check theories, hardware functions, many rooms, the
calibration and an optimum yardstick against it. All of that is post-processing once the trap face is a PORT.

## The idea

The trap's driver is a rigid piston: one uniform normal velocity over its face. So the room solve treats the trap face
exactly like a source port (ideal velocity, 1 m/s, `excitation:trap:<name>`). The room is linear, so per frequency

    p(x) = sum_s H_s(x) w_s + sum_t H_t(x) v_t          (s: source ports, t: trap ports)

and ANY trap behaviour is a law that sets v_t from what the trap can sense: its face-averaged pressure, a mic, the
cone itself. With sensed quantities q = Q_s w + Q_t v (each a face average or a mic point, per port) and a law
v = G(f) q (G: traps x sensed, complex per frequency; a delay is exp(-i omega tau)):

    (I - G Q_t) v = G Q_s w

one small solve per frequency, then every output (seats, face pressures, powers) from the same H's. One room solve
serves every law: no new run for a new law, a new delay, a new gain. New runs only for new geometry, mic positions,
wall loss, frequency grid, or trap positions.

## Outputs

- Face-averaged pressure of each trap face, per port: exact for the FEM field from the P2 surface triangles (the mean
  of a quadratic over a triangle is the mean of its three midside nodes), from build/room.msh's `trap:<name>` group.
- Mic points per trap: embedded mesh nodes like the seats, named, at positions relative to the trap face (its centre,
  normal into the room, along, up).
- Power absorbed by trap t: 0.5 Re(p_face conj(v_t)) A_t, v along the FEM outward normal (into the trap) — positive =
  eating watts. Source input power by the same formula with the sign for "into the air". Watts are at the solve's
  1 m/s unit scale times the weights.
- Seat pressures as now (analysis.RoomSolve), trap off (v = 0) vs on.

## Gates

1. Local unit tests on fake results: the closure algebra against a case with a known answer; the P2 face average exact
   on a quadratic field; the sign of power (a rho c law absorbs, its negative emits).
2. On a pod (after the new image is pinned), one tiny room solved twice:
   a. EQUIVALENCE: the trap as a port closed with G = 1/(rho c) on its face average vs. the same trap as a real
      plane-wave (rho c) face. Not identical: the termination is pointwise, the port is uniform, so the difference
      should fall like (ka)^2 with frequency; the gate is that trend plus a small error at the lowest frequencies.
   b. ENERGY BALANCE: rigid walls, tiny bulk loss: source input power = trap absorbed power + the bulk loss's share,
      the latter shrinking with eta.
3. Later (bass trap project): the Rivet thesis's measured absorber reproduced.

The steady-state numbers mean nothing for an UNSTABLE law; stability is checked separately (bass trap plan 5b). A
passive law (rho c, any positive real impedance) is stable, which is all the gates above use.

## Spec change

`traps[i]` gains "name" and "face": "velocity_port" | "rho_c" (required, no default), and optional "mic_points_m":
{name: [normal, along, up]} in the trap-face frame. "rho_c" stays only as the equivalence reference.
