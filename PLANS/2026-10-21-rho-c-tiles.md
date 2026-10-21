# 2026-10-21 -- the walls' rho c absorption tiled evenly, and the shoebox's modal decay

For the active bass trap's PLANS/2026-10-21-uniform-walls.md: one centred rho c patch per surface damped the low
modes unevenly (an end wall's patch sits on the length modes' antinodes, a side wall's on their node line) and could
not be calibrated to a modal decay time.

## Spec change (room.json; no backwards compatibility)

- `absorption.rho_c_tile_m` (required when `rho_c_patch_fraction` > 0): every surface is cut into
  ceil(side / pitch) equal cells per axis, each holding one centred rho c rectangle (`spec.rho_c_tiles`). No tile
  touches a trap's face or a corner prism's footprint (`spec.surface_keep_outs`); the others grow so the surface
  keeps its fraction of its whole area. A pitch past a surface's size is one centred patch. Wall traps are no longer
  cut out of a patch: tiles avoid them.
- Why equal cells: the area mean of cos^2(n pi u / L) over N equal cells' centred tiles is the wall's for every
  n < N, so to first order every such mode sees the mean admittance beta = f (tests/test_room_build.py).

## The modal decay (room/analysis.py)

`modal_decay_rate(dimensions, mode, beta)`: delta = c beta sum_axis (2 if n > 0 else 1) / L, from the hard-wall
root of Calafà et al. 2026 (arXiv 2602.09594v2, Table 1 group 1; references/fetch_sources.sh), and its inverse
`admittance_for_modal_t60`. Tested against the exact root of their eq. 6 (Newton per axis, K iterated): < 1 % at
beta 0.01.

## Steps

1. [x] spec.rho_c_tiles + build.wall_tiles + analysis.modal_decay_rate; tests; commit + push.
