# 2026-10-21 -- rooms for the bass trap's first visuals: several boxes, listening volumes, wall traps, the field anywhere

For the active bass trap's PLANS/2026-10-21-first-visuals.md: typical rooms with a stereo pair of cardioid boxes,
four corner traps and two wall traps, a listening volume at the seat, and the pressure FIELD on slices over time.

## Spec changes (room.json; no backwards compatibility)

- `sources`: a LIST of boxes, each {"name", "size_m", "centre_m", "aim_m", "ports", optional
  "free_field_points_m"} (replaces `source`). Port names are unique across boxes (a port id stays
  "excitation:<port>"); the rigid rest of every box is the group "box".
- `seats`: {seat: {"centre_m", "step_m": [sx, sy, sz], "half_count": [nx, ny, nz]}} (replaces `seats_m` +
  `seat_grid`): a box of (2nx+1)(2ny+1)(2nz+1) points named "<seat>/x+1y-1z+0" in room axes; the centre is
  "x+0y+0z+0", the seat's ear-height grid its z+0 points. A single point is half_count [0, 0, 0].
- `traps[i]` gains "mount": "corner" (the floor-standing prism, as before: "corner", "width_m", "height_m") |
  "wall" (a face flush in a wall: "wall": "x0" | "x1" | "y0" | "y1", "centre_m" [x, y, z] on that wall, "width_m",
  "height_m"; the cabinet is behind the wall, outside the air). Required, no default. A wall trap's face is cut out
  of any rho c patch it overlaps.
- The free field: every box of the spec in one exterior BEM, in the FIRST box's frame; specs share it when their
  boxes are the same and stand the same way relative to the first (one box: as before, anywhere in the room).

## The field anywhere (room/field.py)

The interior FEM retains every node's pressure (fem_nodal_pressure). A point in the air lies in one P2 tetrahedron;
the field there is the 10 quadratic shape functions on its nodes, exact for the FEM solution (no embedding, no
re-solve for a new slice). Points are located by barycentric coordinates in the straight-sided tetrahedra (refused
loudly if a point is in none, or the mesh's edge nodes are not at their edges' midpoints). Test: a quadratic field
is reproduced exactly on a real gmsh room mesh.

## Steps

1. [ ] spec + build + field + traps/analysis readers; tests; the pod gate's room spec moved to the new grammar.
2. [ ] commit + push.
