"""THE FIELD ANYWHERE in a solved room: the retained P2 nodal pressure (fem_nodal_pressure) interpolated at any point of
the air, exactly as the FEM solution is defined there (the 10 quadratic shape functions of the ten-node tetrahedron
that holds the point), so a slice or a new listening point costs no new solve and no embedded node.

    mesh = P2Mesh.from_result(result)              # the solver's own mesh: points_m + tetrahedra10 of the room domain
    nodes, weights = mesh.interpolation(points)   # (point, 10) each: p(point) = sum weights * p[nodes]
    pressure = plane_pressure(stage, "ear")       # (frequency, port, row, column), NaN outside the air

LOCATING: a point is in the tetrahedron whose four corners give it barycentric coordinates all >= -LOCATE_TOLERANCE
(straight-sided tetrahedra: the corners are the geometry). Candidates are the tetrahedra whose centroids are nearest;
a point found in none RAISES (outside the air, or in a part of the mesh a reduced result did not keep).

THE EDGE NODES: which corner pair each of nodes 4..9 sits between is read off the mesh itself (the node at its edge's
midpoint), never assumed from an ordering convention (gmsh's and VTK's differ); a mesh whose cells disagree RAISES.
A tetrahedron whose edge nodes are off their midpoints is CURVED (a disc port's rim): locating in it with its corners
is not exact, so a point that lands in one RAISES.

THE SHAPE FUNCTIONS (barycentric l_0..l_3): corner a: l_a (2 l_a - 1); edge node between a and b: 4 l_a l_b. They
reproduce every quadratic exactly (tests/test_room_field.py proves it on a gmsh room mesh).

THE REDUCED RESULT: a stage built with field planes (acoustic_solve.room.build) keeps on the pod only the nodes its
readers need (the planes' tetrahedra, the embedded points, the port and trap faces: build/keep_nodes.npy,
acoustic_solve/reduce_nodes.py), so the result's mesh is that sub-mesh; P2Mesh.from_result reads whatever it is.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from acoustic_solve.room import analysis
from acoustic_solve.room.spec import SPEC_FILE, field_plane_points

LOCATE_TOLERANCE = 1e-9
LOCATE_CANDIDATES = (16, 128)            # nearest centroids tried, then more
MIDPOINT_RELATIVE_TOLERANCE = 1e-6       # an edge node this close to its edge's midpoint (x the edge length)
CORNER_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass
class P2Mesh:
    """Ten-node tetrahedra: nodes (node, 3) m, tetrahedra (cell, 10) node indices, corners first; edge_pairs[k] is
    the corner pair node 4 + k sits between; curved[c] is True when cell c's edge nodes are off their midpoints."""
    nodes: np.ndarray
    tetrahedra: np.ndarray
    edge_pairs: tuple
    curved: np.ndarray

    @classmethod
    def from_arrays(cls, nodes, tetrahedra):
        nodes, tetrahedra = np.asarray(nodes, dtype=float), np.asarray(tetrahedra, dtype=np.int64)
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 10:
            raise RuntimeError(f"(╯°□°)╯ tetrahedra of shape {tetrahedra.shape}, not (cell, 10)")
        corners = nodes[tetrahedra[:, :4]]                                       # (cell, 4, 3)
        midpoints = np.stack([(corners[:, a] + corners[:, b]) / 2 for a, b in CORNER_PAIRS], axis=1)   # (cell, 6, 3)
        lengths = np.stack([np.linalg.norm(corners[:, a] - corners[:, b], axis=1) for a, b in CORNER_PAIRS], axis=1)
        edge_nodes = nodes[tetrahedra[:, 4:]]                                    # (cell, 6, 3)
        # distance of each edge node to each midpoint, relative to the longest edge: (cell, node, pair)
        distance = np.linalg.norm(edge_nodes[:, :, None, :] - midpoints[:, None, :, :], axis=3) / lengths.max(
            axis=1)[:, None, None]
        nearest = distance.argmin(axis=2)                                        # (cell, node)
        straight = distance.min(axis=2).max(axis=1) < MIDPOINT_RELATIVE_TOLERANCE
        if not straight.any():
            raise RuntimeError("(╯°□°)╯ no tetrahedron has its edge nodes at its edges' midpoints: not a P2 mesh")
        pattern = nearest[straight]
        if not (pattern == pattern[0]).all() or sorted(pattern[0]) != list(range(6)):
            raise RuntimeError(f"(╯°□°)╯ the straight tetrahedra disagree on which edge each node 4..9 sits on "
                               f"(first {pattern[0].tolist()})")
        return cls(nodes, tetrahedra, tuple(CORNER_PAIRS[k] for k in pattern[0]), ~straight)

    @classmethod
    def from_result(cls, result):
        """The room solve's own mesh (its one fem_volume domain: points_m, tetrahedra10)."""
        domains = result.domains_of_kind("fem_volume")
        if len(domains) != 1:
            raise RuntimeError(f"(╯°□°)╯ {result.directory}: {len(domains)} fem_volume domains, not one")
        return cls.from_arrays(result.domain_coordinate(domains[0], "points_m"),
                               result.domain_topology(domains[0], "tetrahedra10"))

    def locate(self, points):
        """(cell (point,), barycentric (point, 4)) of each point; RAISES for a point in no cell or in a curved one."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        corners = self.nodes[self.tetrahedra[:, :4]]
        tree = cKDTree(corners.mean(axis=1))
        cells = np.full(len(points), -1)
        barycentric = np.zeros((len(points), 4))
        for candidates in LOCATE_CANDIDATES:
            left = np.flatnonzero(cells < 0)
            if not left.size:
                break
            _, nearest = tree.query(points[left], k=min(candidates, len(corners)))
            nearest = nearest.reshape(len(left), -1)
            for row, (point_index, cell_list) in enumerate(zip(left, nearest)):
                origin = corners[cell_list, 0]                                    # (candidate, 3)
                edges = np.swapaxes(corners[cell_list, 1:] - origin[:, None, :], 1, 2)   # (candidate, 3, 3)
                local = np.linalg.solve(edges, (points[point_index] - origin)[..., None])[..., 0]
                weights = np.concatenate([1 - local.sum(axis=1, keepdims=True), local], axis=1)
                inside = np.flatnonzero(weights.min(axis=1) >= -LOCATE_TOLERANCE)
                if inside.size:
                    cells[point_index] = cell_list[inside[0]]
                    barycentric[point_index] = weights[inside[0]]
        missing = np.flatnonzero(cells < 0)
        if missing.size:
            raise RuntimeError(f"(╯°□°)╯ {missing.size} points are in no tetrahedron of the mesh (the first "
                               f"{points[missing[0]].tolist()}): outside the air, or not kept by a reduced result")
        if self.curved[cells].any():
            first = int(np.flatnonzero(self.curved[cells])[0])
            raise RuntimeError(f"(╯°□°)╯ point {points[first].tolist()} is in a curved tetrahedron: its corners do not "
                               f"locate it exactly")
        return cells, barycentric

    def interpolation(self, points):
        """(node indices (point, 10), weights (point, 10)): p(point) = sum_k weights[:, k] p[nodes[:, k]]."""
        cells, l = self.locate(points)
        shapes = np.empty((len(cells), 10))
        shapes[:, :4] = l * (2 * l - 1)
        for k, (a, b) in enumerate(self.edge_pairs):
            shapes[:, 4 + k] = 4 * l[:, a] * l[:, b]
        return self.tetrahedra[cells], shapes

    def nodes_of(self, points):
        """The sorted node indices every point's interpolation uses."""
        nodes, _ = self.interpolation(points)
        return np.unique(nodes)


def pressure_at(result, mesh, points):
    """(frequency, port, point) complex: the retained nodal pressure interpolated at the points."""
    nodes, weights = mesh.interpolation(points)
    used, inverse = np.unique(nodes, return_inverse=True)
    matrix = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(nodes)), 10), inverse.ravel())),
                               shape=(len(nodes), len(used)))                      # (point, used node)
    values = analysis.nodal_pressure(result, used.tolist())                       # (frequency, port, used node)
    frequencies, ports = values.shape[:2]
    return (matrix @ values.reshape(-1, len(used)).T).T.reshape(frequencies, ports, len(nodes))


def plane_pressure(stage, plane):
    """(frequency, port, row, column) of one of the stage spec's field planes (spec.field_plane_points), NaN where
    the grid point is not in the air; and the plane's grid (field_plane_points' dict)."""
    stage = Path(stage)
    spec = json.loads((stage / SPEC_FILE).read_text())
    grid = field_plane_points(spec, plane)
    result, _ = analysis.room_result(stage)
    mesh = P2Mesh.from_result(result)
    inside = grid["points_m"][grid["in_air"]]
    values = pressure_at(result, mesh, inside)
    out = np.full(values.shape[:2] + grid["in_air"].shape, np.nan, dtype=complex)
    out[:, :, grid["in_air"]] = values
    return out, grid
