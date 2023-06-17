import sys
import numpy
import itertools

numpy.set_printoptions(floatmode = 'maxprec', suppress = True)
epsilon = numpy.finfo(numpy.float32).eps
index_bounding_box_edges = numpy.array(list(itertools.combinations(itertools.product([0, 1], repeat=3), 2)))
index_bounding_box_mask = index_bounding_box_edges[:, 0] != index_bounding_box_edges[:, 1]
index_bounding_box_edges = index_bounding_box_edges[numpy.count_nonzero(index_bounding_box_mask, axis=-1) == 1]
index_bounding_box_edge_normal = numpy.nonzero(index_bounding_box_edges[:, 0] != index_bounding_box_edges[:, 1])[1]
del index_bounding_box_mask

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def normalize(x):
    y = x.reshape((-1, x.shape[-1]))
    return (y / numpy.linalg.norm(y, axis = -1)[:, None]).reshape(x.shape)

def dot(x, y):
    return numpy.sum(x * y, axis=-1)

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

class TriangleEdgeBoundingBoxIntersection:
    def __init__(self, triangle_list, bounding_box):
        self.triangle_list = triangle_list
        self.bounding_box = bounding_box
        self.edge_list = self.triangle_list[:, numpy.roll(numpy.arange(3), -1)] - self.triangle_list
        with numpy.errstate(divide='ignore'):
            self.gradient = (self.bounding_box - self.triangle_list[:, :, None]) / self.edge_list[:, :, None]
        self.finite_filter = numpy.isfinite(self.gradient)
        self.edge_filter = numpy.logical_and(self.gradient >= 0.0, self.gradient <= 1.0)
        self.position = self.triangle_list[:, :, None, None] + self.gradient[:, :, :, :, None] * self.edge_list[:, :, None, None]
        self.box_filter = numpy.all(numpy.logical_and(self.position >= bounding_box[0], self.position <= bounding_box[1]), axis=-1)
        self.inner_filter = numpy.logical_not(numpy.all(numpy.all(numpy.logical_and(triangle_list >= bounding_box[0], triangle_list <= bounding_box[1]), axis=-1), axis=-1))
        # While above guarantee proper parameter ranges, there is one case where triangle might be considered even when it does not cross a plane.
        # If one (or more) vertices of the triangle are at (or epsilon-near) the side of the bounding box,
        # the triangle might be counted. To prevent this, we include true crossing mask where triangle must clearly cross a side of the box
        # to be considered.

        # Find all vertices that are below the side of the box.
        below_mask = triangle_list[:, :, None] <= bounding_box
        # Find all vertices that are above the side of the box.
        above_mask = triangle_list[:, :, None] >= bounding_box
        # Find all vertices epsilon-near to the sides of the box.
        at_plane_mask = numpy.abs(triangle_list[:, :, None] - bounding_box) < epsilon
        below_mask = numpy.logical_and(below_mask, numpy.logical_not(at_plane_mask))
        above_mask = numpy.logical_and(above_mask, numpy.logical_not(at_plane_mask))
        self.crossing_mask = numpy.any(numpy.logical_and(numpy.any(below_mask, axis=1), numpy.any(above_mask, axis=1)).reshape((-1, 6)), axis=-1)
        self.filter = numpy.logical_and(self.finite_filter, numpy.logical_and(self.crossing_mask[:, None, None, None], numpy.logical_and(self.edge_filter, self.box_filter)))
        self.index = numpy.nonzero(self.filter)
        self.point_list = self.position[self.filter]

class BoundingBoxEdgeTriangleIntersection:
    def __init__(self, bounding_box, triangle_list):
        self.edge_list = bounding_box[(index_bounding_box_edges, numpy.arange(3))]
        self.bounding_box = bounding_box
        self.triangle_list = triangle_list
        self.normal_list = normalize(numpy.cross(triangle_list[:, 2] - triangle_list[:, 0], triangle_list[:, 1] - triangle_list[:, 0]))
        with numpy.errstate(divide='ignore'):
            self.gradient = dot(triangle_list[:, 0][:, None] - self.edge_list[:, 0][None], self.normal_list[:, None]) / dot((self.edge_list[:, 1] - self.edge_list[:, 0])[None], self.normal_list[:, None])
        self.finite_filter = numpy.isfinite(self.gradient)
        self.edge_filter = numpy.logical_and(self.gradient >= 0.0, self.gradient <= 1.0)
        self.position = self.edge_list[:, 0][None] + self.gradient[:, :, None] * (self.edge_list[:, 1] - self.edge_list[:, 0])[None]
        v0 = triangle_list[:, 1] - triangle_list[:, 0]
        v1 = triangle_list[:, 2] - triangle_list[:, 0]
        v2 = self.position - triangle_list[:, 0][:, None]
        d00 = dot(v0, v0)
        d01 = dot(v0, v1)
        d11 = dot(v1, v1)
        d20 = dot(v2, v0[:, None])
        d21 = dot(v2, v1[:, None])
        denom = d00 * d11 - d01 * d01
        v = (d11[:, None] * d20 - d01[:, None] * d21) / denom[:, None]
        w = (d00[:, None] * d21 - d01[:, None] * d20) / denom[:, None]
        u = 1.0 - v - w
        self.barycentric = numpy.concatenate([u[:, :, None], v[:, :, None], w[:, :, None]], axis=-1)
        self.barycentric_filter = numpy.all(numpy.logical_and(self.barycentric >= 0.0, self.barycentric <= 1.0), axis=-1)
        self.filter = numpy.logical_and(self.finite_filter, numpy.logical_and(self.edge_filter, self.barycentric_filter))
        self.index = numpy.nonzero(self.filter)
        self.point_list = self.position[self.filter]

def split_node(triangle_database, node):
    if node.triangle_indices.shape[0] <= 6:
        node.is_container = False
        return
    
    while True:
        triangle_list = triangle_database[node.triangle_indices]

        triangle_edge_box_intersection = TriangleEdgeBoundingBoxIntersection(triangle_list.position, node.bounding_box)
        box_edge_triangle_intersection = BoundingBoxEdgeTriangleIntersection(node.bounding_box, triangle_list.position)
        
        inner_triangle_vertex_mask = numpy.all(numpy.logical_and(triangle_list.position >= node.bounding_box[0], triangle_list.position <= node.bounding_box[1]), axis=-1)
        inner_triangle_mask = numpy.all(inner_triangle_vertex_mask, axis=-1)
        outer_triangle_mask = numpy.logical_not(inner_triangle_mask)

        # Initialize the vertex list by including all vertices that are contained within the bounding box
        vertex_list = triangle_list[inner_triangle_mask].position.reshape((-1, 3))
        outer_triangle_inner_vertex_mask = inner_triangle_vertex_mask[inner_triangle_mask == False]
        outer_triangle_vertex_list = triangle_list[inner_triangle_mask == False]
        external_triangle_mask = numpy.all(inner_triangle_vertex_mask == False, axis=-1)
        # In each node we pass all triangles that cannot be split.
        # If this is inherited for the parent node, it is possible to be a triangle entirely outside of the bounding box.
        # If that is not the case, triangles with all vertices outside of the bounding box must appear in the box_edge_triangle_intersection
        triangle_relevancy_mask = numpy.logical_or(numpy.logical_not(external_triangle_mask), numpy.logical_and(external_triangle_mask, numpy.any(box_edge_triangle_intersection.filter, axis=-1)))
        if numpy.all(triangle_relevancy_mask):
            break
        node.triangle_indices = node.triangle_indices[triangle_relevancy_mask]

    triangle_edge_box_intersection.filter = numpy.logical_and(triangle_edge_box_intersection.filter, outer_triangle_mask[:, None, None, None])

    # 1: Append all inner vertices of triangles that have at least one outer vertex
    vertex_list = numpy.concatenate([vertex_list, outer_triangle_vertex_list[outer_triangle_inner_vertex_mask].position], axis=0)
    # 2: Append all vertices on the bounding box sides that are intersected by triangle edge
    vertex_list = numpy.concatenate([vertex_list, triangle_edge_box_intersection.point_list], axis=0)
    # 3: Append all vertices where a triangle area is intersected by a bounding box edge
    vertex_list = numpy.concatenate([vertex_list, box_edge_triangle_intersection.point_list], axis=0)
    axis_plane_factor = numpy.median(vertex_list, axis=0)
    axis_mask = numpy.concatenate([
        numpy.all(triangle_list.position <= axis_plane_factor, axis=1)[None],
        numpy.all(triangle_list.position >= axis_plane_factor, axis=1)[None],
    ], axis=0)
    axis_mask = numpy.concatenate([
        axis_mask,
        numpy.logical_and(axis_mask[0] == False, axis_mask[1] == False)[None]
    ], axis=0)
    distribution = numpy.count_nonzero(axis_mask, axis=1).T
    assert numpy.all(numpy.sum(distribution, axis=-1) == triangle_list.shape[0])
    parent_index = -1
    if node.parent is not None:
        for i in range(2):
            if node.parent.children[i] == node:
                parent_index = i
                break
    selected_axis = numpy.argmin(distribution[:,2])
    print('%d: %d: [%d, %d]' % (node.level, parent_index, distribution[selected_axis][0] + distribution[selected_axis][2], distribution[selected_axis][1] + distribution[selected_axis][2]))
    group_mask = axis_mask[:, :, selected_axis]
    # All triangles belong to exactly one group
    assert numpy.all(numpy.count_nonzero(group_mask, axis=0) == 1)
    node.is_container = True
    node.children = [
        Data(
            triangle_indices = numpy.argwhere(numpy.logical_or(group_mask[0], group_mask[2]))[:,0],
            bounding_box = numpy.array(node.bounding_box),
            parent = node,
            level = node.level + 1
        ),
        Data(
            triangle_indices = numpy.argwhere(numpy.logical_or(group_mask[1], group_mask[2]))[:,0],
            bounding_box = numpy.array(node.bounding_box),
            parent = node,
            level = node.level + 1
        )
    ]
    node.children[0].bounding_box[1, selected_axis] = axis_plane_factor[selected_axis]
    node.children[1].bounding_box[0, selected_axis] = axis_plane_factor[selected_axis]
    assert numpy.all(node.children[0].bounding_box[1] >= node.children[0].bounding_box[0])
    # DEBUG START: prevent recursion in depth for more than one level during debugging.
    # if node.parent is not None:
    #    return
    # DEBUG END
    for i in range(2):
        split_node(triangle_database, node.children[i])
    pass


def _main():
    count_vertices = int.from_bytes(sys.stdin.buffer.read(4), byteorder = 'little')
    count_normals = int.from_bytes(sys.stdin.buffer.read(4), byteorder = 'little')
    count_texture_coords = int.from_bytes(sys.stdin.buffer.read(4), byteorder = 'little')
    count_faces = int.from_bytes(sys.stdin.buffer.read(4), byteorder = 'little')

    data_vertices = numpy.frombuffer(sys.stdin.buffer.read(count_vertices * 3 * 4), dtype = numpy.float32).reshape((count_vertices, 3))
    data_normals = numpy.frombuffer(sys.stdin.buffer.read(count_normals * 3 * 4), dtype = numpy.float32).reshape((count_normals, 3))
    data_texture_coords = numpy.frombuffer(sys.stdin.buffer.read(count_texture_coords * 2 * 4), dtype = numpy.float32).reshape((count_texture_coords, 2))
    data_triangles = numpy.frombuffer(sys.stdin.buffer.read(count_faces * 3 * 3 * 4), dtype = numpy.uint32).reshape((count_faces, 3, 3))

    vertex_attributes = numpy.dtype([('position', '3f8'), ('normal', '3f8'), ('texcoord', '2f8')])
    triangles = numpy.core.records.fromarrays([
        data_vertices[data_triangles[:, :, 0]],
        data_normals[data_triangles[:, :, 1]],
        data_texture_coords[data_triangles[:, :, 2]],
    ], dtype = vertex_attributes)
    bounding_box = triangles.position.reshape((-1, 3))
    bounding_box = numpy.concatenate([
        bounding_box.min(axis = 0)[None],
        bounding_box.max(axis = 0)[None]
    ], axis = 0)

    root_node = Data(
        triangle_indices = numpy.arange(triangles.shape[0]),
        bounding_box = bounding_box,
        parent=None,
        level=0
    )
    split_node(triangles, root_node)
    pass

if __name__ == '__main__':
    _main()