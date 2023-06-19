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
        self.gradient = (
            (self.bounding_box - self.triangle_list[:, :, None])
            /
            numpy.ma.masked_where(numpy.abs(self.edge_list[:, :, None]) < epsilon, self.edge_list[:, :, None], copy=False)
        )
        self.edge_filter = numpy.logical_and(self.gradient >= 0.0, self.gradient <= 1.0).filled(False)
        self.position = self.triangle_list[:, :, None, None] + self.gradient[:, :, :, :, None] * self.edge_list[:, :, None, None]
        self.box_filter = numpy.all(numpy.logical_and(self.position >= bounding_box[0], self.position <= bounding_box[1]), axis=-1).filled(False)
        self.inner_filter = numpy.logical_not(numpy.all(numpy.all(numpy.logical_and(triangle_list >= bounding_box[0], triangle_list <= bounding_box[1]), axis=-1), axis=-1))
        # Find all vertices that are below the side of the box.
        below_mask = triangle_list[:, :, None] <= bounding_box
        # Find all vertices that are above the side of the box.
        above_mask = triangle_list[:, :, None] >= bounding_box
        # Find all vertices epsilon-near to the sides of the box.
        at_plane_mask = numpy.abs(triangle_list[:, :, None] - bounding_box) < epsilon
        below_mask = numpy.logical_and(below_mask, numpy.logical_not(at_plane_mask))
        above_mask = numpy.logical_and(above_mask, numpy.logical_not(at_plane_mask))
        self.crossing_mask = numpy.any(numpy.logical_and(numpy.any(below_mask, axis=1), numpy.any(above_mask, axis=1)).reshape((-1, 6)), axis=-1)
        self.filter = numpy.logical_and(self.crossing_mask[:, None, None, None], numpy.logical_and(self.edge_filter, self.box_filter))
        self.index = numpy.nonzero(self.filter)
        self.point_list = numpy.ma.compress_rowcols(self.position[self.filter], 0)

class BoundingBoxEdgeTriangleIntersection:
    def __init__(self, bounding_box, triangle_list):
        self.edge_list = bounding_box[(index_bounding_box_edges, numpy.arange(3))]
        self.bounding_box = bounding_box
        self.triangle_list = triangle_list
        self.normal_list = normalize(numpy.cross(triangle_list[:, 2] - triangle_list[:, 0], triangle_list[:, 1] - triangle_list[:, 0]))
        denom = dot((self.edge_list[:, 1] - self.edge_list[:, 0])[None], self.normal_list[:, None])
        denom_mask = numpy.abs(denom) < epsilon
        denom = numpy.ma.masked_where(denom_mask, denom, copy=False)
        self.gradient = dot(triangle_list[:, 0][:, None] - self.edge_list[:, 0][None], self.normal_list[:, None]) / denom
        self.edge_filter = numpy.logical_and(self.gradient >= 0.0, self.gradient <= 1.0).filled(False)
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
        self.barycentric_filter = numpy.all(numpy.logical_and(self.barycentric >= 0.0, self.barycentric <= 1.0), axis=-1).filled(False)
        self.filter = numpy.logical_and(self.edge_filter, self.barycentric_filter)
        self.index = numpy.nonzero(self.filter)
        self.point_list = numpy.ma.compress_rowcols(self.position[self.filter], 0)

class UniqueFloat:
    def __init__(self, array):
        if not isinstance(array, numpy.ndarray) or not numpy.issubdtype(array.dtype, numpy.floating):
            raise ValueError('Expected NDArray of floating type')
        self.values, self.index = numpy.unique(array.ravel(), return_inverse=True)
        self.index = self.index.reshape(array.shape)

def split_node(triangles, node):
    # First we must determine which triangles to include in the node
    # We include:
    # 1. All triangles with vertex inside the bounding box (including those epsilon-near its sides);
    # 2. All triangles whose edges intersect the bounding box;
    # 3. All triangles that are intersected by the bounding box edges;
    triangle_edge_box_intersection = TriangleEdgeBoundingBoxIntersection(triangles.position, node.bounding_box)
    box_edge_triangle_intersection = BoundingBoxEdgeTriangleIntersection(node.bounding_box, triangles.position)
    inner_triangle_vertex_mask = numpy.all(numpy.logical_and(triangles.position >= node.bounding_box[0], triangles.position <= node.bounding_box[1]), axis=-1)
    
    triangle_mask = numpy.logical_or(
        numpy.any(inner_triangle_vertex_mask, axis=1),
        numpy.logical_or(
            numpy.any(triangle_edge_box_intersection.filter, axis=(1, 2, 3)),
            numpy.any(box_edge_triangle_intersection.filter, axis=1)
        )
    )

    triangle_list = triangles[triangle_mask]
    node.index_list = numpy.nonzero(triangle_mask)[0]
    if (
        node.parent is not None
        and
        numpy.intersect1d(node.index_list, node.parent.index_list).shape[0] == node.parent.index_list.shape[0]
    ):
        return False
    assert node.parent is None or numpy.intersect1d(node.index_list, node.parent.index_list).shape[0] == node.index_list.shape[0]

    print('%d: %d' % (node.level, triangle_list.shape[0]))

    if triangle_list.shape[0] <= 6:
        node.is_container = False
        node.max_triangle_count = triangle_list.shape[0]
        node.max_depth = 0
        return True

    vertex_list = triangles.position[inner_triangle_vertex_mask]
    vertex_list = numpy.concatenate([vertex_list, triangle_edge_box_intersection.point_list], axis=0)
    vertex_list = numpy.concatenate([vertex_list, box_edge_triangle_intersection.point_list], axis=0)

    box_size = node.bounding_box[1] - node.bounding_box[0]
    assert numpy.all(box_size >= epsilon)
    for selected_axis in numpy.flip(numpy.argsort(box_size)):
        # split_factor = numpy.mean(vertex_list[:, selected_axis])
        split_factor = numpy.median(vertex_list[:, selected_axis])
        if node.bounding_box[1, selected_axis] - split_factor < epsilon or split_factor - node.bounding_box[0, selected_axis] < epsilon:
            continue
        children = [
            Data(
                bounding_box = numpy.array(node.bounding_box),
                parent = node,
                level = node.level + 1
            ),
            Data(
                bounding_box = numpy.array(node.bounding_box),
                parent = node,
                level = node.level + 1
            )
        ]
        children[0].bounding_box[1, selected_axis] = split_factor
        children[1].bounding_box[0, selected_axis] = split_factor
        distinct_children = True
        for child in children:
            if not split_node(triangles, child):
                distinct_children = False
                break
        if distinct_children:
            node.is_container = True
            node.children = children
            node.max_triangle_count = max(children[0].max_triangle_count, children[1].max_triangle_count)
            node.max_depth = 1 + max(children[0].max_depth, children[1].max_depth)
            return True
    else:
        node.is_container = False
        node.max_triangle_count = triangle_list.shape[0]
        node.max_depth = 0
        return True


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
        bounding_box = bounding_box,
        parent = None,
        level = 0
    )
    split_node(triangles, root_node)
    pass

if __name__ == '__main__':
    _main()