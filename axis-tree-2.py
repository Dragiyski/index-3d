import sys
import numpy
import itertools

numpy.set_printoptions(floatmode = 'maxprec', suppress = True)
epsilon = numpy.finfo(numpy.float32).eps
index_bounding_box_edges = numpy.array(list(itertools.combinations(itertools.product([0, 1], repeat = 3), 2)))
index_bounding_box_mask = index_bounding_box_edges[:, 0] != index_bounding_box_edges[:, 1]
index_bounding_box_edges = index_bounding_box_edges[numpy.count_nonzero(index_bounding_box_mask, axis = -1) == 1]
index_bounding_box_edge_normal = numpy.nonzero(index_bounding_box_edges[:, 0] != index_bounding_box_edges[:, 1])[1]
del index_bounding_box_mask

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def normalize(x):
    y = x.reshape((-1, x.shape[-1]))
    return (y / numpy.linalg.norm(y, axis = -1)[:, None]).reshape(x.shape)

def dot(x, y):
    return numpy.sum(x * y, axis = -1)

def compile_node(triangle_data, selection_index, parent = None):
    triangle_list = triangle_data[selection_index]
    triangle_flat_vertex_list = triangle_list.position.reshape(-1, 3)
    bounding_box = numpy.concatenate([
        triangle_flat_vertex_list.min(axis = 0)[None],
        triangle_flat_vertex_list.max(axis = 0)[None]
    ], axis = 0)
    if selection_index.shape[0] < 6:
        return Data(
            is_container = False,
            bounding_box=bounding_box,
            index = selection_index,
            depth = 0 if parent is None else (parent.depth + 1),
            parent = parent
        )
    split_factor = numpy.median(triangle_flat_vertex_list, axis=0)
    below_mask = triangle_list.position <= split_factor
    above_mask = triangle_list.position >= split_factor
    below_triangle_mask = numpy.any(below_mask, axis=1)
    above_triangle_mask = numpy.any(above_mask, axis=1)
    mixed_triangle_mask = numpy.logical_and(below_triangle_mask, above_triangle_mask)
    mixed_distribution = numpy.count_nonzero(mixed_triangle_mask, axis=0)
    for selected_axis in numpy.argsort(mixed_distribution):
        axis_below_triangle_mask = below_triangle_mask[:, selected_axis]
        axis_above_triangle_mask = above_triangle_mask[:, selected_axis]
        axis_mixed_triangle_mask = mixed_triangle_mask[:, selected_axis]
        axis_below_pure_triangle_mask = numpy.logical_and(axis_below_triangle_mask, numpy.logical_not(axis_mixed_triangle_mask))
        axis_above_pure_triangle_mask = numpy.logical_and(axis_above_triangle_mask, numpy.logical_not(axis_mixed_triangle_mask))
        axis_distribution = numpy.array([
            numpy.count_nonzero(axis_below_pure_triangle_mask),
            numpy.count_nonzero(axis_above_pure_triangle_mask),
            numpy.count_nonzero(axis_mixed_triangle_mask),
        ])
        if axis_distribution[0] == 0 or axis_distribution[1] == 0:
            continue
        node = Data(
            is_container = True,
            bounding_box = bounding_box,
            triangle_index = selection_index,
            index = selection_index,
            axis = selected_axis,
            split_factor = split_factor[selected_axis],
            depth = 0 if parent is None else (parent.depth + 1),
            parent = parent
        )
        print('%d -> %d: [%d, %d, %d]' % (node.depth, selected_axis, *axis_distribution))
        below_child = compile_node(triangle_data, selection_index[axis_below_triangle_mask], node)
        above_child = compile_node(triangle_data, selection_index[axis_above_triangle_mask], node)
        if below_child is not None and above_child is not None:
            node.children = [below_child, above_child]
            return node
    else:
        print('What Now?')
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
    root_node = compile_node(triangles, numpy.arange(triangles.shape[0]))
    pass

if __name__ == '__main__':
    _main()