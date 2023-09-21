import sys
import numpy

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def compile_node(triangle_data, selection, parent = None):
    if len(selection) <= 0:
        return None
    triangle_list = triangle_data[selection]
    vertex_list = triangle_list.position.reshape(-1, 3)
    bounding_box = numpy.concatenate([
        vertex_list.min(axis = 0)[None],
        vertex_list.max(axis = 0)[None]
    ], axis = 0)
    print(f'Depth: {0 if parent is None else (parent.depth + 1)}, Input: {len(selection)}')
    if len(selection) <= 6:
        return Data(
            is_container = False,
            bounding_box = bounding_box,
            index = selection,
            depth = 0 if parent is None else (parent.depth + 1),
            max_depth = 0,
            parent = parent
        )
    distribution_list = []
    for dim in range(3):
        # Goal: Assign every triangle to a group such that we can reduce the amount of work per node.
        # Idea: We have (axis-aligned) bounding box. We can assign (for each dimension)
        # whether the triangle is closed to the bounding box minimum or maximum.
        # TODO: There is probably some information loss/caveats here.
        min_vertex = triangle_list.position.min(axis=-2)[:, dim]
        max_vertex = triangle_list.position.max(axis=-2)[:, dim]
        min_affinity = (min_vertex - bounding_box[0, dim]) - (bounding_box[1, dim] - max_vertex)
        min_sort = numpy.argsort(min_affinity)
        groups = []
        groups.append(min_sort[0 : selection.shape[0] // 2])
        groups.append(min_sort[selection.shape[0] // 2 : ])
        min_isect = min_vertex[groups[1]].min()
        max_isect = max_vertex[groups[0]].max()
        if min_isect < max_isect:
            group_sort = [
                numpy.argsort(max_vertex[groups[0]])[::-1],
                numpy.argsort(min_vertex[groups[1]]),
            ]
            min_cut = min(len(groups[0]) // 2, numpy.count_nonzero(max_vertex[groups[0][group_sort[0]]] >= min_isect))
            max_cut = min(len(groups[1]) // 2, numpy.count_nonzero(min_vertex[groups[1][group_sort[1]]] <= max_isect))
            g2 = numpy.union1d(groups[0][group_sort[0][:min_cut]], groups[1][group_sort[1][:max_cut]])
            groups[0] = numpy.setdiff1d(groups[0], g2)
            groups[1] = numpy.setdiff1d(groups[1], g2)
            groups.append(g2)
            pass
        assert len(groups[0]) > 0
        assert len(groups[1]) > 0
        distribution_list.append(groups)
    # We split the triangles based on one-dimensional data for each dimensions
    # The best split are those that are more balanced, and with lower amount group[2] triangles
    rating = []
    for distribution in distribution_list:
        r = numpy.abs(distribution[1].shape[0] - distribution[0].shape[0])
        if len(distribution) > 2:
            r += len(distribution[2])
        rating.append(r)
    node = Data(
        is_container = True,
        bounding_box = bounding_box,
        depth = 0 if parent is None else (parent.depth + 1),
        parent = parent
    )
    for dim in numpy.argsort(rating):
        children = []
        count = []
        for distribution in distribution_list[dim]:
            child = compile_node(triangle_data, selection[distribution], node)
            if child is not None:
                children.append(child)
                count.append(len(distribution))
        if len(children) > 1:
            node.children = children
            node.distribution = count
            node.max_depth = 1 + numpy.maximum.reduce(list(c.max_depth for c in node.children))
            return node
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
    # Root node now contains (potentially semi-balanced/non-balanced tree)
    pass


if __name__ == '__main__':
    _main()
