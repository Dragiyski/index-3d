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
    vertex_list = triangle_list.position.reshape(-1, 3)
    bounding_box = numpy.concatenate([
        vertex_list.min(axis = 0)[None],
        vertex_list.max(axis = 0)[None]
    ], axis = 0)
    node = Data(
        bounding_box = bounding_box,
        index = selection_index,
        depth = 0 if parent is None else (parent.depth + 1),
        parent = parent
    )
    if node.depth >= 256:
        raise RuntimeError('Index tree too deep: > 256 levels')
    if selection_index.shape[0] <= 6:
        node.is_container = False
        print('> %d: %d' % (node.depth, selection_index.shape[0]))
        return node
    
    split_factor_list = numpy.concatenate([
        numpy.median(vertex_list, axis = 0)[None],
        ((bounding_box[1] + bounding_box[0]) * 0.5)[None],
        numpy.mean(vertex_list, axis = 0)[None]
    ])
    vertex_split_diff_list = triangle_list.position[:, :, None] - split_factor_list
    # Shape:
    # 3 groups: below plane, above plane, at plane
    # N triangles:
    # 3 vertex per triangle
    # 3 split factors
    # 3 dimensions
    vertex_split_factor_mask = numpy.concatenate([
        (vertex_split_diff_list <= -epsilon)[None],
        (vertex_split_diff_list >= +epsilon)[None],
        (numpy.abs(vertex_split_diff_list) < epsilon)[None]
    ], axis = 0)
    # This gives triangles with *all* vertices below, above, and *all* vertices *at* the plane, which is most likely zero (0).
    # The sum will be <= number of triangles.
    triangle_split_factor_mask = numpy.all(vertex_split_factor_mask, axis = 2)
    # We reassign the group[2] into the mask of all triangles not in group 0 and 1, i.e. those that are split.
    triangle_split_factor_mask[2] = numpy.logical_and(numpy.logical_not(triangle_split_factor_mask[0]), numpy.logical_not(triangle_split_factor_mask[1]))
    # 3 split factors
    # 3 dimensions
    # 3 groups: below plane, above plane, at plane
    distribution_list = numpy.moveaxis(numpy.count_nonzero(triangle_split_factor_mask, axis = 1), 0, -1)
    pass

    # There is no easy way to split the volume in two along the axis and keep the split triangles at zero.
    # But we do not need to do that to implement bounding box optimization.
    # SOLUTION: 3 children nodes:
    # First 2 children is what we expect, they divide the node volume in 2 and include triangles fully contained below or above the splitting plane.
    # The bounding boxes of those two shapes is subset of (or match) the bounding box in the parent up to the split location of the chosen axis.
    # The last children is a subset of (or match) the volume defined by the bounding box in the parent, but even if it matches, it will contain fewer triangles
    # than the parent.
    # First 2 children are expected to have nearly equivalent set of triangles (usually by the mean),
    # if any of those children contains too few triangles we might reject the splitting method or axis.
    # This is especially important if group[2] contains all the triangles: distribution [0, 0, N] will lead to infinite loop (so does [N, 0, 0] or [0, N, 0])
    # A variant [1, 1, N] with large N can also be undesirable, as it can lead to large depth of the generated tree.
    # Ideally, min(distribution) should be > 0, and min(distribution[0], distribution[1]) > max(distribution) / max_depth
    # For example, max_depth = 256, with splitting 8530 triangles means  8530 // 256 = 33. Any divisition where group[0] or group[1] < 33, will be bad.
    for distribution_flat_index in numpy.argsort(distribution_list[:, :, 2] + numpy.abs(distribution_list[:, :, 1] - distribution_list[:, :, 0]), axis=None):
        split_factor_index = distribution_flat_index // 3
        dimension_index = distribution_flat_index % 3
        split_factor = split_factor_list[split_factor_index, dimension_index]
        triangle_mask = triangle_split_factor_mask[:, :, split_factor_index, dimension_index]
        distribution = distribution_list[split_factor_index, dimension_index]
        children = []
        for group_index in range(3):
            group_list = numpy.nonzero(triangle_mask[group_index])[0]
            if group_list.shape[0] > 0:
                if numpy.setdiff1d(selection_index, selection_index[group_list]).shape[0] <= 0:
                    # A child node must be a proper subset.
                    # If a child contains all triangles, since each triangle is exactly in one child, other children would be empty.
                    # As a result, this distribution would be denied and the loop will try another distribution.
                    continue
                child = compile_node(triangle_data, selection_index[group_list], node)
                if child is not None:
                    children.append(child)
        # We only consider a node where triangles are split into two or more (namely, 3) children, each of them will be non-empty.
        if len(children) >= 2:
            node.children = children
            node.split_factor = split_factor
            node.distribution = distribution
            node.is_container = True
            print('+ %d: %r' % (node.depth, list(node.distribution)))
            return node
        numpy.searchsorted
    return None

def insert_float_in_storage(storage, value):
    index_min = 0
    index_max = len(storage.float)
    while index_max > index_min:
        index_mid = (index_max + index_min) // 2
        value_mid = storage.float[index_mid]
        diff_mid = value - value_mid
        if diff_mid <= -epsilon:
            index_max = index_mid
        elif diff_mid >= epsilon:
            index_min = index_mid + 1
        else:
            index_min = index_mid
            break
    storage.float.insert(index_min, value)
    return index_min

def insert_float_ndarray_in_storage(storage, ndarray):
    index_list = numpy.array(list(insert_float_in_storage(storage, value) for value in ndarray.ravel()))
    return index_list.reshape(ndarray.shape)

OBJECT_TYPE_BOUNDING_BOX = 1
OBJECT_TYPE_MESH_TRIANGLE = 2

def insert_node_in_storage(storage, node):
    float_ref = len(storage.float)
    int_ref = len(storage.int)
    object_ref = len(storage.object)
    tree_ref = len(storage.tree)
    if node.is_container:
        storage.object.append([OBJECT_TYPE_BOUNDING_BOX, -1, -1, float_ref])
        storage.float.extend(list(node.bounding_box.ravel()))
        storage.tree.append([-1, -1, -1, object_ref])
        node.tree_ref = tree_ref
        if node.parent is not None:
            storage.tree[tree_ref][0] = node.parent.tree_ref
        child_ref_list = []
        for child in node.children:
            child_ref_list.append(insert_node_in_storage(storage, child))
        if len(child_ref_list) > 0:
            storage.object[object_ref][1] = child_ref_list[0]
        for i in range(1, len(child_ref_list)):
            storage.tree[child_ref_list[i]][1] = child_ref_list[i - 1]
        for i in range(len(child_ref_list) - 1):
            storage.tree[child_ref_list[i]][2] = child_ref_list[i + 1]
    else:
        storage.object.append([OBJECT_TYPE_MESH_TRIANGLE, -1, -1, float_ref])
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
    index_attributes = numpy.dtype([('position', '3u4'), ('normal', '3u4'), ('texcoord', '2u4')])
    triangles = numpy.core.records.fromarrays([
        data_vertices[data_triangles[:, :, 0]],
        data_normals[data_triangles[:, :, 1]],
        data_texture_coords[data_triangles[:, :, 2]],
    ], dtype = vertex_attributes)

    root_node = compile_node(triangles, numpy.arange(triangles.shape[0]))

    storage = Data(
        float=[],
        int=[],
        object=[],
        tree=[]
    )

    insert_node_in_storage(storage, root_node)

    # float_list_ravel = triangles.view('<f8', numpy.ndarray).astype(numpy.float32).ravel()
    # float_unique_value, float_list_index = numpy.unique(float_list_ravel, return_inverse=True)
    # triangle_index = float_list_index.astype(numpy.uint32).view(index_attributes, numpy.recarray)
    # triangle_index_count = float_unique_value.shape[0]
    # triangle_position_index = triangle_index.position.astype(numpy.int64)
    # triangle_position_flat_index = triangle_position_index[:, 0] * triangle_index_count * triangle_index_count + triangle_position_index[:, 1] * triangle_index_count + triangle_position_index[:, 2]
    # triangle_normal_index = triangle_index.normal.astype(numpy.int64)
    # triangle_normal_flat_index = triangle_normal_index[:, 0] * triangle_index_count * triangle_index_count + triangle_normal_index[:, 1] * triangle_index_count + triangle_normal_index[:, 2]
    # triple_flat_index = numpy.concatenate([triangle_position_flat_index, triangle_normal_flat_index])
    # triple_unique_flat_index, triple_value_index = numpy.unique(triple_flat_index, return_inverse = True)
    # triangle_position_value_index = triple_value_index[:triangle_position_flat_index.shape[0]]
    # triangle_normal_value_index = triple_value_index[triangle_position_flat_index.shape[0]:]
    # triple_unique_index = numpy.concatenate([
    #     (triple_unique_flat_index // (triangle_index_count * triangle_index_count))[:, None],
    #     (triple_unique_flat_index % (triangle_index_count * triangle_index_count) // triangle_index_count)[:, None],
    #     (triple_unique_flat_index % triangle_index_count)[:, None],
    # ], axis = -1)
    # triangle_texcoord_index = triangle_index.texcoord.astype(numpy.int64)
    # triangle_texcoord_flat_index = triangle_texcoord_index[:, 0] * triangle_index_count + triangle_texcoord_index[:, 1]
    # double_flat_index = triangle_texcoord_flat_index
    # double_unique_flat_index, double_value_index = numpy.unique(double_flat_index, return_inverse = True)
    # triangle_texcoord_value_index = double_value_index
    # double_unique_index = numpy.concatenate([
    #     (double_unique_flat_index // triangle_index_count)[:, None],
    #     (double_unique_flat_index % triangle_index_count)[:, None],
    # ], axis = -1)

if __name__ == '__main__':
    _main()