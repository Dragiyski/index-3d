import sys
import numpy

numpy.set_printoptions(floatmode='maxprec', suppress=True)
epsilon = numpy.finfo(numpy.float32).eps

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def normalize(x):
    return x / numpy.linalg.norm(x, axis=-1)[:, None]

def array_dot(x, y):
    assert len(x.shape) == len(y.shape)
    assert numpy.all(numpy.array(x.shape) == numpy.array(y.shape))
    return numpy.sum(x * y, axis=-1)

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


vertex_position = numpy.array([0, 1, 2], dtype=numpy.uint8)
vertex_normal = numpy.array([3, 4, 5], dtype=numpy.uint8)
vertex_texcoord = numpy.array([6, 7], dtype=numpy.uint8)

def v1_create_storage(source):
    fl = source.flatten()
    sfl = numpy.argsort(fl)
    ifl = numpy.argsort(sfl)
    ufl, rfl = numpy.unique(fl[sfl], return_inverse=True)
    return Data(
        float=ufl,
        index=rfl[ifl].reshape(source.shape)
    )

def v1_create_node(storage, selected_triangle_index):
    # TODO: If less than 6 triangles are selected, just add them to the node, no need to split
    triangles = storage.index[selected_triangle_index]
    triangle_vertices = triangles[:, :, vertex_position]
    uvx, vx_triangles = numpy.unique(triangle_vertices.reshape((-1, 3)), axis=0, return_inverse=True)
    vx_triangles = vx_triangles.reshape(triangle_vertices.shape[0:2])
    split_info = [None] * 3
    split_dim = -1
    split_separation_count = numpy.inf
    for dim in range(3):
        split = split_info[dim] = Data()
        split.vertex_rejection = storage.float[uvx][:, dim]
        split.plane_factor = numpy.median(split.vertex_rejection)
        split.normal = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
        split.normal[dim] = 1.0
        split.point = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
        split.point[dim] = split.plane_factor
        split.triangle_rejection = storage.float[storage.index[:, :, vertex_position[dim]]]
        split.is_plane_index = numpy.abs(split.triangle_rejection - split.plane_factor) < epsilon
        split.is_left_index = numpy.logical_and(numpy.logical_not(split.is_plane_index), split.triangle_rejection < split.plane_factor)
        split.is_right_index = numpy.logical_and(numpy.logical_not(split.is_plane_index), split.triangle_rejection >= split.plane_factor)
        split.groups = [
            numpy.argwhere(numpy.all(split.is_left_index, axis=-1))[:, 0],
            numpy.argwhere(numpy.all(split.is_right_index, axis=-1))[:, 0],
        ]
        split.offending = numpy.delete(numpy.arange(split.triangle_rejection.shape[0]), numpy.union1d(split.groups[0], split.groups[1]))
        # Offending triangles vertices are in 3 groups:
        # Plane vertices lies near the split plane;
        # Left vertices dot(normal, vertex) < plane_factor
        # Right vertices dot(normal, vertex) > plane_factor

        # Triangle can have:
        # 3 plane vertices = triangle lies on the plane, assign the triangle to the smaller group:
        # - Vertices might move epsilon distance to fit in the box;
        # Note: because plane triangles can apply to each group, they do not count to the imbalance.
        # 2 plane vertices = one vertex is not plane, it determines which group the triangle belongs;
        # 1 plane vertex:
        # Case 1: The other two vertices are in the same half-space = assign the triangle to that group;
        # Case 2: The other two vertices are in the opposite half-space = split the triangle in two;
        # 0 plane vertices:
        # Have a lonely vertex in one half-space and couple vertices in the other half-space,
        # the triangle is split on edges connecting the lonely vertex.

        split.separation_count = 0
        split.distribution = numpy.array(list(x.shape[0] for x in split.groups))
        split.offending_has_plane_vertex = list(numpy.argwhere(numpy.count_nonzero(split.is_plane_index[split.offending], axis=-1) == i)[:, 0] for i in range(4))
        # 3 plane vertices: triangle can go into either group, so we store this count separately;
        split.free_distribution = numpy.count_nonzero(numpy.any(split.is_left_index[split.offending[split.offending_has_plane_vertex[2]]], axis=-1))
        # 2 plane vertices: the triangle goes into the group where the non-plane vertex is located.
        split.distribution[0] += numpy.count_nonzero(numpy.any(split.is_left_index[split.offending[split.offending_has_plane_vertex[2]]], axis=-1))
        split.distribution[1] += numpy.count_nonzero(numpy.any(split.is_right_index[split.offending[split.offending_has_plane_vertex[2]]], axis=-1))
        # 1 plane vertex: has two possible cases
        # Case 1: The two non-plane verties are in the same group - put the triangle into that group
        split.distribution[0] += split.offending[split.offending_has_plane_vertex[1]][numpy.count_nonzero(split.is_left_index[split.offending[split.offending_has_plane_vertex[1]]], axis=-1) == 2].shape[0]
        split.distribution[1] += split.offending[split.offending_has_plane_vertex[1]][numpy.count_nonzero(split.is_right_index[split.offending[split.offending_has_plane_vertex[1]]], axis=-1) == 2].shape[0]
        # Case 2: The two non-plane vertices are in different groups, we split the triangle and put subtriangle in each group
        t1p_split = split.offending[split.offending_has_plane_vertex[1]][numpy.logical_and(numpy.count_nonzero(split.is_left_index[split.offending[split.offending_has_plane_vertex[1]]], axis=-1) == 1, numpy.count_nonzero(split.is_right_index[split.offending[split.offending_has_plane_vertex[1]]], axis=-1) == 1)].shape[0]
        split.distribution[0] += t1p_split
        split.distribution[1] += t1p_split
        split.separation_count += t1p_split
        # 0 plane vertices: put one triangle in the lonely vertex's group and 2 triangles into the couple vertices' group
        tl_left = numpy.count_nonzero(numpy.count_nonzero(split.is_left_index[split.offending[split.offending_has_plane_vertex[0]]], axis=-1) == 1)
        tl_right = numpy.count_nonzero(numpy.count_nonzero(split.is_right_index[split.offending[split.offending_has_plane_vertex[0]]], axis=-1) == 1)
        split.distribution[0] += tl_left + 2 * tl_right
        split.distribution[1] += tl_right + 2 * tl_left
        split.separation_count += split.offending_has_plane_vertex[0].shape[0] * 3

        if split.separation_count < split_separation_count:
            split_separation_count = split.separation_count
            split_dim = dim

    split = split_info[split_dim]
    dim = split_dim
    new_vertices = numpy.array([
        storage.float[split.triangle_order[0]] + split.plane_intersection[0][:, None] * (storage.float[split.triangle_order[1]] - storage.float[split.triangle_order[0]]),
        storage.float[split.triangle_order[0]] + split.plane_intersection[1][:, None] * (storage.float[split.triangle_order[2]] - storage.float[split.triangle_order[0]]),
    ])
    # Append case3 in the existing groups
    numpy.hstack([split.groups[0], split.offending[split.case3[split.case3_dir == False]]])
    numpy.hstack([split.groups[1], split.offending[split.case3[split.case3_dir == True]]])

    new_group_triangle = [numpy.array([], dtype=numpy.float32), numpy.array([], dtype=numpy.float32)]
    new_group_triangle_index = [numpy.array([], dtype=split.groups[0].dtype), numpy.array([], dtype=split.groups[1].dtype)]
    # new_triangle_vertices = numpy.array()

    # Case 4: Only one of the new vertices is used, the other matches an existing triangle vertex
    case4_plane_vertex_index = numpy.argwhere(numpy.abs(storage.float[split.triangle_index[split.case4]][:, :, split_dim] - split.plane_factor) < epsilon)[:, 1]
    case4_other_vertex_index = numpy.tile([0, 1, 2], split.case4.shape[0]).reshape(split.case4.shape[0], 3)
    case4_other_vertex_index = case4_other_vertex_index[numpy.arange(3) != case4_plane_vertex_index[:, None]].reshape((case4_other_vertex_index.shape[0], case4_other_vertex_index.shape[1]-1))
    case4_intersection_index = numpy.argwhere(numpy.logical_and(split.plane_intersection.T[split.case4] >= 0.0, 1.0 - split.plane_intersection.T[split.case4] >= epsilon))
    case4_intersection_factor = split.plane_intersection.T[split.case4][case4_intersection_index[:, 0], case4_intersection_index[:, 1]]
    case4_float_plane_vertex = storage.float[split.triangle_order[case4_plane_vertex_index, split.case4]]
    case4_float_new_vertex = new_vertices[case4_intersection_index[:, 1], split.case4]
    case4_min_vertex_index = numpy.argwhere(storage.float[split.triangle_order[case4_other_vertex_index, split.case4]][:, :, split_dim] < split.plane_factor)
    case4_max_vertex_index = numpy.argwhere(storage.float[split.triangle_order[case4_other_vertex_index, split.case4]][:, :, split_dim] >= split.plane_factor)
    case4_min_vertex_index = case4_other_vertex_index[case4_min_vertex_index[:, 0], case4_min_vertex_index[:, 1]]
    case4_max_vertex_index = case4_other_vertex_index[case4_max_vertex_index[:, 0], case4_max_vertex_index[:, 1]]
    case4_float_min_vertex = storage.float[split.triangle_index[split.case4][numpy.arange(split.case4.shape[0]), case4_min_vertex_index]]
    case4_float_max_vertex = storage.float[split.triangle_index[split.case4][numpy.arange(split.case4.shape[0]), case4_min_vertex_index]]
    case4_min_triangles = numpy.dstack([case4_float_plane_vertex, case4_float_new_vertex, case4_float_min_vertex])
    case4_max_triangles = numpy.dstack([case4_float_plane_vertex, case4_float_new_vertex, case4_float_max_vertex])
    pass
    
    assert case4_plane_vertex_index.shape[0] == split.case4.shape[0]
    assert case4_new_vertex_index.shape[0] == split.case4.shape[0]
    float_case4_new_vertex = new_vertices[case4_new_vertex_index, split.case4]
    case4_plane_vertex = split.triangle_index[split.case4, case4_plane_vertex_index]
    case4_other_index = case4_other_index[numpy.arange(3) != case4_plane_vertex_index[:, None]].reshape((case4_other_index.shape[0], case4_other_index.shape[1]-1))
    case4_other_vertex_index = split.triangle_index[split.case4, case4_other_index]
    case4_other_rejections = split.triangle_rejections[split.offending[split.case4], case4_other_index]
    case4_space_vertex_index = numpy.array([
        numpy.argwhere(case4_other_rejections < 0.0)[:, 1],
        numpy.argwhere(case4_other_rejections >= 0.0)[:, 1],
    ])
    assert numpy.count_nonzero(numpy.abs(case4_other_rejections.flatten()) < epsilon) == 0
    
    # TODO: Do this after finding all vertices to insert
    # 1. We generate array (N, 3, K) where N is the number of new triangles, K=8 is the number of elements per vertex (vec3 position, vec3 normal, vec2 texcoord)
    # 2. Then flatten the result to get a list of floats, pass the result to unique to get unique floats;
    # 3. Optional: search current array for matching floats, that is make a N^2 grid of the new floats by numpy.abs(numpy.subtract.outer),
    # then for all N [x, x] indices (matching in both dimensions), set to inf, then find all elements that are less than epsilon.
    # For this new mapping, remove elements from the old mapping and make inverse index.
    # 4. As shown below, use numpy.subtract.outer (memory intensive) to get mapping of new floats to old ones;
    # 5. Remap the new traingles (N, 3, K) with the total_float index.
    # 6. Remap the old triangles in storage.index into the total_float index.
    # 7. Append the new triangles to old triangles in storage.index
    new_vertices_float = new_vertices.reshape((-1, 3)).flatten()
    new_float, new_vertices_index = numpy.unique(new_vertices_float, return_inverse=True)
    new_vertices_index = new_vertices_index.reshape(new_vertices.shape)
    new_float_offset = storage.float.shape[0]
    new_old_float_remap = numpy.argwhere(numpy.abs(numpy.subtract.outer(new_float, storage.float)) < epsilon)
    filter_float = numpy.delete(new_float, new_old_float_remap[:, 0])
    filter_float_forward_index = numpy.delete(numpy.arange(new_float.shape[0]), new_old_float_remap[:, 0])
    filter_float_index = numpy.arange(filter_float_forward_index.shape[0])
    filter_float_forward_index = numpy.dstack([filter_float_forward_index, filter_float_index])[0]
    del filter_float_index
    filter_float_inverse_index = numpy.repeat(-1, new_float.shape[0])
    filter_float_inverse_index[new_old_float_remap[:, 0]] = new_old_float_remap[:, 1]
    filter_float_inverse_index[filter_float_forward_index[:, 0]] = new_float_offset + filter_float_forward_index[:, 1]
    total_float = numpy.hstack([storage.float, filter_float])
    total_float_forward_map = numpy.argsort(total_float)
    total_float_inverse_map = numpy.argsort(total_float_forward_map)
    assert numpy.count_nonzero(total_float[total_float_forward_map][total_float_inverse_map[storage.index]] != storage.float[storage.index]) == 0, 'Original data is not preserved'
    pass

class DataContext:
    def __init__(self, vertex_position_index=[0, 1, 2], vertex_normal_index=[3, 4, 5], vertex_texcoord_index=[6, 7]):
        self.vertex_position_index = numpy.array(vertex_position_index, dtype=numpy.int32)
        self.vertex_normal_index = numpy.array(vertex_normal_index, dtype=numpy.int32)
        self.vertex_texcoord_index = numpy.array(vertex_texcoord_index, dtype=numpy.int32)

    def create_node(self, triangles):
        assert isinstance(triangles, numpy.ndarray)
        assert len(triangles.shape) == 3
        assert triangles.shape[1] == 3
        assert triangles[:, :, self.vertex_position_index].shape[1:3] == (3, 3)
        assert triangles[:, :, self.vertex_normal_index].shape[1:3] == (3, 3)
        assert triangles[:, :, self.vertex_texcoord_index].shape[1:3] == (3, 2)

        triangle_vertex_position = triangles[:, :, self.vertex_position_index]
        plane_factor = numpy.median(numpy.swapaxes(triangle_vertex_position, 0, -1).reshape((3, -1)), axis=-1)

        at_plane = numpy.array(list(numpy.abs(triangle_vertex_position[:, :, i] - plane_factor[i]) < epsilon for i in range(3)))
        below = numpy.array(list(numpy.less.outer(triangle_vertex_position[:, :, i], plane_factor[i]) for i in range(3)))
        below = numpy.logical_and(numpy.logical_not(at_plane), below)
        above = numpy.array(list(numpy.greater_equal.outer(triangle_vertex_position[:, :, i], plane_factor[i]) for i in range(3)))
        above = numpy.logical_and(numpy.logical_not(at_plane), above)
        all_below = numpy.all(below, axis=-1)
        all_above = numpy.all(above, axis=-1)
        count_below = numpy.count_nonzero(all_below, axis=-1)
        count_above = numpy.count_nonzero(all_above, axis=-1)
        count_conflict = triangles.shape[0] - (count_below + count_above)
        count_metric = numpy.array(list(count_conflict[i] if count_above[i] > 0 and count_below[i] > 0 else numpy.inf for i in range(3)))
        split_dimension = numpy.argmin(count_metric)
        assert count_metric[split_dimension] <= triangles.shape[0]

        groups = [
            triangles[numpy.argwhere(all_below[split_dimension])[:, 0]],
            triangles[numpy.argwhere(all_above[split_dimension])[:, 0]],
        ]

        in_conflict = numpy.argwhere(numpy.logical_not(numpy.logical_or(all_below[split_dimension], all_above[split_dimension])))[:, 0]
        ct_at_plane = at_plane[split_dimension, in_conflict]
        ct_at_plane_count = numpy.count_nonzero(ct_at_plane, axis=-1)

        # If the triangle has 2 vertices at the splitting plane:
        # insert the triangle into the group determined by the non-plane vertex
        ct2_plane_vertices = numpy.argwhere(ct_at_plane_count == 2)[:,0]
        numpy.vstack([groups[0], triangles[in_conflict[ct2_plane_vertices][numpy.any(below[split_dimension, in_conflict[ct2_plane_vertices]], axis=-1)]]])
        numpy.vstack([groups[1], triangles[in_conflict[ct2_plane_vertices][numpy.any(above[split_dimension, in_conflict[ct2_plane_vertices]], axis=-1)]]])

        # If the triangle has 1 vertices at the splitting plane:
        ct1_plane_vertex = numpy.argwhere(ct_at_plane_count == 1)[:,0]
        ct1_below_count = numpy.count_nonzero(below[split_dimension, in_conflict[ct1_plane_vertex]], axis=-1)
        ct1_above_count = numpy.count_nonzero(above[split_dimension, in_conflict[ct1_plane_vertex]], axis=-1)
        # If it has two vertices in the same group, add it to that group:
        numpy.vstack([groups[0], triangles[in_conflict[ct1_plane_vertex[ct1_below_count == 2]]]])
        numpy.vstack([groups[1], triangles[in_conflict[ct1_plane_vertex[ct1_above_count == 2]]]])
        # Otherwise we split the triangle into two, a new vertex is inserted at the intersection point
        # at the edge formed by non-plane vertices
        assert numpy.all(numpy.argwhere(ct1_below_count == 1)[:,0] == numpy.argwhere(ct1_above_count == 1)[:,0])
        ct1_triangle_index = in_conflict[ct1_plane_vertex[numpy.argwhere(ct1_below_count == 1)[:,0]]]
        ct1_below_vertex = numpy.argwhere(below[split_dimension, in_conflict[ct1_plane_vertex]][ct1_below_count == 1])
        ct1_below_vertex[:,0] = ct1_triangle_index[ct1_below_vertex[:,0]]
        ct1_below_vertex_data = triangles[ct1_below_vertex[:,0], ct1_below_vertex[:,1]]
        ct1_above_vertex = numpy.argwhere(above[split_dimension, in_conflict[ct1_plane_vertex]][ct1_below_count == 1])
        ct1_above_vertex[:,0] = ct1_triangle_index[ct1_above_vertex[:,0]]
        ct1_above_vertex_data = triangles[ct1_above_vertex[:,0], ct1_above_vertex[:,1]]
        ct1_plane_vertex = numpy.argwhere(at_plane[split_dimension, ct1_triangle_index])
        ct1_plane_vertex[:,0] = ct1_triangle_index[ct1_plane_vertex[:,0]]
        ct1_plane_vertex_data = triangles[ct1_plane_vertex[:,0], ct1_plane_vertex[:,1]]
        ct1_below_vertex_position = ct1_below_vertex_data[:,vertex_position]
        ct1_above_vertex_position = ct1_above_vertex_data[:,vertex_position]
        ct1_intersection_factor = (plane_factor[split_dimension] - ct1_below_vertex_position[:,split_dimension]) / (ct1_above_vertex_position[:,split_dimension] - ct1_below_vertex_position[:,split_dimension])
        ct1_new_vertex_position = ct1_below_vertex_position + ct1_intersection_factor[:,None] * (ct1_above_vertex_position - ct1_below_vertex_position)
        assert numpy.all(numpy.abs(ct1_new_vertex_position[:,split_dimension] - plane_factor[split_dimension]) < epsilon)
        ct1_new_vertex_normal = ct1_intersection_factor[:,None] * ct1_above_vertex_data[:,vertex_normal] + (1.0 - ct1_intersection_factor)[:,None] * ct1_below_vertex_data[:,vertex_normal]
        ct1_new_vertex_texcoord = ct1_intersection_factor[:,None] * ct1_above_vertex_data[:,vertex_texcoord] + (1.0 - ct1_intersection_factor)[:,None] * ct1_below_vertex_data[:,vertex_texcoord]
        ct1_new_vertex_data = numpy.concatenate([ct1_new_vertex_position, ct1_new_vertex_normal, ct1_new_vertex_texcoord], axis=-1)
        ct1_below_vertex_order = numpy.concatenate([ct1_below_vertex[:,1][:,None], ct1_plane_vertex[:,1][:,None]], axis=-1)
        ct1_below_vertex_order = numpy.concatenate([ct1_below_vertex_order, 3 - numpy.sum(ct1_below_vertex_order, axis=-1)[:,None]], axis=-1)
        # ct1_below_vertex_order = numpy.array(list(numpy.argwhere(ct1_below_vertex_order == i)[:,1] for i in range(3))).T
        ct1_above_vertex_order = numpy.concatenate([ct1_above_vertex[:,1][:,None], ct1_plane_vertex[:,1][:,None]], axis=-1)
        ct1_above_vertex_order = numpy.concatenate([ct1_above_vertex_order, 3 - numpy.sum(ct1_above_vertex_order, axis=-1)[:,None]], axis=-1)
        # ct1_above_vertex_order = numpy.array(list(numpy.argwhere(ct1_above_vertex_order == i)[:,1] for i in range(3))).T
        ct1_below_data = numpy.concatenate([ct1_below_vertex_data[:,None], ct1_plane_vertex_data[:,None], ct1_new_vertex_data[:,None]], axis=1)
        ct1_above_data = numpy.concatenate([ct1_above_vertex_data[:,None], ct1_plane_vertex_data[:,None], ct1_new_vertex_data[:,None]], axis=1)
        ct1_below_triangles = numpy.array(list(ct1_below_data[ct1_below_vertex_order == i] for i in range(3))).swapaxes(0, 1)
        ct1_above_triangles = numpy.array(list(ct1_above_data[ct1_above_vertex_order == i] for i in range(3))).swapaxes(0, 1)
        numpy.vstack([groups[0], ct1_below_triangles])
        numpy.vstack([groups[1], ct1_above_triangles])

        # If the triangle has 0 vertices at the splitting plane (most common case):
        # 1. Identify the lonely vertex and lonely direction
        # 2. The vertex couple and vertex couple direction are opposite of the lonely vertex
        # 3. Get the two edges of the vertex couple with the lonely vertex
        # 4. Intersect the two edges with the plane to generate two plane vertices
        # 5. Create one new triangle from the plane vertices and the lonely vertex
        # 6. Create two new triangles for the quadliteral formed by the plane vertices and vertex couple
        # Note: this can be done in two ways, as there are two diagonals in a quadliteral
        # For all intent and purposes these are equivalent up to the vertex order;
        # The result will always append one triangle in one of the groups and two triangles in the other.

        # If the triangle has all 3 vertices at the splitting plane (very rare case):
        # Put the triangle into the group with smaller number of triangles to increase the balance

        # Next step: ensure all vertices fit their corresponding group
        # Note: newly create vertices can differ from plane_factor up to epsilon in any direction.
        # Since the groups has been assigned, we move all vertices up to epsilon to fit the group sub-space.

        # Next, Optionally, reassign all floating-points to fit a grid of epsilon item size. This can be done with numpy.argwhere, numpy.diff and indexing (perhaps).
        # This will allow easy comparison of vertices, normals and texcoords, so we can store those in a separate arrays.
        # This will also allow triangle vertices to be set of indices to the corresponding attributes (for example 3 uint32 can point to [0] = position, [1] = normal, [2] = texcoord)
        # Finally, the triangle data can be 3 uint32 pointing to 3 vertices.
        # The idea is to minimize the repetitions of values for enclosed space. A mesh having enclosed space will definitely have triangle sharing vertices as well as vertices sharing positions.
        # A container node can be given of set of (2, 3) indices to positions f32[3] specifying the min point and max point of a box.
        # Since those will be selected from coordinates of some vertex positions, f32 will definitely be reusable.
        # Optional: Finally, when a 3-plets or pair indices data is already found in data_int, repetition can be avoided and pointers can be reused.
        # Note: But this should not be done in expensive of additional memory indirections. It should only be used for existing memory indirections.

        # Minimizing models can improve loading of the models over the internet (but ServiceWorker offline cache should be used when available).
        # However, this should be carefully considered against the time it requires for scene to raytrace. The target is <16.66ms for 1080p
        # The optimization should guarantee O(ln(N)) order of complexity with model data maximum 2 times the original binary data.
        pass

def _main():
    count_vertices = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_normals = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_texture_coords = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_faces = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')

    data_vertices = numpy.frombuffer(sys.stdin.buffer.read(count_vertices * 3 * 4), dtype=numpy.float32).reshape((count_vertices, 3))
    data_normals = numpy.frombuffer(sys.stdin.buffer.read(count_normals * 3 * 4), dtype=numpy.float32).reshape((count_normals, 3))
    data_texture_coords = numpy.frombuffer(sys.stdin.buffer.read(count_texture_coords * 2 * 4), dtype=numpy.float32).reshape((count_texture_coords, 2))
    data_triangles = numpy.frombuffer(sys.stdin.buffer.read(count_faces * 3 * 3 * 4), dtype=numpy.uint32).reshape((count_faces, 3, 3))

    # wavefront = Data(vertex=data_vertices, normal=data_normals, texcoord=data_texture_coords, triangle=data_triangles)
    # root_node = Data(
    #     vertices=data_vertices,
    #     normals=data_normals,
    #     texcoords=data_texture_coords,
    #     triangles=data_triangles
    # )
    triangles = numpy.concatenate([data_vertices[data_triangles[:, :, 0]], data_normals[data_triangles[:, :, 1]], data_texture_coords[data_triangles[:, :, 2]]], axis=-1) 
    # Can be modified with insertion of new floating point numbers or triangles
    # storage = create_storage(triangles)
    # root_node = create_node(storage, numpy.arange(storage.index.shape[0]))
    context = DataContext()
    context.create_node(triangles)
    pass

if __name__ == '__main__':
    _main()
