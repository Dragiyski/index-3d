import sys
import numpy

numpy.set_printoptions(floatmode='maxprec', suppress=True)
epsilon = numpy.finfo(numpy.float32).eps

vertex_attributes = numpy.dtype([('position', '3f8'), ('normal', '3f8'), ('texcoord', '2f8')])

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

class DataContext:
    def __init__(self, vertex_position_index=[0, 1, 2], vertex_normal_index=[3, 4, 5], vertex_texcoord_index=[6, 7]):
        self.vertex_position_index = numpy.array(vertex_position_index, dtype=numpy.int32)
        self.vertex_normal_index = numpy.array(vertex_normal_index, dtype=numpy.int32)
        self.vertex_texcoord_index = numpy.array(vertex_texcoord_index, dtype=numpy.int32)

    def create_node(self, triangles, parent=None):
        assert isinstance(triangles, numpy.ndarray)
        assert len(triangles.shape) >= 2
        assert triangles.shape[1] == 3

        if triangles.shape[0] <= 6:
            return Data(
                is_split=False,
                triangles=triangles,
                depth = parent.depth + 1 if parent is not None else 1,
                parent=parent
            )

        plane_factor = numpy.median(numpy.swapaxes(triangles.position, 0, -1).reshape((3, -1)), axis=-1)
        vertex_group_mask = numpy.concatenate([
            (triangles.position < plane_factor)[None],
            (triangles.position > plane_factor)[None],
            (numpy.abs(triangles.position - plane_factor) < epsilon)[None]
        ], axis=0)
        for i in range(2):
            vertex_group_mask[i] = numpy.logical_and(vertex_group_mask[i], numpy.logical_not(vertex_group_mask[2]))
        assert numpy.all(numpy.count_nonzero(vertex_group_mask, axis=0) == 1)

        groups = numpy.all(numpy.moveaxis(vertex_group_mask, -1, 0), axis=-1)
        distribution = numpy.count_nonzero(groups, axis=-1)
        distribution[:, 2] = triangles.shape[0] - numpy.sum(distribution, axis=-1)
        split_dimension = numpy.argmin(distribution[:, 2])
        print('%d: distrib_ini: %d :\n%r' % (parent.depth + 1 if parent is not None else 1, split_dimension, distribution))
        distribution = distribution[split_dimension]
        groups = groups[split_dimension]
        vertex_group_mask = vertex_group_mask[:, :, :, split_dimension]
        plane_factor = plane_factor[split_dimension]
        group_mask = numpy.array([
            groups[0],
            groups[1],
            numpy.all(numpy.logical_not(groups), axis=0)
        ])
        group_index = list(numpy.argwhere(mask)[:, 0] for mask in group_mask)
        groups = list(triangles[group_mask[i]] for i in range(3))
        # Groups 0 and 1 must not have vertices close to the split plane
        assert numpy.count_nonzero(numpy.abs(groups[0].position[:, :, split_dimension] - plane_factor) < epsilon) == 0
        assert numpy.count_nonzero(numpy.abs(groups[1].position[:, :, split_dimension] - plane_factor) < epsilon) == 0
        
        vertex_near_split_plane_mask = numpy.abs(groups[2].position[:, :, split_dimension] - plane_factor) < epsilon
        vertex_near_split_plane_count = numpy.count_nonzero(vertex_near_split_plane_mask, axis=-1)

        # If the triangle has 2 vertices at the splitting plane:
        # insert the triangle into the group determined by the non-plane vertex
        ct2_index = group_index[2][vertex_near_split_plane_count == 2]
        for i in range(2):
            groups[i] = numpy.concatenate([
                groups[i],
                triangles[
                    ct2_index[
                        numpy.any(vertex_group_mask[i, ct2_index], axis=-1)
                    ]
                ]
            ], axis=0)

        # If the triangle has 1 vertices at the splitting plane:
        ct1_index = group_index[2][vertex_near_split_plane_count == 1]
        ct1_count = numpy.count_nonzero(vertex_group_mask[0:2, ct1_index], axis=-1)

        # If it has two vertices in the same group, add it to that group:
        for i in range(2):
            groups[i] = numpy.concatenate([
                groups[i],
                triangles[ct1_index[ct1_count[i] == 2]]
            ], axis=0)

        assert numpy.all((ct1_count[0] == 1) == (ct1_count[1] == 1))
        ct1_split_index = ct1_index[ct1_count[0] == 1]
        ct1_plane_vertex_mask = numpy.abs(triangles[ct1_split_index].position[:,:,split_dimension] - plane_factor) < epsilon
        ct1_min_vertex_mask = triangles[ct1_split_index].position[:,:,split_dimension] <= plane_factor - epsilon
        ct1_max_vertex_mask = triangles[ct1_split_index].position[:,:,split_dimension] >= plane_factor + epsilon
        assert numpy.all(numpy.count_nonzero(ct1_plane_vertex_mask, axis=-1) == 1)
        assert numpy.all(numpy.count_nonzero(ct1_min_vertex_mask, axis=-1) == 1)
        assert numpy.all(numpy.count_nonzero(ct1_max_vertex_mask, axis=-1) == 1)
        ct1_split_factor = (
            plane_factor - triangles[ct1_split_index][ct1_min_vertex_mask].position[:, split_dimension]
        ) / (
            triangles[ct1_split_index][ct1_max_vertex_mask].position[:, split_dimension] - triangles[ct1_split_index][ct1_min_vertex_mask].position[:, split_dimension]
        )
        ct1_triangle_float = triangles[ct1_split_index].view((numpy.float64, 8))
        ct1_new_vertices = ((1.0 - ct1_split_factor)[:,None] * ct1_triangle_float[ct1_min_vertex_mask] + ct1_split_factor[:,None] * ct1_triangle_float[ct1_max_vertex_mask]).astype(numpy.float64).view(vertex_attributes)[:,0].view(numpy.recarray)
        ct1_new_min_triangle = numpy.concatenate([
            ct1_new_vertices[:,None],
            triangles[ct1_split_index][ct1_plane_vertex_mask][:,None],
            triangles[ct1_split_index][ct1_min_vertex_mask][:,None],
        ], axis=-1).view(numpy.recarray)
        ct1_new_max_triangle = numpy.concatenate([
            ct1_new_vertices[:,None],
            triangles[ct1_split_index][ct1_plane_vertex_mask][:,None],
            triangles[ct1_split_index][ct1_min_vertex_mask][:,None],
        ], axis=-1).view(numpy.recarray)
        ct1_split_triangle_normal = normalize(numpy.cross(triangles[ct1_split_index, 2].position - triangles[ct1_split_index, 0].position, triangles[ct1_split_index, 1].position - triangles[ct1_split_index, 0].position))
        ct1_new_min_triangle_normal = normalize(numpy.cross(ct1_new_min_triangle[:, 2].position - ct1_new_min_triangle[:, 0].position, ct1_new_min_triangle[:, 1].position - ct1_new_min_triangle[:, 0].position))
        ct1_new_max_triangle_normal = normalize(numpy.cross(ct1_new_max_triangle[:, 2].position - ct1_new_max_triangle[:, 0].position, ct1_new_max_triangle[:, 1].position - ct1_new_max_triangle[:, 0].position))
        ct1_new_min_triangle_flip_mask = array_dot(ct1_new_min_triangle_normal, ct1_split_triangle_normal) < 0
        ct1_new_max_triangle_flip_mask = array_dot(ct1_new_max_triangle_normal, ct1_split_triangle_normal) < 0
        ct1_new_min_triangle[ct1_new_min_triangle_flip_mask] = numpy.flip(ct1_new_min_triangle[ct1_new_min_triangle_flip_mask], axis=-1)
        ct1_new_max_triangle[ct1_new_max_triangle_flip_mask] = numpy.flip(ct1_new_max_triangle[ct1_new_max_triangle_flip_mask], axis=-1)
        numpy.concatenate([
            groups[0],
            ct1_new_min_triangle
        ], axis=0)
        numpy.concatenate([
            groups[1],
            ct1_new_max_triangle
        ], axis=0)

        # If the triangle has 0 vertices at the splitting plane (most common case):
        # Out of 3 vertices, 2 will be on one side of the splitting plane (a.k.a. couple), the other vertex (a.k.a. lonely) will be on the other side
        # We create 2 new vertices at the intersection point of the plane and the edges between the lonely vertex and the vertex couple.
        # We create 3 new triangles: one with the new plane vertices and the lonely vertex and two for triangulization of the quadliteral
        # between the 4 vertex: the 2 new plane vertices and the vertex couple.
        ct0_index = group_index[2][vertex_near_split_plane_count == 0]
        ct0_lonely_vertex_index = numpy.argwhere((triangles[ct0_index].position[:,:,split_dimension] < plane_factor) == numpy.logical_xor.reduce((triangles[ct0_index].position[:,:,split_dimension] < plane_factor), axis=-1)[:,None])[:,1]
        ct0_couple_vertex_index = numpy.tile(numpy.arange(3), ct0_lonely_vertex_index.shape[0]).reshape((-1, 3))[numpy.not_equal.outer(ct0_lonely_vertex_index, numpy.arange(3))].reshape((-1, 2))
        assert not (numpy.any(ct0_couple_vertex_index[:,0] == ct0_lonely_vertex_index) or numpy.any(ct0_couple_vertex_index[:,1] == ct0_lonely_vertex_index))
        assert not numpy.any(numpy.logical_xor(triangles[ct0_index, ct0_couple_vertex_index[:, 0]].position[:, split_dimension] < plane_factor, triangles[ct0_index, ct0_couple_vertex_index[:, 1]].position[:, split_dimension] < plane_factor)), 'vertex_couple vertices must be in the same group'
        ct0_triangle_normal = normalize(numpy.cross(triangles[ct0_index, 2].position - triangles[ct0_index, 0].position, triangles[ct0_index, 1].position - triangles[ct0_index, 0].position))
        ct0_split_factor = numpy.array(list(
            (
                plane_factor - triangles[ct0_index].position[(numpy.arange(ct0_index.shape[0]), ct0_lonely_vertex_index)][:, split_dimension]
            ) / (
                triangles[ct0_index].position[(numpy.arange(ct0_index.shape[0]), ct0_couple_vertex_index[:,i])][:, split_dimension] - triangles[ct0_index].position[(numpy.arange(ct0_index.shape[0]), ct0_lonely_vertex_index)][:, split_dimension]
            )
            for i in range(2)
        ))
        ct0_triangle_float = triangles[ct0_index].view((numpy.float64, 8))
        ct0_new_vertices = numpy.array(list(
            (1.0 - ct0_split_factor[i])[:,None] * ct0_triangle_float[(numpy.arange(ct0_lonely_vertex_index.shape[0]), ct0_lonely_vertex_index)] + ct0_split_factor[i][:,None] * ct0_triangle_float[(numpy.arange(ct0_couple_vertex_index.shape[0]), ct0_couple_vertex_index[:,i])]
            for i in range(2)
        )).astype(numpy.float64).view(vertex_attributes)[:,:,0].view(numpy.recarray)
        ct0_new_lonely_triangle = numpy.concatenate([
            triangles[ct0_index, ct0_lonely_vertex_index][:,None],
            ct0_new_vertices.T,
        ], axis=-1).view(numpy.recarray)
        ct0_new_lonely_triangle_normal = normalize(numpy.cross(ct0_new_lonely_triangle[:,2].position - ct0_new_lonely_triangle[:,0].position, ct0_new_lonely_triangle[:,1].position - ct0_new_lonely_triangle[:,0].position))
        ct0_new_lonely_triangle_flip_mask = array_dot(ct0_new_lonely_triangle_normal, ct0_triangle_normal) < 0.0
        ct0_new_lonely_triangle[ct0_new_lonely_triangle_flip_mask] = numpy.flip(ct0_new_lonely_triangle[ct0_new_lonely_triangle_flip_mask], axis=-1)
        ct0_new_plane_triangle = numpy.concatenate([
            triangles[(ct0_index, ct0_couple_vertex_index[:,0])][:,None],
            ct0_new_vertices.T
        ], axis=-1).view(numpy.recarray)
        ct0_new_plane_triangle_normal = normalize(numpy.cross(ct0_new_plane_triangle[:,2].position - ct0_new_plane_triangle[:,0].position, ct0_new_plane_triangle[:,1].position - ct0_new_plane_triangle[:,0].position))
        ct0_new_plane_triangle_flip_mask = array_dot(ct0_new_plane_triangle_normal, ct0_triangle_normal) < 0.0
        ct0_new_plane_triangle[ct0_new_plane_triangle_flip_mask] = numpy.flip(ct0_new_plane_triangle[ct0_new_plane_triangle_flip_mask], axis=-1)
        ct0_new_couple_triangle = numpy.concatenate([
            triangles[(ct0_index, ct0_couple_vertex_index[:,0])][:,None],
            triangles[(ct0_index, ct0_couple_vertex_index[:,1])][:,None],
            ct0_new_vertices[1][:,None],
        ], axis=-1).view(numpy.recarray)
        ct0_new_couple_triangle_normal = normalize(numpy.cross(ct0_new_couple_triangle[:,2].position - ct0_new_couple_triangle[:,0].position, ct0_new_couple_triangle[:,1].position - ct0_new_couple_triangle[:,0].position))
        ct0_new_couple_triangle_flip_mask = array_dot(ct0_new_couple_triangle_normal, ct0_triangle_normal) < 0.0
        ct0_new_couple_triangle[ct0_new_couple_triangle_flip_mask] = numpy.flip(ct0_new_couple_triangle[ct0_new_couple_triangle_flip_mask], axis=-1)
        ct0_new_triangles = numpy.concatenate([ct0_new_lonely_triangle, ct0_new_plane_triangle, ct0_new_couple_triangle], axis=0).view(numpy.recarray)
        # assert not numpy.any(numpy.abs(numpy.mean(ct0_new_triangles.position[:,:,split_dimension], axis=-1) - plane_factor) < epsilon)
        ct0_group_mask = (numpy.mean(ct0_new_triangles.position[:,:,split_dimension], axis=-1) >= plane_factor).astype('i')
        for i in range(2):
            groups[i] = numpy.concatenate([
                groups[i],
                ct0_new_triangles[ct0_group_mask == i]
            ], axis=0)

        # A very rare case: the median triangle might be aligned with the axes, thus all 3 vertices are plane.
        # In general we can put this triangles into any group
        ct3_index = group_index[2][vertex_near_split_plane_count == 3]
        if ct3_index.shape[0] > 0:
            raise NotImplementedError('Handling of rare 3-plane-vertices median triangle postponed until a model is found to need this case')
        
        new_distribution = numpy.array([g.shape[0] for g in groups[0:2]])
        print('%d: distrib_cur[%d, %d]' % (parent.depth + 1 if parent is not None else 1, *new_distribution))
        if (numpy.sum(new_distribution) <= numpy.sum(distribution)) or (parent is not None and numpy.sum(parent.distribution) <= numpy.sum(new_distribution)):
            return Data(
                is_split=False,
                triangles=triangles,
                depth = parent.depth + 1 if parent is not None else 1,
                parent=parent
            )
        

        del groups[2]
        for i in range(2):
            groups[i] = groups[i].view(numpy.recarray)

        # This will throw if any plane vertex do not lie exactly on the splitting plane.
        # If throws, we can write code to move all f32 epsilon proximity to the plane to match the plane, so that it never step outside the bounding box
        # assert not numpy.any(numpy.count_nonzero(numpy.abs(groups[i].position[numpy.abs(groups[i].position[:,:,split_dimension] - plane_factor) < epsilon][:,split_dimension] - plane_factor) > 0.0))
        
        node = Data(
            split_dimension=split_dimension,
            split_factor=plane_factor,
            children=[],
            parent=parent,
            depth = parent.depth + 1 if parent is not None else 1,
            distribution = numpy.array([g.shape[0] for g in groups[0:2]]),
            is_split=True
        )

        max_triangles = -1
        for group in groups:
            child = self.create_node(group, node)
            if not child.is_split:
                max_triangles = max(max_triangles, child.triangles.shape[0])
            else:
                max_triangles = max(max_triangles, child.max_triangles)
            node.children.append(self.create_node(group, node))
        node.max_triangles = max_triangles

        return node

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
    triangles = numpy.core.records.fromarrays([triangles[:, :, [0, 1, 2]], triangles[:, :, [3, 4, 5]], triangles[:, :, [6, 7]]], dtype=vertex_attributes)
    # Can be modified with insertion of new floating point numbers or triangles
    # storage = create_storage(triangles)
    # root_node = create_node(storage, numpy.arange(storage.index.shape[0]))
    context = DataContext()
    root_node = context.create_node(triangles)
    print(root_node.max_triangles)
    pass

if __name__ == '__main__':
    _main()
