import numpy
import numpy.matlib
import functools

f32_eps = numpy.finfo(numpy.float32).eps

def find_longest_axis(vertices: numpy.ndarray):
    pivot_index = 0
    pivot_distance = 0.0
    selected_index = 0
    selected_distance = 0.0

    mid_point = numpy.average(vertices, axis=0)
    distances = numpy.linalg.norm(vertices - mid_point, axis=-1)
    sorted_distances = numpy.argsort(distances)
    selected_index = sorted_distances[-1]

    while True:
        selected_point = vertices[selected_index]
        distances = numpy.linalg.norm(vertices - selected_point, axis=-1)
        sorted_distances = numpy.argsort(distances)
        farthest_index = sorted_distances[-1]
        if farthest_index == pivot_index or farthest_index == selected_index:
            break
        farthest_distance = distances[farthest_index]
        if farthest_distance <= selected_distance:
            break
        pivot_index = selected_index
        pivot_distance = selected_distance
        selected_index = farthest_index
        selected_distance = farthest_distance

    return (pivot_index, selected_index, selected_distance)

def compute_axis(vertices):
    p, q, distance = find_longest_axis(vertices)
    axis = vertices[q] - vertices[p]
    normal = axis / numpy.linalg.norm(axis)
    return Data(p=p, q=q, p_coords=vertices[p], q_coords=vertices[q], axis=axis, normal=normal, distance=distance, min=numpy.dot(vertices[p], normal), max=numpy.dot(vertices[q], normal))

def compute_dimensions(vertices):
    dims = []
    for dim in range(3):
        axis = compute_axis(vertices)
        if axis.distance < f32_eps:
            break
        dims.append(axis)
        if dim < 3:
            vertices = vertices - numpy.multiply.outer(numpy.dot(vertices - vertices[axis.p], axis.normal), axis.normal)
    return dims

def find_box_common_point(n1, n2, n3, mx, my,mz):
    m = numpy.array([n1, n2, n3, [mx, my, mz]])
    u = numpy.cross(m[1], m[2])
    v = numpy.cross(m[0], m[3])
    w = numpy.dot(m[0], u)
    if numpy.abs(w) < numpy.finfo(numpy.float32).eps:
        raise ValueError('Single point 3-plane intersection failed')
    return numpy.array([
        numpy.dot(m[3], u) / w,
        numpy.dot(m[2], v) / w,
        -numpy.dot(m[1], v) / w
    ])


def generate_node(int_data, float_data, object_data, vertex_buffer, index_buffer, triangle_index, position_index, *, parent_id=None):
    id = obj_pointer = len(object_data) // 4
    int_pointer = len(int_data) // 4
    float_pointer = len(float_data) // 4
    # Step 1 find how many dimensions are required
    dimensions = []
    point_index = numpy.unique(index_buffer[triangle_index])
    point_list = vertex_buffer[point_index,:][:,position_index]
    bounding_dims = compute_dimensions(point_list)
    flags = 1
    if len(bounding_dims) < 2:
        # We do not render lines and points: models that arrange triangles into a line or when the 3 points are nearly the same (up to f32_eps)
        return None
    elif len(bounding_dims) == 2:
        # This is a Quad node. Quad nodes are not inversible, so they must be raytraced in world space (opposite to model space).
        # The node needs an origin point and two vectors. We would store the two normals 3-d + distance
        j = 0
        type_id = 1
    elif len(bounding_dims) == 3:
        j = 0
        # In order to store the box we need a common vertex and the three normals,
        # The common vertex is based on the matrix
        min_point = find_box_common_point(
            bounding_dims[0].normal,
            bounding_dims[1].normal,
            bounding_dims[2].normal,
            bounding_dims[0].min,
            bounding_dims[1].min,
            bounding_dims[2].min
        )
        max_point = find_box_common_point(
            bounding_dims[0].normal,
            bounding_dims[1].normal,
            bounding_dims[2].normal,
            bounding_dims[0].max,
            bounding_dims[1].max,
            bounding_dims[2].max
        )
        mid_point = numpy.average([min_point, max_point], axis=0)
        model_transform = numpy.matlib.mat([[*bounding_dims[i].normal, 0.0] for i in range(3)] + [[*mid_point, 1.0]]).transpose()
        inverse_transform = numpy.linalg.inv(model_transform)
        for i in range(4):
            float_data.append([model_transform[i, x] for x in range(4)])
        for i in range(4):
            float_data.append([inverse_transform[i, x] for x in range(4)])
        type_id = 2
        flags = flags | 2
    else:
        raise RuntimeError('Maximum working dimensions must be 3')
    if parent_id is None:
        parent_id = -1
        parent_obj = None
    else:
        parent_obj = object_data[parent_id * 4]
    int_data.append([parent_id, 0, 0, 0])
    object_data.append([type_id, flags, int_pointer, float_pointer])
    # TODO: Perform _compute_children, which
    # Should call generate_triangle for triangle nodes
    # Or generate_node for more containers



class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BoxNode:
    def __init__(self, vertex_buffer, index_buffer, triangle_index, position_index, parent=None):
        self.vertex_buffer = vertex_buffer
        self.index_buffer = index_buffer
        self.triangle_index = triangle_index
        self.position_index = position_index
        self.parent = parent
        self.box = []
        self.children = []
        if parent is None:
            self._id_gen = self.id = 0
        else:
            self.id = self._next_id()

        vertex_index = numpy.unique(self.index_buffer[self.triangle_index])
        self._compute_box(self.vertex_buffer[vertex_index,:][:,self.position_index], vertex_index)
        self._compute_vertices(vertex_index)
        self.mid_point = self.points[0] + self.box[0].axis / 2 + self.box[1].axis / 2 + self.box[2].axis / 2
        self.model_transform = numpy.matlib.mat([[*self.box[i].normal, 0.0] for i in range(3)] + [[*self.mid_point, 1.0]]).transpose()
        self.inverse_transform = numpy.linalg.inv(self.model_transform)
        self.volume = self.box[0].distance * self.box[1].distance * self.box[2].distance
        self.surface = 2.0 * self.box[0].distance * self.box[1].distance + 2.0 * self.box[0].distance * self.box[2].distance + 2.0 * self.box[1].distance * self.box[2].distance
        self._compute_children()
    
    def _next_id(self):
        if self.parent is None:
            self._id_gen += 1
            return self._id_gen
        return self.parent._next_id()

    def _compute_axis(self, vertices, indices):
        p, q, distance = find_longest_axis(vertices)
        axis = vertices[q] - vertices[p]
        normal = axis / numpy.linalg.norm(axis)
        return Data(p=p, q=q, axis=axis, normal=normal, distance=distance, min=numpy.dot(vertices[p], normal), max=numpy.dot(vertices[q], normal))

    def _compute_box(self, vertices, indices):
        x = self._compute_axis(vertices, indices)
        x.min = numpy.dot(vertices[x.p], x.normal) - f32_eps
        x.max = numpy.dot(vertices[x.q], x.normal) + f32_eps
        self.box.append(x)
        y_vertices = vertices - numpy.multiply.outer(numpy.dot(vertices - vertices[x.p], x.normal), x.normal)
        y = self._compute_axis(y_vertices, indices)
        if numpy.abs(y.distance) < f32_eps:
            return
        y.min = numpy.dot(vertices[y.p], y.normal) - f32_eps
        y.max = numpy.dot(vertices[y.q], y.normal) + f32_eps
        self.box.append(y)
        z_vertices = y_vertices - numpy.multiply.outer(numpy.dot(y_vertices - y_vertices[y.p], y.normal), y.normal)
        z = self._compute_axis(z_vertices, indices)
        if numpy.abs(z.distance) < f32_eps:
            return
        z.min = numpy.dot(vertices[z.p], z.normal) - f32_eps
        z.max = numpy.dot(vertices[z.q], z.normal) + f32_eps
        self.box.append(z)

    def _compute_vertices(self, indices):
        points = []
        for s in range(8):
            m = numpy.concatenate([list(list(self.box[i].normal[k] for i in range(3)) for k in range(3)), [list(getattr(self.box[i], 'max' if (s & (1 << i)) else 'min') for i in range(3))]])
            u = numpy.cross(m[1], m[2])
            v = numpy.cross(m[0], m[3])
            w = numpy.dot(m[0], u)
            if numpy.abs(w) < numpy.finfo(numpy.float32).eps:
                raise ValueError('Single point 3-plane intersection failed')
            points.append([
                numpy.dot(m[3], u) / w,
                numpy.dot(m[2], v) / w,
                -numpy.dot(m[1], v) / w
            ])
        self.points = numpy.array(points)

    def _compute_children(self):
        if len(self.triangle_index) <= 6:
            print(f'+Distribution: {len(self.triangle_index)}')
            for idx in self.triangle_index:
                self.children.append(idx)
            return
        split_dims = []
        selected_dim = None
        for split_index in range(3):
            origin = self.vertex_buffer[self.box[split_index].p][self.position_index]
            direction = self.box[split_index].normal
            triangle_to_origin = self.vertex_buffer[self.index_buffer[self.triangle_index],:][:,:,self.position_index] - origin
            triangle_projection = numpy.tensordot(triangle_to_origin, direction, axes=([2], [0])) / self.box[0].distance
            line_projection = numpy.stack([numpy.min(triangle_projection, axis=1), numpy.max(triangle_projection, axis=1)], axis=1)
            split_factor = numpy.median(line_projection)
            triangle_zero = numpy.argwhere(numpy.all(triangle_projection <= split_factor, axis=1))
            triangle_zero.shape = (triangle_zero.shape[0],)
            triangle_one = numpy.argwhere(numpy.all(triangle_projection > split_factor, axis=1))
            triangle_one.shape = (triangle_one.shape[0],)
            triangle_mid = numpy.delete(numpy.arange(line_projection.shape[0]), numpy.vstack((triangle_zero[:,None], triangle_one[:,None])), axis=0)
            if triangle_zero.shape[0] + triangle_one.shape[0] < triangle_mid.shape[0] or triangle_zero.shape[0] * triangle_one.shape[0] + triangle_one.shape[0] * triangle_mid.shape[0] + triangle_zero.shape[0] * triangle_mid.shape[0] == 0:
                split_dims.append(Data(triangle_zero=triangle_zero, triangle_one=triangle_one, triangle_mid=triangle_mid))
                continue
            selected_dim = split_index
            print(f'>Distribution: {triangle_zero.shape[0]}, {triangle_one.shape[0]}, {triangle_mid.shape[0]}')
            if triangle_zero.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_zero], self.position_index, self))
            if triangle_one.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_one], self.position_index, self))
            if triangle_mid.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_mid], self.position_index, self))
            break
        if selected_dim is None:
            sorted_dim = sorted(split_dims, key=functools.cmp_to_key(compare_split))
            triangle_zero = sorted_dim[0].triangle_zero
            triangle_one = sorted_dim[0].triangle_one
            triangle_mid = sorted_dim[0].triangle_mid
            if triangle_zero.shape[0] * triangle_one.shape[0] + triangle_one.shape[0] * triangle_mid.shape[0] + triangle_zero.shape[0] * triangle_mid.shape[0] == 0:
                print(f'*Distribution: {len(self.triangle_index)}')
                for idx in self.triangle_index:
                    self.children.append(idx)
                return
            print(f'?Distribution: {triangle_zero.shape[0]}, {triangle_one.shape[0]}, {triangle_mid.shape[0]}')
            if triangle_zero.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_zero], self.position_index, self))
            if triangle_one.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_one], self.position_index, self))
            if triangle_mid.shape[0] > 0:
                self.children.append(BoxNode(self.vertex_buffer, self.index_buffer, self.triangle_index[triangle_mid], self.position_index, self))

def compare_split(a, b):
    return b.triangle_zero.shape[0] * b.triangle_one.shape[0] - a.triangle_zero.shape[0] * a.triangle_one.shape[0]

# def find_bounding_box(vertices: numpy.ndarray):
#     box = []
#     dim_vertices = vertices

#     def compute_dimension(compress=False):
#         nonlocal dim_vertices, box
#         p1, p2, distance = find_longest_axis(dim_vertices)
#         axis = dim_vertices[p2] - dim_vertices[p1]
#         normal = axis / numpy.linalg.norm(axis)
#         dim = CubeDimension(p1, p2, axis, normal, numpy.dot(dim_vertices[p1], normal), numpy.dot(dim_vertices[p2], normal))
#         box.append(dim)
#         if compress:
#             dim_vertices = dim_vertices - numpy.multiply.outer(numpy.dot(dim_vertices - dim_vertices[p1], normal), normal)
#     compute_dimension(True)
#     compute_dimension(True)
#     compute_dimension(False)
#     return box

def main():
    with open('cottage/vertex-buffer.bin', 'rb') as file:
        vertex_buffer = numpy.frombuffer(file.read(), dtype=numpy.float32)
    with open('cottage/index-buffer.bin', 'rb') as file:
        index_buffer = numpy.frombuffer(file.read(), dtype=numpy.uint16)
    vertex_buffer.shape = (len(vertex_buffer) // 8, 8)
    index_buffer.shape = (len(index_buffer) // 3, 3)
    position_index = numpy.array([0, 1, 2])
    triangle_index = numpy.arange(index_buffer.shape[0])

    int_data = []
    float_data = []
    object_data = []


    # To create a surrounding sphere, get the longest axis
    # The center of the sphere is the middle point
    # The radius is the length(axis) / 2
    node = generate_node(int_data, float_data, object_data, vertex_buffer, index_buffer, triangle_index, position_index)
    # box = find_bounding_box(vertices)

    # # Project all triangles onto the longest axis
    # origin = vertices[start_index]
    # direction = vertices[end_index] - origin
    # direction = direction / numpy.linalg.norm(direction)
    # tv = triangles - origin
    # prj = numpy.tensordot(tv, direction, axes=([2], [0])) / distance

    # # Obtain the minimum and the maximum value for the projection.
    # # The triangle has three points, one will be min, one will be max, and one in between. 
    # m1 = numpy.min(prj, axis=1)
    # m2 = numpy.max(prj, axis=1)
    # amm = numpy.stack([m1, m2], axis=1)
    # damm = amm[:,1] - amm[:,0]
    # odamm = (1 - amm[:,1]) + amm[:,0]
    # # Solution 1: Weighted average (reasonable), but small triangles in the middle are not included in any group
    # # wvg = ((1 - amm[:,1]) + amm[:,0]) * numpy.average(amm, axis=1)

    # # Choosing a split point, this will be the median vertex
    # spl = numpy.median(amm)

    # # The spl split triangles in 3 groups:
    # # Group 1: where the entire triangle is projected on the first half of the axis.
    # # Group 2: where the entire triangle is projected on the second half of the axis.
    # s1 = numpy.argwhere(numpy.all(prj <= spl, axis=1))
    # s2 = numpy.argwhere(numpy.all(prj > spl, axis=1))
    # # Group 3: where the triangle crosses the split point.
    # # This group will include two types of triangles:
    # # 1. Small triangles projected in the middle of the axis.
    # # 2. Large triangles*
    # # * To determine what is a "large" triangle is ambigous.
    # # - It might be feasible not to distinguish them and just leave them be...
    # # - Or we might project the triangles onto perpedicular axis and compare their sizes
    # sr = numpy.delete(numpy.arange(amm.shape[0]), numpy.vstack((s1, s2)), axis=0)

    # # Solution 2: 
    # # damm = amm[:,1] - amm[:,0]
    # # odamm = (1 - amm[:,1]) + amm[:,0]
    # # samm = numpy.argsort(damm)
    # # lgti = numpy.argwhere(damm > odamm)
    j = 0

if __name__ == '__main__':
    main()
