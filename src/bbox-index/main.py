import sys
import numpy
import itertools
import struct

numpy.set_printoptions(floatmode = 'maxprec', suppress = True)
epsilon = numpy.finfo(numpy.float32).eps

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def normalize(x):
    y = x.reshape((-1, x.shape[-1]))
    return (y / numpy.linalg.norm(y, axis = -1)[:, None]).reshape(x.shape)

def dot(x, y):
    return numpy.sum(x * y, axis = -1)

# Recursively create a node from triangle data.
# The data is not copied, it is just passed down, the selection_index changes which set of triangle will be contained.
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
    # While this is not perfectly balanced tree, the mininmum leaf node is 6,
    # which means depth of 256 is probably more triangles than what any GPU can load in memory.
    # If this is reached, the split is not working well.
    if node.depth >= 256:
        raise RuntimeError('Index tree too deep: > 256 levels')
    # A leaf node will contain less than 6 triangles. We can examine those triangles one-by-one, which will have nearly the same performance
    # as a bounding box check. Narrowing down triangles to 6 using O(logN) algorithm is a sufficient target.
    if selection_index.shape[0] <= 6:
        node.is_container = False
        print('> %d: %d' % (node.depth, selection_index.shape[0]))
        return node
    
    # We try three different split points:
    # Median - this works the best of splitting triangles groups in half, but it might fail on nodes with too few triangles.
    # Mid-space plane splitting the volume of the bounding box in half.
    # The mean plane based on vertex position.
    # Ideally the split will have 2 goals:
    # 1. Minimize the number of mid-triangles (group[2]);
    # 2. Make the number of group[0] and group[1] triangles relatively equal;
    split_factor_list = numpy.concatenate([
        numpy.median(vertex_list, axis = 0)[None],
        ((bounding_box[1] + bounding_box[0]) * 0.5)[None],
        numpy.mean(vertex_list, axis = 0)[None]
    ])
    vertex_split_diff_list = triangle_list.position[:, :, None] - split_factor_list
    # The split factor mask has the following shape:
    # 3 groups: group[0] is less than, group[1] is more than, and group[2] is triangles that cross the splitting plane.
    # N triangles:
    # 3 vertex per triangle
    # 3 split factors from split_factor_list
    # 3 dimensions: X, Y, Z
    vertex_split_factor_mask = numpy.concatenate([
        (vertex_split_diff_list <= -epsilon)[None],
        (vertex_split_diff_list >= +epsilon)[None],
        (numpy.abs(vertex_split_diff_list) < epsilon)[None]
    ], axis = 0)
    # There are 3 axis-aligned planes that can split a triangle. The normal of those planes correspond to the unit vectors.

    # First isolate triangles that are easy to recognize, we have three groups:
    # [0]: triangles with *all* vertices less than the plane;
    # [1]: triangles with *all* vertices greater than the plane;
    # [2]: triangles with *all* vertices (nearly) equal to the plane (very unlikely);
    # This will not contain triangles that cross the plane;
    triangle_split_factor_mask = numpy.all(vertex_split_factor_mask, axis = 2)
    # We reassign the [2] into the mask of all triangles not in group 0 and 1, i.e. those that are split.
    triangle_split_factor_mask[2] = numpy.logical_and(numpy.logical_not(triangle_split_factor_mask[0]), numpy.logical_not(triangle_split_factor_mask[1]))
    # 3 split factors
    # 3 dimensions
    # 3 groups: less than plane, greater than plane, at plane
    distribution_list = numpy.moveaxis(numpy.count_nonzero(triangle_split_factor_mask, axis = 1), 0, -1)

    # The distribution list is NOT exactly a filter, all possible combinations will be attempted, the list only define the order in which we try.
    # The score received is based on two factors:
    # - Group[2] must be as small as possible;
    # - Group[1] and Group[0] must contain equal number of triangles (split must be approximately in half), or the abs([1] - [0]) must be as small as possible.
    # This works well when the node contains a lot of well distributed rectangles, but approaching the leaf nodes can make group[0] or group[1] have
    # no triangles, i.e. one of the groups contain all triangles.
    # To avoid this, different dimension and split factor can be attempted.
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
                else:
                    continue # Should we have else here? It worked fine without...
        # We only consider a node where triangles are split into two or more (namely, 3) children, each of them will be non-empty.
        if len(children) >= 2:
            node.children = children
            node.split_factor = split_factor
            node.distribution = distribution
            node.is_container = True
            print('+ %d: %r' % (node.depth, list(node.distribution)))
            return node
        pass
        # If a node does not result in at least 2 child nodes, we try another dimension and/or split factor
    return None

def validate_node(node, triangle_count):
    triangle_set = set()
    def validate_recursive(node):
        if node.is_container:
            for child in node.children:
                validate_recursive(child)
        else:
            for index in node.index:
                if index in triangle_set:                    
                    raise ValueError(f'An index node contains a duplicate triangle: {index}')
                triangle_set.add(index)
        return True
    if not validate_recursive(node):
        return False
    if len(triangle_set) != triangle_count:
        raise ValueError(f'An index node contains different number of triangles, expected: {triangle_count}, got {len(triangle_set)}')

class TreeStorage:
    def __init__(self, root):
        self.byte_list = []
        self.node_index = dict()
        self.byte_length = 4
        root_offset = self.insert_node(root)
        self.byte_list.insert(0, struct.pack('<I', root_offset))

    def append(self, data):
        self.byte_list.append(data)
        self.byte_length += len(data)

    def insert_node(self, node):
        if node in self.node_index:
            return self.node_index[node]
        if node.is_container:
            for child in node.children:
                self.insert_node(child)
        node_offset = self.byte_length
        self.node_index[node] = node_offset
        if node.is_container:
            self.append(b'node')
        else:
            self.append(b'leaf')
        for value in node.bounding_box.flat:
            self.append(struct.pack('<f', value))
        if node.is_container:
            self.append(struct.pack('<I', len(node.children)))
            for child in node.children:
                self.append(struct.pack('<I', self.node_index[child]))
        else:
            self.append(struct.pack('<I', len(node.index)))
            for index in node.index:
                self.append(struct.pack('<I', index))
        return node_offset


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
    data_tree = TreeStorage(root_node)
    validate_node(root_node, triangles.shape[0])
    sys.stdout.buffer.write(b''.join(data_tree.byte_list))

if __name__ == '__main__':
    _main()
