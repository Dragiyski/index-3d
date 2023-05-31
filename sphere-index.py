import numpy
import numpy.matlib 
import functools

sphere_container_type = 1

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def verify_node(source, node):
    if hasattr(node, 'container') and node.container is True:
        for child in node.children:
            verify_node(source, child)
    if node.type == 'Triangle':
        if not numpy.all(numpy.linalg.norm(source.vertices[node.indices][:,[0,1,2]] - node.parent.center, axis=-1) <= node.parent.radius):
            raise ValueError(f'Triangle with indices {node.indices} does not fit in sphere at ({node.parent.center}; {node.parent.radius})')

def create_node(source, selected_triangles, position_index, parent=None):
    # Step 1: Determine the bounding sphere
    triangles = source.indices[selected_triangles]
    (indices, triangle_indices) = numpy.unique(triangles, return_inverse=True)
    triangle_indices = triangle_indices.reshape((triangle_indices.shape[0] // 3, 3))
    vertices = source.vertices[indices,:][:,position_index]
    mid_point = numpy.average(vertices, axis=0)
    mid_vectors = vertices - mid_point
    distances = numpy.linalg.norm(mid_vectors, axis=-1)
    distance_sort = numpy.argsort(distances)
    radius = distances[distance_sort[-1]]

    node = Data(type='Sphere', container=True, center=mid_point, radius=radius, children=[], parent=parent, pole=None)

    # Case 1: There are way too few triangles to warrant splitting. It will be faster to manually process them.
    if (triangles.shape[0] <= 4):
        print(f'> [{triangles.shape[0]}]')
        for triangle in triangles:
            node.children.append(Data(type='Triangle', container=False, indices=list(triangle), parent=node))
        return node

    # Splitting
    ref_vector = mid_vectors[distance_sort[-1]] / distances[distance_sort[-1]]
    node.pole = ref_vector
    triangle_vector = mid_vectors[triangle_indices,:]
    triangle_projection = numpy.dot(triangle_vector, ref_vector)
    split_factor = numpy.median(triangle_projection)
    groups = [
        numpy.argwhere(numpy.all(triangle_projection <= split_factor, axis=-1)).squeeze(),
        numpy.argwhere(numpy.all(triangle_projection >= split_factor, axis=-1)).squeeze(),
    ]
    for i in range(len(groups)):
        if len(groups[i].shape) <= 0:
            groups[i] = groups[i][None]
    rest = numpy.delete(numpy.arange(triangle_projection.shape[0]), numpy.vstack((groups[0][:,None], groups[1][:,None])), axis=0)
    if len(rest.shape) > 0:
        groups.append(rest)
    distribution = [x.shape[0] for x in groups]
    # In case we cannot split sufficiently, we should use other methods of splitting
    # For now just process all triangles manually...
    if distribution[0] + distribution[1] < 6:
        print(f'! [{distribution[0]}, {distribution[1]}, {distribution[2]}] => {sum(distribution)}')
        for group in groups:
            for triangle_index in group:
                node.children.append(Data(type='Triangle', container=False, indices=list(source.indices[selected_triangles[triangle_index]]), parent=node))
        return node
    # int_data: array<i32>: will be: object_type, children_count, child[children_count]
    # where each child[*] will be int reference to tree_ref.
    for group in groups:
        if len(group) > 0:
            node.children.append(create_node(source, group, position_index, node))
    node.distribution = distribution
    return node


def main():
    with open('cottage/vertex-buffer.bin', 'rb') as file:
        vertex_buffer = numpy.frombuffer(file.read(), dtype=numpy.float32)
    with open('cottage/index-buffer.bin', 'rb') as file:
        index_buffer = numpy.frombuffer(file.read(), dtype=numpy.uint16)
    vertex_buffer.shape = (len(vertex_buffer) // 8, 8)
    index_buffer.shape = (len(index_buffer) // 3, 3)
    position_index = numpy.array([0, 1, 2])
    triangle_index = numpy.arange(index_buffer.shape[0])
    source = Data(
        vertices=vertex_buffer,
        indices=index_buffer,
        triangles=triangle_index
    )

    storage = Data(int_data=[], float_data=[], data_ref=[], tree_ref=[], root=None)
    root_node = create_node(source, numpy.arange(source.triangles.shape[0]), position_index)
    verify_node(source, root_node)

    # Type: Mesh data: A continuous stream of varying data for a mesh
    # Consists of type: 1 = mesh_data
    # Flags:
    # 0 = normal_flag
    # 1 = texture_coords_flag
    # 2 = tangent_flag
    # 3 = bitangent_flag
    # 4 = color_flag
    # 5 = alpha_flag
    # 6 = symetric_color (for debugging) = makes RGB colors from range [-1, 1] instead of [0, 1], very useful in conjunction of specifying color_index[3] = normal_index[3]
    # Int data:
    # stride = number of floats per vertex (NOT bytes)
    # vertex_count
    # position_index[3] = the indices of the position
    # normal_index[3] = the indices of the normal (only if normal_flag present)
    # texture_coords_index[2] = the indices of the texture coordinates (only if texture_coords_flag present)
    # tangent_index[3] = the indices of the tangent vector (only if tangent_flag present)
    # bitangent_index[3] = the indices of the tangent vector (only if bitangent_flag present)
    # color_index[3] = the indices of the colors (only if color_flag present)
    # alpha_index = the index of the alpha channel (only if alpha_flag present)
    # All indices must be X >= 0 and X < stride
    # Float data: consists of vertex_count * stride f32 values.

    # Mesh: A container type that:
    # * Specify mesh data
    # * Can specify 4x4 matrix (on flag)
    # * Can be a child of another mesh
    # * As a container specify one of more children

    # TODO: If this index creation is working well and triangles are properly separated, an update to the index can be made, that find a better geon:
    # The new geon must be primitive-intersectable, or alternatively a non-flat box.
    # A geon is better when its volume is as smaller as possible, while still contain

    ray_origin = numpy.array([50, 50, 50], dtype=numpy.float32)
    ray_direction = numpy.array([0, 0, 0], dtype=numpy.float32) - ray_origin
    ray_direction = ray_direction / numpy.linalg.norm(ray_direction)

    current_node = root_node
    min_depth = 0.0
    max_depth = numpy.inf
    # Depth optimization:
    # Depth has two modifiers:
    # - For rendering object, actual intersection depth must be between min_depth and max_depth. If true, intersection occurs and max_depth is updated with that value.
    # This is consistent with the operation of the depth buffer in OpenGL. An object will override previous intersection only if it is closer,
    # but not if negative (as initial min_depth will be 0.0).
    # - Within a container, a ray always have two solutions: min_intersect and max_intersect. The allowed intersection is the intersection between min_depth/max_depth line
    # and min_intersection/max_intersection line. If this is empty, no intersection occurs even when there is a solution.
    while True:
        while True:
            has_intersection = False
            if current_node.type == 'Sphere':
                sphere_vector = current_node.center - ray_origin
                b = 2.0 * (numpy.dot(ray_direction, ray_origin) - numpy.dot(ray_direction, current_node.center))
                c = numpy.dot(sphere_vector, sphere_vector) - current_node.radius * current_node.radius

                D = b * b - 4 * c
                if D < 0.0:
                    break
                has_intersection = True
                min_depth = numpy.max([min_depth, (-b - numpy.sqrt(D)) / 2.0])
                max_depth = numpy.min([max_depth, (-b + numpy.sqrt(D)) / 2.0])
            if current_node.type == 'Triangle':
                # TODO:
                pass
            break
        if has_intersection and current_node.container and len(current_node.children) > 0:
            current_node = current_node.children[0]
            continue
        if current_node.parent is None:
            break
        else:
            next_node = None
            child_node = current_node
            while True:
                if child_node.parent is None:
                    next_node = None
                    break
                parent_node = child_node.parent
                child_index = parent_node.children.index(child_node)
                if child_index + 1 < len(parent_node.children):
                    next_node = parent_node.children[child_index + 1]
                    break
                child_node = parent_node
            if next_node is None:
                break
            current_node = next_node
                
                        
                

if __name__ == '__main__':
    main()
