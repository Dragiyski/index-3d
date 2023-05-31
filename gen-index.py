import sys
import numpy
from PIL import Image

epsilon = numpy.finfo(numpy.float32).eps


_triangle_intersection_count = 0
_sphere_intersection_count = 0

state_float_index = 0
state_int_index = 0
state_object_index = 0
state_tree_index = 0

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def make_sphere_distribution(vertices, triangles, selected_triangle_indices, plane_normal):
    selected_triangle_vertices = vertices[triangles[selected_triangle_indices]]
    selected_triangle_rejection = numpy.dot(selected_triangle_vertices, plane_normal)
    split_factor = numpy.median(selected_triangle_rejection)
    groups = [
        numpy.argwhere(numpy.all(selected_triangle_rejection <= split_factor, axis=-1)).squeeze(),
        numpy.argwhere(numpy.all(selected_triangle_rejection >= split_factor, axis=-1)).squeeze(),
    ]
    for i in range(len(groups)):
        if len(groups[i].shape) <= 0:
            groups[i] = groups[i][None]
    rest = numpy.delete(numpy.arange(selected_triangle_rejection.shape[0]), numpy.vstack((groups[0][:,None], groups[1][:,None])), axis=0)
    if len(rest.shape) <= 0:
        rest = rest[None]
    groups.append(rest)
    return [selected_triangle_indices[group] for group in groups]
        
def make_sphere_node(vertices, triangles, selected_triangle_indices, parent=None):
    selected_triangles = triangles[selected_triangle_indices]
    (selected_indices, triangle_indices) = numpy.unique(selected_triangles, return_inverse=True)
    triangle_indices = triangle_indices.reshape(selected_triangles.shape)
    selected_vertices = vertices[selected_indices]
    sphere_center = numpy.average(selected_vertices, axis=0)
    sphere_vertex_vectors = selected_vertices - sphere_center
    sphere_distances = numpy.linalg.norm(sphere_vertex_vectors, axis=-1)
    sphere_distances_sort = numpy.argsort(sphere_distances)
    sphere_radius = sphere_distances[sphere_distances_sort[-1]]

    node = Data(geometry='Sphere', container=True, center=sphere_center, radius=sphere_radius, children=[], parent=parent, distribution=None, depth=0)

    if parent is not None:
        node.depth = parent.depth + 1

    if selected_triangle_indices.shape[0] <= 6:
        for selected_triangle_index in selected_triangle_indices:
            node.children.append(Data(geometry='Triangle', container=False, index=selected_triangle_index, parent=parent))
        node.max_depth = 1
        return node
    
    assert numpy.abs(sphere_distances[sphere_distances_sort[-1]]) >= epsilon
    distribution_vectors = [sphere_vertex_vectors[sphere_distances_sort[-1]] / sphere_distances[sphere_distances_sort[-1]]]
    triangle_vertices = vertices[triangles[selected_triangle_indices]]
    triangle_normals = numpy.cross(triangle_vertices[:,2,:] - triangle_vertices[:,0,:], triangle_vertices[:,1,:] - triangle_vertices[:,0,:])
    triangle_normals = triangle_normals / numpy.linalg.norm(triangle_normals, axis=-1).reshape(-1, 1)
    triangle_normal_dir = numpy.dot(triangle_normals, triangle_normals[0])
    triangle_normal_dir = triangle_normal_dir / abs(triangle_normal_dir)
    triangle_normals = numpy.multiply(triangle_normal_dir[:,None], triangle_normals)
    assert numpy.all(numpy.dot(triangle_normals, triangle_normals[0]) / numpy.abs(numpy.dot(triangle_normals, triangle_normals[0])) == 1.0)
    triangle_normal_average = numpy.average(triangle_normals, axis=0)
    assert numpy.abs(numpy.linalg.norm(triangle_normal_average)) >= epsilon
    triangle_normal_average = triangle_normal_average / numpy.linalg.norm(triangle_normal_average)
    distribution_vectors.append(triangle_normal_average)
    binormal = numpy.cross(distribution_vectors[0], distribution_vectors[1])
    assert numpy.linalg.norm(binormal) >= epsilon
    binormal = binormal / numpy.linalg.norm(binormal)
    distribution_vectors.append(binormal)
    del binormal, triangle_normal_average, triangle_normal_dir, triangle_normals

    distributions = []
    for distribution_vector in distribution_vectors:
        distributions.append(make_sphere_distribution(vertices, triangles, selected_triangle_indices, distribution_vector))

    selected_distribution = 0
    for i in range(1, len(distributions)):
        if distributions[i][2].shape[0] < distributions[selected_distribution][2].shape[0]:
            selected_distribution = i

    node.distribution = list(x.shape[0] for x in distributions[selected_distribution])

    for group in distributions[selected_distribution]:
        if group.shape[0] > 0:
            child = make_sphere_node(vertices, triangles, group, node)
            while child.container and len(child.children) == 1:
                child = child.children[0]
            node.children.append(child)

    node.max_depth = 1 + numpy.max(list(c.max_depth for c in node.children if c.container))

    return node

def raytrace(vertices, triangles, ray_origin, ray_direction, node):
    global _triangle_intersection_count, _sphere_intersection_count
    if node.geometry == 'Sphere':
        _sphere_intersection_count += 1
        sphere_vector = node.center - ray_origin
        b = 2.0 * (numpy.dot(ray_direction, ray_origin) - numpy.dot(ray_direction, node.center))
        c = numpy.dot(sphere_vector, sphere_vector) - node.radius * node.radius

        D = b * b - 4 * c
        if D < 0:
            return None
        depth = (-b - numpy.sqrt(D)) * 0.5
        if depth < 0.0:
            depth = (-b + numpy.sqrt(D)) * 0.5
            if depth < 0.0:
                return None
        
        hit_point = ray_origin + depth * ray_direction
        normal = hit_point - node.center
        normal = normal / numpy.linalg.norm(normal)
        
    elif node.geometry == 'Triangle':
        _triangle_intersection_count += 1
        triangle_vertices = vertices[triangles[node.index]]
        edge1 = triangle_vertices[1] - triangle_vertices[0]
        edge2 = triangle_vertices[2] - triangle_vertices[0]
        h = numpy.cross(ray_direction, edge2)
        a = numpy.dot(edge1, h)
        if numpy.abs(a) < epsilon:
            return None
        f = 1.0 / a
        s = ray_origin - triangle_vertices[0]
        u = f * numpy.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = numpy.cross(s, edge1)
        v = f * numpy.dot(ray_direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        depth = f * numpy.dot(edge2, q)
        if depth < 0.0:
            return None
        
        hit_point = ray_origin + depth * ray_direction
        normal = numpy.cross(edge1, edge2)
        normal = normal / numpy.linalg.norm(normal)
    
    else:
        return None
    
    result = None

    if node.container:
        for child in node.children:
            child_result = raytrace(vertices, triangles, ray_origin, ray_direction, child)
            if child_result is not None:
                if result is not None:
                    if result.depth <= child_result.depth:
                        continue
                result = child_result
    else:
        result = Data(node=node, depth=depth, hit_point=hit_point, normal=normal)
    
    return result

def _main():
    count_vertices = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_normals = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_texture_coords = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_faces = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')

    data_vertices = numpy.frombuffer(sys.stdin.buffer.read(count_vertices * 3 * 4), dtype=numpy.float32).reshape((count_vertices, 3))
    data_normals = numpy.frombuffer(sys.stdin.buffer.read(count_normals * 3 * 4), dtype=numpy.float32).reshape((count_normals, 3))
    data_texture_coords = numpy.frombuffer(sys.stdin.buffer.read(count_texture_coords * 2 * 4), dtype=numpy.float32).reshape((count_texture_coords, 2))
    data_faces = numpy.frombuffer(sys.stdin.buffer.read(count_faces * 3 * 3 * 4), dtype=numpy.uint32).reshape((count_faces, 3, 3))
    
    root_node = make_sphere_node(data_vertices, data_faces[:,:,0], numpy.arange(data_faces.shape[0]))

    camera_origin = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)
    camera_forward = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32) - camera_origin
    camera_forward = camera_forward / numpy.linalg.norm(camera_forward)
    camera_right = numpy.cross(camera_forward, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32))
    camera_right = camera_right / numpy.linalg.norm(camera_right)
    camera_up = numpy.cross(camera_forward, camera_right)

    width = 640
    height = 480
    field_of_view = (60.0 * 0.5) / 180.0 * numpy.pi
    aspect_ratio = float(width) / float(height)
    diagonal_size = numpy.tan(field_of_view)
    screen_height = diagonal_size / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    screen_width = aspect_ratio * screen_height

    world_screen_right = screen_width * camera_right
    world_screen_up = screen_height * camera_up

    color_data = []

    intersect_count = 0

    for y in range(height):
        for x in range(width):
            position = numpy.array([float(x) / float(width), float(y) / float(height)], dtype=numpy.float32)
            position = position * 2.0 - 1.0
            world_screen_center = camera_origin + camera_forward
            world_screen_point = world_screen_center + position[0] * world_screen_right + position[1] * world_screen_up
            ray_direction = world_screen_point - camera_origin
            ray_direction = ray_direction / numpy.linalg.norm(ray_direction)
            intersection = raytrace(data_vertices, data_faces[:,:,0], camera_origin, ray_direction, root_node)
            if intersection is None:
                color_data.append(numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32))
            else:
                intersect_count += 1
                color_data.append(intersection.normal * 0.5 + 0.5)
    # intersection = raytrace(data_vertices, data_faces[:,:,0], ray_origin, ray_direction, root_node)
    color_data = numpy.array(color_data).reshape((height, width, 3))
    color_data = (color_data * 255.0).astype(numpy.uint8)
    image = Image.fromarray(color_data, mode='RGB')
    image.save('test.png', format='PNG', optimize=True)

    print('Average sphere intersection per pixel: %.3f' % (float(_sphere_intersection_count) / float(width * height)))
    print('Average triangle intersection per pixel: %.3f' % (float(_triangle_intersection_count) / float(width * height)))

if __name__ == '__main__':
    _main()
