import sys
import numpy
from PIL import Image

epsilon = numpy.finfo(numpy.float32).eps

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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



    # camera_origin = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)
    # camera_forward = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32) - camera_origin
    # camera_forward = camera_forward / numpy.linalg.norm(camera_forward)
    # camera_right = numpy.cross(camera_forward, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32))
    # camera_right = camera_right / numpy.linalg.norm(camera_right)
    # camera_up = numpy.cross(camera_forward, camera_right)

    # width = 640
    # height = 480
    # field_of_view = (60.0 * 0.5) / 180.0 * numpy.pi
    # aspect_ratio = float(width) / float(height)
    # diagonal_size = numpy.tan(field_of_view)
    # screen_height = diagonal_size / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    # screen_width = aspect_ratio * screen_height

    # world_screen_right = screen_width * camera_right
    # world_screen_up = screen_height * camera_up

    # color_data = []

    # intersect_count = 0

    # for y in range(height):
    #     for x in range(width):
    #         position = numpy.array([float(x) / float(width), float(y) / float(height)], dtype=numpy.float32)
    #         position = position * 2.0 - 1.0
    #         world_screen_center = camera_origin + camera_forward
    #         world_screen_point = world_screen_center + position[0] * world_screen_right + position[1] * world_screen_up
    #         ray_direction = world_screen_point - camera_origin
    #         ray_direction = ray_direction / numpy.linalg.norm(ray_direction)
    #         intersection = raytrace(data_vertices, data_faces[:,:,0], camera_origin, ray_direction, root_node)
    #         if intersection is None:
    #             color_data.append(numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32))
    #         else:
    #             intersect_count += 1
    #             color_data.append(intersection.normal * 0.5 + 0.5)
    # # intersection = raytrace(data_vertices, data_faces[:,:,0], ray_origin, ray_direction, root_node)
    # color_data = numpy.array(color_data).reshape((height, width, 3))
    # color_data = (color_data * 255.0).astype(numpy.uint8)
    # image = Image.fromarray(color_data, mode='RGB')
    # image.save('test.png', format='PNG', optimize=True)

    # print('Average sphere intersection per pixel: %.3f' % (float(_sphere_intersection_count) / float(width * height)))
    # print('Average triangle intersection per pixel: %.3f' % (float(_triangle_intersection_count) / float(width * height)))

NODE_TYPE_MESH=1
NODE_TYPE_CONTAINER=2
NODE_TYPE_MESH_TRIANGLE=3

MESH_FLAG_TRANSFORM=1
MESH_FLAG_NORMALS=2
MESH_FLAG_TEX_COORDS=4
MESH_FLAG_TANGENT=8
MESH_FLAG_BITANGENT=16

NODE_GEOMETRY_SPHERE=1

def raytrace_triangle(triangle, ray_origin, ray_direction):
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = numpy.cross(ray_direction, edge2)
    a = numpy.dot(edge1, h)
    if numpy.abs(a) < epsilon:
        return None
    f = 1.0 / a
    s = ray_origin - triangle[0]
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
    
    return (depth, u, v)

def raytrace_sphere(sphere_origin, sphere_radius, ray_origin, ray_direction, dot_ray_origin_ray_direction):
    sphere_vector = sphere_origin - ray_origin
    b = 2.0 * (dot_ray_origin_ray_direction - numpy.dot(ray_direction, sphere_origin))
    c = numpy.dot(sphere_vector, sphere_vector) - sphere_radius * sphere_radius
    D = b * b - 4 * c
    if D < 0:
        return None
    depth = (-b - numpy.sqrt(D)) * 0.5
    if depth >= 0.0:
        return depth
    depth = (-b + numpy.sqrt(D)) * 0.5
    if depth >= 0.0:
        return depth
    return None

def execute_shader(data, start_index, camera, x, y, width, height):
    position = numpy.array([float(x) / float(width), float(y) / float(height)], dtype=numpy.float32)
    position = position * 2.0 - 1.0
    world_screen_point = camera.screen.center + position[0] * camera.screen.right + position[1] * camera.screen.up
    ray_direction = world_screen_point - camera.origin
    ray_direction = ray_direction / numpy.linalg.norm(ray_direction)
    ray_origin = camera.origin
    dot_ray_origin_ray_direction = numpy.dot(ray_origin, ray_direction)

    current_node_index = start_index
    current_intersection = None
    current_mesh_index = -1
    node_processed = 0
    while current_node_index != -1:
        node_processed += 1
        current_node = data.tree[current_node_index]
        current_object = data.object[current_node[3]]

        if current_object[0] == NODE_TYPE_MESH:
            current_mesh_index = current_node_index
            current_node_index = data.int[current_object[2] + 1]
            continue
        elif current_object[0] == NODE_TYPE_CONTAINER:
            relevant = False
            geometry = data.int[current_object[2] + 0]
            if geometry == NODE_GEOMETRY_SPHERE:
                # print('Sphere: { center: [%.6f, %.6f, %.6f], radius: %.6f }' % (data.float[current_object[3] + 0], data.float[current_object[3] + 1], data.float[current_object[3] + 2], data.float[current_object[3] + 3]))
                depth = raytrace_sphere(
                    data.float[current_object[3] + 0:current_object[3] + 3],
                    data.float[current_object[3] + 3],
                    ray_origin,
                    ray_direction,
                    dot_ray_origin_ray_direction
                )
                if depth is not None:
                    relevant = True
            else:
                raise RuntimeError('Unsupported geometry: %d' % (geometry))
            if relevant:
                if current_intersection is not None and depth >= current_intersection.depth:
                    relevant = False
                if relevant:
                    current_node_index = current_object[1]
                    continue
        elif current_object[0] == NODE_TYPE_MESH_TRIANGLE:
            relevant = False
            mesh_node = data.tree[current_mesh_index]
            mesh = data.object[mesh_node[3]]
            data_flags = (mesh[1] >> 1) & 0xF
            data_item_count = 0
            # For shader do not use while loop, use loop from 1 to 5 and if (flags & (1 << i)) increase data.
            while data_flags != 0:
                data_item_count += 1
                data_flags = data_flags >> 1
            # Note: this is python specific, for an actual shader, here it will be faster to directly get 3 indices and then 3 vertices, potentially with repeated code;
            triangle_indices = data.int[mesh[2] + 4 + data_item_count : mesh[2] + 4 + data_item_count + data.int[3] * (1 + data_item_count) * 3].reshape(data.int[3], 3, 1 + data_item_count)[current_object[1]][:,0]
            triangle_vertices = data.float[0 : data.int[mesh[2] + 2] * 3].reshape(data.int[mesh[2] + 2], 3)[triangle_indices]
            # print('Triangle: { 0: [%.6f, %.6f, %.6f], 1: [%.6f, %.6f, %.6f], 2: [%.6f, %.6f, %.6f] }' % (triangle_vertices[0][0], triangle_vertices[0][1], triangle_vertices[0][2], triangle_vertices[1][0], triangle_vertices[1][1], triangle_vertices[1][2], triangle_vertices[2][0], triangle_vertices[2][1], triangle_vertices[2][2]))
            triangle_intersection = raytrace_triangle(triangle_vertices, ray_origin, ray_direction)
            if triangle_intersection is not None:
                relevant = True
                if current_intersection is not None and triangle_intersection[0] >= current_intersection.depth:
                    relevant = False
                if relevant:
                    normal = numpy.cross(triangle_vertices[2] - triangle_vertices[0], triangle_vertices[1] - triangle_vertices[0])
                    normal = normal / numpy.linalg.norm(normal)
                    # Barycentric here need does not need to be stored in the shader
                    # The shader can use barycentric here to compute the interpolated normals, texture coordinates, tangent and bitangent,
                    # for the point of intersection.
                    # Additional flags for smoothing can potentially affect the normal.
                    current_intersection = Data(
                        depth=triangle_intersection[0],
                        normal=normal,
                        hit_point=ray_origin + triangle_intersection[0] * ray_direction
                    )
        else:
            raise RuntimeError('Unsupported node type: %d' % (current_object[0]))

        current_node_index = -1
        while True:
            if current_node[2] != -1:
                current_node_index = current_node[2]
                break
            if current_node[0] == -1:
                break
            current_node = data.tree[current_node[0]]
    return current_intersection


def _main():
    start_index = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    float_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    int_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    ptr_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    tree_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    float_data = numpy.frombuffer(sys.stdin.buffer.read(float_length * 4), dtype=numpy.float32)
    int_data = numpy.frombuffer(sys.stdin.buffer.read(int_length * 4), dtype=numpy.int32)
    object_data = numpy.frombuffer(sys.stdin.buffer.read(ptr_length * 4 * 4), dtype=numpy.int32).reshape((ptr_length, 4))
    tree_data = numpy.frombuffer(sys.stdin.buffer.read(tree_length * 4 * 4), dtype=numpy.int32).reshape((tree_length, 4))

    data = Data(float=float_data, int=int_data, object=object_data, tree=tree_data)

    camera_origin = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)
    camera_forward = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32) - camera_origin
    camera_forward = camera_forward / numpy.linalg.norm(camera_forward)
    camera_right = numpy.cross(camera_forward, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32))
    camera_right = camera_right / numpy.linalg.norm(camera_right)
    camera_up = numpy.cross(camera_forward, camera_right)

    width = 400
    height = 300
    field_of_view = (60.0 * 0.5) / 180.0 * numpy.pi
    aspect_ratio = float(width) / float(height)
    diagonal_size = numpy.tan(field_of_view)
    screen_height = diagonal_size / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    screen_width = aspect_ratio * screen_height

    world_screen_right = screen_width * camera_right
    world_screen_up = screen_height * camera_up
    world_screen_center = camera_origin + camera_forward

    screen = Data(center=world_screen_center, right=world_screen_right, up=world_screen_up)
    camera = Data(origin=camera_origin, forward=camera_forward, screen=screen)


    # execute_shader(data, start_index, camera, width // 2, height // 2, width, height)

    color_data = []
    for y in range(height):
        for x in range(width):
            intersection = execute_shader(data, start_index, camera, x, y, width, height)
            if intersection is not None:
                color_data.append(intersection.normal * 0.5 + 0.5)
            else:
                color_data.append([0, 0, 0])

    color_data = numpy.array(color_data).reshape((height, width, 3))
    color_data = (color_data * 255.0).astype(numpy.uint8)
    image = Image.fromarray(color_data, mode='RGB')
    image.save('test.png', format='PNG', optimize=True)
    

if __name__ == '__main__':
    _main()