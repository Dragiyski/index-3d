import sys
import numpy
from numpy import dot, cross, sqrt, abs
from PIL import Image

numpy.set_printoptions(floatmode='maxprec', suppress=True)
epsilon = numpy.finfo(numpy.float32).eps

def normalize(v):
    return v / numpy.linalg.norm(v)

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

MESH_FLAG_NORMALS=1
MESH_FLAG_TEX_COORDS=2
MESH_FLAG_TANGENT=4
MESH_FLAG_BITANGENT=8

other_dim = [
    [1, 2],
    [0, 2],
    [0, 1]
]

def action_raytrace_triangle(depth_out, barycentric_out, triangle, ray_origin, ray_direction):
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = numpy.cross(ray_direction, edge2)
    a = numpy.dot(edge1, h)
    if numpy.abs(a) < epsilon:
        return False
    f = 1.0 / a
    s = ray_origin - triangle[0]
    u = f * numpy.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = numpy.cross(s, edge1)
    v = f * numpy.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    depth = f * numpy.dot(edge2, q)
    if depth < 0.0:
        return False
    depth_out.set(depth)
    barycentric_out.set(numpy.array([1.0 - u - v, u, v], dtype=numpy.float32))
    return True

class RayIntersection:
    def __init__(self):
        self.normal = numpy.array([0, 0, 0], dtype=numpy.float32)
        self.depth = numpy.inf

class NDArrayRef:
    def __init__(self, array, *index):
        if not isinstance(array, numpy.ndarray):
            raise ValueError('Specified array is not numpy.ndarray')
        if (len(index) > len(array.shape)):
            raise IndexError('Number of indices %d is larger than number of dimensions %d' % (len(index), len(array.shape)))
        self.__array = array
        self.__index = index

    def get(self):
        ref = self.__array
        for index in self.__index:
            ref = ref[index]
        return ref
    
    def set(self, value):
        ref = self.__array
        for index in self.__index[:-1]:
            ref = ref[index]
        ref[self.__index[-1]] = value

def action_raytrace_container(depth_out, node_origin, node_radius, node_axes, ray_origin, ray_direction, dot_ray_origin_ray_direction):
    s = node_origin - ray_origin
    b = 2.0 * (dot_ray_origin_ray_direction - dot(node_origin, ray_direction))
    c = dot(s, s) - node_radius * node_radius

    D = b * b - 4 * c
    if D < 0.0:
        return False
    depth = (-b - sqrt(D)) * 0.5
    if depth < 0.0:
        depth = (-b + sqrt(D)) * 0.5
        if depth < 0.0:
            return False
    
    # mat_box_transform = numpy.array([
    #     [node_axes[0], 0.0, 0.0, node_origin[0]],
    #     [0.0, node_axes[1], 0.0, node_origin[1]],
    #     [0.0, 0.0, node_axes[2], node_origin[2]],
    #     [0.0, 0.0, 0.0, 1.0]
    # ], dtype=numpy.float32)

    # mat_box_inv_transform = numpy.array([
    #     [1.0 / node_axes[0], 0.0, 0.0, -node_origin[0] / node_axes[0]],
    #     [0.0, 1.0 / node_axes[1], 0.0, -node_origin[1] / node_axes[1]],
    #     [0.0, 0.0, 1.0 / node_axes[2], -node_origin[2] / node_axes[2]],
    #     [0.0, 0.0, 0.0, 1.0]
    # ], dtype=numpy.float32)

    # This is multiplication (*) in shader, but numpy makes mat-vec multiplication using numpy.dot
    # box_ray_origin = numpy.dot(mat_box_inv_transform, [*ray_origin, 1.0])[0:3]
    # box_ray_direction = normalize(numpy.dot(mat_box_inv_transform, [*(ray_origin + ray_direction), 1.0])[0:3] - box_ray_origin)
    has_box_intersection = False
    depth = numpy.inf
    for dim in range(3):
        if abs(ray_direction[dim]) < epsilon:
            continue
        for dir in [-1.0, 1.0]:
            # Normal is: N = [0.0, 0.0, 0.0]; N[dim] = 1.0;
            # then: dot(normal, ray_origin) == ray_origin[dim]
            # and: dot(normal, ray_direction) == ray_direction[dim]
            # Thus intersection: (plane_factor - dot(normal, ray_origin)) / dot(normal, ray_direction)
            # Where plane_factor is: dot(normal, node_origin ± node_axes)
            # then: plane_factor = node_origin[dim] ± node_axes[dim]
            # unless ray_direction lies in the plane of the normal, in which case the below is division by 0.0
            plane_factor = node_origin[dim] + dir * node_axes[dim]
            box_depth = (plane_factor - ray_origin[dim]) / ray_direction[dim]
            if box_depth >= 0.0:
                hit_point = ray_origin + box_depth * ray_direction
                plane_center = numpy.array(node_origin)
                plane_center[dim] = plane_factor
                plane_vector = hit_point - plane_center
                if numpy.abs(plane_vector[other_dim[dim][0]]) <= node_axes[other_dim[dim][0]] and numpy.abs(plane_vector[other_dim[dim][1]]) <= node_axes[other_dim[dim][1]]:
                    depth = numpy.minimum(box_depth, depth)
                    has_box_intersection = True
    if has_box_intersection:
        depth_out.set(depth)
    return has_box_intersection


    (numpy.array([1.0, 1.0, 1.0]) - box_ray_origin) / box_ray_direction

total_invocation_count = 0

def execute_shader(color_out, scene_float, scene_int, scene_object, scene_tree, uniform_data, invocation_id):
    global total_invocation_count
    # Compute shader executes in workgroups, each workgroup with specific integer size.
    # As workgroup size might not divide width and height, some invocation might be out-of-bounds (severe minority of invocations)
    if invocation_id[0] >= uniform_data.image_size[0] or invocation_id[1] >= uniform_data.image_size[1]:
        return
    
    position = numpy.array([
        (float(invocation_id[0]) + 0.5) / float(uniform_data.image_size[0]),
        (float(invocation_id[1]) + 0.5) / float(uniform_data.image_size[1]),
    ], dtype=numpy.float32)
    position = position * 2.0 - 1.0
    world_screen_point = uniform_data.screen_center + position[0] * uniform_data.screen_right + position[1] * uniform_data.screen_up
    ray_origin = uniform_data.camera_origin
    ray_direction = normalize(world_screen_point - ray_origin)
    dot_ray_origin_ray_direction = dot(ray_origin, ray_direction)
    
    current_node_index = uniform_data.start_node_index
    has_intersection = False
    intersection = RayIntersection()
    current_mesh = Data(
        index = -1,
        item_count = 0,
        flags = 0,
        normal_ptr = -1,
        texcoord_ptr = -1,
        tangent_ptr = -1,
        bitangent_ptr = -1,
        vertex_ptr = -1,
        triangle_ptr = -1
    )

    local_invocation_count = 0

    while current_node_index != -1:
        print('%d: [%d, %d]: %d = %d' % (total_invocation_count, invocation_id[0], invocation_id[1], local_invocation_count, current_node_index))
        total_invocation_count += 1
        local_invocation_count += 1
        current_node = scene_tree[current_node_index]
        current_object = scene_object[current_node[3]]

        if current_object[0] == NODE_TYPE_MESH:
            current_mesh.index = current_node_index
            current_node_index = scene_int[current_object[2] + 1]
            current_mesh.flags = current_object[1]
            float_offset = current_object[3]
            int_offset = current_object[2] + 4
            current_mesh.item_count = 1
            current_mesh.vertex_ptr = float_offset
            float_offset += scene_int[current_object[2] + 2] * 3
            for i in range(0, 4):
                if (current_mesh.flags & (1 << i)) != 0:
                    current_mesh.item_count += 1
                    if i == 0:
                        current_mesh.normal_ptr = float_offset
                        float_offset += scene_int[int_offset] * 3
                    elif i == 1:
                        current_mesh.texcoord_ptr = float_offset
                        float_offset += scene_int[int_offset] * 2
                    elif i == 2:
                        current_mesh.tangent_ptr = float_offset
                        float_offset += scene_int[int_offset] * 3
                    elif i == 3:
                        current_mesh.bitangent_ptr = float_offset
                        float_offset += scene_int[int_offset] * 3
                    int_offset += 1
            current_mesh.triangle_ptr = current_object[2] + 3 + current_mesh.item_count
            continue
        elif current_object[0] == NODE_TYPE_CONTAINER:
            relevant = False
            depth = None
            # WSGL uses &depth to pass ptr<function, f32> that can be written in the function
            # Python has better mechanisms to handle that, but to avoid huge difference between simulated shader code and actual WGSL
            # we prefer creating a pointer, instead of using numpy with complex indexing or complex return types (something we cannot do in WGSL)
            def get_depth():
                nonlocal depth
                return depth
            def set_depth(value):
                nonlocal depth
                depth = value
            if action_raytrace_container(
                Data(get=get_depth, set=set_depth),
                numpy.array([scene_float[current_object[3] + 0], scene_float[current_object[3] + 1], scene_float[current_object[3] + 2]], dtype=numpy.float32),
                scene_float[current_object[3] + 3],
                numpy.array([scene_float[current_object[3] + 4], scene_float[current_object[3] + 5], scene_float[current_object[3] + 6]], dtype=numpy.float32),
                ray_origin,
                ray_direction,
                dot_ray_origin_ray_direction
            ):
                relevant = True
            if relevant:
                if has_intersection and depth >= intersection.depth:
                    relevant = False
            if relevant:
                current_node_index = current_object[1]
                continue
        elif current_object[0] == NODE_TYPE_MESH_TRIANGLE:
            relevant = False
            depth = None
            barycentric = None
            triangle_vertices = numpy.array([
                [
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 0 * current_mesh.item_count] * 3 + 0],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 0 * current_mesh.item_count] * 3 + 1],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 0 * current_mesh.item_count] * 3 + 2],
                ],
                [
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 1 * current_mesh.item_count] * 3 + 0],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 1 * current_mesh.item_count] * 3 + 1],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 1 * current_mesh.item_count] * 3 + 2],
                ],
                [
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 2 * current_mesh.item_count] * 3 + 0],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 2 * current_mesh.item_count] * 3 + 1],
                    scene_float[current_mesh.vertex_ptr + scene_int[current_mesh.triangle_ptr + current_object[1] * current_mesh.item_count * 3 + 2 * current_mesh.item_count] * 3 + 2],
                ]
            ], dtype=numpy.float32)
            def get_depth():
                nonlocal depth
                return depth
            def set_depth(value):
                nonlocal depth
                depth = value
            def get_barycentric():
                nonlocal barycentric
                return barycentric
            def set_barycentric(value):
                nonlocal barycentric
                barycentric = value
            if action_raytrace_triangle(
                Data(get=get_depth, set=set_depth),
                Data(get=get_barycentric, set=set_barycentric),
                triangle_vertices,
                ray_origin,
                ray_direction
            ):
                relevant = True
            if relevant:
                if has_intersection and depth >= intersection.depth:
                    relevant = False
            if relevant:
                has_intersection = True
                intersection.depth = depth
                intersection.normal = normalize(cross(triangle_vertices[2] - triangle_vertices[0], triangle_vertices[1] - triangle_vertices[0]))
                if dot(ray_direction, intersection.normal) >= 0.0:
                    intersection.normal = -intersection.normal
        else:
            raise ValueError('Unexpected node type: %d' % (current_object[0]))
        
        current_node_index = -1
        while True:
            if current_node[2] != -1:
                current_node_index = current_node[2]
                break
            if current_node[0] == -1:
                break
            current_node = scene_tree[current_node[0]]
    
    if has_intersection:
        color_out.set(intersection.normal * 0.5 + 0.5)
    else:
        color_out.set([0.0, 0.0, 0.0])


def _main():
    start_node_index = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    float_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    int_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    ptr_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    tree_length = int.from_bytes(sys.stdin.buffer.read(4), 'little')
    scene_float = numpy.frombuffer(sys.stdin.buffer.read(float_length * 4), dtype=numpy.float32)
    scene_int = numpy.frombuffer(sys.stdin.buffer.read(int_length * 4), dtype=numpy.int32)
    scene_object = numpy.frombuffer(sys.stdin.buffer.read(ptr_length * 4 * 4), dtype=numpy.int32).reshape((ptr_length, 4))
    scene_tree = numpy.frombuffer(sys.stdin.buffer.read(tree_length * 4 * 4), dtype=numpy.int32).reshape((tree_length, 4))

    uniform_data = Data(
        image_size = numpy.array([400, 300], dtype=numpy.int32),
        start_node_index = start_node_index,
        camera_origin = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32),
        screen_center = None,
        screen_right = None,
        screen_up = None
    )

    camera_forward = normalize(numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32) - uniform_data.camera_origin)

    camera_right = normalize(numpy.cross(camera_forward, numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)))
    camera_up = numpy.cross(camera_forward, camera_right)
    field_of_view = (60.0 * 0.5) / 180.0 * numpy.pi
    aspect_ratio = float(uniform_data.image_size[0]) / float(uniform_data.image_size[1])
    diagonal_size = numpy.tan(field_of_view)
    screen_height = diagonal_size / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    screen_width = aspect_ratio * screen_height
    uniform_data.screen_center = uniform_data.camera_origin + camera_forward
    uniform_data.screen_right = screen_width * camera_right
    uniform_data.screen_up = screen_height * camera_up
    # execute_shader(data, start_index, camera, width // 2, height // 2, width, height)

    color_data = numpy.zeros((uniform_data.image_size[1], uniform_data.image_size[0], 3), dtype=numpy.float32)
    for y in range(uniform_data.image_size[1]):
        for x in range(uniform_data.image_size[0]):
            global_invocation_id = numpy.array([x, y, 0], dtype=numpy.uint32)
            execute_shader(NDArrayRef(color_data, y, x), scene_float, scene_int, scene_object, scene_tree, uniform_data, global_invocation_id)

    image_data = (color_data * 255.0).astype(numpy.uint8)
    image = Image.fromarray(image_data, mode='RGB')
    image.save('test.png', format='PNG', optimize=True)
    

if __name__ == '__main__':
    _main()