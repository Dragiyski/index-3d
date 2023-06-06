import sys
import numpy
from PIL import Image

epsilon = numpy.finfo(numpy.float32).eps

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def create_mesh_index_child_distribution(vertices, triangles, selected_triangle_indices, plane_normal):
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
    return ([selected_triangle_indices[group] for group in groups], plane_normal, split_factor)
        
def create_mesh_shpere_container_node(vertices, triangles, selected_triangle_indices, parent=None):
    selected_triangles = triangles[selected_triangle_indices]
    (selected_indices, triangle_indices) = numpy.unique(selected_triangles, return_inverse=True)
    triangle_indices = triangle_indices.reshape(selected_triangles.shape)
    selected_vertices = vertices[selected_indices]
    # sphere_center = numpy.average(selected_vertices, axis=0)
    node_center = (selected_vertices.max(axis=0) + selected_vertices.min(axis=0)) * 0.5
    node_box = (selected_vertices.max(axis=0) - selected_vertices.min(axis=0)) * 0.5 + numpy.repeat(epsilon, 3)
    sphere_vertex_vectors = selected_vertices - node_center
    sphere_distances = numpy.linalg.norm(sphere_vertex_vectors, axis=-1)
    sphere_distances_sort = numpy.argsort(sphere_distances)
    sphere_radius = sphere_distances[sphere_distances_sort[-1]] + epsilon

    node = Data(type='Container', container=True, center=node_center, radius=sphere_radius, box=node_box, children=[], parent=parent, distribution=None, depth=0)

    if parent is not None:
        node.depth = parent.depth + 1

    if selected_triangle_indices.shape[0] <= 6:
        for selected_triangle_index in selected_triangle_indices:
            node.children.append(Data(type='Mesh.Triangle', container=False, index=selected_triangle_index, parent=node))
        node.max_depth = 1
        return node
    
    assert numpy.abs(sphere_distances[sphere_distances_sort[-1]]) >= epsilon
    distribution_vectors = [
        numpy.array([1, 0, 0], dtype=numpy.float32),
        numpy.array([0, 1, 0], dtype=numpy.float32),
        numpy.array([0, 0, 1], dtype=numpy.float32),
        sphere_vertex_vectors[sphere_distances_sort[-1]] / sphere_distances[sphere_distances_sort[-1]]
    ]
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
    distribution_vectors.append(numpy.cross(distribution_vectors[-2], distribution_vectors[-1]))
    del binormal, triangle_normal_average, triangle_normal_dir, triangle_normals

    distributions = []
    for distribution_vector in distribution_vectors:
        distributions.append(create_mesh_index_child_distribution(vertices, triangles, selected_triangle_indices, distribution_vector))

    selected_distribution = 0
    for i in range(1, len(distributions)):
        if distributions[i][0][2].shape[0] < distributions[selected_distribution][0][2].shape[0]:
            selected_distribution = i

    node.distribution = list(x.shape[0] for x in distributions[selected_distribution][0])
    node.split_normal = distributions[selected_distribution][1]
    node.split_factor = distributions[selected_distribution][2]

    if distributions[selected_distribution][0][0].shape[0] + distributions[selected_distribution][0][1].shape[0] < 2:
        for selected_triangle_index in selected_triangle_indices:
            node.children.append(Data(type='Mesh.Triangle', container=False, index=selected_triangle_index, parent=node))
        node.max_depth = 1
        return node

    for group in distributions[selected_distribution][0]:
        if group.shape[0] > 0:
            child = create_mesh_shpere_container_node(vertices, triangles, group, node)
            while child.container and len(child.children) == 1:
                child = child.children[0]
                child.parent = node
            assert child.parent is node
            node.children.append(child)

    node.max_depth = 1 + numpy.max(list(c.max_depth for c in node.children if c.container))

    return node

def append_vertices(triangles, storage, node):
    if node.type == 'Container':
        for child in node.children:
            append_vertices(triangles, storage, child)
        return
    if node.type == 'Mesh.Triangle':
        storage.extend(triangles[node.index][:,0])


NODE_TYPE_ROOT=0
NODE_TYPE_MESH=1
NODE_TYPE_CONTAINER=2
NODE_TYPE_MESH_TRIANGLE=3

MESH_FLAG_NORMALS=1
MESH_FLAG_TEX_COORDS=2
MESH_FLAG_TANGENT=4
MESH_FLAG_BITANGENT=8

CONTAINER_FLAG_BOX=1

def insert_node_in_storage(storage, node):
    float_ref = len(storage.float_data)
    int_ref = len(storage.int_data)
    ptr_ref = len(storage.ptr_data)
    node_ref = len(storage.tree_data)
    
    if node.type == 'Mesh':
        storage.ptr_data.append([NODE_TYPE_MESH, MESH_FLAG_NORMALS | MESH_FLAG_TEX_COORDS, int_ref, float_ref])
        storage.int_data.extend([
            -1,
            -1,
            node.vertices.shape[0],
            node.triangles.shape[0],
            node.normals.shape[0],
            node.tex_coords.shape[0],
        ])
        storage.float_data.extend(list(node.vertices.flatten()))
        storage.int_data.extend(list(node.triangles.flatten()))
        storage.float_data.extend(list(node.normals.flatten()))
        storage.float_data.extend(list(node.tex_coords.flatten()))
        storage.tree_data.append([-1, -1, -1, ptr_ref])
        node.tree_ref = node_ref
        if node.parent is not None:
            storage.tree_data[node_ref][0] = node.tree_ref
        index_node_ref = insert_node_in_storage(storage, node.index_node)
        storage.int_data[int_ref + 1] = index_node_ref
    
    elif node.type == 'Container':
        storage.ptr_data.append([NODE_TYPE_CONTAINER, -1, int_ref, float_ref])
        storage.int_data.extend([
            CONTAINER_FLAG_BOX
        ])
        storage.float_data.extend(list(node.center.flatten()))
        storage.float_data.append(node.radius)
        storage.float_data.extend(list(node.box.flatten()))
        storage.tree_data.append([-1, -1, -1, ptr_ref])
        node.tree_ref = node_ref
        if node.parent is not None:
            storage.tree_data[node_ref][0] = node.parent.tree_ref
        child_ref_list = []
        for child in node.children:
            child_ref_list.append(insert_node_in_storage(storage, child))
        if len(child_ref_list) > 0:
            storage.ptr_data[ptr_ref][1] = child_ref_list[0]
        for i in range(1, len(child_ref_list)):
            storage.tree_data[child_ref_list[i]][1] = child_ref_list[i - 1]
        for i in range(len(child_ref_list) - 1):
            storage.tree_data[child_ref_list[i]][2] = child_ref_list[i + 1]
    
    elif node.type == 'Mesh.Triangle':
        mesh_parent = node.parent
        while mesh_parent.type != 'Mesh':
            mesh_parent = mesh_parent.parent
        storage.ptr_data.append([NODE_TYPE_MESH_TRIANGLE, node.index, mesh_parent.tree_ref, -1])
        storage.tree_data.append([node.parent.tree_ref, -1, -1, ptr_ref])
        node.tree_ref = node_ref

    return node_ref

def make_flatten_leaves(storage, node):
    if hasattr(node, 'children') and len(node.children) > 0:
        for child in node.children:
            make_flatten_leaves(storage, child)
        return
    storage.append(node)


def _main():
    count_vertices = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_normals = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_texture_coords = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_faces = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')

    data_vertices = numpy.frombuffer(sys.stdin.buffer.read(count_vertices * 3 * 4), dtype=numpy.float32).reshape((count_vertices, 3))
    data_normals = numpy.frombuffer(sys.stdin.buffer.read(count_normals * 3 * 4), dtype=numpy.float32).reshape((count_normals, 3))
    data_texture_coords = numpy.frombuffer(sys.stdin.buffer.read(count_texture_coords * 2 * 4), dtype=numpy.float32).reshape((count_texture_coords, 2))
    data_faces = numpy.frombuffer(sys.stdin.buffer.read(count_faces * 3 * 3 * 4), dtype=numpy.uint32).reshape((count_faces, 3, 3))
    
    mesh_node = Data(type='Mesh', parent=None, vertices=data_vertices, normals=data_normals, tex_coords=data_texture_coords, triangles=data_faces, index_node=None, depth=0, max_depth=0)
    index_node = create_mesh_shpere_container_node(data_vertices, data_faces[:,:,0], numpy.arange(data_faces.shape[0]), mesh_node)
    mesh_node.index_node = index_node
    storage = Data(float_data=[], int_data=[], ptr_data=[], tree_data=[])

    start_index = insert_node_in_storage(storage, mesh_node)

    storage.float_data = numpy.array(storage.float_data, dtype=numpy.float32)
    storage.int_data = numpy.array(storage.int_data, dtype=numpy.int32)
    storage.ptr_data = numpy.array(storage.ptr_data, dtype=numpy.int32)
    storage.tree_data = numpy.array(storage.tree_data, dtype=numpy.int32)

    sys.stdout.buffer.write(start_index.to_bytes(4, 'little'))
    sys.stdout.buffer.write(storage.float_data.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(storage.int_data.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(storage.ptr_data.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(storage.tree_data.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(storage.float_data.tobytes('C'))
    sys.stdout.buffer.write(storage.int_data.tobytes('C'))
    sys.stdout.buffer.write(storage.ptr_data.tobytes('C'))
    sys.stdout.buffer.write(storage.tree_data.tobytes('C'))

if __name__ == '__main__':
    _main()
