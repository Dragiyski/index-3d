import sys
import numpy
import itertools
import matplotlib.pyplot as plt

numpy.set_printoptions(floatmode = 'maxprec', suppress = True)
epsilon = numpy.finfo(numpy.float32).eps

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
    triangles = numpy.core.records.fromarrays([
        data_vertices[data_triangles[:, :, 0]],
        data_normals[data_triangles[:, :, 1]],
        data_texture_coords[data_triangles[:, :, 2]],
    ], dtype = vertex_attributes)
    triangle_minmax = numpy.concatenate([
        triangles.position.min(axis = 1)[:, :, None],
        triangles.position.max(axis = 1)[:, :, None],
    ], axis=-1)
    triangle_median = numpy.median(triangles.position.reshape((-1, 3)), axis=0)
    triangle_mean = numpy.mean(triangles.position.reshape((-1, 3)), axis=0)
    triangle_sort_index = numpy.argsort(triangle_minmax, axis = 0)
    min_position = triangles.position.reshape((-1, 3)).min(axis=0)
    max_position = triangles.position.reshape((-1, 3)).max(axis=0)
    mid_position = 0.5 * (min_position + max_position)
    for dim in range(3):
        axis_domain_shape = triangle_minmax[:, dim].shape
        axis_domain_values = triangle_minmax[:, dim].ravel()
        axis_domain_sort_index = numpy.argsort(axis_domain_values)
        axis_domain_unsort_index = numpy.argsort(axis_domain_sort_index).reshape(axis_domain_shape)
        axis_position, axis_index = numpy.unique(axis_domain_values[axis_domain_sort_index], return_inverse=True)
        axis_value_add_index, axis_value_add = numpy.unique(axis_index[axis_domain_unsort_index][:, 0], return_counts=True)
        axis_value_sub_index, axis_value_sub = numpy.unique(axis_index[axis_domain_unsort_index][:, 1], return_counts=True)
        axis_value = numpy.repeat(0, axis_position.shape[0])
        axis_value[axis_value_add_index] += axis_value_add
        axis_value[axis_value_sub_index] -= axis_value_sub
        axis_value_sum = numpy.cumsum(axis_value)
        plt.plot(axis_position, axis_value_sum, label=['x', 'y', 'z'][dim], color=['r', 'g', 'b'][dim])
        plt.axvline(triangle_median[dim], color=['r', 'g', 'b'][dim], linestyle='dashed')
        plt.axvline(triangle_mean[dim], color=['r', 'g', 'b'][dim], linestyle='dotted')
        plt.axvline(mid_position[dim], color=['r', 'g', 'b'][dim], linestyle='dashdot')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _main()