import sys
import numpy

def _main():
    count_vertices = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_normals = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_texture_coords = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')
    count_faces = int.from_bytes(sys.stdin.buffer.read(4), byteorder='little')

    data_vertices = numpy.frombuffer(sys.stdin.buffer.read(count_vertices * 3 * 4), dtype=numpy.float32).reshape((count_vertices, 3))
    data_normals = numpy.frombuffer(sys.stdin.buffer.read(count_normals * 3 * 4), dtype=numpy.float32).reshape((count_normals, 3))
    data_texture_coords = numpy.frombuffer(sys.stdin.buffer.read(count_texture_coords * 2 * 4), dtype=numpy.float32).reshape((count_texture_coords, 2))
    data_faces = numpy.frombuffer(sys.stdin.buffer.read(count_faces * 3 * 3 * 4), dtype=numpy.uint32).reshape((count_faces, 3, 3))

    triangle_indices = data_faces[:,:,0]
    triangle_vertices = data_vertices[triangle_indices]
    pass

if __name__ == '__main__':
    _main()    