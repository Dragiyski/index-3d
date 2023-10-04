from collections.abc import Callable, Iterable, Mapping
import sys
from typing import Any
import numpy
import struct
from threading import Thread
from time import sleep

numpy.set_printoptions(floatmode = 'maxprec', suppress = True)
epsilon = numpy.finfo(numpy.float32).eps

total_face_count = 0
finished_faces = set()

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def normalize(x):
    y = x.reshape((-1, x.shape[-1]))
    return (y / numpy.linalg.norm(y, axis = -1)[:, None]).reshape(x.shape)

def dot(x, y):
    return numpy.sum(x * y, axis = -1)

class ProgressThread(Thread):
    def __init__(self, total_count, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__complete = False
        self.__total = total_count
        self.__current = 0

    def run(self) -> None:
        while not self.__complete:
            self.show_progress()
            sleep(1.0)
    
    def complete(self):
        self.__complete = True
        self.show_progress()

    def increment(self, value):
        self.__current += value

    def decrement(self, value):
        self.__current -= value
    
    def show_progress(self):
        print(f'Processed: {self.__current} / {self.__total} triangles', file=sys.stderr)

progress_thread = None

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
    global progress_thread
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

    print(f'Creating index from {count_faces} triangles...', file=sys.stderr)

    progress_thread = ProgressThread(count_faces, daemon=True)
    progress_thread.start()

    root_node = compile_node(triangles, numpy.arange(triangles.shape[0]))
    progress_thread.complete()
    data_tree = TreeStorage(root_node)
    # WARNING: We get less triangles than the number of triangles originally in the model
    validate_node(root_node, triangles.shape[0])
    sys.stdout.buffer.write(b''.join(data_tree.byte_list))

if __name__ == '__main__':
    _main()
