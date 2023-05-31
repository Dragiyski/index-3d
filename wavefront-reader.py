import sys
import numpy
from argparse import ArgumentParser
from pathlib import Path

def _main():
    data_vertices = []
    data_normals = []
    data_textures = []
    data_faces = []
    objects = dict()
    groups = dict()
    active_groups = set()
    active_object = None
    for line in sys.stdin:
        line = line.strip()
        if len(line) <= 0 or line.startswith('#'):
            continue
        try:
            first_hash = line.index('#')
        except ValueError:
            first_hash = -1
        if first_hash >= 0:
            line = line[0:first_hash].strip()
        args = line.split()
        if args[0] == 'g':
            active_groups = set(args[1:])
        elif args[0] == 'o':
            active_object = args[1]
        elif args[0] == 'v':
            if len(args) != 4:
                raise RuntimeError('Vertices must contain 3D coordinates')
            data_vertices.append([float(x) for x in args[1:]])
        elif args[0] == 'vn':
            if len(args) != 4:
                raise RuntimeError('Normals must contain 3D coordinates')
            data_normals.append([float(x) for x in args[1:]])
        elif args[0] == 'vt':
            if len(args) != 3:
                raise RuntimeError('Textures must contain 2D coordinates')
            data_textures.append([float(x) for x in args[1:]])
        elif args[0] == 'f':
            if len(args) != 4:
                raise RuntimeError('Wavefront object must be triangulated')
            face = [[int(coord) - 1 for coord in indices.split('/')] for indices in args[1:]]
            face_index = len(data_faces)
            data_faces.append(face)
            if active_object is not None:
                if active_object not in objects:
                    objects[active_object] = []
                objects[active_object].append(face_index)
            for active_group in active_groups:
                if active_group not in groups:
                    groups[active_group] = []
                groups[active_group].append(face_index)
        
    data_vertices = numpy.array(data_vertices, dtype=numpy.float32)
    data_normals = numpy.array(data_normals, dtype=numpy.float32)
    data_textures = numpy.array(data_textures, dtype=numpy.float32)
    data_faces = numpy.array(data_faces, dtype=numpy.uint32)[:,:,[0,2,1]]

    sys.stdout.buffer.write(data_vertices.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(data_normals.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(data_textures.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(data_faces.shape[0].to_bytes(4, 'little'))
    sys.stdout.buffer.write(data_vertices.tobytes('C'))
    sys.stdout.buffer.write(data_normals.tobytes('C'))
    sys.stdout.buffer.write(data_textures.tobytes('C'))
    sys.stdout.buffer.write(data_faces.tobytes('C'))

    sys.stdout.buffer.write(len(objects.keys()).to_bytes(4, 'little'))
    for name in objects.keys():
        sys.stdout.buffer.write(len(name.encode('utf-8')).to_bytes(4, 'little'))
        sys.stdout.buffer.write(len(objects[name]).to_bytes(4, 'little'))
    for name in objects.keys():
        sys.stdout.buffer.write(name.encode('utf-8'))
    for name in objects.keys():
        sys.stdout.buffer.write(numpy.array(objects[name]).tobytes('C'))

    sys.stdout.buffer.write(len(groups.keys()).to_bytes(4, 'little'))
    for name in groups.keys():
        sys.stdout.buffer.write(len(name.encode('utf-8')).to_bytes(4, 'little'))
        sys.stdout.buffer.write(len(groups[name]).to_bytes(4, 'little'))
    for name in groups.keys():
        sys.stdout.buffer.write(name.encode('utf-8'))
    for name in groups.keys():
        sys.stdout.buffer.write(numpy.array(groups[name]).tobytes('C'))

if __name__ == '__main__':
    _main()
