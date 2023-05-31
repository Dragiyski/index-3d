import numpy
import itertools
from argparse import ArgumentParser
from pathlib import Path

class ParseError(RuntimeError):
    pass

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Wavefront:
    def __init__(self, file):
        self.file = str(file)
        self.object_map = {}
        self.data = {
            'v': [],
            'vt': [],
            'vn': [],
            'vp': []
        }
        self.vertex_map = {}
        self.__current_object = None
        with open(self.file, 'r') as handle:
            text = handle.read()
        lines = text.splitlines()
        self._parse_lines(lines)
    
    def _parse_lines(self, lines: list):
        line_continue = None
        line_length = len(lines)
        for line_index in range(line_length):
            line = lines[line_index].strip()
            if len(line) <= 0:
                continue
            if line.startswith('#'):
                continue
            if line_continue is not None:
                line = line_continue + line
            else:
                line_start = line_index
            if line.endswith('\\'):
                line_continue = line.rstrip('\\')
                continue
            items = line.split()
            if len(items) < 1:
                continue
            if items[0] in ['v', 'vn', 'vt', 'vp']:
                float_list = [float(x) for x in items[1:]]
                if items[0] == 'vn':
                    if len(float_list) != 3:
                        raise ParseError('[%s:%d]: Command "vn" expects 3 arguments' % (self.file, line_start))
                elif items[0] == 'v':
                    if (len(float_list) < 3 or len(float_list) > 4):
                        raise ParseError('[%s:%d]: Command "v" expects 3 or 4 arguments' % (self.file, line_start))
                elif items[0] == 'vt' or items[0] == 'vp':
                    if (len(float_list) < 1 or len(float_list) > 3):
                        raise ParseError('[%s:%d]: Command "%s" expects 1 to 3 arguments' % (self.file, line_start, items[0]))
                if items[0] == 'v':
                    float_list = float_list[0:3]
                elif items[0] == 'vt':
                    while len(float_list) < 2:
                        float_list.append(0.0)
                    float_list = float_list[0:2]
                elif items[0] == 'vp':
                    if len(float_list) < 2:
                        float_list.append(0.0)
                    if len(float_list) < 3:
                        float_list.append(1.0)
                self.data[items[0]].append(float_list)
            elif items[0] == 'o':
                name = line[2:].strip()
                self.__current_object = name
            elif items[0] == 'f':
                polygon = [[int(y) if len(y) > 0 else None for y in x.split('/')] + [None] * (3 - len(x.split('/'))) for x in items[1:]]
                if self.__current_object not in self.object_map:
                    current_object = self.object_map[self.__current_object] = Data(
                        has_texture=polygon[0][1] is not None,
                        has_normal=polygon[0][2] is not None,
                        vertices=[],
                        indices=[],
                        vertex_map={},
                        faces=[],
                        is_parsed=False
                    )
                else:
                    current_object = self.object_map[self.__current_object]
                current_object.faces.append(polygon)
                for i in range(len(polygon)):
                    if (current_object.has_texture and polygon[i][1] is None) or (current_object.has_normal and polygon[i][2] is None):
                        raise ParseError('[%s:%d]: Command "f" inconsistent v/vt/vn format' % (self.file, line_start))
                    for k in range(3):
                        if polygon[i][k] is None or polygon[i][k] == 0:
                            continue
                        if polygon[i][k] < 0:
                            polygon[i][k] = len(self.data[['v', 'vt', 'vn'][k]]) + polygon[i][k]
                            if polygon[i][k] < 0:
                                raise ParseError('[%s:%d]: Command "f" polygon backward reference out of bounds' % (self.file, line_start, items[0]))
                        else:
                            polygon[i][k] -= 1

    def _parse_object(self, target):
        if target.is_parsed:
            return
        for polygon in target.faces:
            for i in range(len(polygon)):
                key = '/'.join([str(x) if x is not None else '' for x in polygon[i]])
                if key not in self.vertex_map:
                    vertex_data = self.data['v'][polygon[i][0]]
                    if polygon[i][2] is not None:
                        vertex_data = vertex_data + self.data['vn'][polygon[i][2]]
                    if polygon[i][1] is not None:
                        vertex_data = vertex_data + self.data['vt'][polygon[i][1]]
                    self.vertex_map[key] = vertex_data
                if key not in target.vertex_map:
                    index = len(target.vertices)
                    target.vertices.append(self.vertex_map[key])
                    target.vertex_map[key] = index
            for i in range(1, len(polygon) - 1):
                triangle = [
                    polygon[0],
                    polygon[i],
                    polygon[i+1]
                ]
                keys = ['/'.join([str(x) if x is not None else '' for x in p]) for p in triangle]
                for k in range(3):
                    target.indices.append(target.vertex_map[keys[k]])

    def get_data(self, name):
        self._parse_object(self.object_map[name])
        return (self.object_map[name].vertices, self.object_map[name].indices)            


def main():
    parser = ArgumentParser(
        description="Reads a WaveFront Object file and generate vertex information",
        add_help=True
    )
    parser.add_argument('-i', '--input', required=True, type=Path, dest='input', help='Input file')
    parser.add_argument('-ov', '--output-vertices', required=True, type=Path, dest='output_vertices', help='Output file for the vertex buffer')
    parser.add_argument('-oi', '--output-indices', required=True, type=Path, dest='output_indices', help='Output file for the index buffer')
    parser.add_argument('--object', dest='object_name', help='Selects an object from the *.obj file')
    args = parser.parse_args()

    input_file = args.input.resolve()
    wave = Wavefront(input_file)
    vertex, index = wave.get_data(args.object_name)
    vertex = numpy.array(list(itertools.chain.from_iterable(vertex)), dtype=numpy.float32)
    if vertex.shape[0] < 256:
        index_type = numpy.uint8
    elif vertex.shape[0] < 65536:
        index_type = numpy.uint16
    else:
        index_type = numpy.uint32
    index = numpy.array(index, dtype=index_type)

    with open(args.output_vertices.resolve(), 'wb') as file:
        file.write(vertex.tobytes())
    
    with open(args.output_indices.resolve(), 'wb') as file:
        file.write(index.tobytes())

if __name__ == '__main__':
    main()