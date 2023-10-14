wavefront_commands = {
    'v': 'parse_position',
    'vn': 'parse_normal',
    'vt': 'parse_texcoord',
    'f': 'parse_face'
}

class WaveFront:
    def __init__(self, objfile: str):
        from pathlib import Path
        objfile = str((Path('.') / Path(str(objfile))).resolve())
        self.__stack = []
        self.__position = []
        self.__normal = []
        self.__texcoord = []
        self.__face_index_ref = [self.__position, self.__texcoord, self.__normal]
        self.__triangles = []
        self.__face_count = 0
        self.__face_has_normal = None
        self.__face_has_texcoord = None
        try:
            self._parse_object_file(objfile)
        except Exception as e:
            if len(e.args) > 0:
                e.args = ('%s:%d: %s' % (self.__stack[-1][0], self.__stack[-1][1], e.args[0]), *e.args[1:])
            else:
                e.args = ('%s:%d' % (self.__stack[-1][0], self.__stack[-1][1]))
            raise
        del self.__stack
        import numpy
        self.position = numpy.array(self.__position, dtype=numpy.float32)
        del self.__position
        self.normal = numpy.array(self.__normal, dtype=numpy.float32)
        del self.__normal
        self.texcoord = numpy.array(self.__texcoord, dtype=numpy.float32)
        del self.__texcoord

        # TODO: Modify floatlist library
        # TODO: C will have a class (not function): FloatList
        # TODO: More than one array can be added to the FloatList
        # TODO: A separate method to generate index from the values found in the FloatList
        # TODO: FloatList will wrap std::set
        # TODO: 

    def _parse_object_file(self, path: str):
        self.__stack.append([path, 0])
        with open(path, 'r') as file:
            for line in file:
                self.__stack[-1][1] += 1
                line = line.strip()
                try:
                    hash_index = line.index('#')
                except ValueError:
                    hash_index = -1
                if hash_index >= 0:
                    line = line[0:hash_index].strip()
                if len(line) <= 0:
                    continue
                line = line.split()
                if line[0] not in wavefront_commands:
                    continue
                getattr(self, wavefront_commands[line[0]])(*line[1:])

    def parse_position(self, x, y, z, w = 1.0):
        from numpy import float32
        self.__position.append([float32(value) for value in [x, y, z]])

    def parse_normal(self, x, y, z):
        from numpy import float32
        self.__normal.append([float32(value) for value in [x, y, z]])

    def parse_texcoord(self, *args):
        if len(args) < 1 or len(args) > 3:
            raise TypeError('vt requires 1-3 arguments')
        if len(args) != 2:
            raise RuntimeError('vt only supported with 2 arguments')
        from numpy import float32
        self.__texcoord.append([float32(value) for value in args])

    def parse_face_arg_first(self, position, texcoord = None, normal = None):
        from numpy import float32
        position = int(position)
        texcoord = int(texcoord) if texcoord is not None and len(texcoord) > 0 else None
        normal = int(normal) if normal is not None and len(normal) > 0 else None
        if texcoord is not None:
            self.__face_has_texcoord = True
        if normal is not None:
            self.__face_has_normal = True
        self.parse_face_arg = self.parse_face_arg_with_consistency_check
        return self.parse_face_arg(self, position, texcoord, normal)
    
    def parse_face_arg_with_consistency_check(self, position, texcoord = None, normal = None):
        indices = [position]
        if self.__face_has_texcoord:
            if texcoord is None or len(texcoord) <= 0:
                raise ValueError('Incosistent face indices: texture coordinate indices present only in some faces')
            texcoord = int(texcoord)
        position = int(position)
        texcoord = int(texcoord) if texcoord is not None and len(texcoord) > 0 else None
        normal = int(normal) if normal is not None and len(normal) > 0 else None
        if self.__face_has_texcoord and texcoord is None or not self.__face_has_texcoord and texcoord is not None:
            raise ValueError('Incosistent face indices: texture coordinate indices present only in some faces')
        if self.__face_has_normal and normal is None or not self.__face_has_normal and normal is not None:
            raise ValueError('Incosistent face indices: normal indices present only in some face')
        indices = [position, texcoord, normal]
        for i in range(3):
            if

    def parse_face(self, *args):
        arg_index = -1
        face = []
        for arg in args:
            arg_index += 1
            indices = 
            index_ref = arg.split('/')
            if len(index_ref) != 3:
                raise RuntimeError('face argument not in format <int>/[int]/[int]')
            for index in index_ref:
                index_index += 1
                if len(index) <= 0:
                    indices.append(0)
                    continue
                index = int(index)
                if index < 0:
                    index = self.__face_count + index
                if index <= 0 or index > len(self.__face_index_ref[index_index]):
                    raise IndexError('face[%d][%d][%d] refer to invalid index reference: %d' % (self.__face_count, arg_index, index_index, index))
                indices.append(index)
            face.append(indices)
        if len(face) < 3:
            raise RuntimeError('f command requires at least 3 arguments')
        for index in range(1, len(face) - 1):
            self.__triangles.append([
                face[0],
                face[index],
                face[index + 1]
            ])
        self.__face_count += 1


def _main():
    from argparse import ArgumentParser
    import sys

    argument_parser = ArgumentParser(
        description="Reads an WaveFront Object file and produce binary format suitable for further processing, especially in shaders.",
    )
    argument_parser.add_argument('objfile', help='Wavefront Object file to parse')
    args = argument_parser.parse_args()
    wavefront = WaveFront(args.objfile)
    pass

if __name__ == '__main__':
    _main()