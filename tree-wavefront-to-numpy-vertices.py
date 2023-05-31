import pywavefront
import numpy

vertices = numpy.array(pywavefront.Wavefront('tree.obj', parse=True, cache=False, collect_faces=False).vertices, dtype=numpy.float64)
numpy.save('tree-vertex.npy', vertices, allow_pickle=False)
