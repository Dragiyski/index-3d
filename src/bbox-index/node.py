import numpy
from .lib import Data

def compile_node(triangle_data, selection, parent = None):
    if len(selection) <= 0:
        return None
    triangle_list = triangle_data[selection]
    vertex_list = triangle_list.position.reshape(-1, 3)
    bounding_box = numpy.concatenate([
        vertex_list.min(axis = 0)[None],
        vertex_list.max(axis = 0)[None]
    ], axis = 0)
    print(f'Depth: {0 if parent is None else (parent.depth + 1)}, Input: {len(selection)}', file=sys.stderr)
    if len(selection) <= 6:
        return Data(
            is_container = False,
            bounding_box = bounding_box,
            index = selection,
            depth = 0 if parent is None else (parent.depth + 1),
            max_depth = 0,
            parent = parent
        )
    distribution_list = []
    for dim in range(3):
        min_vertex = triangle_list.position.min(axis=-2)[:, dim]
        max_vertex = triangle_list.position.max(axis=-2)[:, dim]
        min_affinity = (min_vertex - bounding_box[0, dim]) - (bounding_box[1, dim] - max_vertex)
        min_sort = numpy.argsort(min_affinity)
        groups = []
        groups.append(min_sort[0 : selection.shape[0] // 2])
        groups.append(min_sort[selection.shape[0] // 2 : ])
        min_isect = min_vertex[groups[1]].min()
        max_isect = max_vertex[groups[0]].max()
        if min_isect < max_isect:
            group_sort = [
                numpy.argsort(max_vertex[groups[0]])[::-1],
                numpy.argsort(min_vertex[groups[1]]),
            ]
            min_cut = min(len(groups[0]) // 2, numpy.count_nonzero(max_vertex[groups[0][group_sort[0]]] >= min_isect))
            max_cut = min(len(groups[1]) // 2, numpy.count_nonzero(min_vertex[groups[1][group_sort[1]]] <= max_isect))
            g2 = numpy.union1d(groups[0][group_sort[0][:min_cut]], groups[1][group_sort[1][:max_cut]])
            groups[0] = numpy.setdiff1d(groups[0], g2)
            groups[1] = numpy.setdiff1d(groups[1], g2)
            groups.append(g2)
            pass
        assert len(groups[0]) > 0
        assert len(groups[1]) > 0
        distribution_list.append(groups)
    # We split the triangles based on one-dimensional data for each dimensions
    # The best split are those that are more balanced, and with lower amount group[2] triangles
    rating = []
    for distribution in distribution_list:
        r = numpy.abs(distribution[1].shape[0] - distribution[0].shape[0])
        if len(distribution) > 2:
            r += len(distribution[2])
        rating.append(r)
    node = Data(
        is_container = True,
        bounding_box = bounding_box,
        depth = 0 if parent is None else (parent.depth + 1),
        parent = parent
    )
    for dim in numpy.argsort(rating):
        children = []
        count = []
        for distribution in distribution_list[dim]:
            child = compile_node(triangle_data, selection[distribution], node)
            if child is not None:
                children.append(child)
                count.append(len(distribution))
        if len(children) > 1:
            node.children = children
            node.distribution = count
            index = [child.index for child in children]
            while len(index) >= 2:
                index = [numpy.union1d(index[0], index[1])] + index[2:]
            node.index = index[0]
            node.max_depth = 1 + numpy.maximum.reduce(list(c.max_depth for c in node.children))
            return node
        
def validate_node(node, triangle_count):
    triangle_set = set()
    missing_set = set()
    duplicate_set = set()
    def validate_recursive(node):
        if node.is_container:
            for child in node.children:
                validate_recursive(child)
        else:
            for index in node.index:
                if not isinstance(index, numpy.integer):
                    raise ValueError(f'Expected the index to contain only integers, got {index}')
                if index < 0 or index >= triangle_count:
                    raise ValueError(f'Expected the index to be an integer in range [0, {triangle_count}), got {index}')
                if index in triangle_set:
                    duplicate_set.add(index)
                triangle_set.add(index)
        return True
    if not validate_recursive(node):
        return False
    for index in range(triangle_count):
        if index not in triangle_set:
            missing_set.add(index)
    if len(missing_set) > 0 or len(duplicate_set) > 0:
        message = 'Found inconsistance in the generated index:'
        if len(missing_set) > 0:
            message += '\nMissing:'
            for index in sorted(missing_set):
                message += f'\n{index}'
            message += '\n'
        if len(duplicate_set) > 0:
            message += '\nDuplicated:'
            for index in sorted(duplicate_set):
                message += f'\n{index}'
            message += '\n'
        raise RuntimeError(message)