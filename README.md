# index-3d

Various indexing techniques for 3D data (experimentation).

Currently active experiments:

* `wavefront-reader.py`
  * stdin = wavefront object file with following properties:
    * Y is front;
    * Z is up;
    * All faces must be triangles;
    * Currently, index is build by all faces (no way of filtering by object);
  * stdout = binary containing
    * `uint32`: vertex count
    * `uint32`: normal count
    * `uint32`: texture coordinates count
    * `uint32`: faces count
    * `float32[vertex_count][3]`: array of vertex coordinates
    * `float32[normal_count][3]`: array of normal vectors
    * `float32[texture_coords_count][2]`: array of texture coordinates
    * `uint32[face_count][3][3]`: array of triangles each containing index within vertex, normal, texture coordinate array (in that order)
    * `uint32`: object count
    * `uint32[object_count][2]`: length of object name, then number of indices in object
    * `char[object_count][object_name_length]`: concatenated names of all objects
    * `uint32[object_count][object_index_count]`: indices per object
    * `uint32`: group count
    * `uint32[group_count][2]`: length of group name, then number of indices in group
    * `char[group_count][object_name_length]`: concatenated names of all groups
    * `uint32[group_count][object_index_count]`: indices per group
* `gen-index.py`
  * stdin = binary from `wavefront-reader.py`
  * stdout = TBD

The data will create 4 buffers: `float_data`, `int_data`, `primitive_data`, `tree_data`

* `primitive_data` - contains a fixed amount of data per primitive:
  * It must contain an index to `float_data` and `int_data`
* `tree_data` - contains a fixed amount of data per node:
  * It must contain index to `parent`, `next_sibling`
  * May contain index to `prev_sibling`
  * It must contain index to `primitive_data`
  * Children are not stored in the `tree_data`. If an `primitive_data` identifies a container, it must contain in its `int_data` a `children_count`,
  * followed by `children_count` references to `tree_data`;
* `int_data` - contains variable amount of data per primitive, it must containt (at the start):
  * `type_id` - a way to identify the object
  * `flags` - additional data about the object varying per `type_id` 
  * May contain any other integer data. The total amount of data must be able to be determined by `type_id`, `flags` and any other integer on known index.
* `float_data` - contains variable amount of data per primitive for some primitivies
  * If `primitive_data` index is -1, primitive might not contain any float data.
  * May contain any float data. The total amount of data must be able to be determined by `type_id`, `flags` and any other integer on known index within `int_data`.

The `tree_data` must allow DFS (depth-first search) order of iterating through the tree. The `current_node` is a reference to a `tree_data`. The `tree_data` contains reference to `primitive_data`.

To do DFS raytracing perform the following steps:
1. Set `final_raytrace_result` to `-1`
2. While `true`:
   1. Let `raytrace_result` be the result of `raytrace(current_node)`.
   2. If `raytrace_result` is positive and passess the depth check:
      1. If the `current_node.primitive.int.flags.container`:
         1. Set `current_node` to `current_node.primitive.int.first_child`
         2. continue;
      2. Else:
         1. Replace `final_raytrace_result` with `current_node` index;
         2. Go to the next node;
   3. Else:
      1. Go to the next node;

To go to the next node:
1. While `current_node.next_sibling` is `-1`
   1. If `current_node.parent_id` is `-1`
      1. Break the loop and return `final_raytrace_result`
   2. Else:
      1. Set `current_node` to `current_node.parent_id`
2. Set `current_node` to `current_node.next_sibling`

## Problems

1. Boxes are more versatile, but slower to intersect than ovals.
   * We need to find a way to minimize the container volumes, we can use ovals where a sphere is converted to oval using `mat4` transformation matrix. Intersections can only be done if `mat4` is inversible. Computing the inverse is slower, and must be precomputed.
   * How to minimize the volume? 
     * We can start with the longest axis and derive a plane from it.
     * Project all vertices onto that plane.
     * The projection longest axis is a perpendicular axis to the plane;
     * Repeating we can find 3 axes;
     * Defining an oval by scaling axis `1.0` to the three axis (non-uniform scale) and translate the origin to the center of the sphere.
     * Inverse the above transformation.
     * For each point outside the oval, we might transform it and compute how much longer the projected axes needs to be.
     * Check if the oval volume is lower than the sphere volume;
2. Using boxes:
   * Repeatedly find the longest axis and project onto the plane it defines, do this 3 times for 3 perpendicular axis (the last one can be derived by cross product);
   * We have center point and 3 axes, a box can be made with that coordinate system;
   * Find `mat4` transforming `-1` to `+1` box into that box;
   * Inverse the above transformation.
   * Raytracing can be done into the box space, on all planes at once;
   * For a container, this must provide only a boolean result: if `false` discard the container subtree quickly.

Using box primitive can still use original sphere index. The splitting in the above index is defined by the splitting plane, not the sphere. The sphere only defines boundaries (beyond which no vertex would exist). Thus, the already working index method can be updated by changing the container nodes, but preserving the tree structure.

# TODO

How to format the above data for WebGPU? WebGPU accept storage buffers, perhaps readonly storage buffer?

Incremental testing:

1. Test screen texture for the camera computation in WebGPU;
2. Test compute raytracing with displaying `normal` coordinates as colors:
   1. The computation of a frame in WebGPU can handle `async` tasks. Only when a frame is done, we can call `requestAnimationFrame` to present that frame, and then begin the new frame;
3. Test compute raytracing with a light source and basic lighting;
4. Add a shadow tracing for shadows between primitives to 3;
5. Add a color texture (how to handle textures from different objects)? Perhaps using index reference to the object within `tree_data` and store that into a texture/buffer for the pixel. Once we know what object a pixel displays, we can access its properties. Additionally, we can increase an atomic value and write the pixel coordinates to a screen-size buffer. A second pass can happen, but with less jobs, depending on the count of the atomic describing the size of the subset of the screen-sized buffer that is relevant. The buffer then executes per pixel to find a texture color.
6. Other stuff: reflection, refraction, diffuse illumination (for any object close by) - executing raytracing with limited minimum and maximum value different from `0` and `+Inf`;
7. Partial MSAA: Store total depth of pixel; Then run a pass that reads the total depth and total depth of adjacent pixels; Mark pixels that requires additional processing as described in (5). For the resulting pixel repeat all steps of raytracing (except the partial MSAA) for rays defined from dividing the pixel width and height to N parts for antialiasing xN.

Evaluating performance: cannot be done by default with WebGPU. However, chrome flag allow access to device features required for measuring performance.