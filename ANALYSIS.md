# Analysis of 3D indexing techniques

# Unalinged geons

Testing showed that unaligned geons (ovals, boxes) are not only difficult to compute, but the current attempts on location a proper rotation for the geons failed. Particularly, test on the following shapes:

* Wooden table (550 triangles)
* Locomotive (~2 million triangles)

Showed an optimal direction to be aligned with the axes, as models are aligned with the axes (not in world coordinates).

However, none of the methods produced a vector aligned with the axes.

Tested:

* Using the longest axis from the center;
* Using the longest axis with projection onto the longest axis plane and then repeating until all 3D are exhausted;
* Using the mean or median of the normalized normal vector of the plane of the triangles;

Conclusion: Meshes should be indexed by axis-aligned shapes.

# Splitting planes and spheres

Triangles can be separated in groups by a splitting plane. Then each side can decide on which triangles to take and compute a sphere. This method had promising results, getting the table rendered on 1280x720 in 5ms on 1070. And getting the locomotive rendered in 1280x720 in 35ms on 1070, indicating a proper O(P*log(N)) time of rendering N triangles in P pixels.

# Axis aligned bounding boxes

The above idea is good, but it leads to too many false positive intersection matches as the volume of each sphere can be several time more than the minimum unaligned bounding box or even axes aligned bounding box. Moreover, children spheres are not necessarily exclusive.

Next idea is making axis aligned bounding box by:

* Have a root node creating a bounding box by using the `min` and `max` of each triangle vertex coordinate.
* Split the box on one of the three axes somewhere, so that the number of triangles in each child is roughly the same.

Since each triangle have exactly 3 vertices, having equal amount of vertices (i.e. using median) seems like a good choice.

Problem: some triangles intersects the splitting plane

## Splitting the triangles

If a triangle split the splitting plane and we do not need to keep the original data, we can create additional rectangles that can be assigned to each side of the splitting plane, this have the following cases:

* A triangle is touching the plane, but do not cross;
* A triangle is crossing the plane;
* A triangle is lying on the plane;

Since we generate new triangles, the sum number of triangles in node's children can exceed number of triangles in a node. Also the new vertices should have computed vertex attributes.

In certain situation children generate too mane new triangles and never finish, reaching infinite depth on the stack trace (and crushing the indexer).

Conclusion: Not a viable idea, but some aspects might be useful.

## External triangles

If a raytracer guarantees that it will never match something outside the container node, we can assign the triangles that cross the splitting plane to both groups. This create the following situation:

* Some triangle vertices might lie outside the bounding box.
* All triangle vertices might lie outside the bounding box and no edge crosses the bounding box, but still a part of the triangle can be inside the bounding box. When this happen, a box edge will cross the triangle.

To locate a proper location for splitting using mean/median, we can generate new vertices for the location where a triangle crosses the bounding box or the bounding box crosses a triangle. Combined with the vertices inside the bounding box, this gives proper splitting.

Problem: However, the majority of vertices can be on the bounding box side, resulting in median equal to the box side, thus reducing the volume of the bounding box in one of its children to zero (0).

Partial fix: We notify the parent when the child node triangle list matches the parent (i.e. the child triangles are not proper subset of the parent's triangle). If that happen, we attempt splitting in another axis. If no axis existis, we assign all triangles to the current node (i.e. becoming a leaf node).

Problem: The fix generate nodes with many triangles (13 when using mean, 17 when using medium)

## Using extended boxes instead of extended triangles

As mentioned above the raytracer can impose limits on the depth of the ray.

For example, the raytrace can intersect a bounding box at depth 17.5 and 21.3. Even if the bounding box of a child is larger than the parent, we can only consider intersections within the axis aligned bounding box determined by the points at depth 17.5 and 21.3 or the ray.

This allow optimization where a child container volume can be bigger than the parent container volume.

This gives the following idea:

* Make a splitting plane of the triangles in the node and:
* Assign each triangle fully below or above to the desired group.
* Add all triangles that intersect the splitting plane to both groups.
* Each child compute the bounding box (that can be bigger than the parent bounding box) using all triangles vertices min/max values.
* The node must store the split_index and split_factor.

We might have a way to reject children with too few triangles or too small volume. In such case all three splitting directions must be tried.

Every child compute the triangles included in its bounding box for the set of all triangles. Then it *updates* its bounding box (rather than allowing triangles outside the bounding box). However, the node must guarantee the set of triangles is a subset of the parent set of triangles.

### Example

Let node at level 0 have 550 triangles (wooden table).

#### Splitting plane

A preferred splitting plane will be the one with lowest amount of intersecting triangles. However, if it leads to bad results, other axis might be attempted.

After that a splitting factor is found using the median, so that the number of vertices/triangles is balanced.

---

Two children are created each containing a bounding box determined by the splitting plane.

All triangles that are even partially present of the bounding box are found for each children. Let's say both have 302 triangles with 54 triangles in both group.

A new expanded bounding box is created for each child that encompass the entire triangles, so no triangle crosses the new bounding box.

This will allow the median to work properly, however, there will be a potential problem with the fact that if the expanded bounding box is split again, it might create a bounding box that is entirely outside the parent's bounding box.

TODO: Attempt the above and examine the problematic situation;