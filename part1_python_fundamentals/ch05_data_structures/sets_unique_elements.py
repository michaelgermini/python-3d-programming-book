#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Sets for Unique Elements Example

This example demonstrates how to use Python sets for managing unique elements,
efficient set operations, and data deduplication in 3D graphics applications.
"""

import math

def demonstrate_basic_sets():
    """Demonstrate basic set operations with 3D data"""
    print("=== Basic Set Operations ===\n")
    
    # 1. Creating sets
    print("1. Creating Sets:")
    
    # Set of unique vertex indices
    vertex_indices = {0, 1, 2, 3, 4, 5, 6, 7}
    print(f"   Vertex indices: {vertex_indices}")
    print(f"   Number of vertices: {len(vertex_indices)}")
    
    # Set of unique object types
    object_types = {"cube", "sphere", "cylinder", "light", "camera"}
    print(f"   Object types: {object_types}")
    
    # Set from list (removes duplicates)
    duplicate_vertices = [0, 1, 2, 1, 3, 2, 4, 1, 5]
    unique_vertices = set(duplicate_vertices)
    print(f"   Original list: {duplicate_vertices}")
    print(f"   Unique vertices: {unique_vertices}")
    print(f"   Removed {len(duplicate_vertices) - len(unique_vertices)} duplicates")
    
    # 2. Set operations
    print("\n2. Set Operations:")
    
    # Add elements
    vertex_indices.add(8)
    vertex_indices.add(9)
    print(f"   After adding 8, 9: {vertex_indices}")
    
    # Remove elements
    vertex_indices.remove(0)
    print(f"   After removing 0: {vertex_indices}")
    
    # Discard (safe remove)
    vertex_indices.discard(10)  # Doesn't exist, no error
    print(f"   After discarding 10: {vertex_indices}")
    
    # Pop random element
    popped = vertex_indices.pop()
    print(f"   Popped element: {popped}")
    print(f"   Remaining: {vertex_indices}")
    
    # 3. Set membership
    print("\n3. Set Membership:")
    
    print(f"   Is 1 in vertex_indices? {1 in vertex_indices}")
    print(f"   Is 100 in vertex_indices? {100 in vertex_indices}")
    print(f"   Is 'cube' in object_types? {'cube' in object_types}")
    
    # Check if subset
    small_set = {1, 2}
    print(f"   Is {small_set} subset of {vertex_indices}? {small_set.issubset(vertex_indices)}")

def demonstrate_set_operations():
    """Demonstrate set operations and mathematical operations"""
    print("\n=== Set Operations ===\n")
    
    # 1. Union operations
    print("1. Union Operations:")
    
    # Two sets of vertices
    vertices_a = {0, 1, 2, 3, 4}
    vertices_b = {3, 4, 5, 6, 7}
    
    print(f"   Vertices A: {vertices_a}")
    print(f"   Vertices B: {vertices_b}")
    
    # Union
    union_vertices = vertices_a | vertices_b
    print(f"   Union (A | B): {union_vertices}")
    
    # Union using method
    union_method = vertices_a.union(vertices_b)
    print(f"   Union method: {union_method}")
    
    # 2. Intersection operations
    print("\n2. Intersection Operations:")
    
    # Intersection
    intersection = vertices_a & vertices_b
    print(f"   Intersection (A & B): {intersection}")
    
    # Intersection using method
    intersection_method = vertices_a.intersection(vertices_b)
    print(f"   Intersection method: {intersection_method}")
    
    # 3. Difference operations
    print("\n3. Difference Operations:")
    
    # Difference
    difference_a_b = vertices_a - vertices_b
    difference_b_a = vertices_b - vertices_a
    
    print(f"   Difference (A - B): {difference_a_b}")
    print(f"   Difference (B - A): {difference_b_a}")
    
    # Symmetric difference (elements in either set, but not both)
    symmetric_diff = vertices_a ^ vertices_b
    print(f"   Symmetric difference (A ^ B): {symmetric_diff}")
    
    # 4. Set comparisons
    print("\n4. Set Comparisons:")
    
    subset = {1, 2}
    superset = {0, 1, 2, 3, 4, 5}
    
    print(f"   Is {subset} subset of {superset}? {subset.issubset(superset)}")
    print(f"   Is {superset} superset of {subset}? {superset.issuperset(subset)}")
    print(f"   Are sets disjoint? {subset.isdisjoint({5, 6, 7})}")

def demonstrate_3d_geometry_sets():
    """Demonstrate sets in 3D geometry applications"""
    print("\n=== 3D Geometry Sets ===\n")
    
    # 1. Unique vertex management
    print("1. Unique Vertex Management:")
    
    # Multiple meshes sharing vertices
    mesh1_vertices = {0, 1, 2, 3, 4, 5}
    mesh2_vertices = {2, 3, 4, 5, 6, 7}
    mesh3_vertices = {4, 5, 6, 7, 8, 9}
    
    # Find all unique vertices across all meshes
    all_vertices = mesh1_vertices | mesh2_vertices | mesh3_vertices
    print(f"   Mesh 1 vertices: {mesh1_vertices}")
    print(f"   Mesh 2 vertices: {mesh2_vertices}")
    print(f"   Mesh 3 vertices: {mesh3_vertices}")
    print(f"   All unique vertices: {all_vertices}")
    print(f"   Total unique vertices: {len(all_vertices)}")
    
    # 2. Shared vertices between meshes
    print("\n2. Shared Vertices:")
    
    # Vertices shared between mesh1 and mesh2
    shared_1_2 = mesh1_vertices & mesh2_vertices
    print(f"   Shared between mesh1 and mesh2: {shared_1_2}")
    
    # Vertices shared between all three meshes
    shared_all = mesh1_vertices & mesh2_vertices & mesh3_vertices
    print(f"   Shared between all meshes: {shared_all}")
    
    # 3. Unique face indices
    print("\n3. Unique Face Indices:")
    
    # Face indices (may have duplicates)
    face_indices = [
        [0, 1, 2], [1, 2, 3], [2, 3, 4],  # Mesh 1 faces
        [2, 3, 4], [3, 4, 5], [4, 5, 6],  # Mesh 2 faces
        [4, 5, 6], [5, 6, 7], [6, 7, 8]   # Mesh 3 faces
    ]
    
    # Flatten and find unique indices
    all_face_indices = set()
    for face in face_indices:
        all_face_indices.update(face)
    
    print(f"   All face indices: {all_face_indices}")
    print(f"   Unique indices used in faces: {len(all_face_indices)}")

def demonstrate_object_management():
    """Demonstrate object management with sets"""
    print("\n=== Object Management ===\n")
    
    # 1. Object type tracking
    print("1. Object Type Tracking:")
    
    # Objects in scene
    scene_objects = {
        "cube1", "sphere1", "cylinder1", "light1", "camera1",
        "cube2", "sphere2", "light2", "cube3"
    }
    
    # Categorize objects by type
    geometry_objects = {obj for obj in scene_objects if any(geom in obj for geom in ["cube", "sphere", "cylinder"])}
    light_objects = {obj for obj in scene_objects if "light" in obj}
    camera_objects = {obj for obj in scene_objects if "camera" in obj}
    
    print(f"   All scene objects: {scene_objects}")
    print(f"   Geometry objects: {geometry_objects}")
    print(f"   Light objects: {light_objects}")
    print(f"   Camera objects: {camera_objects}")
    
    # 2. Object visibility tracking
    print("\n2. Object Visibility Tracking:")
    
    # Objects that are visible
    visible_objects = {"cube1", "sphere1", "light1", "camera1"}
    
    # Objects that are selected
    selected_objects = {"cube1", "sphere2"}
    
    # Objects that are both visible and selected
    visible_selected = visible_objects & selected_objects
    print(f"   Visible objects: {visible_objects}")
    print(f"   Selected objects: {selected_objects}")
    print(f"   Visible and selected: {visible_selected}")
    
    # Objects that are visible but not selected
    visible_not_selected = visible_objects - selected_objects
    print(f"   Visible but not selected: {visible_not_selected}")
    
    # 3. Object layer management
    print("\n3. Object Layer Management:")
    
    # Objects in different layers
    layer_0 = {"cube1", "sphere1", "light1"}
    layer_1 = {"sphere1", "cylinder1", "light2"}
    layer_2 = {"cube2", "camera1"}
    
    # Objects in multiple layers
    objects_in_multiple_layers = (layer_0 & layer_1) | (layer_0 & layer_2) | (layer_1 & layer_2)
    print(f"   Layer 0: {layer_0}")
    print(f"   Layer 1: {layer_1}")
    print(f"   Layer 2: {layer_2}")
    print(f"   Objects in multiple layers: {objects_in_multiple_layers}")

def demonstrate_spatial_indexing():
    """Demonstrate spatial indexing with sets"""
    print("\n=== Spatial Indexing ===\n")
    
    # 1. Grid-based spatial indexing
    print("1. Grid-based Spatial Indexing:")
    
    # Objects in different grid cells
    grid_cell_0_0 = {"cube1", "sphere1"}
    grid_cell_0_1 = {"sphere1", "light1"}
    grid_cell_1_0 = {"cube2", "cylinder1"}
    grid_cell_1_1 = {"light2", "camera1"}
    
    # Find objects in a region (2x2 grid)
    region_objects = grid_cell_0_0 | grid_cell_0_1 | grid_cell_1_0 | grid_cell_1_1
    print(f"   Objects in 2x2 region: {region_objects}")
    
    # Find objects that span multiple cells
    spanning_objects = (grid_cell_0_0 & grid_cell_0_1) | (grid_cell_0_0 & grid_cell_1_0) | (grid_cell_0_1 & grid_cell_1_1) | (grid_cell_1_0 & grid_cell_1_1)
    print(f"   Objects spanning multiple cells: {spanning_objects}")
    
    # 2. Frustum culling simulation
    print("\n2. Frustum Culling Simulation:")
    
    # Objects in different distance ranges
    near_objects = {"cube1", "sphere1", "light1"}
    medium_objects = {"sphere1", "cylinder1", "cube2"}
    far_objects = {"cylinder1", "light2", "camera1"}
    
    # Objects visible in frustum (near + medium)
    frustum_objects = near_objects | medium_objects
    print(f"   Near objects: {near_objects}")
    print(f"   Medium objects: {medium_objects}")
    print(f"   Far objects: {far_objects}")
    print(f"   Objects in frustum: {frustum_objects}")
    
    # 3. Occlusion culling simulation
    print("\n3. Occlusion Culling Simulation:")
    
    # Objects that might occlude others
    occluders = {"cube1", "sphere1", "cylinder1"}
    
    # Objects that are occluded
    occluded = {"cube2", "light2"}
    
    # Objects that are definitely visible (not occluded)
    definitely_visible = (near_objects | medium_objects) - occluded
    print(f"   Occluders: {occluders}")
    print(f"   Occluded objects: {occluded}")
    print(f"   Definitely visible: {definitely_visible}")

def demonstrate_performance_benefits():
    """Demonstrate performance benefits of sets"""
    print("\n=== Performance Benefits ===\n")
    
    import time
    
    # 1. Membership testing performance
    print("1. Membership Testing Performance:")
    
    # Create large collections
    large_list = list(range(10000))
    large_set = set(range(10000))
    
    # Test membership in list vs set
    test_value = 5000
    
    start_time = time.time()
    for _ in range(100000):
        test_value in large_list
    list_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(100000):
        test_value in large_set
    set_time = time.time() - start_time
    
    print(f"   List membership test: {list_time*1000:.2f} ms")
    print(f"   Set membership test: {set_time*1000:.2f} ms")
    print(f"   Set is {list_time/set_time:.1f}x faster")
    
    # 2. Deduplication performance
    print("\n2. Deduplication Performance:")
    
    # Create list with many duplicates
    duplicate_list = [i % 1000 for i in range(10000)]  # 1000 unique values, 10x each
    
    # Deduplicate using list
    start_time = time.time()
    unique_list = []
    for item in duplicate_list:
        if item not in unique_list:
            unique_list.append(item)
    list_dedup_time = time.time() - start_time
    
    # Deduplicate using set
    start_time = time.time()
    unique_set = set(duplicate_list)
    set_dedup_time = time.time() - start_time
    
    print(f"   List deduplication: {list_dedup_time*1000:.2f} ms")
    print(f"   Set deduplication: {set_dedup_time*1000:.2f} ms")
    print(f"   Set is {list_dedup_time/set_dedup_time:.1f}x faster")
    
    # 3. Set operations performance
    print("\n3. Set Operations Performance:")
    
    # Create two large sets
    set_a = set(range(5000))
    set_b = set(range(3000, 8000))
    
    # Test intersection performance
    start_time = time.time()
    for _ in range(1000):
        intersection = set_a & set_b
    intersection_time = time.time() - start_time
    
    # Test union performance
    start_time = time.time()
    for _ in range(1000):
        union = set_a | set_b
    union_time = time.time() - start_time
    
    print(f"   Intersection operation: {intersection_time*1000:.2f} ms")
    print(f"   Union operation: {union_time*1000:.2f} ms")
    print(f"   Intersection result size: {len(set_a & set_b)}")
    print(f"   Union result size: {len(set_a | set_b)}")

def demonstrate_practical_examples():
    """Demonstrate practical examples with sets"""
    print("\n=== Practical Examples ===\n")
    
    # 1. Material management
    print("1. Material Management:")
    
    # Objects using different materials
    metal_objects = {"cube1", "sphere1", "cylinder1"}
    plastic_objects = {"sphere1", "cube2", "light1"}
    glass_objects = {"sphere2", "light2"}
    
    # Objects that can use multiple materials
    multi_material_objects = (metal_objects & plastic_objects) | (metal_objects & glass_objects) | (plastic_objects & glass_objects)
    
    # All objects that need materials
    all_material_objects = metal_objects | plastic_objects | glass_objects
    
    print(f"   Metal objects: {metal_objects}")
    print(f"   Plastic objects: {plastic_objects}")
    print(f"   Glass objects: {glass_objects}")
    print(f"   Multi-material objects: {multi_material_objects}")
    print(f"   Total objects needing materials: {len(all_material_objects)}")
    
    # 2. Animation tracking
    print("\n2. Animation Tracking:")
    
    # Objects with different animation states
    animating_objects = {"cube1", "sphere1", "light1"}
    paused_objects = {"sphere1", "cylinder1"}
    finished_objects = {"cube2", "light2"}
    
    # Currently active animations
    active_animations = animating_objects - paused_objects
    print(f"   Animating objects: {animating_objects}")
    print(f"   Paused objects: {paused_objects}")
    print(f"   Finished objects: {finished_objects}")
    print(f"   Active animations: {active_animations}")
    
    # 3. Collision detection optimization
    print("\n3. Collision Detection Optimization:")
    
    # Objects in different collision groups
    static_objects = {"cube1", "sphere1", "cylinder1"}
    dynamic_objects = {"sphere1", "cube2", "light1"}
    trigger_objects = {"sphere2", "light2"}
    
    # Objects that need collision detection
    collision_candidates = dynamic_objects | trigger_objects
    
    # Objects that can collide with each other
    dynamic_vs_dynamic = dynamic_objects & dynamic_objects
    dynamic_vs_static = dynamic_objects & static_objects
    trigger_vs_dynamic = trigger_objects & dynamic_objects
    
    print(f"   Static objects: {static_objects}")
    print(f"   Dynamic objects: {dynamic_objects}")
    print(f"   Trigger objects: {trigger_objects}")
    print(f"   Collision candidates: {collision_candidates}")
    print(f"   Dynamic vs dynamic pairs: {len(dynamic_vs_dynamic)}")
    print(f"   Dynamic vs static pairs: {len(dynamic_vs_static)}")
    print(f"   Trigger vs dynamic pairs: {len(trigger_vs_dynamic)}")

def main():
    """Main function to run all set demonstrations"""
    print("=== Python Sets for Unique Elements ===\n")
    
    # Run all demonstrations
    demonstrate_basic_sets()
    demonstrate_set_operations()
    demonstrate_3d_geometry_sets()
    demonstrate_object_management()
    demonstrate_spatial_indexing()
    demonstrate_performance_benefits()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered set data structures:")
    print("✓ Basic set operations and uniqueness")
    print("✓ Set operations (union, intersection, difference)")
    print("✓ 3D geometry applications (vertex management)")
    print("✓ Object management and categorization")
    print("✓ Spatial indexing and optimization")
    print("✓ Performance benefits and efficiency")
    print("✓ Practical applications in graphics")
    
    print("\nSets are essential for:")
    print("- Managing unique elements and deduplication")
    print("- Efficient membership testing")
    print("- Mathematical set operations")
    print("- Spatial indexing and optimization")
    print("- Object categorization and filtering")
    print("- Performance-critical operations")

if __name__ == "__main__":
    main()
