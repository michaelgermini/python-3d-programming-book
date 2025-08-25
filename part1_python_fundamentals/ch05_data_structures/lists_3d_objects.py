#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Lists for 3D Objects Example

This example demonstrates how to use Python lists for managing 3D objects,
collections, and dynamic data in graphics applications.
"""

import math
import random

def demonstrate_basic_lists():
    """Demonstrate basic list operations with 3D data"""
    print("=== Basic List Operations ===\n")
    
    # 1. Creating lists of 3D coordinates
    print("1. Creating Lists of 3D Coordinates:")
    
    # List of vertex positions
    vertices = [
        [0, 0, 0],    # Origin
        [1, 0, 0],    # X-axis
        [0, 1, 0],    # Y-axis
        [0, 0, 1],    # Z-axis
        [1, 1, 1]     # Diagonal point
    ]
    
    print(f"   Vertices: {vertices}")
    print(f"   Number of vertices: {len(vertices)}")
    print(f"   First vertex: {vertices[0]}")
    print(f"   Last vertex: {vertices[-1]}")
    
    # 2. List slicing and indexing
    print("\n2. List Slicing and Indexing:")
    
    # Get first 3 vertices
    first_three = vertices[:3]
    print(f"   First 3 vertices: {first_three}")
    
    # Get vertices with step 2
    every_other = vertices[::2]
    print(f"   Every other vertex: {every_other}")
    
    # Get vertices from index 1 to 3
    middle_vertices = vertices[1:4]
    print(f"   Middle vertices (1:4): {middle_vertices}")
    
    # 3. Modifying lists
    print("\n3. Modifying Lists:")
    
    # Add new vertex
    vertices.append([2, 2, 2])
    print(f"   After append: {vertices}")
    
    # Insert vertex at specific position
    vertices.insert(2, [0.5, 0.5, 0])
    print(f"   After insert at index 2: {vertices}")
    
    # Remove vertex by value
    vertices.remove([0.5, 0.5, 0])
    print(f"   After remove: {vertices}")
    
    # Pop last vertex
    last_vertex = vertices.pop()
    print(f"   Popped vertex: {last_vertex}")
    print(f"   Remaining vertices: {vertices}")

def demonstrate_3d_object_management():
    """Demonstrate managing 3D objects with lists"""
    print("\n=== 3D Object Management ===\n")
    
    # 1. List of 3D objects
    print("1. List of 3D Objects:")
    
    objects = [
        {"name": "cube1", "position": [0, 0, 0], "scale": [1, 1, 1], "visible": True},
        {"name": "sphere1", "position": [2, 0, 0], "scale": [0.5, 0.5, 0.5], "visible": True},
        {"name": "cylinder1", "position": [0, 2, 0], "scale": [1, 2, 1], "visible": False},
        {"name": "light1", "position": [5, 5, 5], "scale": [1, 1, 1], "visible": True}
    ]
    
    print(f"   Total objects: {len(objects)}")
    for obj in objects:
        print(f"     {obj['name']}: {obj['position']}, visible: {obj['visible']}")
    
    # 2. Filtering objects
    print("\n2. Filtering Objects:")
    
    # Get visible objects
    visible_objects = [obj for obj in objects if obj["visible"]]
    print(f"   Visible objects: {len(visible_objects)}")
    for obj in visible_objects:
        print(f"     {obj['name']}")
    
    # Get objects by type (assuming name contains type)
    geometry_objects = [obj for obj in objects if any(geom in obj["name"] for geom in ["cube", "sphere", "cylinder"])]
    print(f"   Geometry objects: {len(geometry_objects)}")
    for obj in geometry_objects:
        print(f"     {obj['name']}")
    
    # 3. Sorting objects
    print("\n3. Sorting Objects:")
    
    # Sort by distance from origin
    def distance_from_origin(obj):
        pos = obj["position"]
        return math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    
    sorted_by_distance = sorted(objects, key=distance_from_origin)
    print("   Objects sorted by distance from origin:")
    for obj in sorted_by_distance:
        dist = distance_from_origin(obj)
        print(f"     {obj['name']}: {dist:.2f} units")
    
    # Sort by name
    sorted_by_name = sorted(objects, key=lambda obj: obj["name"])
    print("   Objects sorted by name:")
    for obj in sorted_by_name:
        print(f"     {obj['name']}")

def demonstrate_vertex_data_management():
    """Demonstrate managing vertex data with lists"""
    print("\n=== Vertex Data Management ===\n")
    
    # 1. Vertex positions
    print("1. Vertex Positions:")
    
    # Create a simple cube
    cube_vertices = [
        # Front face
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
        # Back face
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1]
    ]
    
    print(f"   Cube vertices: {len(cube_vertices)}")
    print(f"   First vertex: {cube_vertices[0]}")
    print(f"   Last vertex: {cube_vertices[-1]}")
    
    # 2. Face indices
    print("\n2. Face Indices:")
    
    # Define faces as lists of vertex indices
    cube_faces = [
        [0, 1, 2, 3],  # Front face
        [1, 5, 6, 2],  # Right face
        [5, 4, 7, 6],  # Back face
        [4, 0, 3, 7],  # Left face
        [3, 2, 6, 7],  # Top face
        [4, 5, 1, 0]   # Bottom face
    ]
    
    print(f"   Cube faces: {len(cube_faces)}")
    for i, face in enumerate(cube_faces):
        print(f"     Face {i}: vertices {face}")
    
    # 3. Vertex attributes
    print("\n3. Vertex Attributes:")
    
    # Create vertex data with positions and colors
    vertex_data = []
    for i, vertex in enumerate(cube_vertices):
        # Assign different colors to each vertex
        color = [random.random(), random.random(), random.random()]
        vertex_data.append({
            "position": vertex,
            "color": color,
            "index": i
        })
    
    print(f"   Vertex data entries: {len(vertex_data)}")
    for i, data in enumerate(vertex_data[:3]):  # Show first 3
        print(f"     Vertex {i}: pos={data['position']}, color={[f'{c:.2f}' for c in data['color']]}")

def demonstrate_dynamic_collections():
    """Demonstrate dynamic collections with lists"""
    print("\n=== Dynamic Collections ===\n")
    
    # 1. Building collections dynamically
    print("1. Building Collections Dynamically:")
    
    # Start with empty list
    particle_system = []
    
    # Add particles dynamically
    for i in range(5):
        particle = {
            "id": i,
            "position": [random.uniform(-10, 10) for _ in range(3)],
            "velocity": [random.uniform(-1, 1) for _ in range(3)],
            "life": random.uniform(0, 100)
        }
        particle_system.append(particle)
    
    print(f"   Created {len(particle_system)} particles")
    for particle in particle_system[:3]:  # Show first 3
        print(f"     Particle {particle['id']}: pos={[f'{p:.1f}' for p in particle['position']]}")
    
    # 2. Updating collections
    print("\n2. Updating Collections:")
    
    # Update particle positions
    for particle in particle_system:
        # Simple physics update
        for i in range(3):
            particle["position"][i] += particle["velocity"][i]
        particle["life"] -= 1
    
    print("   Updated particle positions:")
    for particle in particle_system[:3]:
        print(f"     Particle {particle['id']}: pos={[f'{p:.1f}' for p in particle['position']]}, life={particle['life']:.1f}")
    
    # 3. Removing elements conditionally
    print("\n3. Removing Elements Conditionally:")
    
    # Remove dead particles
    alive_particles = [p for p in particle_system if p["life"] > 0]
    print(f"   Alive particles: {len(alive_particles)} out of {len(particle_system)}")
    
    # Alternative: remove in-place
    particle_system[:] = [p for p in particle_system if p["life"] > 0]
    print(f"   Particles after cleanup: {len(particle_system)}")

def demonstrate_list_comprehensions():
    """Demonstrate list comprehensions for 3D data"""
    print("\n=== List Comprehensions ===\n")
    
    # 1. Basic list comprehensions
    print("1. Basic List Comprehensions:")
    
    # Generate grid of points
    grid_size = 3
    grid_points = [(x, y, z) for x in range(grid_size) 
                                  for y in range(grid_size) 
                                  for z in range(grid_size)]
    
    print(f"   Grid points: {len(grid_points)}")
    print(f"   First few points: {grid_points[:5]}")
    
    # 2. Conditional comprehensions
    print("\n2. Conditional Comprehensions:")
    
    # Only points within unit sphere
    unit_sphere_points = [point for point in grid_points 
                         if sum(p**2 for p in point) <= 1]
    
    print(f"   Points in unit sphere: {len(unit_sphere_points)}")
    print(f"   Sphere points: {unit_sphere_points}")
    
    # 3. Complex comprehensions
    print("\n3. Complex Comprehensions:")
    
    # Create objects with calculated properties
    objects_with_volume = [
        {
            "position": list(point),
            "volume": abs(point[0] * point[1] * point[2]),
            "distance": math.sqrt(sum(p**2 for p in point))
        }
        for point in grid_points
    ]
    
    print(f"   Objects with properties: {len(objects_with_volume)}")
    for obj in objects_with_volume[:3]:
        print(f"     Pos: {obj['position']}, Volume: {obj['volume']:.2f}, Distance: {obj['distance']:.2f}")

def demonstrate_performance_considerations():
    """Demonstrate performance considerations with lists"""
    print("\n=== Performance Considerations ===\n")
    
    import time
    
    # 1. Appending vs inserting
    print("1. Appending vs Inserting:")
    
    # Test appending
    start_time = time.time()
    append_list = []
    for i in range(10000):
        append_list.append(i)
    append_time = time.time() - start_time
    
    # Test inserting at beginning
    start_time = time.time()
    insert_list = []
    for i in range(10000):
        insert_list.insert(0, i)
    insert_time = time.time() - start_time
    
    print(f"   Append 10,000 elements: {append_time*1000:.2f} ms")
    print(f"   Insert 10,000 elements: {insert_time*1000:.2f} ms")
    print(f"   Insert is {insert_time/append_time:.1f}x slower")
    
    # 2. List vs generator for large datasets
    print("\n2. List vs Generator:")
    
    # Using list comprehension
    start_time = time.time()
    large_list = [i**2 for i in range(100000)]
    list_time = time.time() - start_time
    
    # Using generator expression
    start_time = time.time()
    large_gen = (i**2 for i in range(100000))
    gen_time = time.time() - start_time
    
    print(f"   List comprehension: {list_time*1000:.2f} ms")
    print(f"   Generator expression: {gen_time*1000:.2f} ms")
    print(f"   Memory efficient: Generator")
    
    # 3. Efficient filtering
    print("\n3. Efficient Filtering:")
    
    # Create test data
    test_objects = [
        {"id": i, "position": [random.uniform(-10, 10) for _ in range(3)], "visible": random.choice([True, False])}
        for i in range(10000)
    ]
    
    # Filter with list comprehension
    start_time = time.time()
    visible_objects = [obj for obj in test_objects if obj["visible"]]
    filter_time = time.time() - start_time
    
    print(f"   Filtered {len(visible_objects)} visible objects from {len(test_objects)}")
    print(f"   Filter time: {filter_time*1000:.2f} ms")

def demonstrate_practical_examples():
    """Demonstrate practical examples with lists"""
    print("\n=== Practical Examples ===\n")
    
    # 1. Scene graph implementation
    print("1. Scene Graph Implementation:")
    
    class SceneNode:
        def __init__(self, name, position=[0, 0, 0]):
            self.name = name
            self.position = position
            self.children = []
            self.parent = None
        
        def add_child(self, child):
            child.parent = self
            self.children.append(child)
        
        def get_all_children(self):
            """Get all descendants recursively"""
            all_children = []
            for child in self.children:
                all_children.append(child)
                all_children.extend(child.get_all_children())
            return all_children
    
    # Create scene hierarchy
    root = SceneNode("Root", [0, 0, 0])
    camera = SceneNode("Camera", [0, 0, 5])
    world = SceneNode("World", [0, 0, 0])
    
    # Add objects to world
    cube1 = SceneNode("Cube1", [1, 0, 0])
    cube2 = SceneNode("Cube2", [-1, 0, 0])
    sphere1 = SceneNode("Sphere1", [0, 1, 0])
    
    world.add_child(cube1)
    world.add_child(cube2)
    world.add_child(sphere1)
    
    root.add_child(camera)
    root.add_child(world)
    
    # Get all objects in scene
    all_objects = [root] + root.get_all_children()
    print(f"   Scene objects: {len(all_objects)}")
    for obj in all_objects:
        print(f"     {obj.name}: {obj.position}")
    
    # 2. Object pooling
    print("\n2. Object Pooling:")
    
    class ObjectPool:
        def __init__(self, object_type, initial_size=10):
            self.object_type = object_type
            self.available = []
            self.in_use = []
            
            # Pre-populate pool
            for i in range(initial_size):
                obj = {"id": i, "active": False, "position": [0, 0, 0]}
                self.available.append(obj)
        
        def get_object(self):
            if self.available:
                obj = self.available.pop()
                obj["active"] = True
                self.in_use.append(obj)
                return obj
            else:
                # Create new object if pool is empty
                obj = {"id": len(self.in_use) + len(self.available), "active": True, "position": [0, 0, 0]}
                self.in_use.append(obj)
                return obj
        
        def return_object(self, obj):
            if obj in self.in_use:
                self.in_use.remove(obj)
                obj["active"] = False
                obj["position"] = [0, 0, 0]  # Reset position
                self.available.append(obj)
    
    # Use object pool
    pool = ObjectPool("particle", 5)
    
    # Get objects from pool
    obj1 = pool.get_object()
    obj2 = pool.get_object()
    obj3 = pool.get_object()
    
    print(f"   Objects in use: {len(pool.in_use)}")
    print(f"   Available objects: {len(pool.available)}")
    
    # Return objects to pool
    pool.return_object(obj2)
    print(f"   After returning obj2 - In use: {len(pool.in_use)}, Available: {len(pool.available)}")

def main():
    """Main function to run all list demonstrations"""
    print("=== Python Lists for 3D Objects ===\n")
    
    # Run all demonstrations
    demonstrate_basic_lists()
    demonstrate_3d_object_management()
    demonstrate_vertex_data_management()
    demonstrate_dynamic_collections()
    demonstrate_list_comprehensions()
    demonstrate_performance_considerations()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered list data structures:")
    print("✓ Basic list operations and indexing")
    print("✓ Managing 3D objects and collections")
    print("✓ Vertex data and mesh management")
    print("✓ Dynamic collections and updates")
    print("✓ List comprehensions for data processing")
    print("✓ Performance considerations and optimization")
    print("✓ Practical applications (scene graphs, object pooling)")
    
    print("\nLists are essential for:")
    print("- Managing dynamic collections of 3D objects")
    print("- Storing vertex data and mesh information")
    print("- Implementing scene hierarchies and graphs")
    print("- Building efficient data processing pipelines")
    print("- Creating flexible and extensible systems")
    print("- Prototyping and rapid development")

if __name__ == "__main__":
    main()
