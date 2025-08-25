#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Tuples for Coordinates Example

This example demonstrates how to use Python tuples for immutable coordinate data,
fixed collections, and efficient data storage in 3D graphics applications.
"""

import math

def demonstrate_basic_tuples():
    """Demonstrate basic tuple operations with 3D coordinates"""
    print("=== Basic Tuple Operations ===\n")
    
    # 1. Creating coordinate tuples
    print("1. Creating Coordinate Tuples:")
    
    # 3D coordinates as tuples
    origin = (0, 0, 0)
    x_axis = (1, 0, 0)
    y_axis = (0, 1, 0)
    z_axis = (0, 0, 1)
    diagonal = (1, 1, 1)
    
    print(f"   Origin: {origin}")
    print(f"   X-axis: {x_axis}")
    print(f"   Y-axis: {y_axis}")
    print(f"   Z-axis: {z_axis}")
    print(f"   Diagonal: {diagonal}")
    
    # 2. Tuple unpacking
    print("\n2. Tuple Unpacking:")
    
    # Unpack coordinates
    x, y, z = diagonal
    print(f"   Diagonal coordinates: x={x}, y={y}, z={z}")
    
    # Multiple assignment
    point1, point2 = (1, 2, 3), (4, 5, 6)
    print(f"   Point 1: {point1}")
    print(f"   Point 2: {point2}")
    
    # 3. Tuple indexing and slicing
    print("\n3. Tuple Indexing and Slicing:")
    
    vertex = (2.5, -1.3, 4.7)
    print(f"   Vertex: {vertex}")
    print(f"   X coordinate: {vertex[0]}")
    print(f"   Y coordinate: {vertex[1]}")
    print(f"   Z coordinate: {vertex[2]}")
    print(f"   First two coordinates: {vertex[:2]}")
    print(f"   Last two coordinates: {vertex[-2:]}")
    
    # 4. Tuple immutability
    print("\n4. Tuple Immutability:")
    
    try:
        # This will raise an error
        vertex[0] = 5.0
    except TypeError as e:
        print(f"   Error: {e}")
        print("   Tuples are immutable - cannot modify individual elements")
    
    # Create new tuple instead
    new_vertex = (5.0, vertex[1], vertex[2])
    print(f"   New vertex: {new_vertex}")

def demonstrate_coordinate_operations():
    """Demonstrate coordinate operations with tuples"""
    print("\n=== Coordinate Operations ===\n")
    
    # 1. Vector operations with tuples
    print("1. Vector Operations:")
    
    def add_vectors(v1, v2):
        """Add two 3D vectors"""
        return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])
    
    def subtract_vectors(v1, v2):
        """Subtract two 3D vectors"""
        return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])
    
    def scale_vector(v, scalar):
        """Scale a 3D vector"""
        return (v[0] * scalar, v[1] * scalar, v[2] * scalar)
    
    def dot_product(v1, v2):
        """Calculate dot product of two 3D vectors"""
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    
    def vector_magnitude(v):
        """Calculate magnitude of a 3D vector"""
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    # Test vector operations
    v1 = (1, 2, 3)
    v2 = (4, 5, 6)
    
    print(f"   v1 = {v1}")
    print(f"   v2 = {v2}")
    print(f"   v1 + v2 = {add_vectors(v1, v2)}")
    print(f"   v1 - v2 = {subtract_vectors(v1, v2)}")
    print(f"   v1 * 2 = {scale_vector(v1, 2)}")
    print(f"   v1 · v2 = {dot_product(v1, v2)}")
    print(f"   |v1| = {vector_magnitude(v1):.3f}")
    
    # 2. Distance calculations
    print("\n2. Distance Calculations:")
    
    def distance_between_points(p1, p2):
        """Calculate distance between two 3D points"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def manhattan_distance(p1, p2):
        """Calculate Manhattan distance between two 3D points"""
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])
    
    point_a = (0, 0, 0)
    point_b = (3, 4, 0)
    
    print(f"   Point A: {point_a}")
    print(f"   Point B: {point_b}")
    print(f"   Euclidean distance: {distance_between_points(point_a, point_b):.3f}")
    print(f"   Manhattan distance: {manhattan_distance(point_a, point_b)}")
    
    # 3. Coordinate transformations
    print("\n3. Coordinate Transformations:")
    
    def translate_point(point, offset):
        """Translate a point by an offset"""
        return (point[0] + offset[0], point[1] + offset[1], point[2] + offset[2])
    
    def scale_point(point, scale_factors):
        """Scale a point by scale factors"""
        return (point[0] * scale_factors[0], point[1] * scale_factors[1], point[2] * scale_factors[2])
    
    def rotate_point_2d(point, angle, center=(0, 0, 0)):
        """Rotate a point around Z-axis (2D rotation)"""
        # Translate to origin
        px, py, pz = point[0] - center[0], point[1] - center[1], point[2]
        
        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        
        # Translate back
        return (rx + center[0], ry + center[1], pz)
    
    # Test transformations
    original_point = (1, 0, 0)
    print(f"   Original point: {original_point}")
    
    translated = translate_point(original_point, (2, 3, 1))
    print(f"   Translated by (2, 3, 1): {translated}")
    
    scaled = scale_point(original_point, (2, 1, 1))
    print(f"   Scaled by (2, 1, 1): {scaled}")
    
    rotated = rotate_point_2d(original_point, math.pi/2)  # 90 degrees
    print(f"   Rotated 90° around Z: {rotated}")

def demonstrate_tuple_collections():
    """Demonstrate collections of tuples"""
    print("\n=== Tuple Collections ===\n")
    
    # 1. Tuple of coordinates
    print("1. Tuple of Coordinates:")
    
    # Fixed set of vertices for a triangle
    triangle_vertices = (
        (0, 0, 0),      # Vertex 1
        (1, 0, 0),      # Vertex 2
        (0.5, 1, 0)     # Vertex 3
    )
    
    print(f"   Triangle vertices: {triangle_vertices}")
    print(f"   Number of vertices: {len(triangle_vertices)}")
    print(f"   First vertex: {triangle_vertices[0]}")
    print(f"   Last vertex: {triangle_vertices[-1]}")
    
    # 2. Tuple of tuples (matrix)
    print("\n2. Tuple of Tuples (Matrix):")
    
    # 3x3 identity matrix
    identity_matrix = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    )
    
    print(f"   Identity matrix: {identity_matrix}")
    print(f"   Matrix size: {len(identity_matrix)}x{len(identity_matrix[0])}")
    print(f"   Element at [1][1]: {identity_matrix[1][1]}")
    
    # 3. Named tuples for better readability
    print("\n3. Named Tuples:")
    
    from collections import namedtuple
    
    # Define named tuples for better code readability
    Point3D = namedtuple('Point3D', ['x', 'y', 'z'])
    Vector3D = namedtuple('Vector3D', ['x', 'y', 'z'])
    Color = namedtuple('Color', ['r', 'g', 'b'])
    
    # Create named tuples
    origin = Point3D(0, 0, 0)
    direction = Vector3D(1, 0, 0)
    red_color = Color(255, 0, 0)
    
    print(f"   Origin: {origin}")
    print(f"   Direction: {direction}")
    print(f"   Red color: {red_color}")
    
    # Access by name
    print(f"   Origin x-coordinate: {origin.x}")
    print(f"   Direction magnitude: {math.sqrt(direction.x**2 + direction.y**2 + direction.z**2):.3f}")
    print(f"   Red color intensity: {red_color.r}")

def demonstrate_immutable_advantages():
    """Demonstrate advantages of tuple immutability"""
    print("\n=== Immutable Advantages ===\n")
    
    # 1. Hashable tuples
    print("1. Hashable Tuples:")
    
    # Tuples can be used as dictionary keys
    coordinate_colors = {
        (0, 0, 0): "black",
        (1, 0, 0): "red",
        (0, 1, 0): "green",
        (0, 0, 1): "blue",
        (1, 1, 1): "white"
    }
    
    print(f"   Coordinate-color mapping: {len(coordinate_colors)} entries")
    for coord, color in coordinate_colors.items():
        print(f"     {coord} -> {color}")
    
    # 2. Tuple as function return values
    print("\n2. Tuple Return Values:")
    
    def calculate_bounding_box(points):
        """Calculate bounding box for a set of points"""
        if not points:
            return None
        
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        min_z = min(point[2] for point in points)
        
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        max_z = max(point[2] for point in points)
        
        return ((min_x, min_y, min_z), (max_x, max_y, max_z))
    
    # Test bounding box calculation
    test_points = [(0, 0, 0), (1, 2, 3), (-1, -1, -1), (5, 5, 5)]
    bbox = calculate_bounding_box(test_points)
    
    print(f"   Test points: {test_points}")
    print(f"   Bounding box: min={bbox[0]}, max={bbox[1]}")
    
    # 3. Tuple unpacking in loops
    print("\n3. Tuple Unpacking in Loops:")
    
    # List of coordinate pairs
    coordinate_pairs = [
        ((0, 0, 0), (1, 1, 1)),
        ((2, 2, 2), (3, 3, 3)),
        ((4, 4, 4), (5, 5, 5))
    ]
    
    print("   Coordinate pairs:")
    for start, end in coordinate_pairs:
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)
        print(f"     {start} to {end}: distance = {distance:.3f}")

def demonstrate_performance_benefits():
    """Demonstrate performance benefits of tuples"""
    print("\n=== Performance Benefits ===\n")
    
    import time
    
    # 1. Memory efficiency
    print("1. Memory Efficiency:")
    
    # Compare memory usage
    import sys
    
    list_coords = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    tuple_coords = ((0, 0, 0), (1, 1, 1), (2, 2, 2))
    
    list_size = sys.getsizeof(list_coords)
    tuple_size = sys.getsizeof(tuple_coords)
    
    print(f"   List size: {list_size} bytes")
    print(f"   Tuple size: {tuple_size} bytes")
    print(f"   Tuple is {list_size/tuple_size:.1f}x more memory efficient")
    
    # 2. Creation speed
    print("\n2. Creation Speed:")
    
    # Test creation speed
    start_time = time.time()
    for _ in range(100000):
        coords = [0, 0, 0]
    list_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(100000):
        coords = (0, 0, 0)
    tuple_time = time.time() - start_time
    
    print(f"   List creation: {list_time*1000:.2f} ms")
    print(f"   Tuple creation: {tuple_time*1000:.2f} ms")
    print(f"   Tuple is {list_time/tuple_time:.1f}x faster")
    
    # 3. Access speed
    print("\n3. Access Speed:")
    
    # Test access speed
    test_list = [0, 0, 0]
    test_tuple = (0, 0, 0)
    
    start_time = time.time()
    for _ in range(1000000):
        x = test_list[0]
        y = test_list[1]
        z = test_list[2]
    list_access_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(1000000):
        x = test_tuple[0]
        y = test_tuple[1]
        z = test_tuple[2]
    tuple_access_time = time.time() - start_time
    
    print(f"   List access: {list_access_time*1000:.2f} ms")
    print(f"   Tuple access: {tuple_access_time*1000:.2f} ms")
    print(f"   Tuple is {list_access_time/tuple_access_time:.1f}x faster")

def demonstrate_practical_applications():
    """Demonstrate practical applications of tuples"""
    print("\n=== Practical Applications ===\n")
    
    # 1. Configuration settings
    print("1. Configuration Settings:")
    
    # Graphics settings as tuples
    resolution = (1920, 1080)
    viewport = (0, 0, 1920, 1080)
    color_depth = (8, 8, 8, 8)  # RGBA bits
    
    print(f"   Resolution: {resolution[0]}x{resolution[1]}")
    print(f"   Viewport: {viewport}")
    print(f"   Color depth: {color_depth}")
    
    # 2. Geometric primitives
    print("\n2. Geometric Primitives:")
    
    # Define geometric primitives using tuples
    cube_vertices = (
        # Front face
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
        # Back face
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1)
    )
    
    cube_faces = (
        (0, 1, 2, 3),  # Front
        (1, 5, 6, 2),  # Right
        (5, 4, 7, 6),  # Back
        (4, 0, 3, 7),  # Left
        (3, 2, 6, 7),  # Top
        (4, 5, 1, 0)   # Bottom
    )
    
    print(f"   Cube vertices: {len(cube_vertices)}")
    print(f"   Cube faces: {len(cube_faces)}")
    print(f"   First face vertices: {cube_faces[0]}")
    
    # 3. Coordinate systems
    print("\n3. Coordinate Systems:")
    
    # Define coordinate system axes
    world_axes = (
        (1, 0, 0),  # X-axis
        (0, 1, 0),  # Y-axis
        (0, 0, 1)   # Z-axis
    )
    
    # Camera coordinate system
    camera_position = (0, 0, 5)
    camera_target = (0, 0, 0)
    camera_up = (0, 1, 0)
    
    print(f"   World axes: {world_axes}")
    print(f"   Camera position: {camera_position}")
    print(f"   Camera target: {camera_target}")
    print(f"   Camera up vector: {camera_up}")
    
    # 4. Color palettes
    print("\n4. Color Palettes:")
    
    # Define color palette as tuples
    basic_colors = (
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 255, 255), # White
        (0, 0, 0)       # Black
    )
    
    print(f"   Basic color palette: {len(basic_colors)} colors")
    for i, color in enumerate(basic_colors):
        print(f"     Color {i}: RGB{color}")

def main():
    """Main function to run all tuple demonstrations"""
    print("=== Python Tuples for Coordinates ===\n")
    
    # Run all demonstrations
    demonstrate_basic_tuples()
    demonstrate_coordinate_operations()
    demonstrate_tuple_collections()
    demonstrate_immutable_advantages()
    demonstrate_performance_benefits()
    demonstrate_practical_applications()
    
    print("\n=== Summary ===")
    print("This chapter covered tuple data structures:")
    print("✓ Basic tuple operations and immutability")
    print("✓ Coordinate operations and vector math")
    print("✓ Tuple collections and named tuples")
    print("✓ Immutability advantages (hashable, keys)")
    print("✓ Performance benefits and memory efficiency")
    print("✓ Practical applications in 3D graphics")
    
    print("\nTuples are essential for:")
    print("- Storing immutable coordinate data")
    print("- Creating hashable data structures")
    print("- Efficient memory usage and performance")
    print("- Fixed collections and configurations")
    print("- Function return values and unpacking")
    print("- Geometric primitives and transformations")

if __name__ == "__main__":
    main()
