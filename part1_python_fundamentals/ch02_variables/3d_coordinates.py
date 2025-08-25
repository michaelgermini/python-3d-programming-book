#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
3D Coordinates Example

This example demonstrates working with 3D coordinates, vectors, and mathematical
operations essential for 3D graphics programming.
"""

import math

def demonstrate_3d_coordinates():
    """Demonstrate 3D coordinate systems and basic operations"""
    print("=== 3D Coordinates and Vectors ===\n")
    
    # 1. Basic 3D coordinates
    print("1. Basic 3D Coordinates:")
    
    # Representing 3D points
    origin = [0, 0, 0]
    point_a = [1, 2, 3]
    point_b = [-2, 4, 1]
    
    print(f"   Origin: {origin}")
    print(f"   Point A: {point_a}")
    print(f"   Point B: {point_b}")
    
    # Accessing individual coordinates
    print(f"   Point A - X: {point_a[0]}, Y: {point_a[1]}, Z: {point_a[2]}")
    
    # 2. Vector operations
    print("\n2. Vector Operations:")
    
    # Vector addition
    def add_vectors(v1, v2):
        """Add two 3D vectors"""
        return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
    
    # Vector subtraction
    def subtract_vectors(v1, v2):
        """Subtract vector v2 from v1"""
        return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]
    
    # Vector scaling
    def scale_vector(v, scalar):
        """Scale a vector by a scalar value"""
        return [v[0] * scalar, v[1] * scalar, v[2] * scalar]
    
    # Demonstrate vector operations
    print(f"   Vector A: {point_a}")
    print(f"   Vector B: {point_b}")
    
    # Addition
    sum_vectors = add_vectors(point_a, point_b)
    print(f"   A + B = {sum_vectors}")
    
    # Subtraction
    diff_vectors = subtract_vectors(point_a, point_b)
    print(f"   A - B = {diff_vectors}")
    
    # Scaling
    scaled_a = scale_vector(point_a, 2)
    print(f"   2 * A = {scaled_a}")
    
    # 3. Distance calculations
    print("\n3. Distance Calculations:")
    
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def calculate_manhattan_distance(p1, p2):
        """Calculate Manhattan distance between two 3D points"""
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) + abs(p2[2] - p1[2])
    
    # Calculate distances
    euclidean_dist = calculate_distance(point_a, point_b)
    manhattan_dist = calculate_manhattan_distance(point_a, point_b)
    
    print(f"   Point A: {point_a}")
    print(f"   Point B: {point_b}")
    print(f"   Euclidean distance: {euclidean_dist:.3f}")
    print(f"   Manhattan distance: {manhattan_dist:.3f}")
    
    # 4. Vector magnitude and normalization
    print("\n4. Vector Magnitude and Normalization:")
    
    def vector_magnitude(v):
        """Calculate the magnitude (length) of a vector"""
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    
    def normalize_vector(v):
        """Normalize a vector (make it unit length)"""
        mag = vector_magnitude(v)
        if mag == 0:
            return [0, 0, 0]  # Avoid division by zero
        return [v[0]/mag, v[1]/mag, v[2]/mag]
    
    # Demonstrate magnitude and normalization
    direction_vector = [3, 4, 0]  # 3-4-5 triangle
    magnitude = vector_magnitude(direction_vector)
    normalized = normalize_vector(direction_vector)
    
    print(f"   Direction vector: {direction_vector}")
    print(f"   Magnitude: {magnitude}")
    print(f"   Normalized vector: {normalized}")
    print(f"   Normalized magnitude: {vector_magnitude(normalized):.6f}")

def demonstrate_3d_transformations():
    """Demonstrate 3D transformation operations"""
    print("\n=== 3D Transformations ===\n")
    
    # 1. Translation (moving objects)
    print("1. Translation (Moving Objects):")
    
    def translate_point(point, translation):
        """Translate a point by a translation vector"""
        return [
            point[0] + translation[0],
            point[1] + translation[1],
            point[2] + translation[2]
        ]
    
    # Original object position
    object_pos = [0, 0, 0]
    movement = [5, 3, 2]
    
    new_pos = translate_point(object_pos, movement)
    print(f"   Original position: {object_pos}")
    print(f"   Movement vector: {movement}")
    print(f"   New position: {new_pos}")
    
    # 2. Scaling (resizing objects)
    print("\n2. Scaling (Resizing Objects):")
    
    def scale_point(point, scale_factors):
        """Scale a point by scale factors [sx, sy, sz]"""
        return [
            point[0] * scale_factors[0],
            point[1] * scale_factors[1],
            point[2] * scale_factors[2]
        ]
    
    # Object vertices (simplified cube)
    cube_vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top face
    ]
    
    # Scale the cube
    scale_factors = [2, 1.5, 3]  # Make it wider, taller, and deeper
    scaled_vertices = [scale_point(vertex, scale_factors) for vertex in cube_vertices]
    
    print(f"   Original cube size: 2x2x2")
    print(f"   Scale factors: {scale_factors}")
    print(f"   New cube size: {scale_factors[0]*2}x{scale_factors[1]*2}x{scale_factors[2]*2}")
    
    # Show first few vertices
    print(f"   First 3 original vertices: {cube_vertices[:3]}")
    print(f"   First 3 scaled vertices: {scaled_vertices[:3]}")
    
    # 3. Rotation (simplified 2D rotation around Z-axis)
    print("\n3. Rotation (2D Rotation around Z-axis):")
    
    def rotate_point_2d(point, angle_degrees):
        """Rotate a point around the Z-axis (2D rotation)"""
        angle_radians = math.radians(angle_degrees)
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        
        new_x = point[0] * cos_a - point[1] * sin_a
        new_y = point[0] * sin_a + point[1] * cos_a
        
        return [new_x, new_y, point[2]]  # Z remains unchanged
    
    # Rotate a point
    point_to_rotate = [2, 0, 0]  # Point on X-axis
    rotation_angle = 90  # degrees
    
    rotated_point = rotate_point_2d(point_to_rotate, rotation_angle)
    print(f"   Original point: {point_to_rotate}")
    print(f"   Rotation angle: {rotation_angle}°")
    print(f"   Rotated point: {rotated_point}")

def demonstrate_coordinate_systems():
    """Demonstrate different coordinate systems"""
    print("\n=== Coordinate Systems ===\n")
    
    # 1. World coordinates vs Local coordinates
    print("1. World vs Local Coordinates:")
    
    # World coordinates (global reference system)
    world_origin = [0, 0, 0]
    world_camera = [0, 0, 10]
    world_object = [5, 3, 0]
    
    print(f"   World origin: {world_origin}")
    print(f"   World camera position: {world_camera}")
    print(f"   World object position: {world_object}")
    
    # Local coordinates (relative to camera)
    def world_to_camera(world_point, camera_pos):
        """Convert world coordinates to camera-relative coordinates"""
        return [
            world_point[0] - camera_pos[0],
            world_point[1] - camera_pos[1],
            world_point[2] - camera_pos[2]
        ]
    
    camera_relative_object = world_to_camera(world_object, world_camera)
    print(f"   Object in camera coordinates: {camera_relative_object}")
    
    # 2. Screen coordinates (2D projection)
    print("\n2. Screen Coordinates (2D Projection):")
    
    def project_to_screen(camera_point, screen_distance=5):
        """Simple perspective projection to screen coordinates"""
        if camera_point[2] <= 0:
            return None  # Behind camera
        
        # Simple perspective projection
        scale = screen_distance / camera_point[2]
        screen_x = camera_point[0] * scale
        screen_y = camera_point[1] * scale
        
        return [screen_x, screen_y]
    
    # Project 3D point to 2D screen
    screen_point = project_to_screen(camera_relative_object)
    print(f"   Camera-relative object: {camera_relative_object}")
    print(f"   Screen coordinates: {screen_point}")

def demonstrate_practical_examples():
    """Demonstrate practical 3D coordinate examples"""
    print("\n=== Practical 3D Examples ===\n")
    
    # 1. Object positioning in a scene
    print("1. Object Positioning in a Scene:")
    
    # Scene setup
    scene_center = [0, 0, 0]
    player_position = [0, 0, 0]
    
    # Place objects around the scene
    objects = [
        {"name": "Tree 1", "position": [10, 0, 5]},
        {"name": "Tree 2", "position": [-8, 0, 3]},
        {"name": "Rock", "position": [5, 0, -7]},
        {"name": "House", "position": [-3, 0, 8]}
    ]
    
    print(f"   Scene center: {scene_center}")
    print(f"   Player position: {player_position}")
    print("   Objects in scene:")
    
    for obj in objects:
        distance = math.sqrt(
            (obj["position"][0] - player_position[0])**2 +
            (obj["position"][1] - player_position[1])**2 +
            (obj["position"][2] - player_position[2])**2
        )
        print(f"     {obj['name']}: {obj['position']} (distance: {distance:.2f})")
    
    # 2. Camera movement
    print("\n2. Camera Movement:")
    
    def move_camera(camera_pos, direction, speed):
        """Move camera in a given direction"""
        normalized_dir = normalize_vector(direction)
        return [
            camera_pos[0] + normalized_dir[0] * speed,
            camera_pos[1] + normalized_dir[1] * speed,
            camera_pos[2] + normalized_dir[2] * speed
        ]
    
    camera_pos = [0, 0, 10]
    forward_direction = [0, 0, -1]  # Looking forward
    movement_speed = 2
    
    new_camera_pos = move_camera(camera_pos, forward_direction, movement_speed)
    print(f"   Original camera position: {camera_pos}")
    print(f"   Movement direction: {forward_direction}")
    print(f"   Movement speed: {movement_speed}")
    print(f"   New camera position: {new_camera_pos}")
    
    # 3. Collision detection (simple sphere-sphere)
    print("\n3. Simple Collision Detection:")
    
    def check_sphere_collision(pos1, radius1, pos2, radius2):
        """Check if two spheres are colliding"""
        distance = calculate_distance(pos1, pos2)
        return distance < (radius1 + radius2)
    
    # Two objects
    object1_pos = [0, 0, 0]
    object1_radius = 2
    
    object2_pos = [3, 0, 0]
    object2_radius = 1.5
    
    collision = check_sphere_collision(object1_pos, object1_radius, 
                                     object2_pos, object2_radius)
    
    print(f"   Object 1: position {object1_pos}, radius {object1_radius}")
    print(f"   Object 2: position {object2_pos}, radius {object2_radius}")
    print(f"   Distance between centers: {calculate_distance(object1_pos, object2_pos):.2f}")
    print(f"   Sum of radii: {object1_radius + object2_radius}")
    print(f"   Collision detected: {collision}")

def main():
    """Main function to run all 3D coordinate demonstrations"""
    print("=== Python 3D Coordinates and Vectors ===\n")
    
    # Run all demonstrations
    demonstrate_3d_coordinates()
    demonstrate_3d_transformations()
    demonstrate_coordinate_systems()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered 3D coordinate operations:")
    print("✓ Vector addition, subtraction, and scaling")
    print("✓ Distance calculations (Euclidean and Manhattan)")
    print("✓ Vector magnitude and normalization")
    print("✓ 3D transformations (translation, scaling, rotation)")
    print("✓ Coordinate system conversions")
    print("✓ Practical applications in 3D graphics")
    
    print("\nThese operations are fundamental for:")
    print("- Positioning objects in 3D space")
    print("- Calculating object movements and animations")
    print("- Implementing camera systems")
    print("- Collision detection and physics")
    print("- Rendering and projection calculations")

if __name__ == "__main__":
    main()
