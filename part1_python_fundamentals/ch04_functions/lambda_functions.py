#!/usr/bin/env python3
"""
Chapter 4: Functions
Lambda Functions Example

This example demonstrates lambda functions (anonymous functions) and functional
programming concepts, focusing on applications in 3D graphics programming.
"""

import math
from functools import reduce

def demonstrate_basic_lambdas():
    """Demonstrate basic lambda function syntax and usage"""
    print("=== Basic Lambda Functions ===\n")
    
    # 1. Simple lambda functions
    print("1. Simple Lambda Functions:")
    
    # Basic arithmetic
    add = lambda x, y: x + y
    multiply = lambda x, y: x * y
    square = lambda x: x ** 2
    
    print(f"   add(5, 3) = {add(5, 3)}")
    print(f"   multiply(4, 7) = {multiply(4, 7)}")
    print(f"   square(6) = {square(6)}")
    
    # 2. Lambda with conditional logic
    print("\n2. Lambda with Conditionals:")
    
    # Absolute value
    abs_value = lambda x: x if x >= 0 else -x
    
    # Clamp value between min and max
    clamp = lambda x, min_val, max_val: max(min_val, min(x, max_val))
    
    # Check if point is in bounds
    in_bounds = lambda x, y, z, bounds: (
        bounds[0] <= x <= bounds[3] and 
        bounds[1] <= y <= bounds[4] and 
        bounds[2] <= z <= bounds[5]
    )
    
    print(f"   abs_value(-7) = {abs_value(-7)}")
    print(f"   clamp(15, 0, 10) = {clamp(15, 0, 10)}")
    print(f"   in_bounds(5, 3, 2, [0, 0, 0, 10, 10, 10]) = {in_bounds(5, 3, 2, [0, 0, 0, 10, 10, 10])}")
    
    # 3. Lambda with multiple operations
    print("\n3. Lambda with Multiple Operations:")
    
    # Calculate distance between two 2D points
    distance_2d = lambda x1, y1, x2, y2: math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate magnitude of 3D vector
    magnitude_3d = lambda x, y, z: math.sqrt(x**2 + y**2 + z**2)
    
    # Normalize 3D vector (returns tuple)
    normalize_3d = lambda x, y, z: (
        x / math.sqrt(x**2 + y**2 + z**2),
        y / math.sqrt(x**2 + y**2 + z**2),
        z / math.sqrt(x**2 + y**2 + z**2)
    ) if (x**2 + y**2 + z**2) > 0 else (0, 0, 0)
    
    print(f"   distance_2d(0, 0, 3, 4) = {distance_2d(0, 0, 3, 4):.2f}")
    print(f"   magnitude_3d(1, 2, 2) = {magnitude_3d(1, 2, 2):.2f}")
    print(f"   normalize_3d(3, 0, 0) = {normalize_3d(3, 0, 0)}")

def demonstrate_lambda_with_functions():
    """Demonstrate lambda functions used with built-in functions"""
    print("\n=== Lambda with Built-in Functions ===\n")
    
    # 1. Lambda with map()
    print("1. Lambda with map():")
    
    # Transform list of positions
    positions = [[0, 0, 0], [1, 2, 3], [5, 10, 15], [-2, -4, -6]]
    
    # Scale all positions by 2
    scaled_positions = list(map(lambda pos: [pos[0] * 2, pos[1] * 2, pos[2] * 2], positions))
    print(f"   Original positions: {positions}")
    print(f"   Scaled positions: {scaled_positions}")
    
    # Calculate distances from origin
    distances = list(map(lambda pos: math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2), positions))
    print(f"   Distances from origin: {[f'{d:.2f}' for d in distances]}")
    
    # 2. Lambda with filter()
    print("\n2. Lambda with filter():")
    
    # Filter objects by distance
    objects = [
        {"name": "cube1", "position": [0, 0, 0], "visible": True},
        {"name": "sphere1", "position": [10, 0, 0], "visible": True},
        {"name": "cube2", "position": [50, 0, 0], "visible": False},
        {"name": "light1", "position": [5, 5, 5], "visible": True}
    ]
    
    # Filter visible objects
    visible_objects = list(filter(lambda obj: obj["visible"], objects))
    print(f"   Visible objects: {[obj['name'] for obj in visible_objects]}")
    
    # Filter objects within distance 20
    nearby_objects = list(filter(
        lambda obj: math.sqrt(obj["position"][0]**2 + obj["position"][1]**2 + obj["position"][2]**2) <= 20,
        objects
    ))
    print(f"   Nearby objects (≤20 units): {[obj['name'] for obj in nearby_objects]}")
    
    # 3. Lambda with sorted()
    print("\n3. Lambda with sorted():")
    
    # Sort objects by distance from origin
    sorted_by_distance = sorted(objects, key=lambda obj: 
        math.sqrt(obj["position"][0]**2 + obj["position"][1]**2 + obj["position"][2]**2)
    )
    print("   Objects sorted by distance:")
    for obj in sorted_by_distance:
        distance = math.sqrt(obj["position"][0]**2 + obj["position"][1]**2 + obj["position"][2]**2)
        print(f"     {obj['name']}: {distance:.2f} units")
    
    # Sort by name
    sorted_by_name = sorted(objects, key=lambda obj: obj["name"])
    print(f"   Objects sorted by name: {[obj['name'] for obj in sorted_by_name]}")

def demonstrate_functional_programming():
    """Demonstrate functional programming concepts with lambdas"""
    print("\n=== Functional Programming ===\n")
    
    # 1. Lambda with reduce()
    print("1. Lambda with reduce():")
    
    # Sum all coordinates
    coordinates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    total_sum = reduce(lambda acc, coord: [acc[0] + coord[0], acc[1] + coord[1], acc[2] + coord[2]], coordinates)
    print(f"   Coordinates: {coordinates}")
    print(f"   Total sum: {total_sum}")
    
    # Find maximum distance from origin
    distances = [5, 12, 8, 15, 3, 20]
    max_distance = reduce(lambda max_val, dist: max(max_val, dist), distances)
    print(f"   Distances: {distances}")
    print(f"   Maximum distance: {max_distance}")
    
    # 2. Lambda with list comprehensions
    print("\n2. Lambda with List Comprehensions:")
    
    # Create grid of points
    grid_size = 3
    grid_points = [(x, y, z) for x in range(grid_size) for y in range(grid_size) for z in range(grid_size)]
    
    # Apply transformation to each point
    transformed_points = [(lambda x, y, z: (x*2, y*2, z*2))(x, y, z) for x, y, z in grid_points]
    print(f"   Grid points: {grid_points[:5]}... (total: {len(grid_points)})")
    print(f"   Transformed: {transformed_points[:5]}...")
    
    # 3. Lambda with any() and all()
    print("\n3. Lambda with any() and all():")
    
    # Check if any object is at origin
    object_positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    at_origin = any(lambda pos: pos == [0, 0, 0] for pos in object_positions)
    print(f"   Any object at origin: {at_origin}")
    
    # Check if all objects are within bounds
    bounds = [-10, -10, -10, 10, 10, 10]
    all_in_bounds = all(
        lambda pos: bounds[0] <= pos[0] <= bounds[3] and 
                   bounds[1] <= pos[1] <= bounds[4] and 
                   bounds[2] <= pos[2] <= bounds[5]
        for pos in object_positions
    )
    print(f"   All objects in bounds: {all_in_bounds}")

def demonstrate_3d_graphics_lambdas():
    """Demonstrate lambda functions in 3D graphics contexts"""
    print("\n=== 3D Graphics Lambda Examples ===\n")
    
    # 1. Vector operations with lambdas
    print("1. Vector Operations:")
    
    # Vector addition
    add_vectors = lambda v1, v2: [v1[i] + v2[i] for i in range(3)]
    
    # Vector subtraction
    sub_vectors = lambda v1, v2: [v1[i] - v2[i] for i in range(3)]
    
    # Vector scaling
    scale_vector = lambda v, s: [v[i] * s for i in range(3)]
    
    # Dot product
    dot_product = lambda v1, v2: sum(v1[i] * v2[i] for i in range(3))
    
    # Test vector operations
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    print(f"   v1 = {v1}, v2 = {v2}")
    print(f"   v1 + v2 = {add_vectors(v1, v2)}")
    print(f"   v1 - v2 = {sub_vectors(v1, v2)}")
    print(f"   v1 * 2 = {scale_vector(v1, 2)}")
    print(f"   v1 · v2 = {dot_product(v1, v2)}")
    
    # 2. Transformation lambdas
    print("\n2. Transformations:")
    
    # Translation
    translate = lambda point, offset: [point[i] + offset[i] for i in range(3)]
    
    # Scaling around origin
    scale = lambda point, factor: [point[i] * factor for i in range(3)]
    
    # Simple 2D rotation around Z-axis
    rotate_z = lambda point, angle: [
        point[0] * math.cos(angle) - point[1] * math.sin(angle),
        point[0] * math.sin(angle) + point[1] * math.cos(angle),
        point[2]
    ]
    
    # Test transformations
    point = [1, 0, 0]
    print(f"   Original point: {point}")
    print(f"   Translated by [2, 3, 1]: {translate(point, [2, 3, 1])}")
    print(f"   Scaled by 2: {scale(point, 2)}")
    print(f"   Rotated 90° around Z: {rotate_z(point, math.pi/2)}")
    
    # 3. Utility lambdas for 3D graphics
    print("\n3. 3D Graphics Utilities:")
    
    # Calculate bounding box
    get_bounds = lambda points: {
        "min": [min(p[i] for p in points) for i in range(3)],
        "max": [max(p[i] for p in points) for i in range(3)]
    }
    
    # Check if point is inside bounding box
    point_in_bounds = lambda point, bounds: all(
        bounds["min"][i] <= point[i] <= bounds["max"][i] for i in range(3)
    )
    
    # Calculate center of points
    center_of_points = lambda points: [
        sum(p[i] for p in points) / len(points) for i in range(3)
    ]
    
    # Test utilities
    test_points = [[0, 0, 0], [1, 2, 3], [-1, -1, -1], [5, 5, 5]]
    bounds = get_bounds(test_points)
    center = center_of_points(test_points)
    
    print(f"   Test points: {test_points}")
    print(f"   Bounding box: {bounds}")
    print(f"   Center: {center}")
    print(f"   [0, 0, 0] in bounds: {point_in_bounds([0, 0, 0], bounds)}")

def demonstrate_advanced_lambda_patterns():
    """Demonstrate advanced lambda function patterns"""
    print("\n=== Advanced Lambda Patterns ===\n")
    
    # 1. Lambda factories
    print("1. Lambda Factories:")
    
    # Create transformation function
    def create_transformer(scale_factor, offset):
        return lambda point: [point[i] * scale_factor + offset[i] for i in range(3)]
    
    # Create different transformers
    scale_up = create_transformer(2, [0, 0, 0])
    move_right = create_transformer(1, [5, 0, 0])
    scale_and_move = create_transformer(1.5, [1, 1, 1])
    
    test_point = [1, 1, 1]
    print(f"   Test point: {test_point}")
    print(f"   Scale up: {scale_up(test_point)}")
    print(f"   Move right: {move_right(test_point)}")
    print(f"   Scale and move: {scale_and_move(test_point)}")
    
    # 2. Lambda with default arguments (using closure)
    print("\n2. Lambda with Default Arguments:")
    
    def create_physics_calculator(gravity=9.81, air_resistance=0.1):
        return lambda position, velocity, delta_time: [
            position[i] + velocity[i] * delta_time for i in range(3)
        ]
    
    # Create different physics calculators
    earth_physics = create_physics_calculator(9.81, 0.1)
    moon_physics = create_physics_calculator(1.62, 0.0)
    
    pos = [0, 10, 0]
    vel = [5, 0, 0]
    dt = 0.1
    
    print(f"   Position: {pos}, Velocity: {vel}, Delta time: {dt}")
    print(f"   Earth physics: {earth_physics(pos, vel, dt)}")
    print(f"   Moon physics: {moon_physics(pos, vel, dt)}")
    
    # 3. Lambda with conditional logic
    print("\n3. Lambda with Conditional Logic:")
    
    # Level of detail selector
    lod_selector = lambda distance: (
        "high" if distance < 10 else
        "medium" if distance < 50 else
        "low" if distance < 100 else
        "none"
    )
    
    # Collision response selector
    collision_response = lambda obj1_type, obj2_type: (
        "damage" if obj1_type == "player" and obj2_type == "enemy" else
        "pickup" if obj1_type == "player" and obj2_type == "item" else
        "bounce" if obj1_type == "ball" else
        "ignore"
    )
    
    # Test conditional lambdas
    distances = [5, 25, 75, 150]
    print("   LOD Selection:")
    for dist in distances:
        print(f"     Distance {dist}: {lod_selector(dist)} LOD")
    
    collision_pairs = [("player", "enemy"), ("player", "item"), ("ball", "wall"), ("light", "player")]
    print("   Collision Responses:")
    for obj1, obj2 in collision_pairs:
        print(f"     {obj1} + {obj2}: {collision_response(obj1, obj2)}")

def demonstrate_lambda_limitations():
    """Demonstrate limitations and alternatives to lambda functions"""
    print("\n=== Lambda Limitations ===\n")
    
    # 1. Lambda limitations
    print("1. Lambda Limitations:")
    
    # Lambda can't have multiple statements
    # This would not work: lambda x: print(x); return x * 2
    
    # Lambda can't have complex logic
    # This would be hard to read: lambda x: x * 2 if x > 0 else x * 3 if x < 0 else 0
    
    # Lambda can't have docstrings
    # This doesn't work: lambda x: """docstring"""; x * 2
    
    print("   ✓ Lambda functions are limited to single expressions")
    print("   ✓ Complex logic should use regular functions")
    print("   ✓ Lambda functions can't have docstrings")
    print("   ✓ Lambda functions can't have multiple statements")
    
    # 2. When to use regular functions instead
    print("\n2. When to Use Regular Functions:")
    
    def complex_physics_calculation(position, velocity, time, gravity=9.81, air_resistance=0.1):
        """Calculate complex physics with multiple steps"""
        # Multiple calculations
        new_velocity = [v - gravity * time for v in velocity]
        new_velocity = [v * (1 - air_resistance * time) for v in new_velocity]
        new_position = [p + v * time for p, v in zip(position, new_velocity)]
        
        # Additional logic
        if any(p < -1000 for p in new_position):
            return None  # Object out of bounds
        
        return new_position, new_velocity
    
    # Test complex function
    pos = [0, 100, 0]
    vel = [10, 0, 0]
    result = complex_physics_calculation(pos, vel, 1.0)
    print(f"   Complex physics result: {result}")
    
    # 3. Best practices
    print("\n3. Best Practices:")
    print("   ✓ Use lambdas for simple, one-line operations")
    print("   ✓ Use lambdas with map(), filter(), sorted()")
    print("   ✓ Use regular functions for complex logic")
    print("   ✓ Use regular functions when you need docstrings")
    print("   ✓ Use regular functions for reusable code")
    print("   ✓ Use regular functions when debugging is important")

def main():
    """Main function to run all lambda demonstrations"""
    print("=== Python Lambda Functions for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_lambdas()
    demonstrate_lambda_with_functions()
    demonstrate_functional_programming()
    demonstrate_3d_graphics_lambdas()
    demonstrate_advanced_lambda_patterns()
    demonstrate_lambda_limitations()
    
    print("\n=== Summary ===")
    print("This chapter covered lambda functions and functional programming:")
    print("✓ Basic lambda function syntax and usage")
    print("✓ Lambda functions with map(), filter(), sorted()")
    print("✓ Functional programming concepts")
    print("✓ 3D graphics applications of lambdas")
    print("✓ Advanced lambda patterns and factories")
    print("✓ Lambda limitations and best practices")
    
    print("\nLambda functions are useful for:")
    print("- Simple, one-line operations")
    print("- Functional programming patterns")
    print("- Callback functions")
    print("- Data transformation pipelines")
    print("- Quick calculations and filters")
    print("- Creating function factories")

if __name__ == "__main__":
    main()
