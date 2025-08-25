#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
For Loops Example

This example demonstrates for loops in Python, focusing on applications
in 3D graphics programming and data processing.
"""

def demonstrate_basic_for_loops():
    """Demonstrate basic for loop concepts"""
    print("=== Basic For Loops ===\n")
    
    # 1. Iterating over a list
    print("1. Iterating Over Lists:")
    
    object_types = ["cube", "sphere", "cylinder", "cone", "pyramid"]
    
    for obj_type in object_types:
        print(f"   Creating {obj_type} object")
    
    # 2. Iterating with range
    print("\n2. Iterating with Range:")
    
    # Create objects with sequential IDs
    for i in range(5):
        print(f"   Creating object with ID: {i}")
    
    # 3. Iterating with enumerate
    print("\n3. Iterating with Enumerate:")
    
    for index, obj_type in enumerate(object_types):
        print(f"   Object {index}: {obj_type}")
    
    # 4. Iterating over dictionary
    print("\n4. Iterating Over Dictionary:")
    
    object_properties = {
        "position": [10, 5, 0],
        "rotation": [0, 45, 0],
        "scale": [1.0, 1.0, 1.0],
        "color": [255, 128, 64]
    }
    
    for property_name, value in object_properties.items():
        print(f"   {property_name}: {value}")

def demonstrate_3d_for_loops():
    """Demonstrate for loops in 3D context"""
    print("\n=== 3D Graphics For Loops ===\n")
    
    # 1. Processing vertex data
    print("1. Processing Vertex Data:")
    
    # Simulate vertex data (x, y, z coordinates)
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]
    
    # Calculate center point
    center_x = sum(vertex[0] for vertex in vertices) / len(vertices)
    center_y = sum(vertex[1] for vertex in vertices) / len(vertices)
    center_z = sum(vertex[2] for vertex in vertices) / len(vertices)
    
    print(f"   Object center: [{center_x:.2f}, {center_y:.2f}, {center_z:.2f}]")
    
    # 2. Processing multiple objects
    print("\n2. Processing Multiple Objects:")
    
    objects = [
        {"name": "player", "position": [0, 0, 0], "health": 100},
        {"name": "enemy1", "position": [10, 0, 0], "health": 50},
        {"name": "enemy2", "position": [-5, 0, 0], "health": 75},
        {"name": "item1", "position": [5, 5, 0], "health": 0}
    ]
    
    # Find all enemies
    enemies = []
    for obj in objects:
        if obj["name"].startswith("enemy"):
            enemies.append(obj)
    
    print(f"   Found {len(enemies)} enemies:")
    for enemy in enemies:
        print(f"     {enemy['name']} at {enemy['position']} (health: {enemy['health']})")
    
    # 3. Creating procedural geometry
    print("\n3. Creating Procedural Geometry:")
    
    def create_circle_vertices(center, radius, segments):
        """Create vertices for a circle"""
        vertices = []
        for i in range(segments):
            angle = (2 * 3.14159 * i) / segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]
            vertices.append([x, y, z])
        return vertices
    
    import math
    
    circle_center = [0, 0, 0]
    circle_radius = 5
    circle_segments = 8
    
    circle_vertices = create_circle_vertices(circle_center, circle_radius, circle_segments)
    
    print(f"   Created circle with {len(circle_vertices)} vertices:")
    for i, vertex in enumerate(circle_vertices):
        print(f"     Vertex {i}: [{vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f}]")

def demonstrate_nested_for_loops():
    """Demonstrate nested for loops"""
    print("\n=== Nested For Loops ===\n")
    
    # 1. Creating a 3D grid
    print("1. Creating 3D Grid:")
    
    grid_size = 3
    grid_spacing = 2
    
    grid_points = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                pos_x = x * grid_spacing
                pos_y = y * grid_spacing
                pos_z = z * grid_spacing
                grid_points.append([pos_x, pos_y, pos_z])
    
    print(f"   Created {len(grid_points)} grid points:")
    for i, point in enumerate(grid_points):
        print(f"     Point {i}: {point}")
    
    # 2. Processing texture coordinates
    print("\n2. Processing Texture Coordinates:")
    
    texture_width = 4
    texture_height = 3
    
    # Generate UV coordinates for a texture
    uv_coords = []
    for v in range(texture_height):
        for u in range(texture_width):
            u_coord = u / (texture_width - 1)
            v_coord = v / (texture_height - 1)
            uv_coords.append([u_coord, v_coord])
    
    print(f"   Generated {len(uv_coords)} UV coordinates:")
    for i, uv in enumerate(uv_coords):
        print(f"     UV {i}: [{uv[0]:.2f}, {uv[1]:.2f}]")
    
    # 3. Matrix operations
    print("\n3. Matrix Operations:")
    
    # Simple 3x3 matrix multiplication
    matrix_a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    matrix_b = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ]
    
    # Initialize result matrix
    result = [[0 for _ in range(3)] for _ in range(3)]
    
    # Matrix multiplication
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    print("   Matrix multiplication result:")
    for row in result:
        print(f"     {row}")

def demonstrate_for_loops_with_conditions():
    """Demonstrate for loops with conditional logic"""
    print("\n=== For Loops with Conditions ===\n")
    
    # 1. Filtering objects
    print("1. Filtering Objects:")
    
    scene_objects = [
        {"name": "tree1", "type": "vegetation", "visible": True, "distance": 50},
        {"name": "player", "type": "character", "visible": True, "distance": 0},
        {"name": "rock1", "type": "terrain", "visible": False, "distance": 100},
        {"name": "enemy1", "type": "character", "visible": True, "distance": 25},
        {"name": "cloud1", "type": "sky", "visible": True, "distance": 200}
    ]
    
    # Find visible objects within render distance
    visible_objects = []
    for obj in scene_objects:
        if obj["visible"] and obj["distance"] <= 100:
            visible_objects.append(obj)
    
    print(f"   Found {len(visible_objects)} visible objects within render distance:")
    for obj in visible_objects:
        print(f"     {obj['name']} ({obj['type']}) at distance {obj['distance']}")
    
    # 2. Processing with break
    print("\n2. Processing with Break:")
    
    # Find first enemy within attack range
    attack_range = 30
    target_enemy = None
    
    for obj in scene_objects:
        if (obj["type"] == "character" and 
            obj["name"] != "player" and 
            obj["distance"] <= attack_range):
            target_enemy = obj
            break
    
    if target_enemy:
        print(f"   Found target: {target_enemy['name']} at distance {target_enemy['distance']}")
    else:
        print("   No enemies in attack range")
    
    # 3. Processing with continue
    print("\n3. Processing with Continue:")
    
    # Process only visible objects
    processed_count = 0
    for obj in scene_objects:
        if not obj["visible"]:
            continue  # Skip invisible objects
        
        # Simulate processing
        processed_count += 1
        print(f"     Processing {obj['name']}...")
    
    print(f"   Processed {processed_count} visible objects")

def demonstrate_list_comprehensions():
    """Demonstrate list comprehensions (advanced for loops)"""
    print("\n=== List Comprehensions ===\n")
    
    # 1. Basic list comprehension
    print("1. Basic List Comprehension:")
    
    # Create list of squares
    numbers = [1, 2, 3, 4, 5]
    squares = [x**2 for x in numbers]
    
    print(f"   Numbers: {numbers}")
    print(f"   Squares: {squares}")
    
    # 2. List comprehension with condition
    print("\n2. List Comprehension with Condition:")
    
    # Get only even numbers
    even_numbers = [x for x in numbers if x % 2 == 0]
    print(f"   Even numbers: {even_numbers}")
    
    # 3. 3D-specific list comprehensions
    print("\n3. 3D Graphics List Comprehensions:")
    
    # Generate random positions
    import random
    
    num_objects = 5
    random_positions = [
        [random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)]
        for _ in range(num_objects)
    ]
    
    print(f"   Generated {num_objects} random positions:")
    for i, pos in enumerate(random_positions):
        print(f"     Object {i}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    
    # 4. Processing object data
    print("\n4. Processing Object Data:")
    
    # Extract positions from objects
    object_positions = [obj["position"] for obj in scene_objects]
    print(f"   Object positions: {object_positions}")
    
    # Calculate distances from origin
    distances = [math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2) for pos in object_positions]
    print(f"   Distances from origin: {[f'{d:.2f}' for d in distances]}")

def demonstrate_practical_examples():
    """Demonstrate practical for loop examples"""
    print("\n=== Practical For Loop Examples ===\n")
    
    # 1. Animation frame processing
    print("1. Animation Frame Processing:")
    
    def process_animation_frames(num_frames, start_value, end_value):
        """Process animation frames with interpolation"""
        frames = []
        for frame in range(num_frames):
            # Linear interpolation
            t = frame / (num_frames - 1) if num_frames > 1 else 0
            interpolated_value = start_value + t * (end_value - start_value)
            frames.append(interpolated_value)
        return frames
    
    # Animate object scale
    scale_frames = process_animation_frames(5, 1.0, 2.0)
    print(f"   Scale animation frames: {[f'{s:.2f}' for s in scale_frames]}")
    
    # 2. Batch processing
    print("\n2. Batch Processing:")
    
    def process_object_batch(objects, operation):
        """Process a batch of objects with the same operation"""
        results = []
        for obj in objects:
            if operation == "scale":
                # Scale object
                scaled_obj = obj.copy()
                scaled_obj["scale"] = [s * 2 for s in obj.get("scale", [1, 1, 1])]
                results.append(scaled_obj)
            elif operation == "move":
                # Move object
                moved_obj = obj.copy()
                moved_obj["position"] = [p + 5 for p in obj.get("position", [0, 0, 0])]
                results.append(moved_obj)
        return results
    
    test_objects = [
        {"name": "obj1", "position": [0, 0, 0], "scale": [1, 1, 1]},
        {"name": "obj2", "position": [10, 0, 0], "scale": [2, 2, 2]}
    ]
    
    scaled_objects = process_object_batch(test_objects, "scale")
    print("   Scaled objects:")
    for obj in scaled_objects:
        print(f"     {obj['name']}: scale {obj['scale']}")
    
    # 3. Data validation
    print("\n3. Data Validation:")
    
    def validate_object_data(objects):
        """Validate object data and report issues"""
        issues = []
        for i, obj in enumerate(objects):
            # Check required fields
            if "name" not in obj:
                issues.append(f"Object {i}: Missing name")
            
            if "position" not in obj:
                issues.append(f"Object {i}: Missing position")
            else:
                # Check position bounds
                pos = obj["position"]
                if not (isinstance(pos, list) and len(pos) == 3):
                    issues.append(f"Object {i}: Invalid position format")
                elif any(not isinstance(p, (int, float)) for p in pos):
                    issues.append(f"Object {i}: Position contains non-numeric values")
        
        return issues
    
    test_data = [
        {"name": "valid_obj", "position": [0, 0, 0]},
        {"position": [10, 20, 30]},  # Missing name
        {"name": "invalid_pos", "position": "not_a_list"},  # Invalid position
        {"name": "valid_obj2", "position": [1, 2, 3]}
    ]
    
    validation_issues = validate_object_data(test_data)
    if validation_issues:
        print("   Validation issues found:")
        for issue in validation_issues:
            print(f"     {issue}")
    else:
        print("   All objects passed validation")

def main():
    """Main function to run all for loop demonstrations"""
    print("=== Python For Loops for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_for_loops()
    demonstrate_3d_for_loops()
    demonstrate_nested_for_loops()
    demonstrate_for_loops_with_conditions()
    demonstrate_list_comprehensions()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered for loops:")
    print("✓ Basic for loop syntax and iteration")
    print("✓ Iterating over sequences (lists, dictionaries)")
    print("✓ Using range() and enumerate()")
    print("✓ Nested for loops for complex operations")
    print("✓ For loops with conditional logic (break, continue)")
    print("✓ List comprehensions for concise iteration")
    print("✓ Practical applications in 3D graphics")
    
    print("\nFor loops are essential for:")
    print("- Processing collections of data")
    print("- Creating procedural content")
    print("- Implementing batch operations")
    print("- Data validation and transformation")
    print("- Animation and simulation systems")

if __name__ == "__main__":
    main()
