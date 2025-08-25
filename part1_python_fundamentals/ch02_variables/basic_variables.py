#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
Basic Variables Example

This example demonstrates Python's variable system, data types, and operators
in the context of 3D graphics programming.
"""

def demonstrate_basic_variables():
    """Demonstrate basic variable creation and data types"""
    print("=== Basic Variables and Data Types ===\n")
    
    # 1. Numbers (int and float)
    print("1. Numbers:")
    
    # Integer variables
    object_count = 10
    vertices_per_cube = 8
    faces_per_cube = 6
    
    print(f"   Object count: {object_count} (type: {type(object_count)})")
    print(f"   Vertices per cube: {vertices_per_cube} (type: {type(vertices_per_cube)})")
    print(f"   Faces per cube: {faces_per_cube} (type: {type(faces_per_cube)})")
    
    # Float variables (for precise calculations)
    cube_size = 2.5
    rotation_angle = 45.0
    scale_factor = 1.75
    
    print(f"   Cube size: {cube_size} (type: {type(cube_size)})")
    print(f"   Rotation angle: {rotation_angle} degrees (type: {type(rotation_angle)})")
    print(f"   Scale factor: {scale_factor} (type: {type(scale_factor)})")
    
    # 2. Strings
    print("\n2. Strings:")
    
    object_name = "player_cube"
    material_type = "metal"
    texture_path = "/textures/wood.jpg"
    
    print(f"   Object name: '{object_name}' (type: {type(object_name)})")
    print(f"   Material type: '{material_type}' (type: {type(material_type)})")
    print(f"   Texture path: '{texture_path}' (type: {type(texture_path)})")
    
    # String operations
    full_name = object_name + "_" + material_type
    print(f"   Combined name: '{full_name}'")
    
    # 3. Booleans
    print("\n3. Booleans:")
    
    is_visible = True
    is_selected = False
    has_collision = True
    
    print(f"   Is visible: {is_visible} (type: {type(is_visible)})")
    print(f"   Is selected: {is_selected} (type: {type(is_selected)})")
    print(f"   Has collision: {has_collision} (type: {type(has_collision)})")
    
    # 4. Lists (for storing multiple values)
    print("\n4. Lists:")
    
    # 3D coordinates as a list
    cube_position = [0, 0, 0]  # [x, y, z]
    cube_rotation = [45, 0, 90]  # [rx, ry, rz] in degrees
    cube_scale = [1.0, 1.0, 1.0]  # [sx, sy, sz]
    
    print(f"   Cube position: {cube_position} (type: {type(cube_position)})")
    print(f"   Cube rotation: {cube_rotation} (type: {type(cube_rotation)})")
    print(f"   Cube scale: {cube_scale} (type: {type(cube_scale)})")
    
    # Accessing list elements
    print(f"   X coordinate: {cube_position[0]}")
    print(f"   Y coordinate: {cube_position[1]}")
    print(f"   Z coordinate: {cube_position[2]}")
    
    # 5. Tuples (immutable lists)
    print("\n5. Tuples:")
    
    # RGB color as a tuple (immutable)
    cube_color = (255, 128, 64)  # Red, Green, Blue
    light_position = (10.0, 5.0, 0.0)  # Light position
    
    print(f"   Cube color (RGB): {cube_color} (type: {type(cube_color)})")
    print(f"   Light position: {light_position} (type: {type(light_position)})")
    
    # 6. Dictionaries (key-value pairs)
    print("\n6. Dictionaries:")
    
    # Object properties as a dictionary
    cube_properties = {
        'name': 'player_cube',
        'position': [0, 0, 0],
        'rotation': [45, 0, 90],
        'scale': [1.0, 1.0, 1.0],
        'material': 'metal',
        'visible': True,
        'color': (255, 128, 64)
    }
    
    print(f"   Cube properties: {cube_properties}")
    print(f"   Type: {type(cube_properties)}")
    print(f"   Material: {cube_properties['material']}")
    print(f"   Is visible: {cube_properties['visible']}")

def demonstrate_operators():
    """Demonstrate various operators in Python"""
    print("\n=== Operators ===\n")
    
    # 1. Arithmetic Operators
    print("1. Arithmetic Operators:")
    
    x, y, z = 10, 3, 2
    
    print(f"   x = {x}, y = {y}, z = {z}")
    print(f"   Addition: x + y = {x + y}")
    print(f"   Subtraction: x - y = {x - y}")
    print(f"   Multiplication: x * y = {x * y}")
    print(f"   Division: x / y = {x / y}")
    print(f"   Floor division: x // y = {x // y}")
    print(f"   Modulus: x % y = {x % y}")
    print(f"   Exponentiation: x ** z = {x ** z}")
    
    # 2. Assignment Operators
    print("\n2. Assignment Operators:")
    
    position_x = 0
    print(f"   Initial position_x: {position_x}")
    
    position_x += 5  # Same as position_x = position_x + 5
    print(f"   After position_x += 5: {position_x}")
    
    position_x *= 2  # Same as position_x = position_x * 2
    print(f"   After position_x *= 2: {position_x}")
    
    position_x -= 3  # Same as position_x = position_x - 3
    print(f"   After position_x -= 3: {position_x}")
    
    # 3. Comparison Operators
    print("\n3. Comparison Operators:")
    
    cube_size = 5
    min_size = 1
    max_size = 10
    
    print(f"   Cube size: {cube_size}")
    print(f"   Is cube_size > min_size? {cube_size > min_size}")
    print(f"   Is cube_size < max_size? {cube_size < max_size}")
    print(f"   Is cube_size >= min_size? {cube_size >= min_size}")
    print(f"   Is cube_size <= max_size? {cube_size <= max_size}")
    print(f"   Is cube_size == 5? {cube_size == 5}")
    print(f"   Is cube_size != 0? {cube_size != 0}")
    
    # 4. Logical Operators
    print("\n4. Logical Operators:")
    
    is_visible = True
    is_selected = False
    has_collision = True
    
    print(f"   Is visible: {is_visible}")
    print(f"   Is selected: {is_selected}")
    print(f"   Has collision: {has_collision}")
    
    # AND operator
    should_render = is_visible and not is_selected
    print(f"   Should render (visible AND not selected)? {should_render}")
    
    # OR operator
    should_process = is_selected or has_collision
    print(f"   Should process (selected OR has collision)? {should_process}")
    
    # NOT operator
    is_hidden = not is_visible
    print(f"   Is hidden (NOT visible)? {is_hidden}")
    
    # 5. Bitwise Operators (useful for flags and masks)
    print("\n5. Bitwise Operators:")
    
    # Using bits to represent object flags
    FLAG_VISIBLE = 1      # 001
    FLAG_SELECTED = 2     # 010
    FLAG_COLLISION = 4    # 100
    
    object_flags = FLAG_VISIBLE | FLAG_COLLISION  # 101 (visible + collision)
    
    print(f"   Object flags: {object_flags}")
    print(f"   Is visible? {bool(object_flags & FLAG_VISIBLE)}")
    print(f"   Is selected? {bool(object_flags & FLAG_SELECTED)}")
    print(f"   Has collision? {bool(object_flags & FLAG_COLLISION)}")

def demonstrate_3d_calculations():
    """Demonstrate operators in 3D calculations"""
    print("\n=== 3D Calculations with Operators ===\n")
    
    # 1. Vector operations
    print("1. Vector Operations:")
    
    # Position vectors
    player_pos = [0, 0, 0]
    target_pos = [10, 5, 3]
    
    print(f"   Player position: {player_pos}")
    print(f"   Target position: {target_pos}")
    
    # Calculate direction vector
    direction = [
        target_pos[0] - player_pos[0],
        target_pos[1] - player_pos[1],
        target_pos[2] - player_pos[2]
    ]
    print(f"   Direction vector: {direction}")
    
    # 2. Scaling operations
    print("\n2. Scaling Operations:")
    
    base_size = 2.0
    scale_factor = 1.5
    
    new_size = base_size * scale_factor
    print(f"   Base size: {base_size}")
    print(f"   Scale factor: {scale_factor}")
    print(f"   New size: {base_size} * {scale_factor} = {new_size}")
    
    # 3. Rotation calculations
    print("\n3. Rotation Calculations:")
    
    current_angle = 45
    rotation_speed = 5
    max_angle = 360
    
    # Rotate object
    new_angle = (current_angle + rotation_speed) % max_angle
    print(f"   Current angle: {current_angle}°")
    print(f"   Rotation speed: {rotation_speed}°")
    print(f"   New angle: ({current_angle} + {rotation_speed}) % {max_angle} = {new_angle}°")
    
    # 4. Distance calculations
    print("\n4. Distance Calculations:")
    
    # Calculate distance between two points
    point1 = [0, 0, 0]
    point2 = [3, 4, 0]
    
    # Using Pythagorean theorem
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    
    distance = (dx**2 + dy**2 + dz**2)**0.5
    print(f"   Point 1: {point1}")
    print(f"   Point 2: {point2}")
    print(f"   Distance: √({dx}² + {dy}² + {dz}²) = {distance}")

def demonstrate_variable_scope():
    """Demonstrate variable scope and naming"""
    print("\n=== Variable Scope and Naming ===\n")
    
    # Global variables (accessible throughout the module)
    GAME_VERSION = "1.0.0"
    MAX_OBJECTS = 1000
    
    print(f"   Game version: {GAME_VERSION}")
    print(f"   Max objects: {MAX_OBJECTS}")
    
    # Local variables (inside function scope)
    def create_object():
        object_id = 1
        object_type = "cube"
        return object_id, object_type
    
    obj_id, obj_type = create_object()
    print(f"   Created object: ID {obj_id}, Type {obj_type}")
    
    # Variable naming conventions
    print("\nVariable Naming Conventions:")
    print("   ✓ Use descriptive names: player_position instead of pos")
    print("   ✓ Use snake_case for variables: cube_size, rotation_angle")
    print("   ✓ Use UPPER_CASE for constants: MAX_OBJECTS, GAME_VERSION")
    print("   ✓ Avoid single letters except for counters: i, j, k")
    print("   ✓ Use meaningful names for 3D coordinates: x, y, z")

def main():
    """Main function to run all demonstrations"""
    print("=== Python Variables, Data Types, and Operators ===\n")
    
    # Run all demonstrations
    demonstrate_basic_variables()
    demonstrate_operators()
    demonstrate_3d_calculations()
    demonstrate_variable_scope()
    
    print("\n=== Summary ===")
    print("This chapter covered:")
    print("✓ Variable creation and naming conventions")
    print("✓ Different data types (int, float, str, bool, list, tuple, dict)")
    print("✓ Arithmetic, comparison, logical, and assignment operators")
    print("✓ Practical applications in 3D graphics programming")
    print("✓ Variable scope and best practices")
    
    print("\nThese concepts are essential for:")
    print("- Storing 3D object properties and coordinates")
    print("- Performing mathematical calculations for transformations")
    print("- Managing object states and flags")
    print("- Creating efficient and readable code")

if __name__ == "__main__":
    main()
