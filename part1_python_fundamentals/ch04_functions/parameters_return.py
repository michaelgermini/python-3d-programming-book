#!/usr/bin/env python3
"""
Chapter 4: Functions
Parameters and Return Values Example

This example demonstrates different types of function parameters and
return values in Python, focusing on applications in 3D graphics programming.
"""

import math

def demonstrate_parameter_types():
    """Demonstrate different types of function parameters"""
    print("=== Parameter Types ===\n")
    
    # 1. Required parameters
    print("1. Required Parameters:")
    
    def create_3d_object(name, position, scale):
        """Create a 3D object with required parameters"""
        return {
            "name": name,
            "position": position,
            "scale": scale,
            "type": "object"
        }
    
    # Must provide all required parameters
    cube = create_3d_object("MyCube", [0, 0, 0], 1.0)
    print(f"   Created object: {cube}")
    
    # 2. Default parameters
    print("\n2. Default Parameters:")
    
    def create_camera(position, target=[0, 0, 0], fov=60, near=0.1, far=1000):
        """Create a camera with default parameters"""
        return {
            "position": position,
            "target": target,
            "fov": fov,
            "near": near,
            "far": far,
            "type": "camera"
        }
    
    # Use defaults
    camera1 = create_camera([0, 0, 5])
    print(f"   Camera with defaults: {camera1}")
    
    # Override some defaults
    camera2 = create_camera([10, 5, 10], target=[0, 0, 0], fov=90)
    print(f"   Camera with custom FOV: {camera2}")
    
    # 3. Keyword arguments
    print("\n3. Keyword Arguments:")
    
    def create_light(light_type, position, color=[255, 255, 255], intensity=1.0, cast_shadows=True):
        """Create a light with keyword arguments"""
        return {
            "type": light_type,
            "position": position,
            "color": color,
            "intensity": intensity,
            "cast_shadows": cast_shadows
        }
    
    # Use keyword arguments for clarity
    point_light = create_light(
        light_type="point",
        position=[5, 10, 5],
        color=[255, 200, 100],  # Warm light
        intensity=2.0
    )
    print(f"   Point light: {point_light}")
    
    # 4. Variable arguments (*args)
    print("\n4. Variable Arguments (*args):")
    
    def calculate_bounding_box(*points):
        """Calculate bounding box from multiple points"""
        if not points:
            return None
        
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        min_z = min(point[2] for point in points)
        
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)
        max_z = max(point[2] for point in points)
        
        return {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
            "center": [(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2]
        }
    
    # Pass multiple points
    bbox = calculate_bounding_box(
        [0, 0, 0],
        [5, 3, 2],
        [-2, 1, 4],
        [3, -1, 1]
    )
    print(f"   Bounding box: {bbox}")
    
    # 5. Keyword variable arguments (**kwargs)
    print("\n5. Keyword Variable Arguments (**kwargs):")
    
    def create_material(name, **properties):
        """Create a material with flexible properties"""
        material = {
            "name": name,
            "type": "material"
        }
        material.update(properties)
        return material
    
    # Pass various material properties
    metal_material = create_material(
        "Metal",
        diffuse_color=[0.8, 0.8, 0.8],
        metallic=1.0,
        roughness=0.2,
        texture="metal_diffuse.png"
    )
    print(f"   Metal material: {metal_material}")
    
    wood_material = create_material(
        "Wood",
        diffuse_color=[0.6, 0.4, 0.2],
        metallic=0.0,
        roughness=0.8,
        normal_map="wood_normal.png"
    )
    print(f"   Wood material: {wood_material}")

def demonstrate_return_values():
    """Demonstrate different types of return values"""
    print("\n=== Return Values ===\n")
    
    # 1. Single return value
    print("1. Single Return Value:")
    
    def calculate_magnitude(vector):
        """Calculate magnitude of a 3D vector"""
        return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    
    vector = [3, 4, 5]
    magnitude = calculate_magnitude(vector)
    print(f"   Vector {vector} magnitude: {magnitude:.2f}")
    
    # 2. Multiple return values (tuples)
    print("\n2. Multiple Return Values:")
    
    def decompose_vector(vector):
        """Decompose vector into magnitude and direction"""
        magnitude = calculate_magnitude(vector)
        if magnitude == 0:
            return 0, [0, 0, 0]
        
        direction = [vector[0]/magnitude, vector[1]/magnitude, vector[2]/magnitude]
        return magnitude, direction
    
    mag, direction = decompose_vector(vector)
    print(f"   Vector {vector}:")
    print(f"     Magnitude: {mag:.2f}")
    print(f"     Direction: {[f'{x:.3f}' for x in direction]}")
    
    # 3. Return different types
    print("\n3. Return Different Types:")
    
    def process_object_data(object_data):
        """Process object data and return different types based on input"""
        if not object_data:
            return None
        
        if isinstance(object_data, str):
            return {"name": object_data, "type": "string_object"}
        
        if isinstance(object_data, list):
            if len(object_data) == 3:
                return {"position": object_data, "type": "position"}
            else:
                return {"vertices": object_data, "type": "mesh"}
        
        if isinstance(object_data, dict):
            return object_data
        
        return {"data": object_data, "type": "unknown"}
    
    # Test with different input types
    test_inputs = [
        "Player",
        [10, 20, 30],
        [1, 2, 3, 4, 5, 6],
        {"name": "Cube", "position": [0, 0, 0]},
        None
    ]
    
    for test_input in test_inputs:
        result = process_object_data(test_input)
        print(f"   Input {test_input}: {result}")
    
    # 4. Return functions (closures)
    print("\n4. Return Functions (Closures):")
    
    def create_animation_function(duration, start_value, end_value):
        """Create an animation function with closure"""
        def animate(time):
            if time >= duration:
                return end_value
            
            progress = time / duration
            return start_value + progress * (end_value - start_value)
        
        return animate
    
    # Create animation functions
    fade_in = create_animation_function(2.0, 0.0, 1.0)  # 2 second fade in
    scale_up = create_animation_function(1.5, 0.5, 2.0)  # 1.5 second scale
    
    # Test animations
    times = [0, 0.5, 1.0, 1.5, 2.0]
    print("   Fade in animation:")
    for t in times:
        alpha = fade_in(t)
        print(f"     Time {t}s: Alpha = {alpha:.2f}")
    
    print("   Scale animation:")
    for t in times:
        scale = scale_up(t)
        print(f"     Time {t}s: Scale = {scale:.2f}")

def demonstrate_advanced_parameters():
    """Demonstrate advanced parameter techniques"""
    print("\n=== Advanced Parameters ===\n")
    
    # 1. Parameter unpacking
    print("1. Parameter Unpacking:")
    
    def create_transform_matrix(position, rotation, scale):
        """Create transformation matrix from position, rotation, scale"""
        # Simplified matrix creation
        return {
            "position": position,
            "rotation": rotation,
            "scale": scale,
            "matrix": f"Transform({position}, {rotation}, {scale})"
        }
    
    # Unpack parameters from lists
    pos = [10, 5, 0]
    rot = [0, 45, 0]
    scl = [2, 2, 2]
    
    transform = create_transform_matrix(*pos, *rot, *scl)
    print(f"   Transform: {transform}")
    
    # 2. Lambda functions as parameters
    print("\n2. Lambda Functions as Parameters:")
    
    def apply_to_vertices(vertices, transform_func):
        """Apply transformation function to all vertices"""
        return [transform_func(vertex) for vertex in vertices]
    
    # Define transformation functions
    translate_up = lambda v: [v[0], v[1] + 5, v[2]]
    scale_2x = lambda v: [v[0] * 2, v[1] * 2, v[2] * 2]
    rotate_90 = lambda v: [-v[1], v[0], v[2]]
    
    # Test vertices
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    
    translated = apply_to_vertices(vertices, translate_up)
    scaled = apply_to_vertices(vertices, scale_2x)
    rotated = apply_to_vertices(vertices, rotate_90)
    
    print(f"   Original vertices: {vertices}")
    print(f"   Translated: {translated}")
    print(f"   Scaled: {scaled}")
    print(f"   Rotated: {rotated}")
    
    # 3. Optional parameters with None
    print("\n3. Optional Parameters with None:")
    
    def create_shader(vertex_shader, fragment_shader, geometry_shader=None, compute_shader=None):
        """Create a shader program with optional stages"""
        shader = {
            "vertex": vertex_shader,
            "fragment": fragment_shader,
            "stages": ["vertex", "fragment"]
        }
        
        if geometry_shader:
            shader["geometry"] = geometry_shader
            shader["stages"].append("geometry")
        
        if compute_shader:
            shader["compute"] = compute_shader
            shader["stages"].append("compute")
        
        return shader
    
    # Create different shader configurations
    basic_shader = create_shader("basic.vert", "basic.frag")
    advanced_shader = create_shader("advanced.vert", "advanced.frag", "advanced.geom")
    compute_shader = create_shader("compute.vert", "compute.frag", compute_shader="compute.comp")
    
    print(f"   Basic shader: {basic_shader['stages']}")
    print(f"   Advanced shader: {advanced_shader['stages']}")
    print(f"   Compute shader: {compute_shader['stages']}")

def demonstrate_return_patterns():
    """Demonstrate common return value patterns"""
    print("\n=== Return Patterns ===\n")
    
    # 1. Success/Error pattern
    print("1. Success/Error Pattern:")
    
    def load_texture(filename):
        """Load texture file with error handling"""
        # Simulate texture loading
        if not filename:
            return False, "No filename provided"
        
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            return False, f"Unsupported format: {filename}"
        
        # Simulate successful loading
        texture_data = {
            "filename": filename,
            "width": 512,
            "height": 512,
            "format": "RGBA8"
        }
        return True, texture_data
    
    # Test texture loading
    test_files = ["", "texture.txt", "texture.png", "image.jpg"]
    
    for filename in test_files:
        success, result = load_texture(filename)
        if success:
            print(f"   Loaded: {result}")
        else:
            print(f"   Error: {result}")
    
    # 2. Result object pattern
    print("\n2. Result Object Pattern:")
    
    def create_mesh(vertices, indices):
        """Create mesh with result object"""
        if not vertices:
            return {
                "success": False,
                "error": "No vertices provided",
                "mesh": None
            }
        
        if not indices:
            return {
                "success": False,
                "error": "No indices provided",
                "mesh": None
            }
        
        mesh_data = {
            "vertices": vertices,
            "indices": indices,
            "vertex_count": len(vertices),
            "index_count": len(indices)
        }
        
        return {
            "success": True,
            "error": None,
            "mesh": mesh_data
        }
    
    # Test mesh creation
    test_cases = [
        ([], [0, 1, 2]),  # No vertices
        ([1, 2, 3], []),  # No indices
        ([1, 2, 3, 4, 5, 6], [0, 1, 2])  # Valid mesh
    ]
    
    for vertices, indices in test_cases:
        result = create_mesh(vertices, indices)
        if result["success"]:
            print(f"   Created mesh: {result['mesh']}")
        else:
            print(f"   Failed: {result['error']}")
    
    # 3. Generator pattern
    print("\n3. Generator Pattern:")
    
    def generate_fibonacci_sequence(n):
        """Generate Fibonacci sequence up to n terms"""
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
    
    def generate_3d_points(center, radius, count):
        """Generate 3D points in a sphere"""
        import random
        for _ in range(count):
            # Generate random point in unit sphere
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            
            # Normalize and scale
            length = math.sqrt(x*x + y*y + z*z)
            if length > 0:
                x, y, z = x/length, y/length, z/length
            
            # Scale by radius and offset by center
            point = [
                center[0] + x * radius,
                center[1] + y * radius,
                center[2] + z * radius
            ]
            yield point
    
    # Test generators
    print("   Fibonacci sequence (first 10):")
    fib_gen = generate_fibonacci_sequence(10)
    fib_list = list(fib_gen)
    print(f"     {fib_list}")
    
    print("   Random 3D points in sphere:")
    points_gen = generate_3d_points([0, 0, 0], 5.0, 5)
    for i, point in enumerate(points_gen):
        print(f"     Point {i+1}: {[f'{x:.2f}' for x in point]}")

def main():
    """Main function to run all parameter and return value demonstrations"""
    print("=== Python Parameters and Return Values for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_parameter_types()
    demonstrate_return_values()
    demonstrate_advanced_parameters()
    demonstrate_return_patterns()
    
    print("\n=== Summary ===")
    print("This chapter covered parameter and return value concepts:")
    print("✓ Required, default, and keyword parameters")
    print("✓ Variable arguments (*args) and keyword arguments (**kwargs)")
    print("✓ Single and multiple return values")
    print("✓ Return different data types and functions")
    print("✓ Parameter unpacking and lambda functions")
    print("✓ Common return patterns (success/error, result objects, generators)")
    
    print("\nParameter and return value techniques are essential for:")
    print("- Creating flexible and reusable functions")
    print("- Building robust APIs and libraries")
    print("- Handling errors and edge cases")
    print("- Optimizing performance with generators")
    print("- Creating clean and maintainable code")

if __name__ == "__main__":
    main()
