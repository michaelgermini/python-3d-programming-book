#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
Mathematical Operations Example

This example demonstrates mathematical operations in Python, focusing on
applications in 3D graphics programming and game development.
"""

import math
import random

def demonstrate_basic_math():
    """Demonstrate basic mathematical operations"""
    print("=== Basic Mathematical Operations ===\n")
    
    # 1. Basic arithmetic
    print("1. Basic Arithmetic:")
    
    # Object dimensions
    width = 10
    height = 5
    depth = 3
    
    # Calculate volume
    volume = width * height * depth
    print(f"   Object dimensions: {width} x {height} x {depth}")
    print(f"   Volume: {width} * {height} * {depth} = {volume}")
    
    # Calculate surface area
    surface_area = 2 * (width * height + width * depth + height * depth)
    print(f"   Surface area: 2 * ({width}*{height} + {width}*{depth} + {height}*{depth}) = {surface_area}")
    
    # 2. Division and modulo
    print("\n2. Division and Modulo:")
    
    # Calculate frames per second
    total_frames = 3600  # 1 minute at 60 FPS
    fps = 60
    
    seconds = total_frames // fps  # Integer division
    remaining_frames = total_frames % fps  # Modulo
    
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Seconds: {total_frames} // {fps} = {seconds}")
    print(f"   Remaining frames: {total_frames} % {fps} = {remaining_frames}")
    
    # 3. Exponentiation
    print("\n3. Exponentiation:")
    
    # Calculate distance using Pythagorean theorem
    x = 3
    y = 4
    distance = math.sqrt(x**2 + y**2)
    print(f"   Point: ({x}, {y})")
    print(f"   Distance from origin: √({x}² + {y}²) = √{x**2 + y**2} = {distance}")
    
    # Calculate area of a circle
    radius = 5
    area = math.pi * radius**2
    print(f"   Circle radius: {radius}")
    print(f"   Area: π * {radius}² = {math.pi} * {radius**2} = {area:.2f}")

def demonstrate_3d_math():
    """Demonstrate 3D mathematical operations"""
    print("\n=== 3D Mathematical Operations ===\n")
    
    # 1. Vector operations
    print("1. Vector Operations:")
    
    # 3D vectors
    vector_a = [1, 2, 3]
    vector_b = [4, 5, 6]
    
    # Vector addition
    sum_vector = [vector_a[0] + vector_b[0], 
                  vector_a[1] + vector_b[1], 
                  vector_a[2] + vector_b[2]]
    print(f"   Vector A: {vector_a}")
    print(f"   Vector B: {vector_b}")
    print(f"   A + B = {sum_vector}")
    
    # Vector magnitude
    magnitude_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[2]**2)
    print(f"   Magnitude of A: √({vector_a[0]}² + {vector_a[1]}² + {vector_a[2]}²) = {magnitude_a:.3f}")
    
    # 2. Distance calculations
    print("\n2. Distance Calculations:")
    
    # 3D points
    point1 = [0, 0, 0]
    point2 = [3, 4, 5]
    
    # Euclidean distance
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    print(f"   Point 1: {point1}")
    print(f"   Point 2: {point2}")
    print(f"   Distance: √({dx}² + {dy}² + {dz}²) = √{dx**2 + dy**2 + dz**2} = {distance:.3f}")
    
    # 3. Rotation calculations
    print("\n3. Rotation Calculations:")
    
    # Convert degrees to radians
    angle_degrees = 45
    angle_radians = math.radians(angle_degrees)
    
    print(f"   Angle in degrees: {angle_degrees}°")
    print(f"   Angle in radians: {angle_degrees}° * π/180 = {angle_radians:.3f}")
    
    # Calculate sine and cosine
    sin_value = math.sin(angle_radians)
    cos_value = math.cos(angle_radians)
    
    print(f"   sin({angle_degrees}°) = {sin_value:.3f}")
    print(f"   cos({angle_degrees}°) = {cos_value:.3f}")

def demonstrate_advanced_math():
    """Demonstrate advanced mathematical operations"""
    print("\n=== Advanced Mathematical Operations ===\n")
    
    # 1. Interpolation
    print("1. Linear Interpolation:")
    
    # Interpolate between two values
    start_value = 0
    end_value = 100
    t = 0.3  # Interpolation factor (0 to 1)
    
    interpolated = start_value + t * (end_value - start_value)
    print(f"   Start value: {start_value}")
    print(f"   End value: {end_value}")
    print(f"   Interpolation factor: {t}")
    print(f"   Interpolated value: {start_value} + {t} * ({end_value} - {start_value}) = {interpolated}")
    
    # 2. Clamping values
    print("\n2. Value Clamping:")
    
    def clamp(value, min_val, max_val):
        """Clamp a value between min and max"""
        return max(min_val, min(value, max_val))
    
    # Test clamping
    test_values = [-5, 0, 50, 150, 200]
    min_val = 0
    max_val = 100
    
    print(f"   Clamping range: [{min_val}, {max_val}]")
    for value in test_values:
        clamped = clamp(value, min_val, max_val)
        print(f"   clamp({value}, {min_val}, {max_val}) = {clamped}")
    
    # 3. Random number generation
    print("\n3. Random Number Generation:")
    
    # Generate random positions
    min_pos = -10
    max_pos = 10
    
    random_positions = []
    for i in range(3):
        x = random.uniform(min_pos, max_pos)
        y = random.uniform(min_pos, max_pos)
        z = random.uniform(min_pos, max_pos)
        random_positions.append([x, y, z])
    
    print(f"   Random positions in range [{min_pos}, {max_pos}]:")
    for i, pos in enumerate(random_positions):
        print(f"     Position {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

def demonstrate_math_functions():
    """Demonstrate useful math functions"""
    print("\n=== Math Functions ===\n")
    
    # 1. Rounding functions
    print("1. Rounding Functions:")
    
    pi = math.pi
    print(f"   π = {pi}")
    print(f"   round(π, 2) = {round(pi, 2)}")
    print(f"   math.floor(π) = {math.floor(pi)}")
    print(f"   math.ceil(π) = {math.ceil(pi)}")
    
    # 2. Absolute value
    print("\n2. Absolute Value:")
    
    negative_values = [-5, -3.14, -100]
    for value in negative_values:
        abs_value = abs(value)
        print(f"   abs({value}) = {abs_value}")
    
    # 3. Power and square root
    print("\n3. Power and Square Root:")
    
    base = 2
    exponent = 8
    power_result = base ** exponent
    sqrt_result = math.sqrt(power_result)
    
    print(f"   {base}^{exponent} = {power_result}")
    print(f"   √{power_result} = {sqrt_result}")
    
    # 4. Trigonometric functions
    print("\n4. Trigonometric Functions:")
    
    angles = [0, 30, 45, 60, 90]
    for angle in angles:
        radians = math.radians(angle)
        sin_val = math.sin(radians)
        cos_val = math.cos(radians)
        print(f"   {angle}°: sin = {sin_val:.3f}, cos = {cos_val:.3f}")

def demonstrate_practical_examples():
    """Demonstrate practical mathematical examples in 3D graphics"""
    print("\n=== Practical 3D Graphics Examples ===\n")
    
    # 1. Camera field of view calculation
    print("1. Camera Field of View:")
    
    # Calculate field of view in radians
    fov_degrees = 60
    fov_radians = math.radians(fov_degrees)
    
    # Calculate tangent for perspective projection
    tan_half_fov = math.tan(fov_radians / 2)
    
    print(f"   Field of view: {fov_degrees}°")
    print(f"   FOV in radians: {fov_radians:.3f}")
    print(f"   tan(FOV/2) = {tan_half_fov:.3f}")
    
    # 2. Bounding sphere calculation
    print("\n2. Bounding Sphere:")
    
    # Object vertices (simplified)
    vertices = [
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
    ]
    
    # Calculate center (average of all vertices)
    center_x = sum(v[0] for v in vertices) / len(vertices)
    center_y = sum(v[1] for v in vertices) / len(vertices)
    center_z = sum(v[2] for v in vertices) / len(vertices)
    center = [center_x, center_y, center_z]
    
    # Calculate radius (maximum distance from center)
    max_distance = 0
    for vertex in vertices:
        dx = vertex[0] - center[0]
        dy = vertex[1] - center[1]
        dz = vertex[2] - center[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        max_distance = max(max_distance, distance)
    
    print(f"   Object vertices: {vertices}")
    print(f"   Bounding sphere center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"   Bounding sphere radius: {max_distance:.2f}")
    
    # 3. Animation easing
    print("\n3. Animation Easing:")
    
    # Smooth step function (ease in-out)
    def smooth_step(t):
        """Smooth step function for easing"""
        return t * t * (3 - 2 * t)
    
    # Test easing at different times
    times = [0, 0.25, 0.5, 0.75, 1.0]
    print("   Smooth step easing:")
    for t in times:
        eased = smooth_step(t)
        print(f"     t={t}: {eased:.3f}")

def main():
    """Main function to run all mathematical demonstrations"""
    print("=== Python Mathematical Operations for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_math()
    demonstrate_3d_math()
    demonstrate_advanced_math()
    demonstrate_math_functions()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered mathematical operations:")
    print("✓ Basic arithmetic (addition, subtraction, multiplication, division)")
    print("✓ Advanced operations (exponentiation, modulo, square root)")
    print("✓ 3D vector operations and distance calculations")
    print("✓ Trigonometric functions for rotations")
    print("✓ Interpolation and value clamping")
    print("✓ Random number generation")
    print("✓ Practical applications in 3D graphics")
    
    print("\nThese mathematical concepts are essential for:")
    print("- Calculating object positions and movements")
    print("- Implementing camera systems and projections")
    print("- Creating smooth animations and transitions")
    print("- Performing collision detection and physics")
    print("- Generating procedural content")

if __name__ == "__main__":
    main()

