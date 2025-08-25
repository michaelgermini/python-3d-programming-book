#!/usr/bin/env python3
"""
Chapter 4: Functions
3D Math Functions Example

This example demonstrates a practical 3D math function library for graphics
applications, showcasing how functions can be organized into reusable modules.
"""

import math

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def vector_add(v1, v2):
    """Add two 3D vectors"""
    return [v1[i] + v2[i] for i in range(3)]

def vector_subtract(v1, v2):
    """Subtract two 3D vectors"""
    return [v1[i] - v2[i] for i in range(3)]

def vector_scale(v, scalar):
    """Scale a 3D vector by a scalar"""
    return [v[i] * scalar for i in range(3)]

def vector_dot(v1, v2):
    """Calculate dot product of two 3D vectors"""
    return sum(v1[i] * v2[i] for i in range(3))

def vector_cross(v1, v2):
    """Calculate cross product of two 3D vectors"""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def vector_magnitude(v):
    """Calculate magnitude (length) of a 3D vector"""
    return math.sqrt(sum(v[i] ** 2 for i in range(3)))

def vector_normalize(v):
    """Normalize a 3D vector (make it unit length)"""
    mag = vector_magnitude(v)
    if mag == 0:
        return [0, 0, 0]
    return [v[i] / mag for i in range(3)]

def vector_distance(v1, v2):
    """Calculate distance between two 3D points"""
    return vector_magnitude(vector_subtract(v2, v1))

def vector_lerp(v1, v2, t):
    """Linear interpolation between two 3D vectors"""
    return [v1[i] + (v2[i] - v1[i]) * t for i in range(3)]

# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def matrix_identity():
    """Create 4x4 identity matrix"""
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

def matrix_multiply(m1, m2):
    """Multiply two 4x4 matrices"""
    result = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def matrix_translate(x, y, z):
    """Create translation matrix"""
    return [
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ]

def matrix_scale(x, y, z):
    """Create scaling matrix"""
    return [
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ]

def matrix_rotate_x(angle):
    """Create rotation matrix around X-axis"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [1, 0, 0, 0],
        [0, cos_a, -sin_a, 0],
        [0, sin_a, cos_a, 0],
        [0, 0, 0, 1]
    ]

def matrix_rotate_y(angle):
    """Create rotation matrix around Y-axis"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [cos_a, 0, sin_a, 0],
        [0, 1, 0, 0],
        [-sin_a, 0, cos_a, 0],
        [0, 0, 0, 1]
    ]

def matrix_rotate_z(angle):
    """Create rotation matrix around Z-axis"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

def matrix_transform_point(matrix, point):
    """Transform a 3D point by a 4x4 matrix"""
    # Convert point to homogeneous coordinates
    homogeneous = [point[0], point[1], point[2], 1]
    
    # Apply transformation
    result = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            result[i] += matrix[i][j] * homogeneous[j]
    
    # Convert back to 3D coordinates
    if result[3] != 0:
        return [result[0] / result[3], result[1] / result[3], result[2] / result[3]]
    else:
        return [result[0], result[1], result[2]]

# ============================================================================
# QUATERNION OPERATIONS
# ============================================================================

def quaternion_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle representation"""
    half_angle = angle / 2
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    
    normalized_axis = vector_normalize(axis)
    return [
        cos_half,
        normalized_axis[0] * sin_half,
        normalized_axis[1] * sin_half,
        normalized_axis[2] * sin_half
    ]

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    return [
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    ]

def quaternion_rotate_vector(quat, vector):
    """Rotate a 3D vector by a quaternion"""
    # Convert vector to quaternion (w=0)
    vec_quat = [0, vector[0], vector[1], vector[2]]
    
    # q * v * q^(-1)
    q_conjugate = [quat[0], -quat[1], -quat[2], -quat[3]]
    temp = quaternion_multiply(quat, vec_quat)
    result = quaternion_multiply(temp, q_conjugate)
    
    return [result[1], result[2], result[3]]

# ============================================================================
# GEOMETRY FUNCTIONS
# ============================================================================

def point_in_sphere(point, sphere_center, sphere_radius):
    """Check if a point is inside a sphere"""
    distance = vector_distance(point, sphere_center)
    return distance <= sphere_radius

def point_in_box(point, box_min, box_max):
    """Check if a point is inside an axis-aligned bounding box"""
    return all(box_min[i] <= point[i] <= box_max[i] for i in range(3))

def sphere_sphere_intersection(center1, radius1, center2, radius2):
    """Check if two spheres intersect"""
    distance = vector_distance(center1, center2)
    return distance <= (radius1 + radius2)

def box_box_intersection(box1_min, box1_max, box2_min, box2_max):
    """Check if two axis-aligned bounding boxes intersect"""
    return all(
        box1_min[i] <= box2_max[i] and box2_min[i] <= box1_max[i]
        for i in range(3)
    )

def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    """Find intersection of ray with sphere"""
    # Vector from ray origin to sphere center
    oc = vector_subtract(ray_origin, sphere_center)
    
    # Project oc onto ray direction
    a = vector_dot(ray_direction, ray_direction)
    b = 2.0 * vector_dot(oc, ray_direction)
    c = vector_dot(oc, oc) - sphere_radius * sphere_radius
    
    # Discriminant
    discriminant = b * b - 4 * a * c
    
    if discriminant < 0:
        return None  # No intersection
    
    # Find intersection points
    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)
    
    # Return closest positive intersection
    if t1 > 0:
        return vector_add(ray_origin, vector_scale(ray_direction, t1))
    elif t2 > 0:
        return vector_add(ray_origin, vector_scale(ray_direction, t2))
    else:
        return None

# ============================================================================
# INTERPOLATION AND EASING FUNCTIONS
# ============================================================================

def lerp(a, b, t):
    """Linear interpolation between two values"""
    return a + (b - a) * t

def smoothstep(edge0, edge1, x):
    """Smooth interpolation function"""
    t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
    return t * t * (3 - 2 * t)

def ease_in_out(t):
    """Ease-in-out function"""
    return t * t * (3 - 2 * t)

def ease_in(t):
    """Ease-in function"""
    return t * t

def ease_out(t):
    """Ease-out function"""
    return 1 - (1 - t) * (1 - t)

def bezier_interpolate(p0, p1, p2, p3, t):
    """Cubic Bezier interpolation between four points"""
    # Cubic Bezier formula
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    
    return [
        mt3 * p0[i] + 3 * mt2 * t * p1[i] + 3 * mt * t2 * p2[i] + t3 * p3[i]
        for i in range(3)
    ]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clamp(value, min_val, max_val):
    """Clamp a value between min and max"""
    return max(min_val, min(value, max_val))

def clamp_vector(vector, min_val, max_val):
    """Clamp all components of a vector"""
    return [clamp(vector[i], min_val, max_val) for i in range(3)]

def degrees_to_radians(degrees):
    """Convert degrees to radians"""
    return degrees * math.pi / 180

def radians_to_degrees(radians):
    """Convert radians to degrees"""
    return radians * 180 / math.pi

def random_unit_vector():
    """Generate a random unit vector"""
    import random
    
    # Generate random spherical coordinates
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(random.uniform(-1, 1))
    
    # Convert to Cartesian coordinates
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    
    return [x, y, z]

def random_point_in_sphere(center, radius):
    """Generate a random point inside a sphere"""
    import random
    
    # Generate random direction
    direction = random_unit_vector()
    
    # Generate random distance (cube root for uniform distribution)
    distance = radius * (random.random() ** (1/3))
    
    # Scale direction by distance
    offset = vector_scale(direction, distance)
    
    return vector_add(center, offset)

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_vector_operations():
    """Demonstrate vector operations"""
    print("=== Vector Operations ===\n")
    
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {vector_add(v1, v2)}")
    print(f"v1 - v2 = {vector_subtract(v1, v2)}")
    print(f"v1 * 2 = {vector_scale(v1, 2)}")
    print(f"v1 · v2 = {vector_dot(v1, v2)}")
    print(f"v1 × v2 = {vector_cross(v1, v2)}")
    print(f"|v1| = {vector_magnitude(v1):.3f}")
    print(f"normalize(v1) = {[f'{x:.3f}' for x in vector_normalize(v1)]}")
    print(f"distance(v1, v2) = {vector_distance(v1, v2):.3f}")

def demonstrate_matrix_operations():
    """Demonstrate matrix operations"""
    print("\n=== Matrix Operations ===\n")
    
    # Create transformation matrices
    translate_matrix = matrix_translate(5, 0, 0)
    scale_matrix = matrix_scale(2, 1, 1)
    rotate_matrix = matrix_rotate_z(degrees_to_radians(45))
    
    point = [1, 0, 0]
    
    print(f"Original point: {point}")
    
    # Apply transformations
    translated = matrix_transform_point(translate_matrix, point)
    print(f"After translation: {[f'{x:.3f}' for x in translated]}")
    
    scaled = matrix_transform_point(scale_matrix, point)
    print(f"After scaling: {[f'{x:.3f}' for x in scaled]}")
    
    rotated = matrix_transform_point(rotate_matrix, point)
    print(f"After rotation: {[f'{x:.3f}' for x in rotated]}")
    
    # Combine transformations
    combined = matrix_multiply(translate_matrix, matrix_multiply(scale_matrix, rotate_matrix))
    transformed = matrix_transform_point(combined, point)
    print(f"After combined transform: {[f'{x:.3f}' for x in transformed]}")

def demonstrate_quaternion_operations():
    """Demonstrate quaternion operations"""
    print("\n=== Quaternion Operations ===\n")
    
    # Create quaternion for 90-degree rotation around Z-axis
    axis = [0, 0, 1]
    angle = degrees_to_radians(90)
    quat = quaternion_from_axis_angle(axis, angle)
    
    print(f"Quaternion for 90° Z rotation: {[f'{x:.3f}' for x in quat]}")
    
    # Rotate a vector
    vector = [1, 0, 0]
    rotated_vector = quaternion_rotate_vector(quat, vector)
    print(f"Rotating {vector} by quaternion: {[f'{x:.3f}' for x in rotated_vector]}")

def demonstrate_geometry_functions():
    """Demonstrate geometry functions"""
    print("\n=== Geometry Functions ===\n")
    
    # Sphere intersection test
    sphere_center = [0, 0, 0]
    sphere_radius = 5
    point_inside = [1, 1, 1]
    point_outside = [10, 10, 10]
    
    print(f"Point {point_inside} in sphere: {point_in_sphere(point_inside, sphere_center, sphere_radius)}")
    print(f"Point {point_outside} in sphere: {point_in_sphere(point_outside, sphere_center, sphere_radius)}")
    
    # Box intersection test
    box_min = [-1, -1, -1]
    box_max = [1, 1, 1]
    point_in_box_test = [0.5, 0.5, 0.5]
    
    print(f"Point {point_in_box_test} in box: {point_in_box(point_in_box_test, box_min, box_max)}")
    
    # Ray-sphere intersection
    ray_origin = [0, 0, -10]
    ray_direction = [0, 0, 1]
    intersection = ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius)
    print(f"Ray-sphere intersection: {intersection}")

def demonstrate_interpolation():
    """Demonstrate interpolation functions"""
    print("\n=== Interpolation Functions ===\n")
    
    start_point = [0, 0, 0]
    end_point = [10, 10, 10]
    
    print("Linear interpolation:")
    for t in [0, 0.25, 0.5, 0.75, 1]:
        point = vector_lerp(start_point, end_point, t)
        print(f"  t={t}: {[f'{x:.1f}' for x in point]}")
    
    print("\nBezier interpolation:")
    control1 = [5, 0, 0]
    control2 = [5, 10, 0]
    for t in [0, 0.25, 0.5, 0.75, 1]:
        point = bezier_interpolate(start_point, control1, control2, end_point, t)
        print(f"  t={t}: {[f'{x:.1f}' for x in point]}")

def demonstrate_utility_functions():
    """Demonstrate utility functions"""
    print("\n=== Utility Functions ===\n")
    
    # Clamping
    value = 15
    clamped = clamp(value, 0, 10)
    print(f"clamp({value}, 0, 10) = {clamped}")
    
    # Vector clamping
    vector = [15, -5, 8]
    clamped_vector = clamp_vector(vector, 0, 10)
    print(f"clamp_vector({vector}, 0, 10) = {clamped_vector}")
    
    # Angle conversion
    degrees = 45
    radians = degrees_to_radians(degrees)
    back_to_degrees = radians_to_degrees(radians)
    print(f"{degrees}° = {radians:.3f} radians = {back_to_degrees:.1f}°")
    
    # Random vectors
    try:
        random_vec = random_unit_vector()
        print(f"Random unit vector: {[f'{x:.3f}' for x in random_vec]}")
        
        random_point = random_point_in_sphere([0, 0, 0], 5)
        print(f"Random point in sphere: {[f'{x:.3f}' for x in random_point]}")
    except ImportError:
        print("Random functions require random module")

def demonstrate_performance_example():
    """Demonstrate performance considerations"""
    print("\n=== Performance Example ===\n")
    
    import time
    
    # Test vector operations performance
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    start_time = time.time()
    for _ in range(100000):
        result = vector_add(v1, v2)
    end_time = time.time()
    
    print(f"100,000 vector additions: {(end_time - start_time)*1000:.2f} ms")
    
    # Test matrix multiplication performance
    m1 = matrix_identity()
    m2 = matrix_translate(1, 2, 3)
    
    start_time = time.time()
    for _ in range(10000):
        result = matrix_multiply(m1, m2)
    end_time = time.time()
    
    print(f"10,000 matrix multiplications: {(end_time - start_time)*1000:.2f} ms")

def main():
    """Main function to demonstrate 3D math functions"""
    print("=== 3D Math Functions Library ===\n")
    
    # Run all demonstrations
    demonstrate_vector_operations()
    demonstrate_matrix_operations()
    demonstrate_quaternion_operations()
    demonstrate_geometry_functions()
    demonstrate_interpolation()
    demonstrate_utility_functions()
    demonstrate_performance_example()
    
    print("\n=== Summary ===")
    print("This library provides essential 3D math functions:")
    print("✓ Vector operations (add, subtract, scale, dot, cross, normalize)")
    print("✓ Matrix operations (identity, multiply, transform, rotation)")
    print("✓ Quaternion operations (creation, multiplication, rotation)")
    print("✓ Geometry functions (intersection tests, ray casting)")
    print("✓ Interpolation functions (linear, Bezier, easing)")
    print("✓ Utility functions (clamping, angle conversion, random generation)")
    
    print("\nThese functions are essential for:")
    print("- 3D graphics and game development")
    print("- Physics simulations")
    print("- Animation systems")
    print("- Camera and view transformations")
    print("- Collision detection")
    print("- Procedural content generation")

if __name__ == "__main__":
    main()
