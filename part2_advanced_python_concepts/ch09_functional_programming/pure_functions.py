"""
Chapter 9: Functional Programming - Pure Functions
=================================================

This module demonstrates pure functions in Python, applied to 3D graphics
and mathematical operations. Pure functions have no side effects and
always return the same output for the same input.

Key Concepts:
- Pure functions (no side effects)
- Referential transparency
- Immutable data structures
- Mathematical functions
- Function purity testing
"""

import math
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class Vector3D:
    """Immutable 3D vector ensuring pure function behavior."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass(frozen=True)
class Color:
    """Immutable color representation."""
    r: float  # 0.0 to 1.0
    g: float  # 0.0 to 1.0
    b: float  # 0.0 to 1.0
    a: float = 1.0  # 0.0 to 1.0
    
    def __str__(self):
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"


# Pure Mathematical Functions
def calculate_distance(v1: Vector3D, v2: Vector3D) -> float:
    """
    Pure function: calculates Euclidean distance between two 3D points.
    
    Args:
        v1: First 3D vector
        v2: Second 3D vector
    
    Returns:
        Distance between the two points
    """
    diff = v1 - v2
    return diff.magnitude()


def calculate_angle_between_vectors(v1: Vector3D, v2: Vector3D) -> float:
    """
    Pure function: calculates angle between two 3D vectors.
    
    Args:
        v1: First 3D vector
        v2: Second 3D vector
    
    Returns:
        Angle in radians between the vectors
    """
    dot_product = v1.dot(v2)
    mag1 = v1.magnitude()
    mag2 = v2.magnitude()
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cos_angle = dot_product / (mag1 * mag2)
    # Clamp to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def interpolate_vectors(v1: Vector3D, v2: Vector3D, t: float) -> Vector3D:
    """
    Pure function: linear interpolation between two 3D vectors.
    
    Args:
        v1: First 3D vector
        v2: Second 3D vector
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated vector
    """
    t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
    return Vector3D(
        v1.x + (v2.x - v1.x) * t,
        v1.y + (v2.y - v1.y) * t,
        v1.z + (v2.z - v1.z) * t
    )


def calculate_centroid(points: List[Vector3D]) -> Vector3D:
    """
    Pure function: calculates the centroid of a list of 3D points.
    
    Args:
        points: List of 3D points
    
    Returns:
        Centroid (average) of all points
    """
    if not points:
        return Vector3D(0, 0, 0)
    
    total_x = sum(point.x for point in points)
    total_y = sum(point.y for point in points)
    total_z = sum(point.z for point in points)
    count = len(points)
    
    return Vector3D(total_x / count, total_y / count, total_z / count)


def calculate_bounding_box(points: List[Vector3D]) -> Tuple[Vector3D, Vector3D]:
    """
    Pure function: calculates the bounding box of a list of 3D points.
    
    Args:
        points: List of 3D points
    
    Returns:
        Tuple of (min_point, max_point) defining the bounding box
    """
    if not points:
        return Vector3D(0, 0, 0), Vector3D(0, 0, 0)
    
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    min_z = min(point.z for point in points)
    
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    max_z = max(point.z for point in points)
    
    return Vector3D(min_x, min_y, min_z), Vector3D(max_x, max_y, max_z)


# Pure Color Functions
def blend_colors(color1: Color, color2: Color, factor: float) -> Color:
    """
    Pure function: blends two colors using linear interpolation.
    
    Args:
        color1: First color
        color2: Second color
        factor: Blend factor (0.0 to 1.0)
    
    Returns:
        Blended color
    """
    factor = max(0.0, min(1.0, factor))
    
    return Color(
        color1.r + (color2.r - color1.r) * factor,
        color1.g + (color2.g - color1.g) * factor,
        color1.b + (color2.b - color1.b) * factor,
        color1.a + (color2.a - color1.a) * factor
    )


def adjust_brightness(color: Color, factor: float) -> Color:
    """
    Pure function: adjusts the brightness of a color.
    
    Args:
        color: Input color
        factor: Brightness factor (> 1.0 brightens, < 1.0 darkens)
    
    Returns:
        Adjusted color
    """
    return Color(
        max(0.0, min(1.0, color.r * factor)),
        max(0.0, min(1.0, color.g * factor)),
        max(0.0, min(1.0, color.b * factor)),
        color.a
    )


def calculate_luminance(color: Color) -> float:
    """
    Pure function: calculates the luminance of a color.
    
    Args:
        color: Input color
    
    Returns:
        Luminance value (0.0 to 1.0)
    """
    return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b


# Pure Geometry Functions
def calculate_triangle_area(v1: Vector3D, v2: Vector3D, v3: Vector3D) -> float:
    """
    Pure function: calculates the area of a triangle defined by three points.
    
    Args:
        v1: First vertex
        v2: Second vertex
        v3: Third vertex
    
    Returns:
        Area of the triangle
    """
    # Calculate two edge vectors
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    # Calculate cross product
    cross = edge1.cross(edge2)
    
    # Area is half the magnitude of the cross product
    return cross.magnitude() / 2.0


def calculate_triangle_normal(v1: Vector3D, v2: Vector3D, v3: Vector3D) -> Vector3D:
    """
    Pure function: calculates the normal vector of a triangle.
    
    Args:
        v1: First vertex
        v2: Second vertex
        v3: Third vertex
    
    Returns:
        Normalized normal vector
    """
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = edge1.cross(edge2)
    return normal.normalize()


def point_in_triangle(point: Vector3D, v1: Vector3D, v2: Vector3D, v3: Vector3D) -> bool:
    """
    Pure function: determines if a point is inside a triangle.
    
    Args:
        point: Point to test
        v1: First vertex of triangle
        v2: Second vertex of triangle
        v3: Third vertex of triangle
    
    Returns:
        True if point is inside triangle, False otherwise
    """
    # Calculate barycentric coordinates
    def sign(p1: Vector3D, p2: Vector3D, p3: Vector3D) -> float:
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
    
    d1 = sign(point, v1, v2)
    d2 = sign(point, v2, v3)
    d3 = sign(point, v3, v1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


# Pure Physics Functions
def calculate_reflection(incident: Vector3D, normal: Vector3D) -> Vector3D:
    """
    Pure function: calculates reflection vector.
    
    Args:
        incident: Incident vector
        normal: Surface normal (must be normalized)
    
    Returns:
        Reflection vector
    """
    # R = I - 2(N · I)N
    dot_product = normal.dot(incident)
    return incident - normal * (2 * dot_product)


def calculate_refraction(incident: Vector3D, normal: Vector3D, 
                        ior_ratio: float) -> Vector3D:
    """
    Pure function: calculates refraction vector.
    
    Args:
        incident: Incident vector (must be normalized)
        normal: Surface normal (must be normalized)
        ior_ratio: Index of refraction ratio (n1/n2)
    
    Returns:
        Refraction vector
    """
    cos_i = normal.dot(incident)
    sin_t2 = ior_ratio * ior_ratio * (1.0 - cos_i * cos_i)
    
    if sin_t2 > 1.0:
        # Total internal reflection
        return calculate_reflection(incident, normal)
    
    cos_t = math.sqrt(1.0 - sin_t2)
    return incident * ior_ratio - normal * (ior_ratio * cos_i + cos_t)


# Pure Utility Functions
def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Pure function: clamps a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Pure function: smoothstep interpolation.
    
    Args:
        edge0: Lower edge
        edge1: Upper edge
        x: Input value
    
    Returns:
        Smoothly interpolated value
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    """
    Pure function: linear interpolation.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0.0 to 1.0)
    
    Returns:
        Interpolated value
    """
    return a + (b - a) * clamp(t, 0.0, 1.0)


# Function Purity Testing
def test_function_purity(func: Callable, test_inputs: List[Any], 
                        num_runs: int = 3) -> bool:
    """
    Tests if a function is pure by running it multiple times with the same inputs.
    
    Args:
        func: Function to test
        test_inputs: List of test input arguments
        num_runs: Number of times to run the function
    
    Returns:
        True if function appears pure, False otherwise
    """
    results = []
    
    for _ in range(num_runs):
        try:
            result = func(*test_inputs)
            results.append(result)
        except Exception:
            return False
    
    # Check if all results are the same
    return all(result == results[0] for result in results)


# Example Usage and Demonstration
def demonstrate_pure_functions():
    """Demonstrates pure functions with 3D examples."""
    print("=== Pure Functions with 3D Applications ===\n")
    
    # Create sample data
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    v3 = Vector3D(7, 8, 9)
    
    points = [v1, v2, v3, Vector3D(10, 11, 12)]
    
    color1 = Color(1.0, 0.0, 0.0)  # Red
    color2 = Color(0.0, 0.0, 1.0)  # Blue
    
    print("Sample vectors:")
    print(f"  v1: {v1}")
    print(f"  v2: {v2}")
    print(f"  v3: {v3}")
    
    # Demonstrate mathematical functions
    print("\n=== Mathematical Functions ===")
    
    distance = calculate_distance(v1, v2)
    print(f"Distance between v1 and v2: {distance:.3f}")
    
    angle = calculate_angle_between_vectors(v1, v2)
    print(f"Angle between v1 and v2: {math.degrees(angle):.1f}°")
    
    interpolated = interpolate_vectors(v1, v2, 0.5)
    print(f"Interpolated vector (t=0.5): {interpolated}")
    
    centroid = calculate_centroid(points)
    print(f"Centroid of all points: {centroid}")
    
    min_point, max_point = calculate_bounding_box(points)
    print(f"Bounding box: min={min_point}, max={max_point}")
    
    # Demonstrate color functions
    print("\n=== Color Functions ===")
    
    blended = blend_colors(color1, color2, 0.5)
    print(f"Blended color (50%): {blended}")
    
    brightened = adjust_brightness(color1, 1.5)
    print(f"Brightened red: {brightened}")
    
    luminance = calculate_luminance(color1)
    print(f"Luminance of red: {luminance:.3f}")
    
    # Demonstrate geometry functions
    print("\n=== Geometry Functions ===")
    
    area = calculate_triangle_area(v1, v2, v3)
    print(f"Triangle area: {area:.3f}")
    
    normal = calculate_triangle_normal(v1, v2, v3)
    print(f"Triangle normal: {normal}")
    
    test_point = Vector3D(4, 5, 6)
    inside = point_in_triangle(test_point, v1, v2, v3)
    print(f"Point {test_point} in triangle: {inside}")
    
    # Demonstrate physics functions
    print("\n=== Physics Functions ===")
    
    incident = Vector3D(1, -1, 0).normalize()
    surface_normal = Vector3D(0, 1, 0)
    
    reflection = calculate_reflection(incident, surface_normal)
    print(f"Incident: {incident}")
    print(f"Surface normal: {surface_normal}")
    print(f"Reflection: {reflection}")
    
    refraction = calculate_refraction(incident, surface_normal, 1.5)
    print(f"Refraction (IOR=1.5): {refraction}")
    
    # Demonstrate utility functions
    print("\n=== Utility Functions ===")
    
    clamped = clamp(1.5, 0.0, 1.0)
    print(f"Clamped 1.5 to [0,1]: {clamped}")
    
    smoothed = smoothstep(0.0, 1.0, 0.5)
    print(f"Smoothstep(0.5): {smoothed:.3f}")
    
    interpolated_val = lerp(0.0, 100.0, 0.25)
    print(f"Lerp(0, 100, 0.25): {interpolated_val}")
    
    # Test function purity
    print("\n=== Function Purity Testing ===")
    
    pure_functions = [
        (calculate_distance, [v1, v2]),
        (calculate_centroid, [points]),
        (blend_colors, [color1, color2, 0.5]),
        (clamp, [1.5, 0.0, 1.0])
    ]
    
    for func, args in pure_functions:
        is_pure = test_function_purity(func, args)
        print(f"{func.__name__} is pure: {is_pure}")


if __name__ == "__main__":
    demonstrate_pure_functions()
