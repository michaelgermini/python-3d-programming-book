"""
Chapter 9: Functional Programming - Functional Basics
====================================================

This module demonstrates the core concepts of functional programming in Python,
applied to 3D graphics and mathematical operations.

Key Concepts:
- Pure functions
- Immutability
- Function composition
- Higher-order functions
- Lambda expressions
"""

import math
from typing import List, Tuple, Callable, Any
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class Vector3D:
    """Immutable 3D vector representing functional programming principles."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Pure function: creates new vector without modifying original."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        """Pure function: scalar multiplication."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        """Pure function: calculates vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        """Pure function: returns normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)


# Pure Functions Examples
def calculate_distance(v1: Vector3D, v2: Vector3D) -> float:
    """
    Pure function: calculates distance between two 3D points.
    
    Args:
        v1: First 3D vector
        v2: Second 3D vector
    
    Returns:
        Distance between the two points
    """
    diff = Vector3D(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)
    return diff.magnitude()


def rotate_point_around_axis(point: Vector3D, axis: Vector3D, angle: float) -> Vector3D:
    """
    Pure function: rotates a point around an axis using Rodrigues' rotation formula.
    
    Args:
        point: Point to rotate
        axis: Rotation axis (normalized)
        angle: Rotation angle in radians
    
    Returns:
        Rotated point
    """
    # Ensure axis is normalized
    axis = axis.normalize()
    
    # Rodrigues' rotation formula
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # v_rot = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
    # where k is the rotation axis
    
    # k·v (dot product)
    dot_product = axis.x * point.x + axis.y * point.y + axis.z * point.z
    
    # k×v (cross product)
    cross_x = axis.y * point.z - axis.z * point.y
    cross_y = axis.z * point.x - axis.x * point.z
    cross_z = axis.x * point.y - axis.y * point.x
    
    # Calculate rotated point
    rotated_x = point.x * cos_angle + cross_x * sin_angle + axis.x * dot_product * (1 - cos_angle)
    rotated_y = point.y * cos_angle + cross_y * sin_angle + axis.y * dot_product * (1 - cos_angle)
    rotated_z = point.z * cos_angle + cross_z * sin_angle + axis.z * dot_product * (1 - cos_angle)
    
    return Vector3D(rotated_x, rotated_y, rotated_z)


# Function Composition
def compose(*functions: Callable) -> Callable:
    """
    Higher-order function: composes multiple functions.
    
    Args:
        *functions: Functions to compose
    
    Returns:
        Composed function
    """
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner


# Example of function composition for 3D transformations
def scale_vector(scale_factor: float) -> Callable[[Vector3D], Vector3D]:
    """Returns a function that scales vectors by the given factor."""
    return lambda v: v * scale_factor


def translate_vector(offset: Vector3D) -> Callable[[Vector3D], Vector3D]:
    """Returns a function that translates vectors by the given offset."""
    return lambda v: v + offset


# Lambda Expressions and Higher-Order Functions
def apply_transformation_to_points(points: List[Vector3D], 
                                 transform: Callable[[Vector3D], Vector3D]) -> List[Vector3D]:
    """
    Higher-order function: applies a transformation to a list of points.
    
    Args:
        points: List of 3D points
        transform: Transformation function
    
    Returns:
        List of transformed points
    """
    return list(map(transform, points))


def filter_points_by_distance(points: List[Vector3D], 
                            center: Vector3D, 
                            max_distance: float) -> List[Vector3D]:
    """
    Higher-order function: filters points by distance from center.
    
    Args:
        points: List of 3D points
        center: Center point for distance calculation
        max_distance: Maximum allowed distance
    
    Returns:
        Filtered list of points
    """
    return list(filter(lambda p: calculate_distance(p, center) <= max_distance, points))


def reduce_points_to_center(points: List[Vector3D]) -> Vector3D:
    """
    Higher-order function: reduces list of points to their center.
    
    Args:
        points: List of 3D points
    
    Returns:
        Center point (average of all points)
    """
    if not points:
        return Vector3D(0, 0, 0)
    
    def add_vectors(v1: Vector3D, v2: Vector3D) -> Vector3D:
        return v1 + v2
    
    total = reduce(add_vectors, points)
    count = len(points)
    return Vector3D(total.x / count, total.y / count, total.z / count)


# Immutability and Data Processing
def process_3d_scene(points: List[Vector3D], 
                    transformations: List[Callable[[Vector3D], Vector3D]]) -> List[Vector3D]:
    """
    Demonstrates functional processing of 3D scene data.
    
    Args:
        points: Original 3D points
        transformations: List of transformation functions
    
    Returns:
        Processed points
    """
    # Compose all transformations
    combined_transform = compose(*transformations)
    
    # Apply transformation to all points
    transformed_points = apply_transformation_to_points(points, combined_transform)
    
    # Filter points within reasonable bounds
    center = reduce_points_to_center(transformed_points)
    filtered_points = filter_points_by_distance(transformed_points, center, 100.0)
    
    return filtered_points


# Example Usage and Demonstration
def demonstrate_functional_programming():
    """Demonstrates functional programming concepts with 3D examples."""
    print("=== Functional Programming Basics with 3D Applications ===\n")
    
    # Create some 3D points
    points = [
        Vector3D(1, 2, 3),
        Vector3D(4, 5, 6),
        Vector3D(7, 8, 9),
        Vector3D(10, 11, 12)
    ]
    
    print("Original points:")
    for i, point in enumerate(points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate pure functions
    print(f"\nDistance between first two points: {calculate_distance(points[0], points[1]):.2f}")
    
    # Demonstrate function composition
    scale_by_2 = scale_vector(2.0)
    translate_by_offset = translate_vector(Vector3D(1, 1, 1))
    
    # Compose transformations
    combined_transform = compose(scale_by_2, translate_by_offset)
    
    print("\nApplying combined transformation (scale by 2, then translate by (1,1,1)):")
    transformed_points = apply_transformation_to_points(points, combined_transform)
    for i, point in enumerate(transformed_points):
        print(f"  Transformed Point {i}: {point}")
    
    # Demonstrate filtering
    center = Vector3D(5, 5, 5)
    max_dist = 10.0
    filtered_points = filter_points_by_distance(points, center, max_dist)
    
    print(f"\nPoints within {max_dist} units of {center}:")
    for i, point in enumerate(filtered_points):
        print(f"  Filtered Point {i}: {point}")
    
    # Demonstrate reduction
    scene_center = reduce_points_to_center(points)
    print(f"\nScene center (average of all points): {scene_center}")
    
    # Demonstrate rotation
    rotation_axis = Vector3D(0, 0, 1)  # Z-axis
    rotation_angle = math.pi / 4  # 45 degrees
    
    print(f"\nRotating first point by {math.degrees(rotation_angle):.1f}° around Z-axis:")
    original_point = points[0]
    rotated_point = rotate_point_around_axis(original_point, rotation_axis, rotation_angle)
    print(f"  Original: {original_point}")
    print(f"  Rotated:  {rotated_point}")
    
    # Demonstrate complete scene processing
    print("\n=== Complete Scene Processing ===")
    transformations = [
        scale_vector(0.5),
        translate_vector(Vector3D(-2, -2, -2)),
        lambda v: rotate_point_around_axis(v, Vector3D(0, 1, 0), math.pi / 6)
    ]
    
    processed_points = process_3d_scene(points, transformations)
    print("Processed scene points:")
    for i, point in enumerate(processed_points):
        print(f"  Processed Point {i}: {point}")


if __name__ == "__main__":
    demonstrate_functional_programming()
