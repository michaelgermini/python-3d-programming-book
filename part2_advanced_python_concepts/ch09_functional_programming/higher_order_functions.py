"""
Chapter 9: Functional Programming - Higher-Order Functions
========================================================

This module demonstrates higher-order functions in Python,
applied to 3D graphics processing and mathematical operations.

Key Concepts:
- Functions that take functions as arguments
- Functions that return functions
- Currying and partial application
- Function factories
- Decorators as higher-order functions
"""

import math
from typing import List, Callable, Any, TypeVar, Generic
from functools import partial, wraps
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class Point3D:
    """3D point for demonstrating higher-order functions."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Point3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


# Higher-Order Functions that Take Functions as Arguments
def apply_to_points(points: List[Point3D], 
                   transform_func: Callable[[Point3D], Point3D]) -> List[Point3D]:
    """
    Higher-order function: applies a transformation function to all points.
    
    Args:
        points: List of 3D points
        transform_func: Function to apply to each point
    
    Returns:
        List of transformed points
    """
    return [transform_func(point) for point in points]


def filter_points(points: List[Point3D], 
                 predicate: Callable[[Point3D], bool]) -> List[Point3D]:
    """
    Higher-order function: filters points based on a predicate function.
    
    Args:
        points: List of 3D points
        predicate: Function that returns True/False for each point
    
    Returns:
        Filtered list of points
    """
    return [point for point in points if predicate(point)]


def reduce_points(points: List[Point3D], 
                 reducer: Callable[[Point3D, Point3D], Point3D],
                 initial: Point3D = None) -> Point3D:
    """
    Higher-order function: reduces points using a reducer function.
    
    Args:
        points: List of 3D points
        reducer: Function that combines two points
        initial: Initial value (optional)
    
    Returns:
        Reduced point
    """
    if not points:
        return initial
    
    if initial is None:
        result = points[0]
        remaining = points[1:]
    else:
        result = initial
        remaining = points
    
    for point in remaining:
        result = reducer(result, point)
    
    return result


# Higher-Order Functions that Return Functions
def create_translation(offset_x: float, offset_y: float, offset_z: float) -> Callable[[Point3D], Point3D]:
    """
    Function factory: creates a translation function.
    
    Args:
        offset_x: X-axis offset
        offset_y: Y-axis offset
        offset_z: Z-axis offset
    
    Returns:
        Translation function
    """
    def translate(point: Point3D) -> Point3D:
        return Point3D(point.x + offset_x, point.y + offset_y, point.z + offset_z)
    return translate


def create_scaling(scale_x: float, scale_y: float, scale_z: float) -> Callable[[Point3D], Point3D]:
    """
    Function factory: creates a scaling function.
    
    Args:
        scale_x: X-axis scale factor
        scale_y: Y-axis scale factor
        scale_z: Z-axis scale factor
    
    Returns:
        Scaling function
    """
    def scale(point: Point3D) -> Point3D:
        return Point3D(point.x * scale_x, point.y * scale_y, point.z * scale_z)
    return scale


def create_rotation(axis: str, angle: float) -> Callable[[Point3D], Point3D]:
    """
    Function factory: creates a rotation function around specified axis.
    
    Args:
        axis: Rotation axis ('x', 'y', or 'z')
        angle: Rotation angle in radians
    
    Returns:
        Rotation function
    """
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    if axis.lower() == 'x':
        def rotate_x(point: Point3D) -> Point3D:
            return Point3D(
                point.x,
                point.y * cos_angle - point.z * sin_angle,
                point.y * sin_angle + point.z * cos_angle
            )
        return rotate_x
    
    elif axis.lower() == 'y':
        def rotate_y(point: Point3D) -> Point3D:
            return Point3D(
                point.x * cos_angle + point.z * sin_angle,
                point.y,
                -point.x * sin_angle + point.z * cos_angle
            )
        return rotate_y
    
    elif axis.lower() == 'z':
        def rotate_z(point: Point3D) -> Point3D:
            return Point3D(
                point.x * cos_angle - point.y * sin_angle,
                point.x * sin_angle + point.y * cos_angle,
                point.z
            )
        return rotate_z
    
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")


# Currying and Partial Application
def distance_from_origin(point: Point3D) -> float:
    """Calculates distance from origin."""
    return math.sqrt(point.x**2 + point.y**2 + point.z**2)


def distance_from_point(reference_point: Point3D) -> Callable[[Point3D], float]:
    """
    Curried function: creates a distance calculator from a reference point.
    
    Args:
        reference_point: Reference point for distance calculation
    
    Returns:
        Function that calculates distance from reference point
    """
    def distance_to_reference(point: Point3D) -> float:
        dx = point.x - reference_point.x
        dy = point.y - reference_point.y
        dz = point.z - reference_point.z
        return math.sqrt(dx**2 + dy**2 + dz**2)
    return distance_to_reference


# Using partial application
def create_distance_filter(max_distance: float, reference_point: Point3D) -> Callable[[Point3D], bool]:
    """
    Creates a filter function using partial application.
    
    Args:
        max_distance: Maximum allowed distance
        reference_point: Reference point for distance calculation
    
    Returns:
        Filter function
    """
    distance_func = distance_from_point(reference_point)
    return lambda point: distance_func(point) <= max_distance


# Function Composition with Higher-Order Functions
def compose_functions(*functions: Callable) -> Callable:
    """
    Higher-order function: composes multiple functions.
    
    Args:
        *functions: Functions to compose
    
    Returns:
        Composed function
    """
    def composed(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return composed


def create_transformation_pipeline(*transformations: Callable[[Point3D], Point3D]) -> Callable[[List[Point3D]], List[Point3D]]:
    """
    Creates a transformation pipeline for processing multiple points.
    
    Args:
        *transformations: Transformation functions to apply
    
    Returns:
        Function that applies all transformations to a list of points
    """
    combined_transform = compose_functions(*transformations)
    
    def pipeline(points: List[Point3D]) -> List[Point3D]:
        return apply_to_points(points, combined_transform)
    
    return pipeline


# Decorators as Higher-Order Functions
def timing_decorator(func: Callable) -> Callable:
    """
    Decorator that measures execution time of a function.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def validation_decorator(validator: Callable[[Any], bool]) -> Callable:
    """
    Decorator that validates input using a validator function.
    
    Args:
        validator: Function that validates input
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError(f"Validation failed for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Example validation functions
def validate_points(points: List[Point3D]) -> bool:
    """Validates that input is a non-empty list of points."""
    return isinstance(points, list) and len(points) > 0 and all(isinstance(p, Point3D) for p in points)


def validate_positive_scale(scale_x: float, scale_y: float, scale_z: float) -> bool:
    """Validates that scale factors are positive."""
    return scale_x > 0 and scale_y > 0 and scale_z > 0


# Applying decorators
@timing_decorator
@validation_decorator(validate_points)
def process_scene_points(points: List[Point3D]) -> List[Point3D]:
    """
    Processes scene points with multiple transformations.
    
    Args:
        points: List of 3D points
    
    Returns:
        Processed points
    """
    # Create transformation pipeline
    pipeline = create_transformation_pipeline(
        create_translation(1, 1, 1),
        create_scaling(2, 2, 2),
        create_rotation('z', math.pi / 4)
    )
    
    return pipeline(points)


# Advanced Higher-Order Functions
def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for expensive computations.
    
    Args:
        func: Function to memoize
    
    Returns:
        Memoized function
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper


@memoize
def fibonacci(n: int) -> int:
    """Memoized Fibonacci function for demonstration."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def create_point_generator(center: Point3D, radius: float, count: int) -> Callable[[], Point3D]:
    """
    Creates a generator function for random points within a sphere.
    
    Args:
        center: Center of the sphere
        radius: Radius of the sphere
        count: Number of points to generate
    
    Returns:
        Generator function
    """
    import random
    
    def generate_point() -> Point3D:
        # Generate random point within unit sphere
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            
            if x**2 + y**2 + z**2 <= 1:
                # Scale to desired radius and translate to center
                return Point3D(
                    center.x + x * radius,
                    center.y + y * radius,
                    center.z + z * radius
                )
    
    return generate_point


# Demonstration
def demonstrate_higher_order_functions():
    """Demonstrates higher-order functions with 3D examples."""
    print("=== Higher-Order Functions with 3D Applications ===\n")
    
    # Create sample points
    points = [
        Point3D(1, 2, 3),
        Point3D(4, 5, 6),
        Point3D(7, 8, 9),
        Point3D(10, 11, 12)
    ]
    
    print("Original points:")
    for i, point in enumerate(points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate function factories
    print("\n=== Function Factories ===")
    
    translate_func = create_translation(2, 3, 4)
    scale_func = create_scaling(0.5, 0.5, 0.5)
    rotate_func = create_rotation('z', math.pi / 6)
    
    print("Applying translation (2, 3, 4):")
    translated_points = apply_to_points(points, translate_func)
    for i, point in enumerate(translated_points):
        print(f"  Translated Point {i}: {point}")
    
    print("\nApplying scaling (0.5, 0.5, 0.5):")
    scaled_points = apply_to_points(points, scale_func)
    for i, point in enumerate(scaled_points):
        print(f"  Scaled Point {i}: {point}")
    
    # Demonstrate currying and partial application
    print("\n=== Currying and Partial Application ===")
    
    origin = Point3D(0, 0, 0)
    distance_from_origin_func = distance_from_point(origin)
    
    print("Distances from origin:")
    for i, point in enumerate(points):
        distance = distance_from_origin_func(point)
        print(f"  Point {i} distance: {distance:.2f}")
    
    # Demonstrate filtering
    print("\n=== Filtering ===")
    
    center = Point3D(5, 5, 5)
    max_dist = 8.0
    distance_filter = create_distance_filter(max_dist, center)
    
    filtered_points = filter_points(points, distance_filter)
    print(f"Points within {max_dist} units of {center}:")
    for i, point in enumerate(filtered_points):
        print(f"  Filtered Point {i}: {point}")
    
    # Demonstrate reduction
    print("\n=== Reduction ===")
    
    def average_points(p1: Point3D, p2: Point3D) -> Point3D:
        return Point3D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
    
    center_point = reduce_points(points, average_points)
    print(f"Center point (average): {center_point}")
    
    # Demonstrate transformation pipeline
    print("\n=== Transformation Pipeline ===")
    
    pipeline = create_transformation_pipeline(
        create_translation(-1, -1, -1),
        create_scaling(2, 2, 2),
        create_rotation('y', math.pi / 4)
    )
    
    processed_points = pipeline(points)
    print("Processed points through pipeline:")
    for i, point in enumerate(processed_points):
        print(f"  Processed Point {i}: {point}")
    
    # Demonstrate decorators
    print("\n=== Decorators ===")
    
    try:
        processed_scene = process_scene_points(points)
        print("Scene processing completed successfully")
    except Exception as e:
        print(f"Error: {e}")
    
    # Demonstrate memoization
    print("\n=== Memoization ===")
    
    print("Calculating Fibonacci numbers (memoized):")
    for i in range(10):
        result = fibonacci(i)
        print(f"  fibonacci({i}) = {result}")


if __name__ == "__main__":
    demonstrate_higher_order_functions()
