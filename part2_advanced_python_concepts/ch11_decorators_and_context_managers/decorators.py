"""
Chapter 11: Decorators and Context Managers - Decorators
=======================================================

This module demonstrates how to use decorators to enhance 3D graphics
functions with additional functionality like timing, validation,
caching, and logging.

Key Concepts:
- Function decorators
- Class decorators
- Decorator factories
- Parameterized decorators
- Decorator composition
- Performance monitoring for 3D operations
"""

import time
import functools
import math
import random
from typing import Callable, Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class Vector3D:
    """3D vector for decorator examples."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def distance_to(self, other: 'Vector3D') -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class Matrix4x4:
    """4x4 matrix for 3D transformations."""
    m: List[List[float]]
    
    def __init__(self, matrix: Optional[List[List[float]]] = None):
        if matrix is None:
            self.m = [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]
        else:
            self.m = matrix
    
    def __str__(self):
        return f"Matrix4x4({self.m})"


# Basic Decorators
def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"  {func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper


def validation_decorator(func: Callable) -> Callable:
    """Decorator to validate 3D vector inputs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if any Vector3D arguments have NaN or infinite values
        for arg in args:
            if isinstance(arg, Vector3D):
                if (math.isnan(arg.x) or math.isnan(arg.y) or math.isnan(arg.z) or
                    math.isinf(arg.x) or math.isinf(arg.y) or math.isinf(arg.z)):
                    raise ValueError(f"Invalid Vector3D in {func.__name__}: {arg}")
        
        result = func(*args, **kwargs)
        return result
    return wrapper


def logging_decorator(func: Callable) -> Callable:
    """Decorator to log function calls and results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"  Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"  {func.__name__} returned: {result}")
        return result
    return wrapper


# Parameterized Decorators
def retry_decorator(max_attempts: int = 3, delay: float = 0.1):
    """Decorator factory for retrying failed operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"  Attempt {attempt + 1} failed, retrying in {delay}s...")
                        time.sleep(delay)
            
            print(f"  All {max_attempts} attempts failed")
            raise last_exception
        return wrapper
    return decorator


def cache_decorator(max_size: int = 100):
    """Decorator factory for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                print(f"  Cache hit for {func.__name__}")
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Implement LRU cache eviction
            if len(cache) >= max_size:
                # Remove oldest entry (simple implementation)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            print(f"  Cache miss for {func.__name__}, stored result")
            return result
        return wrapper
    return decorator


def performance_monitor_decorator(threshold: float = 0.1):
    """Decorator to monitor performance and warn about slow operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold:
                print(f"  ⚠️  {func.__name__} took {execution_time:.6f}s (threshold: {threshold}s)")
            
            return result
        return wrapper
    return decorator


# 3D Graphics Specific Decorators
def bounds_check_decorator(min_value: float = -1000.0, max_value: float = 1000.0):
    """Decorator to check if 3D coordinates are within bounds."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check Vector3D arguments
            for arg in args:
                if isinstance(arg, Vector3D):
                    if not (min_value <= arg.x <= max_value and 
                           min_value <= arg.y <= max_value and 
                           min_value <= arg.z <= max_value):
                        raise ValueError(f"Vector3D {arg} out of bounds [{min_value}, {max_value}]")
            
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


def normalize_result_decorator(func: Callable) -> Callable:
    """Decorator to normalize Vector3D results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if isinstance(result, Vector3D):
            magnitude = result.magnitude()
            if magnitude > 0:
                return Vector3D(result.x / magnitude, result.y / magnitude, result.z / magnitude)
        
        return result
    return wrapper


def matrix_validation_decorator(func: Callable) -> Callable:
    """Decorator to validate matrix operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if any Matrix4x4 arguments are valid
        for arg in args:
            if isinstance(arg, Matrix4x4):
                if len(arg.m) != 4 or any(len(row) != 4 for row in arg.m):
                    raise ValueError(f"Invalid Matrix4x4 in {func.__name__}: {arg}")
        
        result = func(*args, **kwargs)
        return result
    return wrapper


# Decorator Composition
def comprehensive_3d_decorator(func: Callable) -> Callable:
    """Comprehensive decorator combining multiple 3D graphics checks."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Apply multiple decorators
        decorated_func = timing_decorator(
            validation_decorator(
                bounds_check_decorator(-1000, 1000)(func)
            )
        )
        return decorated_func(*args, **kwargs)
    return wrapper


# Class Decorators
def singleton_decorator(cls):
    """Class decorator to implement singleton pattern."""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def add_methods_decorator(*methods):
    """Class decorator to add methods to a class."""
    def decorator(cls):
        for method_name, method_func in methods:
            setattr(cls, method_name, method_func)
        return cls
    return decorator


# Example Functions to Decorate
@timing_decorator
@validation_decorator
def calculate_distance(v1: Vector3D, v2: Vector3D) -> float:
    """Calculate distance between two 3D vectors."""
    return v1.distance_to(v2)


@retry_decorator(max_attempts=3, delay=0.1)
@performance_monitor_decorator(threshold=0.01)
def complex_3d_calculation(v1: Vector3D, v2: Vector3D, iterations: int = 1000) -> Vector3D:
    """Complex 3D calculation that might fail or be slow."""
    if random.random() < 0.1:  # 10% chance of failure
        raise RuntimeError("Random calculation failure")
    
    # Simulate complex calculation
    result = Vector3D(0, 0, 0)
    for i in range(iterations):
        result.x += math.sin(i * 0.1) * v1.x
        result.y += math.cos(i * 0.1) * v1.y
        result.z += math.sin(i * 0.1) * math.cos(i * 0.1) * v1.z
    
    return result


@cache_decorator(max_size=50)
@bounds_check_decorator(-100, 100)
def expensive_vector_operation(v: Vector3D) -> Vector3D:
    """Expensive vector operation that benefits from caching."""
    # Simulate expensive computation
    time.sleep(0.01)
    return Vector3D(v.x * 2, v.y * 2, v.z * 2)


@normalize_result_decorator
def vector_cross_product(v1: Vector3D, v2: Vector3D) -> Vector3D:
    """Calculate cross product of two vectors."""
    x = v1.y * v2.z - v1.z * v2.y
    y = v1.z * v2.x - v1.x * v2.z
    z = v1.x * v2.y - v1.y * v2.x
    return Vector3D(x, y, z)


@matrix_validation_decorator
def create_translation_matrix(translation: Vector3D) -> Matrix4x4:
    """Create a translation matrix."""
    matrix = [
        [1.0, 0.0, 0.0, translation.x],
        [0.0, 1.0, 0.0, translation.y],
        [0.0, 0.0, 1.0, translation.z],
        [0.0, 0.0, 0.0, 1.0]
    ]
    return Matrix4x4(matrix)


@comprehensive_3d_decorator
def advanced_3d_operation(v1: Vector3D, v2: Vector3D, scale: float = 1.0) -> Vector3D:
    """Advanced 3D operation with comprehensive decoration."""
    # Complex 3D operation
    result = Vector3D(
        (v1.x + v2.x) * scale,
        (v1.y + v2.y) * scale,
        (v1.z + v2.z) * scale
    )
    
    # Simulate some processing time
    time.sleep(0.05)
    
    return result


# Class with Decorators
@singleton_decorator
class Renderer:
    """Singleton renderer class."""
    
    def __init__(self):
        self.render_count = 0
    
    def render(self, scene_data):
        self.render_count += 1
        print(f"  Rendering scene #{self.render_count}")
        return f"Rendered scene with {len(scene_data)} objects"


def add_utility_methods(cls):
    """Add utility methods to a class."""
    def get_info(self):
        return f"Class: {self.__class__.__name__}"
    
    def clone(self):
        return self.__class__()
    
    cls.get_info = get_info
    cls.clone = clone
    return cls


@add_utility_methods
class SceneObject:
    """Scene object with added utility methods."""
    
    def __init__(self, position: Vector3D):
        self.position = position
    
    def update(self):
        print(f"  Updating object at {self.position}")


# Example Usage and Demonstration
def demonstrate_decorators():
    """Demonstrates various decorator patterns for 3D graphics."""
    print("=== Decorators for 3D Graphics ===\n")
    
    # Create test vectors
    v1 = Vector3D(1.0, 2.0, 3.0)
    v2 = Vector3D(4.0, 5.0, 6.0)
    
    print("=== Basic Decorators ===")
    
    print("Timing decorator:")
    distance = calculate_distance(v1, v2)
    print(f"  Distance: {distance:.3f}")
    
    print("\nLogging decorator:")
    @logging_decorator
    def test_function(x, y):
        return x + y
    
    result = test_function(5, 3)
    
    print("\n=== Parameterized Decorators ===")
    
    print("Retry decorator:")
    try:
        result = complex_3d_calculation(v1, v2, iterations=100)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed after retries: {e}")
    
    print("\nCache decorator:")
    for i in range(3):
        result = expensive_vector_operation(v1)
        print(f"  Operation {i+1}: {result}")
    
    print("\nPerformance monitor decorator:")
    result = complex_3d_calculation(v1, v2, iterations=10)
    
    print("\n=== 3D Graphics Specific Decorators ===")
    
    print("Bounds check decorator:")
    try:
        result = expensive_vector_operation(v1)  # Should work
        print(f"  Valid operation: {result}")
        
        invalid_v = Vector3D(1000, 1000, 1000)
        result = expensive_vector_operation(invalid_v)  # Should fail
    except ValueError as e:
        print(f"  Bounds check caught: {e}")
    
    print("\nNormalize result decorator:")
    cross_product = vector_cross_product(v1, v2)
    print(f"  Cross product: {cross_product}")
    print(f"  Magnitude: {cross_product.magnitude():.6f}")
    
    print("\nMatrix validation decorator:")
    translation = create_translation_matrix(Vector3D(1, 2, 3))
    print(f"  Translation matrix: {translation}")
    
    print("\n=== Decorator Composition ===")
    
    print("Comprehensive 3D decorator:")
    result = advanced_3d_operation(v1, v2, scale=2.0)
    print(f"  Advanced operation result: {result}")
    
    print("\n=== Class Decorators ===")
    
    print("Singleton decorator:")
    renderer1 = Renderer()
    renderer2 = Renderer()
    print(f"  Same instance: {renderer1 is renderer2}")
    
    scene_data = ["cube", "sphere", "cylinder"]
    renderer1.render(scene_data)
    renderer2.render(scene_data)
    
    print("\nAdd methods decorator:")
    obj = SceneObject(Vector3D(0, 0, 0))
    print(f"  Object info: {obj.get_info()}")
    obj.update()
    
    cloned_obj = obj.clone()
    print(f"  Cloned object: {cloned_obj.get_info()}")


if __name__ == "__main__":
    demonstrate_decorators()
