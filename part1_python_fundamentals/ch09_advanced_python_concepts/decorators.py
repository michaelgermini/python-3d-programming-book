#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Decorators Example

Demonstrates function and class decorators, parameterized decorators,
and common decorator patterns for 3D graphics applications.
"""

import time
import functools
import math
import random
from typing import List, Tuple, Callable, Any, Optional, Union, Dict
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Decorators"
__description__ = "Decorators for 3D graphics applications"

# ============================================================================
# BASIC FUNCTION DECORATORS
# ============================================================================

def timer_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def cache_result(func: Callable) -> Callable:
    """Decorator to cache function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            print(f"💾 {func.__name__}: Cached new result")
        else:
            print(f"📋 {func.__name__}: Using cached result")
        
        return cache[key]
    return wrapper

def validate_3d_point(func: Callable) -> Callable:
    """Decorator to validate 3D point arguments"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
            point = args[0]
            if all(isinstance(coord, (int, float)) for coord in point):
                return func(*args, **kwargs)
            else:
                raise ValueError(f"Invalid 3D point: {point}")
        else:
            raise ValueError("First argument must be a 3D point")
    return wrapper

# ============================================================================
# PARAMETERIZED DECORATORS
# ============================================================================

def retry_on_error(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function on error"""
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
                        print(f"⚠️  {func.__name__} failed (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(delay)
                    else:
                        print(f"❌ {func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        return wrapper
    return decorator

def performance_threshold(threshold_seconds: float):
    """Decorator to warn if function takes too long"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > threshold_seconds:
                print(f"🐌 {func.__name__} took {execution_time:.4f}s (threshold: {threshold_seconds}s)")
            
            return result
        return wrapper
    return decorator

def validate_vector_magnitude(min_magnitude: float = 0.0, max_magnitude: float = float('inf')):
    """Decorator to validate vector magnitude"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, (list, tuple)) and len(result) == 3:
                magnitude = math.sqrt(sum(coord ** 2 for coord in result))
                if not (min_magnitude <= magnitude <= max_magnitude):
                    print(f"⚠️  {func.__name__} returned vector with magnitude {magnitude:.4f}")
            
            return result
        return wrapper
    return decorator

# ============================================================================
# CLASS DECORATORS
# ============================================================================

def singleton(cls):
    """Class decorator to implement singleton pattern"""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def add_performance_monitoring(cls):
    """Class decorator to add performance monitoring to all methods"""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, timer_decorator(attr))
    return cls

# ============================================================================
# 3D GRAPHICS SPECIFIC DECORATORS
# ============================================================================

def validate_3d_vector(func: Callable) -> Callable:
    """Decorator to validate 3D vector arguments and results"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Validate input
        for arg in args:
            if isinstance(arg, (list, tuple)) and len(arg) == 3:
                if not all(isinstance(coord, (int, float)) for coord in arg):
                    raise ValueError(f"Invalid 3D vector: {arg}")
        
        result = func(*args, **kwargs)
        
        # Validate output
        if isinstance(result, (list, tuple)) and len(result) == 3:
            if not all(isinstance(coord, (int, float)) for coord in result):
                raise ValueError(f"Invalid 3D vector result: {result}")
        
        return result
    return wrapper

def normalize_result(func: Callable) -> Callable:
    """Decorator to normalize 3D vector results"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if isinstance(result, (list, tuple)) and len(result) == 3:
            magnitude = math.sqrt(sum(coord ** 2 for coord in result))
            if magnitude > 0:
                result = [coord / magnitude for coord in result]
        
        return result
    return wrapper

# ============================================================================
# EXAMPLE FUNCTIONS TO DECORATE
# ============================================================================

@timer_decorator
@validate_3d_point
def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate distance between two 3D points"""
    return math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(3)))

@cache_result
@validate_3d_vector
def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a 3D vector"""
    magnitude = math.sqrt(sum(coord ** 2 for coord in vector))
    if magnitude == 0:
        return [0.0, 0.0, 0.0]
    return [coord / magnitude for coord in vector]

@retry_on_error(max_attempts=3, delay=0.1)
def generate_random_point(bounds: List[float]) -> List[float]:
    """Generate a random 3D point within bounds"""
    if random.random() < 0.1:  # 10% chance of failure for demo
        raise ValueError("Random failure for demonstration")
    return [random.uniform(-bounds[i], bounds[i]) for i in range(3)]

@performance_threshold(0.001)
def expensive_calculation(n: int) -> float:
    """Simulate expensive calculation"""
    time.sleep(0.002)  # Simulate work
    return sum(i ** 2 for i in range(n))

@validate_vector_magnitude(min_magnitude=0.1, max_magnitude=10.0)
def scale_vector(vector: List[float], factor: float) -> List[float]:
    """Scale a 3D vector by a factor"""
    return [coord * factor for coord in vector]

@timer_decorator
@normalize_result
def process_vector(vector: List[float]) -> List[float]:
    """Process a 3D vector with multiple decorators"""
    return [coord * 2 for coord in vector]

# ============================================================================
# EXAMPLE CLASSES TO DECORATE
# ============================================================================

@singleton
class GraphicsManager:
    """Singleton graphics manager"""
    
    def __init__(self):
        self.objects = []
        print("🎮 GraphicsManager initialized")
    
    def add_object(self, obj):
        self.objects.append(obj)
        print(f"➕ Added object: {obj}")

@add_performance_monitoring
class Transform3D:
    """3D transform class with performance monitoring"""
    
    def __init__(self, position: List[float], rotation: List[float], scale: List[float]):
        self.position = position
        self.rotation = rotation
        self.scale = scale
    
    def apply_to_point(self, point: List[float]) -> List[float]:
        """Apply transform to a point (simplified)"""
        time.sleep(0.001)  # Simulate work
        return [point[i] + self.position[i] for i in range(3)]
    
    def compose(self, other: 'Transform3D') -> 'Transform3D':
        """Compose two transforms"""
        time.sleep(0.002)  # Simulate work
        return Transform3D(
            [self.position[i] + other.position[i] for i in range(3)],
            [self.rotation[i] + other.rotation[i] for i in range(3)],
            [self.scale[i] * other.scale[i] for i in range(3)]
        )

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_decorators():
    """Demonstrate basic function decorators"""
    print("=== Basic Decorators Demo ===\n")
    
    # Timer decorator
    print("1. Timer decorator:")
    result = calculate_distance([0, 0, 0], [1, 1, 1])
    print(f"   Distance: {result:.4f}")
    
    # Cache decorator
    print("\n2. Cache decorator:")
    for i in range(3):
        result = normalize_vector([3, 4, 0])
        print(f"   Normalized: {[round(v, 4) for v in result]}")
    
    # Retry decorator
    print("\n3. Retry decorator:")
    try:
        point = generate_random_point([5, 5, 5])
        print(f"   Generated point: {[round(p, 2) for p in point]}")
    except ValueError as e:
        print(f"   Failed after retries: {e}")
    
    print()

def demonstrate_parameterized_decorators():
    """Demonstrate parameterized decorators"""
    print("=== Parameterized Decorators Demo ===\n")
    
    # Performance threshold
    print("1. Performance threshold decorator:")
    result = expensive_calculation(1000)
    print(f"   Result: {result}")
    
    # Vector magnitude validation
    print("\n2. Vector magnitude validation:")
    result = scale_vector([1, 1, 1], 15.0)  # Will trigger warning
    print(f"   Scaled vector: {result}")
    
    print()

def demonstrate_class_decorators():
    """Demonstrate class decorators"""
    print("=== Class Decorators Demo ===\n")
    
    # Singleton decorator
    print("1. Singleton decorator:")
    manager1 = GraphicsManager()
    manager2 = GraphicsManager()
    print(f"   Same instance: {manager1 is manager2}")
    
    # Performance monitoring decorator
    print("\n2. Performance monitoring decorator:")
    transform1 = Transform3D([1, 0, 0], [0, 0, 0], [1, 1, 1])
    transform2 = Transform3D([0, 1, 0], [0, 0, 0], [2, 2, 2])
    
    point = [0, 0, 0]
    transformed = transform1.apply_to_point(point)
    print(f"   Transformed point: {transformed}")
    
    composed = transform1.compose(transform2)
    print(f"   Composed transform position: {composed.position}")
    
    print()

def demonstrate_3d_graphics_decorators():
    """Demonstrate 3D graphics specific decorators"""
    print("=== 3D Graphics Decorators Demo ===\n")
    
    # Vector validation
    @validate_3d_vector
    def create_vector(x, y, z):
        return [x, y, z]
    
    print("1. 3D vector validation:")
    try:
        vector = create_vector(1, 2, 3)
        print(f"   Valid vector: {vector}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # Normalize result
    print("\n2. Normalize result decorator:")
    for i in range(3):
        vector = process_vector([random.uniform(-1, 1) for _ in range(3)])
        magnitude = math.sqrt(sum(v**2 for v in vector))
        print(f"   Vector {i+1}: {[round(v, 4) for v in vector]} (magnitude: {magnitude:.4f})")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate decorators"""
    print("=== Decorators Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_decorators()
    demonstrate_parameterized_decorators()
    demonstrate_class_decorators()
    demonstrate_3d_graphics_decorators()
    
    print("="*60)
    print("Decorators demo completed successfully!")
    print("\nKey features:")
    print("✓ Function decorators: Timer, cache, validation")
    print("✓ Parameterized decorators: Retry, performance, validation")
    print("✓ Class decorators: Singleton, performance monitoring")
    print("✓ 3D graphics decorators: Vector validation, normalization")
    print("✓ Code enhancement: Non-intrusive functionality addition")

if __name__ == "__main__":
    main()
