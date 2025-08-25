#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Functional Programming Example

Demonstrates functional programming concepts including pure functions,
higher-order functions, map/filter/reduce operations, and lambda functions
applied to 3D graphics and mathematics.
"""

import math
import random
from typing import List, Tuple, Callable, Any, Optional, Union
from functools import reduce, partial
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Functional Programming"
__description__ = "Functional programming concepts for 3D graphics applications"

# ============================================================================
# PURE FUNCTIONS FOR 3D MATH
# ============================================================================

def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """Pure function: Add two 3D vectors"""
    return [v1[i] + v2[i] for i in range(3)]

def vector_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """Pure function: Subtract two 3D vectors"""
    return [v1[i] - v2[i] for i in range(3)]

def vector_scale(v: List[float], scalar: float) -> List[float]:
    """Pure function: Scale a 3D vector by a scalar"""
    return [v[i] * scalar for i in range(3)]

def vector_dot(v1: List[float], v2: List[float]) -> float:
    """Pure function: Calculate dot product of two 3D vectors"""
    return sum(v1[i] * v2[i] for i in range(3))

def vector_cross(v1: List[float], v2: List[float]) -> List[float]:
    """Pure function: Calculate cross product of two 3D vectors"""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def vector_magnitude(v: List[float]) -> float:
    """Pure function: Calculate magnitude of a 3D vector"""
    return math.sqrt(sum(v[i] ** 2 for i in range(3)))

def vector_normalize(v: List[float]) -> List[float]:
    """Pure function: Normalize a 3D vector"""
    mag = vector_magnitude(v)
    if mag == 0:
        return [0.0, 0.0, 0.0]
    return vector_scale(v, 1.0 / mag)

def distance_between_points(p1: List[float], p2: List[float]) -> float:
    """Pure function: Calculate distance between two 3D points"""
    return vector_magnitude(vector_subtract(p2, p1))

def lerp(a: float, b: float, t: float) -> float:
    """Pure function: Linear interpolation between two values"""
    return a + (b - a) * t

def vector_lerp(v1: List[float], v2: List[float], t: float) -> List[float]:
    """Pure function: Linear interpolation between two vectors"""
    return [lerp(v1[i], v2[i], t) for i in range(3)]

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Pure function: Clamp a value between min and max"""
    return max(min_val, min(max_val, value))

# ============================================================================
# HIGHER-ORDER FUNCTIONS
# ============================================================================

def apply_transform(transform_func: Callable, vectors: List[List[float]]) -> List[List[float]]:
    """Higher-order function: Apply a transformation to a list of vectors"""
    return [transform_func(v) for v in vectors]

def filter_by_condition(condition_func: Callable, vectors: List[List[float]]) -> List[List[float]]:
    """Higher-order function: Filter vectors based on a condition"""
    return [v for v in vectors if condition_func(v)]

def reduce_vectors(reduce_func: Callable, vectors: List[List[float]], initial: List[float] = None) -> List[float]:
    """Higher-order function: Reduce a list of vectors to a single vector"""
    if initial is None:
        initial = [0.0, 0.0, 0.0]
    return reduce(reduce_func, vectors, initial)

def compose(*functions: Callable) -> Callable:
    """Higher-order function: Compose multiple functions"""
    def composed(x):
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return composed

def curry(func: Callable, *args, **kwargs) -> Callable:
    """Higher-order function: Curry a function with partial arguments"""
    return partial(func, *args, **kwargs)

# ============================================================================
# FUNCTIONAL DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class Vector3:
    """Immutable 3D vector using functional programming principles"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        """Immutable addition"""
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """Immutable subtraction"""
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        """Immutable scalar multiplication"""
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        """Calculate magnitude"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3':
        """Return normalized vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0.0, 0.0, 0.0)
        return self * (1.0 / mag)
    
    def to_list(self) -> List[float]:
        """Convert to list"""
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_list(cls, data: List[float]) -> 'Vector3':
        """Create from list"""
        return cls(data[0], data[1], data[2])

@dataclass(frozen=True)
class Transform3D:
    """Immutable 3D transform"""
    position: Vector3
    rotation: Vector3
    scale: Vector3
    
    def apply_to_point(self, point: Vector3) -> Vector3:
        """Apply transform to a point (simplified)"""
        # Simplified transform - in practice would use matrices
        return point + self.position
    
    def compose(self, other: 'Transform3D') -> 'Transform3D':
        """Compose two transforms"""
        return Transform3D(
            position=self.position + other.position,
            rotation=Vector3(
                self.rotation.x + other.rotation.x,
                self.rotation.y + other.rotation.y,
                self.rotation.z + other.rotation.z
            ),
            scale=Vector3(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
                self.scale.z * other.scale.z
            )
        )

# ============================================================================
# FUNCTIONAL UTILITIES
# ============================================================================

def pipeline(*functions: Callable) -> Callable:
    """Create a pipeline of functions"""
    def pipeline_func(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return pipeline_func

def memoize(func: Callable) -> Callable:
    """Memoization decorator for pure functions"""
    cache = {}
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return memoized

def once(func: Callable) -> Callable:
    """Execute function only once and cache result"""
    cache = {}
    def once_func(*args, **kwargs):
        if 'result' not in cache:
            cache['result'] = func(*args, **kwargs)
        return cache['result']
    return once_func

# ============================================================================
# 3D GRAPHICS FUNCTIONAL OPERATIONS
# ============================================================================

def generate_cube_vertices(center: Vector3, size: float) -> List[Vector3]:
    """Generate cube vertices using functional approach"""
    half_size = size / 2
    offsets = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
    ]
    
    def create_vertex(offset):
        return Vector3(
            center.x + offset[0] * half_size,
            center.y + offset[1] * half_size,
            center.z + offset[2] * half_size
        )
    
    return list(map(create_vertex, offsets))

def filter_visible_objects(objects: List[Vector3], camera_pos: Vector3, max_distance: float) -> List[Vector3]:
    """Filter objects based on visibility criteria"""
    def is_visible(obj):
        distance = (obj - camera_pos).magnitude()
        return distance <= max_distance
    
    return list(filter(is_visible, objects))

def calculate_bounding_box(points: List[Vector3]) -> Tuple[Vector3, Vector3]:
    """Calculate bounding box using functional approach"""
    if not points:
        return Vector3(0, 0, 0), Vector3(0, 0, 0)
    
    def min_coords(acc, point):
        return Vector3(
            min(acc.x, point.x),
            min(acc.y, point.y),
            min(acc.z, point.z)
        )
    
    def max_coords(acc, point):
        return Vector3(
            max(acc.x, point.x),
            max(acc.y, point.y),
            max(acc.z, point.z)
        )
    
    min_point = reduce(min_coords, points[1:], points[0])
    max_point = reduce(max_coords, points[1:], points[0])
    
    return min_point, max_point

def interpolate_vertices(v1: Vector3, v2: Vector3, steps: int) -> List[Vector3]:
    """Generate interpolated vertices between two points"""
    def interpolate_step(step):
        t = step / (steps - 1) if steps > 1 else 0
        return Vector3(
            lerp(v1.x, v2.x, t),
            lerp(v1.y, v2.y, t),
            lerp(v1.z, v2.z, t)
        )
    
    return list(map(interpolate_step, range(steps)))

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_pure_functions():
    """Demonstrate pure functions"""
    print("=== Pure Functions Demo ===\n")
    
    # Test vector operations
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    
    print("1. Vector operations:")
    print(f"   v1: {v1}")
    print(f"   v2: {v2}")
    print(f"   v1 + v2: {vector_add(v1, v2)}")
    print(f"   v1 - v2: {vector_subtract(v1, v2)}")
    print(f"   v1 * 2: {vector_scale(v1, 2.0)}")
    print(f"   v1 · v2: {vector_dot(v1, v2)}")
    print(f"   v1 × v2: {vector_cross(v1, v2)}")
    print(f"   |v1|: {vector_magnitude(v1)}")
    print(f"   normalize(v1): {vector_normalize(v1)}")
    
    # Test interpolation
    print(f"\n2. Interpolation:")
    print(f"   lerp(0, 10, 0.5): {lerp(0, 10, 0.5)}")
    print(f"   vector_lerp(v1, v2, 0.5): {vector_lerp(v1, v2, 0.5)}")
    print(f"   clamp(15, 0, 10): {clamp(15, 0, 10)}")
    
    print()

def demonstrate_higher_order_functions():
    """Demonstrate higher-order functions"""
    print("=== Higher-Order Functions Demo ===\n")
    
    # Test data
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]
    
    print("1. Apply transform:")
    doubled = apply_transform(lambda v: vector_scale(v, 2.0), vectors)
    print(f"   Original: {vectors}")
    print(f"   Doubled: {doubled}")
    
    print("\n2. Filter by condition:")
    non_zero = filter_by_condition(lambda v: vector_magnitude(v) > 0, vectors)
    print(f"   Non-zero vectors: {non_zero}")
    
    print("\n3. Reduce vectors:")
    sum_vectors = reduce_vectors(vector_add, vectors)
    print(f"   Sum of all vectors: {sum_vectors}")
    
    print("\n4. Function composition:")
    transform_pipeline = compose(
        lambda v: vector_scale(v, 2.0),
        lambda v: vector_add(v, [1, 1, 1]),
        lambda v: vector_normalize(v)
    )
    result = transform_pipeline([3, 4, 0])
    print(f"   Pipeline result: {result}")
    
    print("\n5. Currying:")
    scale_by_2 = curry(vector_scale, scalar=2.0)
    add_to_origin = curry(vector_add, v2=[1, 1, 1])
    print(f"   Scale by 2: {scale_by_2([1, 2, 3])}")
    print(f"   Add to origin: {add_to_origin([1, 2, 3])}")
    
    print()

def demonstrate_immutable_data_structures():
    """Demonstrate immutable data structures"""
    print("=== Immutable Data Structures Demo ===\n")
    
    # Create immutable vectors
    v1 = Vector3(1.0, 2.0, 3.0)
    v2 = Vector3(4.0, 5.0, 6.0)
    
    print("1. Vector operations:")
    print(f"   v1: {v1}")
    print(f"   v2: {v2}")
    print(f"   v1 + v2: {v1 + v2}")
    print(f"   v1 - v2: {v1 - v2}")
    print(f"   v1 * 2: {v1 * 2.0}")
    print(f"   |v1|: {v1.magnitude()}")
    print(f"   normalize(v1): {v1.normalize()}")
    
    # Test transforms
    print("\n2. Transform operations:")
    transform1 = Transform3D(Vector3(1, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1))
    transform2 = Transform3D(Vector3(0, 1, 0), Vector3(0, 0, 0), Vector3(2, 2, 2))
    composed = transform1.compose(transform2)
    
    point = Vector3(0, 0, 0)
    transformed = transform1.apply_to_point(point)
    
    print(f"   Transform1: {transform1}")
    print(f"   Transform2: {transform2}")
    print(f"   Composed: {composed}")
    print(f"   Point {point} transformed: {transformed}")
    
    print()

def demonstrate_3d_graphics_operations():
    """Demonstrate 3D graphics functional operations"""
    print("=== 3D Graphics Operations Demo ===\n")
    
    # Generate cube vertices
    center = Vector3(0, 0, 0)
    cube_vertices = generate_cube_vertices(center, 2.0)
    print("1. Cube vertices:")
    for i, vertex in enumerate(cube_vertices):
        print(f"   Vertex {i}: {vertex}")
    
    # Filter visible objects
    camera_pos = Vector3(5, 5, 5)
    objects = [
        Vector3(1, 1, 1),
        Vector3(10, 10, 10),
        Vector3(2, 2, 2),
        Vector3(15, 15, 15)
    ]
    visible = filter_visible_objects(objects, camera_pos, 10.0)
    print(f"\n2. Visible objects (distance <= 10): {len(visible)}/{len(objects)}")
    for obj in visible:
        distance = (obj - camera_pos).magnitude()
        print(f"   {obj} (distance: {distance:.2f})")
    
    # Calculate bounding box
    min_point, max_point = calculate_bounding_box(objects)
    print(f"\n3. Bounding box:")
    print(f"   Min: {min_point}")
    print(f"   Max: {max_point}")
    
    # Interpolate vertices
    start = Vector3(0, 0, 0)
    end = Vector3(1, 1, 1)
    interpolated = interpolate_vertices(start, end, 5)
    print(f"\n4. Interpolated vertices:")
    for i, vertex in enumerate(interpolated):
        print(f"   Step {i}: {vertex}")
    
    print()

def demonstrate_functional_utilities():
    """Demonstrate functional utilities"""
    print("=== Functional Utilities Demo ===\n")
    
    # Test memoization
    @memoize
    def expensive_calculation(n):
        print(f"    Computing for {n}...")
        return sum(i**2 for i in range(n))
    
    print("1. Memoization:")
    print(f"   First call: {expensive_calculation(5)}")
    print(f"   Second call: {expensive_calculation(5)}")  # Should use cache
    
    # Test pipeline
    def double(x): return x * 2
    def add_one(x): return x + 1
    def square(x): return x ** 2
    
    pipeline_func = pipeline(double, add_one, square)
    result = pipeline_func(3)
    print(f"\n2. Pipeline (3 → double → add_one → square): {result}")
    
    # Test once function
    @once
    def expensive_initialization():
        print("    Initializing...")
        return "Initialized"
    
    print("\n3. Once function:")
    print(f"   First call: {expensive_initialization()}")
    print(f"   Second call: {expensive_initialization()}")  # Should use cache
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate functional programming concepts"""
    print("=== Functional Programming Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_pure_functions()
    demonstrate_higher_order_functions()
    demonstrate_immutable_data_structures()
    demonstrate_3d_graphics_operations()
    demonstrate_functional_utilities()
    
    print("="*60)
    print("Functional Programming demo completed successfully!")
    print("\nKey features:")
    print("✓ Pure functions: No side effects, predictable results")
    print("✓ Higher-order functions: Functions that take/return functions")
    print("✓ Immutable data structures: Thread-safe, predictable state")
    print("✓ Function composition: Building complex operations from simple ones")
    print("✓ Memory efficiency: Lazy evaluation and functional pipelines")
    print("✓ Mathematical elegance: Clean, declarative code")

if __name__ == "__main__":
    main()
