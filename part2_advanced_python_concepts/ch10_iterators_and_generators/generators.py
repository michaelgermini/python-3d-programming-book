"""
Chapter 10: Iterators and Generators - Generators
===============================================

This module demonstrates generators in Python, applied to 3D graphics
and mathematical operations. Generators provide memory-efficient ways
to create iterators and handle large datasets.

Key Concepts:
- Generator functions
- Generator expressions
- Memory efficiency
- Lazy evaluation
- 3D data generation
"""

import math
import random
from typing import Generator, Iterator, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Vector3D:
    """3D vector for demonstrating generators."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class Color:
    """Color representation."""
    r: float  # 0.0 to 1.0
    g: float  # 0.0 to 1.0
    b: float  # 0.0 to 1.0
    a: float = 1.0  # 0.0 to 1.0
    
    def __str__(self):
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"


# Basic Generator Functions
def fibonacci_generator(n: int) -> Generator[int, None, None]:
    """Generate Fibonacci numbers up to n."""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1


def range_generator(start: float, stop: float, step: float) -> Generator[float, None, None]:
    """Generate floating-point range values."""
    current = start
    while current < stop:
        yield current
        current += step


def vector_interpolation_generator(start: Vector3D, end: Vector3D, steps: int) -> Generator[Vector3D, None, None]:
    """Generate interpolated vectors between start and end."""
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = start.x + (end.x - start.x) * t
        y = start.y + (end.y - start.y) * t
        z = start.z + (end.z - start.z) * t
        yield Vector3D(x, y, z)


# 3D Graphics Generators
def grid_generator(x_range: Tuple[float, float, int],
                  y_range: Tuple[float, float, int],
                  z_range: Tuple[float, float, int]) -> Generator[Vector3D, None, None]:
    """
    Generate points in a 3D grid.
    
    Args:
        x_range: (start, end, steps) for x-axis
        y_range: (start, end, steps) for y-axis
        z_range: (start, end, steps) for z-axis
    """
    x_start, x_end, x_steps = x_range
    y_start, y_end, y_steps = y_range
    z_start, z_end, z_steps = z_range
    
    for i in range(x_steps):
        for j in range(y_steps):
            for k in range(z_steps):
                x = x_start + (x_end - x_start) * i / (x_steps - 1)
                y = y_start + (y_end - y_start) * j / (y_steps - 1)
                z = z_start + (z_end - z_start) * k / (z_steps - 1)
                yield Vector3D(x, y, z)


def sphere_generator(center: Vector3D, radius: float,
                    latitude_steps: int, longitude_steps: int) -> Generator[Vector3D, None, None]:
    """Generate points on a sphere surface."""
    for lat_idx in range(latitude_steps):
        for lon_idx in range(longitude_steps):
            # Calculate spherical coordinates
            lat = math.pi * (lat_idx + 1) / (latitude_steps + 1)  # 0 to π
            lon = 2 * math.pi * lon_idx / longitude_steps  # 0 to 2π
            
            # Convert to Cartesian coordinates
            x = center.x + radius * math.sin(lat) * math.cos(lon)
            y = center.y + radius * math.sin(lat) * math.sin(lon)
            z = center.z + radius * math.cos(lat)
            
            yield Vector3D(x, y, z)


def cylinder_generator(center: Vector3D, radius: float, height: float,
                      radial_steps: int, height_steps: int) -> Generator[Vector3D, None, None]:
    """Generate points on a cylinder surface."""
    for h_idx in range(height_steps):
        for r_idx in range(radial_steps):
            # Calculate cylindrical coordinates
            angle = 2 * math.pi * r_idx / radial_steps
            h = height * h_idx / (height_steps - 1)
            
            # Convert to Cartesian coordinates
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            z = center.z + h
            
            yield Vector3D(x, y, z)


def random_points_generator(center: Vector3D, radius: float, count: int) -> Generator[Vector3D, None, None]:
    """Generate random points within a sphere."""
    for _ in range(count):
        # Generate random point within unit sphere
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            
            if x**2 + y**2 + z**2 <= 1:
                # Scale to desired radius and translate to center
                yield Vector3D(
                    center.x + x * radius,
                    center.y + y * radius,
                    center.z + z * radius
                )
                break


def noise_generator(base_points: List[Vector3D], amplitude: float, frequency: float) -> Generator[Vector3D, None, None]:
    """Generate noisy versions of base points."""
    for point in base_points:
        # Simple noise function
        noise_x = amplitude * math.sin(frequency * point.x)
        noise_y = amplitude * math.cos(frequency * point.y)
        noise_z = amplitude * math.sin(frequency * point.z)
        
        yield Vector3D(
            point.x + noise_x,
            point.y + noise_y,
            point.z + noise_z
        )


# Color and Material Generators
def color_gradient_generator(start_color: Color, end_color: Color, steps: int) -> Generator[Color, None, None]:
    """Generate color gradient between two colors."""
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        r = start_color.r + (end_color.r - start_color.r) * t
        g = start_color.g + (end_color.g - start_color.g) * t
        b = start_color.b + (end_color.b - start_color.b) * t
        a = start_color.a + (end_color.a - start_color.a) * t
        yield Color(r, g, b, a)


def rainbow_generator(steps: int) -> Generator[Color, None, None]:
    """Generate rainbow colors."""
    for i in range(steps):
        hue = i / steps
        # Convert HSV to RGB (simplified)
        r = abs(3 * hue - 1.5) - 0.5
        g = abs(3 * hue - 0.5) - 0.5
        b = abs(3 * hue - 2.5) - 0.5
        
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        yield Color(r, g, b)


# Animation and Time-based Generators
def animation_generator(base_points: List[Vector3D], 
                       animation_func, 
                       duration: float, 
                       fps: int) -> Generator[Tuple[float, List[Vector3D]], None, None]:
    """Generate animation frames."""
    frame_count = int(duration * fps)
    for frame in range(frame_count):
        time = frame / fps
        animated_points = animation_func(base_points, time)
        yield time, animated_points


def rotation_animation_generator(points: List[Vector3D], 
                               axis: Vector3D, 
                               duration: float, 
                               fps: int) -> Generator[Tuple[float, List[Vector3D]], None, None]:
    """Generate rotation animation frames."""
    frame_count = int(duration * fps)
    for frame in range(frame_count):
        time = frame / fps
        angle = 2 * math.pi * time / duration
        
        # Simple rotation around axis
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        rotated_points = []
        for point in points:
            # Apply rotation matrix (simplified)
            if axis.x == 1:  # Rotate around X-axis
                new_y = point.y * cos_a - point.z * sin_a
                new_z = point.y * sin_a + point.z * cos_a
                rotated_points.append(Vector3D(point.x, new_y, new_z))
            elif axis.y == 1:  # Rotate around Y-axis
                new_x = point.x * cos_a + point.z * sin_a
                new_z = -point.x * sin_a + point.z * cos_a
                rotated_points.append(Vector3D(new_x, point.y, new_z))
            elif axis.z == 1:  # Rotate around Z-axis
                new_x = point.x * cos_a - point.y * sin_a
                new_y = point.x * sin_a + point.y * cos_a
                rotated_points.append(Vector3D(new_x, new_y, point.z))
        
        yield time, rotated_points


# Data Processing Generators
def filter_generator(iterator: Iterator, predicate) -> Generator:
    """Filter items from an iterator based on a predicate."""
    for item in iterator:
        if predicate(item):
            yield item


def transform_generator(iterator: Iterator, transform_func) -> Generator:
    """Transform items from an iterator using a function."""
    for item in iterator:
        yield transform_func(item)


def batch_generator(iterator: Iterator, batch_size: int) -> Generator[List, None, None]:
    """Group items from an iterator into batches."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:  # Yield remaining items
        yield batch


def window_generator(iterator: Iterator, window_size: int) -> Generator[List, None, None]:
    """Generate sliding windows over an iterator."""
    window = []
    for item in iterator:
        window.append(item)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            yield window.copy()


# Advanced Generators
def recursive_generator(node, get_children_func) -> Generator:
    """Recursively traverse a tree structure."""
    yield node
    for child in get_children_func(node):
        yield from recursive_generator(child, get_children_func)


def infinite_generator(start_value, next_func) -> Generator:
    """Generate infinite sequence using a function."""
    current = start_value
    while True:
        yield current
        current = next_func(current)


def merge_generators(*generators) -> Generator:
    """Merge multiple generators into one."""
    active_generators = list(generators)
    while active_generators:
        for i, gen in enumerate(active_generators):
            try:
                yield next(gen)
            except StopIteration:
                active_generators.pop(i)
                break


# Utility Functions
def distance_predicate(max_distance: float, reference_point: Vector3D):
    """Creates a predicate that filters points by distance."""
    def predicate(point: Vector3D) -> bool:
        dx = point.x - reference_point.x
        dy = point.y - reference_point.y
        dz = point.z - reference_point.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance <= max_distance
    return predicate


def magnitude_transform():
    """Creates a transform function that returns vector magnitudes."""
    return lambda vector: vector.magnitude()


# Example Usage and Demonstration
def demonstrate_generators():
    """Demonstrates various generator patterns with 3D examples."""
    print("=== Generators with 3D Graphics Applications ===\n")
    
    # Basic generators
    print("=== Basic Generators ===")
    
    print("Fibonacci numbers (first 10):")
    fib_gen = fibonacci_generator(10)
    for num in fib_gen:
        print(f"  {num}", end=" ")
    print()
    
    print("\nVector interpolation:")
    start_vec = Vector3D(0, 0, 0)
    end_vec = Vector3D(1, 1, 1)
    vec_gen = vector_interpolation_generator(start_vec, end_vec, 5)
    for vec in vec_gen:
        print(f"  {vec}")
    
    # 3D graphics generators
    print("\n=== 3D Graphics Generators ===")
    
    print("Grid generator (2x2x2):")
    grid_gen = grid_generator((0, 1, 2), (0, 1, 2), (0, 1, 2))
    for point in grid_gen:
        print(f"  {point}")
    
    print("\nSphere generator (4 points):")
    center = Vector3D(0, 0, 0)
    sphere_gen = sphere_generator(center, 1.0, 2, 2)
    for point in sphere_gen:
        print(f"  {point}")
    
    print("\nCylinder generator (4 points):")
    cylinder_gen = cylinder_generator(center, 1.0, 2.0, 2, 2)
    for point in cylinder_gen:
        print(f"  {point}")
    
    print("\nRandom points generator (3 points):")
    random_gen = random_points_generator(center, 1.0, 3)
    for point in random_gen:
        print(f"  {point}")
    
    # Color generators
    print("\n=== Color Generators ===")
    
    print("Color gradient:")
    start_color = Color(1.0, 0.0, 0.0)  # Red
    end_color = Color(0.0, 0.0, 1.0)    # Blue
    color_gen = color_gradient_generator(start_color, end_color, 5)
    for color in color_gen:
        print(f"  {color}")
    
    print("\nRainbow colors:")
    rainbow_gen = rainbow_generator(5)
    for color in rainbow_gen:
        print(f"  {color}")
    
    # Data processing generators
    print("\n=== Data Processing Generators ===")
    
    # Create some base points
    base_points = [
        Vector3D(0, 0, 0),
        Vector3D(1, 1, 1),
        Vector3D(2, 2, 2),
        Vector3D(3, 3, 3),
        Vector3D(4, 4, 4)
    ]
    
    print("Filtered points (distance <= 2.5 from origin):")
    grid_gen = grid_generator((0, 2, 3), (0, 2, 3), (0, 1, 1))
    filtered_gen = filter_generator(grid_gen, distance_predicate(2.5, Vector3D(0, 0, 0)))
    for point in filtered_gen:
        print(f"  {point}")
    
    print("\nTransformed vectors (magnitudes):")
    vec_gen = vector_interpolation_generator(Vector3D(0, 0, 0), Vector3D(1, 1, 1), 5)
    transformed_gen = transform_generator(vec_gen, magnitude_transform())
    for magnitude in transformed_gen:
        print(f"  {magnitude:.3f}")
    
    print("\nBatched points (batch size 2):")
    grid_gen = grid_generator((0, 1, 3), (0, 1, 3), (0, 1, 1))
    batched_gen = batch_generator(grid_gen, 2)
    for i, batch in enumerate(batched_gen):
        print(f"  Batch {i}: {[str(p) for p in batch]}")
    
    # Animation generators
    print("\n=== Animation Generators ===")
    
    def simple_animation(points, time):
        """Simple animation that moves points up and down."""
        return [Vector3D(p.x, p.y + math.sin(time * 2 * math.pi) * 0.5, p.z) for p in points]
    
    print("Animation frames (3 frames):")
    anim_gen = animation_generator(base_points[:3], simple_animation, 1.0, 3)
    for time, points in anim_gen:
        print(f"  Time {time:.2f}: {[str(p) for p in points]}")
    
    print("\nRotation animation (3 frames):")
    rot_gen = rotation_animation_generator(base_points[:3], Vector3D(0, 0, 1), 1.0, 3)
    for time, points in rot_gen:
        print(f"  Time {time:.2f}: {[str(p) for p in points]}")
    
    # Advanced generators
    print("\n=== Advanced Generators ===")
    
    print("Infinite sequence (first 5):")
    def next_fib(a, b):
        return b, a + b
    
    inf_gen = infinite_generator((0, 1), lambda pair: next_fib(*pair))
    count = 0
    for a, b in inf_gen:
        print(f"  {a}", end=" ")
        count += 1
        if count >= 5:
            break
    print()
    
    print("\nMerged generators:")
    gen1 = vector_interpolation_generator(Vector3D(0, 0, 0), Vector3D(1, 0, 0), 2)
    gen2 = vector_interpolation_generator(Vector3D(0, 0, 0), Vector3D(0, 1, 0), 2)
    merged_gen = merge_generators(gen1, gen2)
    for point in merged_gen:
        print(f"  {point}")


if __name__ == "__main__":
    demonstrate_generators()
