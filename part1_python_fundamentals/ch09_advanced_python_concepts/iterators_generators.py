#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Iterators and Generators Example

Demonstrates iterators and generators for memory-efficient data processing,
lazy evaluation, infinite sequences, and generator pipelines for 3D graphics.
"""

import math
import random
import time
from typing import List, Tuple, Iterator, Generator, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Iterators and Generators"
__description__ = "Iterators and generators for 3D graphics applications"

# ============================================================================
# CUSTOM ITERATORS
# ============================================================================

class Vector3Iterator:
    """Custom iterator for 3D vectors"""
    
    def __init__(self, vectors: List[List[float]]):
        self.vectors = vectors
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List[float]:
        if self.index >= len(self.vectors):
            raise StopIteration
        vector = self.vectors[self.index]
        self.index += 1
        return vector

class Range3DIterator:
    """Iterator for 3D coordinate ranges"""
    
    def __init__(self, start: List[int], end: List[int], step: List[int] = None):
        self.start = start
        self.end = end
        self.step = step or [1, 1, 1]
        self.current = start.copy()
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List[int]:
        if (self.current[0] >= self.end[0] or 
            self.current[1] >= self.end[1] or 
            self.current[2] >= self.end[2]):
            raise StopIteration
        
        result = self.current.copy()
        
        # Increment coordinates
        self.current[2] += self.step[2]
        if self.current[2] >= self.end[2]:
            self.current[2] = self.start[2]
            self.current[1] += self.step[1]
            if self.current[1] >= self.end[1]:
                self.current[1] = self.start[1]
                self.current[0] += self.step[0]
        
        return result

class SpiralIterator:
    """Iterator for spiral pattern generation"""
    
    def __init__(self, center: List[float], radius: float, steps: int):
        self.center = center
        self.radius = radius
        self.steps = steps
        self.current_step = 0
        self.angle_step = 2 * math.pi / steps
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List[float]:
        if self.current_step >= self.steps:
            raise StopIteration
        
        angle = self.current_step * self.angle_step
        radius_factor = self.current_step / self.steps
        
        x = self.center[0] + radius_factor * self.radius * math.cos(angle)
        y = self.center[1] + radius_factor * self.radius * math.sin(angle)
        z = self.center[2]
        
        self.current_step += 1
        return [x, y, z]

# ============================================================================
# GENERATOR FUNCTIONS
# ============================================================================

def generate_vertices_2d(width: int, height: int, spacing: float = 1.0) -> Generator[List[float], None, None]:
    """Generate 2D grid of vertices"""
    for y in range(height):
        for x in range(width):
            yield [x * spacing, y * spacing, 0.0]

def generate_vertices_3d(width: int, height: int, depth: int, spacing: float = 1.0) -> Generator[List[float], None, None]:
    """Generate 3D grid of vertices"""
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                yield [x * spacing, y * spacing, z * spacing]

def generate_sphere_vertices(center: List[float], radius: float, segments: int, rings: int) -> Generator[List[float], None, None]:
    """Generate sphere vertices using spherical coordinates"""
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            
            x = center[0] + radius * math.sin(phi) * math.cos(theta)
            y = center[1] + radius * math.sin(phi) * math.sin(theta)
            z = center[2] + radius * math.cos(phi)
            
            yield [x, y, z]

def generate_cube_vertices(center: List[float], size: float) -> Generator[List[float], None, None]:
    """Generate cube vertices"""
    half_size = size / 2
    offsets = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
    ]
    
    for offset in offsets:
        yield [
            center[0] + offset[0] * half_size,
            center[1] + offset[1] * half_size,
            center[2] + offset[2] * half_size
        ]

def generate_random_points(count: int, bounds: List[float]) -> Generator[List[float], None, None]:
    """Generate random 3D points within bounds"""
    for _ in range(count):
        yield [
            random.uniform(-bounds[0], bounds[0]),
            random.uniform(-bounds[1], bounds[1]),
            random.uniform(-bounds[2], bounds[2])
        ]

def generate_fibonacci_spiral(center: List[float], max_radius: float, points: int) -> Generator[List[float], None, None]:
    """Generate Fibonacci spiral pattern"""
    golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle
    
    for i in range(points):
        radius = max_radius * math.sqrt(i) / math.sqrt(points)
        angle = golden_angle * i
        
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        
        yield [x, y, z]

# ============================================================================
# INFINITE GENERATORS
# ============================================================================

def infinite_counter(start: int = 0) -> Generator[int, None, None]:
    """Infinite counter generator"""
    i = start
    while True:
        yield i
        i += 1

def infinite_random_vectors() -> Generator[List[float], None, None]:
    """Infinite generator of random 3D vectors"""
    while True:
        yield [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ]

def infinite_spiral(center: List[float], radius: float, angle_step: float = 0.1) -> Generator[List[float], None, None]:
    """Infinite spiral generator"""
    angle = 0
    while True:
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        
        yield [x, y, z]
        angle += angle_step

def infinite_wave(amplitude: float, frequency: float, phase: float = 0) -> Generator[float, None, None]:
    """Infinite sine wave generator"""
    t = 0
    while True:
        yield amplitude * math.sin(frequency * t + phase)
        t += 0.1

# ============================================================================
# GENERATOR PIPELINES
# ============================================================================

def filter_by_distance(vertices: Generator[List[float], None, None], 
                      center: List[float], max_distance: float) -> Generator[List[float], None, None]:
    """Filter vertices by distance from center"""
    for vertex in vertices:
        distance = math.sqrt(sum((vertex[i] - center[i]) ** 2 for i in range(3)))
        if distance <= max_distance:
            yield vertex

def transform_vertices(vertices: Generator[List[float], None, None], 
                      transform_func) -> Generator[List[float], None, None]:
    """Transform vertices using a function"""
    for vertex in vertices:
        yield transform_func(vertex)

def batch_vertices(vertices: Generator[List[float], None, None], 
                  batch_size: int) -> Generator[List[List[float]], None, None]:
    """Group vertices into batches"""
    batch = []
    for vertex in vertices:
        batch.append(vertex)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:  # Yield remaining vertices
        yield batch

def limit_vertices(vertices: Generator[List[float], None, None], 
                  limit: int) -> Generator[List[float], None, None]:
    """Limit the number of vertices generated"""
    count = 0
    for vertex in vertices:
        if count >= limit:
            break
        yield vertex
        count += 1

def add_noise(vertices: Generator[List[float], None, None], 
              noise_amplitude: float) -> Generator[List[float], None, None]:
    """Add random noise to vertices"""
    for vertex in vertices:
        noisy_vertex = [
            vertex[0] + random.uniform(-noise_amplitude, noise_amplitude),
            vertex[1] + random.uniform(-noise_amplitude, noise_amplitude),
            vertex[2] + random.uniform(-noise_amplitude, noise_amplitude)
        ]
        yield noisy_vertex

# ============================================================================
# MEMORY-EFFICIENT PROCESSING
# ============================================================================

class VertexProcessor:
    """Memory-efficient vertex processor using generators"""
    
    def __init__(self):
        self.processed_count = 0
    
    def process_large_dataset(self, vertex_generator, processor_func, batch_size: int = 1000):
        """Process large datasets without loading everything into memory"""
        batch = []
        
        for vertex in vertex_generator:
            processed_vertex = processor_func(vertex)
            batch.append(processed_vertex)
            
            if len(batch) >= batch_size:
                # Process batch
                yield from self._process_batch(batch)
                batch = []
                self.processed_count += batch_size
        
        # Process remaining vertices
        if batch:
            yield from self._process_batch(batch)
            self.processed_count += len(batch)
    
    def _process_batch(self, batch: List[List[float]]) -> Generator[List[float], None, None]:
        """Process a batch of vertices"""
        for vertex in batch:
            yield vertex
    
    def get_statistics(self) -> dict:
        """Get processing statistics"""
        return {
            "processed_count": self.processed_count,
            "memory_efficient": True
        }

# ============================================================================
# LAZY EVALUATION EXAMPLES
# ============================================================================

def lazy_distance_calculation(vertices: Generator[List[float], None, None], 
                            target: List[float]) -> Generator[Tuple[List[float], float], None, None]:
    """Lazy distance calculation - only calculate when needed"""
    for vertex in vertices:
        distance = math.sqrt(sum((vertex[i] - target[i]) ** 2 for i in range(3)))
        yield vertex, distance

def lazy_normal_calculation(vertices: Generator[List[float], None, None]) -> Generator[List[float], None, None]:
    """Lazy normal calculation for vertices"""
    for vertex in vertices:
        magnitude = math.sqrt(sum(vertex[i] ** 2 for i in range(3)))
        if magnitude > 0:
            normal = [vertex[i] / magnitude for i in range(3)]
        else:
            normal = [0.0, 0.0, 0.0]
        yield normal

def lazy_bounding_box(vertices: Generator[List[float], None, None]) -> Generator[Tuple[List[float], List[float]], None, None]:
    """Lazy bounding box calculation"""
    min_coords = None
    max_coords = None
    
    for vertex in vertices:
        if min_coords is None:
            min_coords = vertex.copy()
            max_coords = vertex.copy()
        else:
            for i in range(3):
                min_coords[i] = min(min_coords[i], vertex[i])
                max_coords[i] = max(max_coords[i], vertex[i])
        
        yield min_coords.copy(), max_coords.copy()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_custom_iterators():
    """Demonstrate custom iterators"""
    print("=== Custom Iterators Demo ===\n")
    
    # Vector3Iterator
    vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    vector_iter = Vector3Iterator(vectors)
    
    print("1. Vector3Iterator:")
    for vector in vector_iter:
        print(f"   {vector}")
    
    # Range3DIterator
    print("\n2. Range3DIterator:")
    range_iter = Range3DIterator([0, 0, 0], [2, 2, 2])
    for coords in range_iter:
        print(f"   {coords}")
    
    # SpiralIterator
    print("\n3. SpiralIterator:")
    spiral_iter = SpiralIterator([0, 0, 0], 5.0, 8)
    for point in spiral_iter:
        print(f"   {[round(p, 2) for p in point]}")
    
    print()

def demonstrate_generators():
    """Demonstrate generator functions"""
    print("=== Generator Functions Demo ===\n")
    
    # 2D Grid
    print("1. 2D Grid vertices (3x3):")
    for vertex in generate_vertices_2d(3, 3):
        print(f"   {vertex}")
    
    # Sphere vertices
    print("\n2. Sphere vertices (segments=4, rings=2):")
    sphere_vertices = list(generate_sphere_vertices([0, 0, 0], 2.0, 4, 2))
    for i, vertex in enumerate(sphere_vertices[:8]):  # Show first 8
        print(f"   Vertex {i}: {[round(v, 2) for v in vertex]}")
    
    # Cube vertices
    print("\n3. Cube vertices:")
    for vertex in generate_cube_vertices([0, 0, 0], 2.0):
        print(f"   {vertex}")
    
    # Random points
    print("\n4. Random points (5 points):")
    for i, point in enumerate(generate_random_points(5, [5, 5, 5])):
        print(f"   Point {i}: {[round(p, 2) for p in point]}")
    
    # Fibonacci spiral
    print("\n5. Fibonacci spiral (5 points):")
    for i, point in enumerate(generate_fibonacci_spiral([0, 0, 0], 3.0, 5)):
        print(f"   Point {i}: {[round(p, 2) for p in point]}")
    
    print()

def demonstrate_infinite_generators():
    """Demonstrate infinite generators"""
    print("=== Infinite Generators Demo ===\n")
    
    # Infinite counter
    print("1. Infinite counter (first 5):")
    counter = infinite_counter(10)
    for i, num in enumerate(counter):
        if i >= 5:
            break
        print(f"   {num}")
    
    # Infinite random vectors
    print("\n2. Infinite random vectors (first 3):")
    random_vectors = infinite_random_vectors()
    for i, vector in enumerate(random_vectors):
        if i >= 3:
            break
        print(f"   {[round(v, 2) for v in vector]}")
    
    # Infinite spiral
    print("\n3. Infinite spiral (first 5 points):")
    spiral = infinite_spiral([0, 0, 0], 2.0)
    for i, point in enumerate(spiral):
        if i >= 5:
            break
        print(f"   {[round(p, 2) for p in point]}")
    
    # Infinite wave
    print("\n4. Infinite wave (first 5 values):")
    wave = infinite_wave(amplitude=2.0, frequency=1.0)
    for i, value in enumerate(wave):
        if i >= 5:
            break
        print(f"   {round(value, 2)}")
    
    print()

def demonstrate_generator_pipelines():
    """Demonstrate generator pipelines"""
    print("=== Generator Pipelines Demo ===\n")
    
    # Create a pipeline
    print("1. Pipeline: Generate vertices → Add noise → Filter by distance:")
    vertices = generate_vertices_2d(5, 5)
    noisy_vertices = add_noise(vertices, 0.1)
    filtered_vertices = filter_by_distance(noisy_vertices, [2, 2, 0], 3.0)
    
    count = 0
    for vertex in filtered_vertices:
        print(f"   {[round(v, 2) for v in vertex]}")
        count += 1
        if count >= 5:  # Limit output
            break
    
    # Transform pipeline
    print("\n2. Transform pipeline: Scale vertices by 2:")
    vertices = generate_cube_vertices([0, 0, 0], 1.0)
    scaled_vertices = transform_vertices(vertices, lambda v: [v[i] * 2 for i in range(3)])
    
    for vertex in scaled_vertices:
        print(f"   {vertex}")
    
    # Batch processing
    print("\n3. Batch processing:")
    vertices = generate_random_points(10, [2, 2, 2])
    batched = batch_vertices(vertices, 3)
    
    for i, batch in enumerate(batched):
        print(f"   Batch {i}: {len(batch)} vertices")
        for vertex in batch:
            print(f"     {[round(v, 2) for v in vertex]}")
    
    print()

def demonstrate_memory_efficiency():
    """Demonstrate memory-efficient processing"""
    print("=== Memory Efficiency Demo ===\n")
    
    # Large dataset processing
    print("1. Processing large dataset (simulated):")
    processor = VertexProcessor()
    
    def large_vertex_generator():
        """Simulate large vertex dataset"""
        for i in range(10000):
            yield [i, i * 0.5, i * 0.25]
    
    def process_vertex(vertex):
        """Process individual vertex"""
        return [vertex[0] * 2, vertex[1] * 2, vertex[2] * 2]
    
    processed_count = 0
    for processed_vertex in processor.process_large_dataset(
        large_vertex_generator(), process_vertex, batch_size=1000
    ):
        processed_count += 1
        if processed_count <= 5:  # Show first 5
            print(f"   Processed: {processed_vertex}")
    
    print(f"   Total processed: {processor.get_statistics()['processed_count']}")
    
    # Memory comparison
    print("\n2. Memory usage comparison:")
    print("   Generator approach: Processes one vertex at a time")
    print("   List approach: Would load all vertices into memory")
    print("   Memory savings: Significant for large datasets")
    
    print()

def demonstrate_lazy_evaluation():
    """Demonstrate lazy evaluation"""
    print("=== Lazy Evaluation Demo ===\n")
    
    # Lazy distance calculation
    print("1. Lazy distance calculation:")
    vertices = generate_random_points(5, [2, 2, 2])
    target = [1, 1, 1]
    
    for vertex, distance in lazy_distance_calculation(vertices, target):
        print(f"   Vertex: {[round(v, 2) for v in vertex]}, Distance: {round(distance, 2)}")
    
    # Lazy normal calculation
    print("\n2. Lazy normal calculation:")
    vertices = generate_cube_vertices([0, 0, 0], 2.0)
    
    for vertex, normal in zip(vertices, lazy_normal_calculation(generate_cube_vertices([0, 0, 0], 2.0))):
        print(f"   Vertex: {vertex} → Normal: {[round(n, 2) for n in normal]}")
    
    # Lazy bounding box
    print("\n3. Lazy bounding box calculation:")
    vertices = generate_random_points(5, [3, 3, 3])
    
    for i, (min_coords, max_coords) in enumerate(lazy_bounding_box(vertices)):
        print(f"   After vertex {i}: Min={[round(m, 2) for m in min_coords]}, Max={[round(m, 2) for m in max_coords]}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate iterators and generators"""
    print("=== Iterators and Generators Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_custom_iterators()
    demonstrate_generators()
    demonstrate_infinite_generators()
    demonstrate_generator_pipelines()
    demonstrate_memory_efficiency()
    demonstrate_lazy_evaluation()
    
    print("="*60)
    print("Iterators and Generators demo completed successfully!")
    print("\nKey features:")
    print("✓ Custom iterators: Specialized iteration patterns")
    print("✓ Generator functions: Memory-efficient data generation")
    print("✓ Infinite generators: Lazy infinite sequences")
    print("✓ Generator pipelines: Composable data processing")
    print("✓ Memory efficiency: Process large datasets without loading into memory")
    print("✓ Lazy evaluation: Compute values only when needed")

if __name__ == "__main__":
    main()
