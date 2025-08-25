"""
Chapter 10: Iterators and Generators - Iterators
===============================================

This module demonstrates iterators in Python, applied to 3D graphics
and mathematical operations. Iterators provide a way to traverse data
structures efficiently and create custom iteration patterns.

Key Concepts:
- Iterator protocol
- Custom iterators
- Iterator composition
- Memory-efficient iteration
- 3D data traversal
"""

import math
from typing import Iterator, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Vector3D:
    """3D vector for demonstrating iterators."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __iter__(self):
        """Make Vector3D iterable - yields x, y, z components."""
        yield self.x
        yield self.y
        yield self.z
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class Triangle:
    """Triangle defined by three vertices."""
    v1: Vector3D
    v2: Vector3D
    v3: Vector3D
    
    def __iter__(self):
        """Make Triangle iterable - yields vertices."""
        yield self.v1
        yield self.v2
        yield self.v3
    
    def edges(self) -> Iterator[Tuple[Vector3D, Vector3D]]:
        """Iterator over triangle edges."""
        yield (self.v1, self.v2)
        yield (self.v2, self.v3)
        yield (self.v3, self.v1)


# Basic Iterator Examples
class RangeIterator:
    """Custom iterator that mimics range behavior."""
    
    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        
        result = self.current
        self.current += self.step
        return result


class VectorIterator:
    """Iterator that generates vectors in a pattern."""
    
    def __init__(self, start: Vector3D, end: Vector3D, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.current_step = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Vector3D:
        if self.current_step >= self.steps:
            raise StopIteration
        
        # Linear interpolation between start and end
        t = self.current_step / (self.steps - 1) if self.steps > 1 else 0
        
        x = self.start.x + (self.end.x - self.start.x) * t
        y = self.start.y + (self.end.y - self.start.y) * t
        z = self.start.z + (self.end.z - self.start.z) * t
        
        self.current_step += 1
        return Vector3D(x, y, z)


# 3D Graphics Specific Iterators
class GridIterator:
    """Iterator that generates points in a 3D grid."""
    
    def __init__(self, 
                 x_range: Tuple[float, float, int],
                 y_range: Tuple[float, float, int],
                 z_range: Tuple[float, float, int]):
        """
        Initialize grid iterator.
        
        Args:
            x_range: (start, end, steps) for x-axis
            y_range: (start, end, steps) for y-axis
            z_range: (start, end, steps) for z-axis
        """
        self.x_start, self.x_end, self.x_steps = x_range
        self.y_start, self.y_end, self.y_steps = y_range
        self.z_start, self.z_end, self.z_steps = z_range
        
        self.total_points = self.x_steps * self.y_steps * self.z_steps
        self.current_point = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Vector3D:
        if self.current_point >= self.total_points:
            raise StopIteration
        
        # Calculate 3D indices
        point_index = self.current_point
        z_index = point_index % self.z_steps
        point_index //= self.z_steps
        y_index = point_index % self.y_steps
        x_index = point_index // self.y_steps
        
        # Calculate coordinates
        x = self.x_start + (self.x_end - self.x_start) * x_index / (self.x_steps - 1)
        y = self.y_start + (self.y_end - self.y_start) * y_index / (self.y_steps - 1)
        z = self.z_start + (self.z_end - self.z_start) * z_index / (self.z_steps - 1)
        
        self.current_point += 1
        return Vector3D(x, y, z)


class SphereIterator:
    """Iterator that generates points on a sphere surface."""
    
    def __init__(self, center: Vector3D, radius: float, 
                 latitude_steps: int, longitude_steps: int):
        self.center = center
        self.radius = radius
        self.latitude_steps = latitude_steps
        self.longitude_steps = longitude_steps
        self.current_lat = 0
        self.current_lon = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Vector3D:
        if self.current_lat >= self.latitude_steps:
            raise StopIteration
        
        # Calculate spherical coordinates
        lat = math.pi * (self.current_lat + 1) / (self.latitude_steps + 1)  # 0 to π
        lon = 2 * math.pi * self.current_lon / self.longitude_steps  # 0 to 2π
        
        # Convert to Cartesian coordinates
        x = self.center.x + self.radius * math.sin(lat) * math.cos(lon)
        y = self.center.y + self.radius * math.sin(lat) * math.sin(lon)
        z = self.center.z + self.radius * math.cos(lat)
        
        # Update indices
        self.current_lon += 1
        if self.current_lon >= self.longitude_steps:
            self.current_lon = 0
            self.current_lat += 1
        
        return Vector3D(x, y, z)


class MeshIterator:
    """Iterator that traverses a 3D mesh structure."""
    
    def __init__(self, vertices: List[Vector3D], faces: List[Tuple[int, int, int]]):
        self.vertices = vertices
        self.faces = faces
        self.current_face = 0
        self.current_vertex_in_face = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Vector3D:
        if self.current_face >= len(self.faces):
            raise StopIteration
        
        face = self.faces[self.current_face]
        vertex_index = face[self.current_vertex_in_face]
        vertex = self.vertices[vertex_index]
        
        # Move to next vertex in face
        self.current_vertex_in_face += 1
        if self.current_vertex_in_face >= 3:  # Triangle faces
            self.current_vertex_in_face = 0
            self.current_face += 1
        
        return vertex


# Iterator Composition and Utilities
class FilteredIterator:
    """Iterator that filters another iterator based on a predicate."""
    
    def __init__(self, iterator: Iterator, predicate):
        self.iterator = iterator
        self.predicate = predicate
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            item = next(self.iterator)
            if self.predicate(item):
                return item


class TransformedIterator:
    """Iterator that transforms items from another iterator."""
    
    def __init__(self, iterator: Iterator, transform_func):
        self.iterator = iterator
        self.transform_func = transform_func
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = next(self.iterator)
        return self.transform_func(item)


class ChunkedIterator:
    """Iterator that yields chunks of items from another iterator."""
    
    def __init__(self, iterator: Iterator, chunk_size: int):
        self.iterator = iterator
        self.chunk_size = chunk_size
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List:
        chunk = []
        try:
            for _ in range(self.chunk_size):
                chunk.append(next(self.iterator))
        except StopIteration:
            if not chunk:
                raise
        return chunk


# Advanced Iterator Patterns
class BreadthFirstIterator:
    """Breadth-first traversal iterator for hierarchical structures."""
    
    def __init__(self, root):
        self.queue = [root] if root else []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.queue:
            raise StopIteration
        
        current = self.queue.pop(0)
        
        # Add children to queue (assuming children property)
        if hasattr(current, 'children'):
            self.queue.extend(current.children)
        
        return current


class DepthFirstIterator:
    """Depth-first traversal iterator for hierarchical structures."""
    
    def __init__(self, root):
        self.stack = [root] if root else []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        
        current = self.stack.pop()
        
        # Add children to stack in reverse order (assuming children property)
        if hasattr(current, 'children'):
            self.stack.extend(reversed(current.children))
        
        return current


# Utility Functions
def distance_filter(max_distance: float, reference_point: Vector3D):
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
def demonstrate_iterators():
    """Demonstrates various iterator patterns with 3D examples."""
    print("=== Iterators with 3D Graphics Applications ===\n")
    
    # Basic iterator examples
    print("=== Basic Iterators ===")
    
    # Range iterator
    print("Range iterator (0 to 10, step 2):")
    range_iter = RangeIterator(0, 10, 2)
    for i in range_iter:
        print(f"  {i}", end=" ")
    print()
    
    # Vector iterator
    print("\nVector iterator (interpolation):")
    start_vec = Vector3D(0, 0, 0)
    end_vec = Vector3D(1, 1, 1)
    vector_iter = VectorIterator(start_vec, end_vec, 5)
    for vec in vector_iter:
        print(f"  {vec}")
    
    # 3D graphics iterators
    print("\n=== 3D Graphics Iterators ===")
    
    # Grid iterator
    print("Grid iterator (2x2x2 grid):")
    grid_iter = GridIterator((0, 1, 2), (0, 1, 2), (0, 1, 2))
    for point in grid_iter:
        print(f"  {point}")
    
    # Sphere iterator
    print("\nSphere iterator (8 points on sphere):")
    center = Vector3D(0, 0, 0)
    sphere_iter = SphereIterator(center, 1.0, 2, 4)
    for point in sphere_iter:
        print(f"  {point}")
    
    # Iterator composition
    print("\n=== Iterator Composition ===")
    
    # Create a grid and filter by distance
    print("Filtered grid (points within 1.5 units of origin):")
    grid_iter = GridIterator((0, 2, 3), (0, 2, 3), (0, 2, 3))
    filtered_iter = FilteredIterator(grid_iter, distance_filter(1.5, Vector3D(0, 0, 0)))
    for point in filtered_iter:
        print(f"  {point}")
    
    # Transform vectors to magnitudes
    print("\nTransformed vectors (magnitudes):")
    vector_iter = VectorIterator(Vector3D(0, 0, 0), Vector3D(1, 1, 1), 5)
    transformed_iter = TransformedIterator(vector_iter, magnitude_transform())
    for magnitude in transformed_iter:
        print(f"  {magnitude:.3f}")
    
    # Chunked iteration
    print("\nChunked iteration (groups of 3):")
    grid_iter = GridIterator((0, 1, 3), (0, 1, 3), (0, 1, 1))
    chunked_iter = ChunkedIterator(grid_iter, 3)
    for i, chunk in enumerate(chunked_iter):
        print(f"  Chunk {i}: {[str(p) for p in chunk]}")
    
    # Mesh iteration
    print("\n=== Mesh Iteration ===")
    
    # Create a simple cube mesh
    vertices = [
        Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(1, 1, 0), Vector3D(0, 1, 0),
        Vector3D(0, 0, 1), Vector3D(1, 0, 1), Vector3D(1, 1, 1), Vector3D(0, 1, 1)
    ]
    
    # Cube faces (triangles)
    faces = [
        (0, 1, 2), (0, 2, 3),  # Front face
        (4, 6, 5), (4, 7, 6),  # Back face
        (0, 4, 5), (0, 5, 1),  # Bottom face
        (2, 6, 7), (2, 7, 3),  # Top face
        (0, 3, 7), (0, 7, 4),  # Left face
        (1, 5, 6), (1, 6, 2)   # Right face
    ]
    
    print("Mesh vertex iteration:")
    mesh_iter = MeshIterator(vertices, faces)
    for vertex in mesh_iter:
        print(f"  {vertex}")
    
    # Triangle edge iteration
    print("\nTriangle edge iteration:")
    triangle = Triangle(Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 1, 0))
    for edge in triangle.edges():
        print(f"  Edge: {edge[0]} to {edge[1]}")


if __name__ == "__main__":
    demonstrate_iterators()
