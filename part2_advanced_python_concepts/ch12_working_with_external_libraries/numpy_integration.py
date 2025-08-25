"""
Chapter 12: Working with External Libraries - NumPy Integration
==============================================================

This module demonstrates how to integrate NumPy for efficient 3D graphics
operations, including vector operations, matrix transformations, and
performance optimization.

Key Concepts:
- NumPy arrays for 3D data
- Vectorized operations
- Matrix transformations
- Performance optimization
- Memory management
- Broadcasting and advanced indexing
"""

import numpy as np
import time
import math
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Vector3D:
    """3D vector using NumPy for efficient operations."""
    x: float
    y: float
    z: float
    
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        """Create from NumPy array."""
        return cls(arr[0], arr[1], arr[2])
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude using NumPy."""
        return np.linalg.norm(self.to_array())
    
    def normalize(self) -> 'Vector3D':
        """Normalize the vector."""
        arr = self.to_array()
        norm = np.linalg.norm(arr)
        if norm > 0:
            return Vector3D.from_array(arr / norm)
        return Vector3D(0, 0, 0)
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product with another vector."""
        return np.dot(self.to_array(), other.to_array())
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product with another vector."""
        result = np.cross(self.to_array(), other.to_array())
        return Vector3D.from_array(result)


class Matrix4x4:
    """4x4 transformation matrix using NumPy."""
    
    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self.matrix = np.eye(4, dtype=np.float32)
        else:
            self.matrix = matrix.astype(np.float32)
    
    def __str__(self):
        return f"Matrix4x4(\n{self.matrix}\n)"
    
    @classmethod
    def identity(cls) -> 'Matrix4x4':
        """Create identity matrix."""
        return cls(np.eye(4))
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create translation matrix."""
        matrix = np.eye(4)
        matrix[0:3, 3] = [x, y, z]
        return cls(matrix)
    
    @classmethod
    def scaling(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create scaling matrix."""
        matrix = np.eye(4)
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
        return cls(matrix)
    
    @classmethod
    def rotation_x(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around X axis."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4)
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_y(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Y axis."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4)
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return cls(matrix)
    
    @classmethod
    def rotation_z(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Z axis."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        matrix = np.eye(4)
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return cls(matrix)
    
    def multiply(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Multiply with another matrix."""
        result = np.dot(self.matrix, other.matrix)
        return Matrix4x4(result)
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transform a 3D point."""
        point_array = np.array([point.x, point.y, point.z, 1.0])
        transformed = np.dot(self.matrix, point_array)
        return Vector3D(transformed[0], transformed[1], transformed[2])
    
    def transform_vector(self, vector: Vector3D) -> Vector3D:
        """Transform a 3D vector (ignores translation)."""
        vector_array = np.array([vector.x, vector.y, vector.z, 0.0])
        transformed = np.dot(self.matrix, vector_array)
        return Vector3D(transformed[0], transformed[1], transformed[2])


class NumPy3DProcessor:
    """Processor for 3D operations using NumPy."""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
    
    def load_mesh(self, vertices: List[Vector3D], faces: List[List[int]]):
        """Load mesh data into NumPy arrays."""
        # Convert vertices to NumPy array
        self.vertices = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    def calculate_normals(self) -> np.ndarray:
        """Calculate face normals using NumPy."""
        if self.vertices is None or self.faces is None:
            raise ValueError("No mesh loaded")
        
        normals = np.zeros((len(self.faces), 3), dtype=np.float32)
        
        for i, face in enumerate(self.faces):
            # Get vertices of the face
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            # Calculate edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            # Calculate normal using cross product
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            
            if norm > 0:
                normal = normal / norm
            
            normals[i] = normal
        
        return normals
    
    def transform_mesh(self, matrix: Matrix4x4) -> np.ndarray:
        """Transform all vertices using NumPy."""
        if self.vertices is None:
            raise ValueError("No mesh loaded")
        
        # Add homogeneous coordinate
        homogeneous_vertices = np.column_stack([self.vertices, np.ones(len(self.vertices))])
        
        # Transform vertices
        transformed = np.dot(homogeneous_vertices, matrix.matrix.T)
        
        # Return 3D coordinates
        return transformed[:, :3]
    
    def calculate_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Calculate bounding box using NumPy."""
        if self.vertices is None:
            raise ValueError("No mesh loaded")
        
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        
        return Vector3D.from_array(min_coords), Vector3D.from_array(max_coords)
    
    def calculate_centroid(self) -> Vector3D:
        """Calculate mesh centroid using NumPy."""
        if self.vertices is None:
            raise ValueError("No mesh loaded")
        
        centroid = np.mean(self.vertices, axis=0)
        return Vector3D.from_array(centroid)


class NumPyVectorizedOperations:
    """Demonstrates vectorized operations with NumPy."""
    
    @staticmethod
    def batch_vector_operations(vectors: List[Vector3D], operation: str) -> List[Vector3D]:
        """Perform batch operations on vectors."""
        # Convert to NumPy array
        vectors_array = np.array([[v.x, v.y, v.z] for v in vectors])
        
        if operation == "normalize":
            # Vectorized normalization
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            normalized = vectors_array / np.where(norms > 0, norms, 1)
            return [Vector3D.from_array(v) for v in normalized]
        
        elif operation == "scale":
            # Vectorized scaling
            scaled = vectors_array * 2.0
            return [Vector3D.from_array(v) for v in scaled]
        
        elif operation == "magnitude":
            # Vectorized magnitude calculation
            magnitudes = np.linalg.norm(vectors_array, axis=1)
            return magnitudes.tolist()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def distance_matrix(points: List[Vector3D]) -> np.ndarray:
        """Calculate distance matrix between all points."""
        points_array = np.array([[p.x, p.y, p.z] for p in points])
        
        # Calculate pairwise distances using broadcasting
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        
        return distances
    
    @staticmethod
    def interpolate_points(start: Vector3D, end: Vector3D, steps: int) -> List[Vector3D]:
        """Interpolate between two points using NumPy."""
        start_array = start.to_array()
        end_array = end.to_array()
        
        # Create interpolation parameter
        t = np.linspace(0, 1, steps)
        
        # Vectorized interpolation
        interpolated = start_array + t[:, np.newaxis] * (end_array - start_array)
        
        return [Vector3D.from_array(point) for point in interpolated]


class NumPyPerformanceComparison:
    """Compares performance between pure Python and NumPy operations."""
    
    @staticmethod
    def compare_vector_operations(num_vectors: int = 10000):
        """Compare vector operation performance."""
        print(f"=== Performance Comparison ({num_vectors} vectors) ===\n")
        
        # Generate test data
        vectors = [Vector3D(np.random.random(), np.random.random(), np.random.random()) 
                  for _ in range(num_vectors)]
        
        # Pure Python magnitude calculation
        start_time = time.time()
        python_magnitudes = [v.magnitude() for v in vectors]
        python_time = time.time() - start_time
        
        # NumPy magnitude calculation
        start_time = time.time()
        vectors_array = np.array([[v.x, v.y, v.z] for v in vectors])
        numpy_magnitudes = np.linalg.norm(vectors_array, axis=1)
        numpy_time = time.time() - start_time
        
        print(f"Pure Python magnitude calculation: {python_time:.6f}s")
        print(f"NumPy magnitude calculation: {numpy_time:.6f}s")
        print(f"Speedup: {python_time / numpy_time:.2f}x")
        
        # Verify results are similar
        python_array = np.array(python_magnitudes)
        numpy_array = numpy_magnitudes
        max_diff = np.max(np.abs(python_array - numpy_array))
        print(f"Maximum difference: {max_diff:.2e}")
    
    @staticmethod
    def compare_matrix_operations(num_matrices: int = 1000):
        """Compare matrix operation performance."""
        print(f"\n=== Matrix Operations Comparison ({num_matrices} matrices) ===\n")
        
        # Generate test matrices
        matrices = [Matrix4x4(np.random.random((4, 4))) for _ in range(num_matrices)]
        
        # Pure Python matrix multiplication
        start_time = time.time()
        for i in range(num_matrices - 1):
            result = matrices[i].multiply(matrices[i + 1])
        python_time = time.time() - start_time
        
        # NumPy matrix multiplication
        start_time = time.time()
        matrices_array = np.array([m.matrix for m in matrices])
        for i in range(num_matrices - 1):
            result = np.dot(matrices_array[i], matrices_array[i + 1])
        numpy_time = time.time() - start_time
        
        print(f"Pure Python matrix multiplication: {python_time:.6f}s")
        print(f"NumPy matrix multiplication: {numpy_time:.6f}s")
        print(f"Speedup: {python_time / numpy_time:.2f}x")


# Example Usage and Demonstration
def demonstrate_numpy_integration():
    """Demonstrates NumPy integration for 3D graphics."""
    print("=== NumPy Integration for 3D Graphics ===\n")
    
    # Vector operations
    print("=== Vector Operations ===")
    
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1.magnitude(): {v1.magnitude():.3f}")
    print(f"v1.normalize(): {v1.normalize()}")
    print(f"v1.dot(v2): {v1.dot(v2):.3f}")
    print(f"v1.cross(v2): {v1.cross(v2)}")
    
    # Matrix operations
    print("\n=== Matrix Operations ===")
    
    translation = Matrix4x4.translation(1, 2, 3)
    scaling = Matrix4x4.scaling(2, 2, 2)
    rotation = Matrix4x4.rotation_z(np.pi / 4)
    
    print(f"Translation matrix:\n{translation}")
    print(f"Scaling matrix:\n{scaling}")
    print(f"Rotation matrix:\n{rotation}")
    
    # Transform point
    point = Vector3D(1, 0, 0)
    transformed = translation.transform_point(point)
    print(f"Transformed point {point} -> {transformed}")
    
    # Mesh processing
    print("\n=== Mesh Processing ===")
    
    # Create a simple cube mesh
    vertices = [
        Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(1, 1, 0), Vector3D(0, 1, 0),
        Vector3D(0, 0, 1), Vector3D(1, 0, 1), Vector3D(1, 1, 1), Vector3D(0, 1, 1)
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # Front
        [4, 6, 5], [4, 7, 6],  # Back
        [0, 4, 5], [0, 5, 1],  # Bottom
        [2, 6, 7], [2, 7, 3],  # Top
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2]   # Right
    ]
    
    processor = NumPy3DProcessor()
    processor.load_mesh(vertices, faces)
    
    # Calculate normals
    normals = processor.calculate_normals()
    print(f"Calculated {len(normals)} face normals")
    
    # Calculate bounding box
    min_point, max_point = processor.calculate_bounding_box()
    print(f"Bounding box: {min_point} to {max_point}")
    
    # Calculate centroid
    centroid = processor.calculate_centroid()
    print(f"Centroid: {centroid}")
    
    # Vectorized operations
    print("\n=== Vectorized Operations ===")
    
    vectors = [Vector3D(i, i+1, i+2) for i in range(5)]
    print(f"Original vectors: {[str(v) for v in vectors]}")
    
    normalized = NumPyVectorizedOperations.batch_vector_operations(vectors, "normalize")
    print(f"Normalized vectors: {[str(v) for v in normalized]}")
    
    scaled = NumPyVectorizedOperations.batch_vector_operations(vectors, "scale")
    print(f"Scaled vectors: {[str(v) for v in scaled]}")
    
    # Distance matrix
    distances = NumPyVectorizedOperations.distance_matrix(vectors)
    print(f"Distance matrix shape: {distances.shape}")
    
    # Interpolation
    start = Vector3D(0, 0, 0)
    end = Vector3D(1, 1, 1)
    interpolated = NumPyVectorizedOperations.interpolate_points(start, end, 5)
    print(f"Interpolated points: {[str(p) for p in interpolated]}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    NumPyPerformanceComparison.compare_vector_operations(1000)
    NumPyPerformanceComparison.compare_matrix_operations(100)


if __name__ == "__main__":
    demonstrate_numpy_integration()
