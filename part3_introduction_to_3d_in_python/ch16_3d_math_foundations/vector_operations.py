"""
Chapter 16: 3D Math Foundations - Vector Operations
==================================================

This module demonstrates fundamental 3D vector operations and mathematics
essential for 3D graphics programming.

Key Concepts:
- 3D vector representation and operations
- Vector arithmetic (addition, subtraction, multiplication)
- Vector properties (magnitude, normalization, dot product, cross product)
- Vector transformations and geometric operations
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Vector3D:
    """3D vector with comprehensive mathematical operations."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Vector addition."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        """Vector subtraction."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        """Scalar multiplication."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        """Right scalar multiplication."""
        return self * scalar
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        """Scalar division."""
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        """Calculate vector magnitude (length)."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        """Normalize vector to unit length."""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product of two vectors."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product of two vectors."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def distance_to(self, other: 'Vector3D') -> float:
        """Distance between two vectors."""
        return (self - other).magnitude()
    
    def angle_to(self, other: 'Vector3D') -> float:
        """Angle between two vectors in radians."""
        dot_product = self.dot(other)
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            return 0
        cos_angle = max(-1, min(1, dot_product / mag_product))
        return math.acos(cos_angle)
    
    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        """Linear interpolation between two vectors."""
        t = max(0, min(1, t))
        return Vector3D(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )
    
    def reflect(self, normal: 'Vector3D') -> 'Vector3D':
        """Reflect vector off a surface with given normal."""
        normal = normal.normalize()
        return self - normal * (2 * self.dot(normal))
    
    def to_array(self) -> List[float]:
        """Convert to array representation."""
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_array(cls, arr: List[float]) -> 'Vector3D':
        """Create vector from array."""
        if len(arr) != 3:
            raise ValueError("Array must have exactly 3 elements")
        return cls(arr[0], arr[1], arr[2])
    
    @classmethod
    def zero(cls) -> 'Vector3D':
        """Create zero vector."""
        return cls(0, 0, 0)
    
    @classmethod
    def unit_x(cls) -> 'Vector3D':
        """Create unit vector in X direction."""
        return cls(1, 0, 0)
    
    @classmethod
    def unit_y(cls) -> 'Vector3D':
        """Create unit vector in Y direction."""
        return cls(0, 1, 0)
    
    @classmethod
    def unit_z(cls) -> 'Vector3D':
        """Create unit vector in Z direction."""
        return cls(0, 0, 1)


class Vector3DArray:
    """Optimized array of 3D vectors using NumPy for performance."""
    
    def __init__(self, vectors: Optional[List[Vector3D]] = None):
        """Initialize vector array."""
        if vectors is None:
            self.data = np.zeros((0, 3), dtype=np.float32)
        else:
            self.data = np.array([v.to_array() for v in vectors], dtype=np.float32)
    
    def add_vector(self, vector: Vector3D):
        """Add a vector to the array."""
        self.data = np.vstack([self.data, vector.to_array()])
    
    def get_vector(self, index: int) -> Vector3D:
        """Get vector at index."""
        if index < 0 or index >= len(self.data):
            raise IndexError("Vector index out of range")
        return Vector3D.from_array(self.data[index].tolist())
    
    def magnitude_all(self) -> np.ndarray:
        """Calculate magnitude of all vectors."""
        return np.sqrt(np.sum(self.data**2, axis=1))
    
    def normalize_all(self) -> 'Vector3DArray':
        """Normalize all vectors."""
        magnitudes = self.magnitude_all()
        magnitudes = np.where(magnitudes == 0, 1, magnitudes)
        normalized_data = self.data / magnitudes[:, np.newaxis]
        result = Vector3DArray()
        result.data = normalized_data
        return result
    
    def dot_all(self, other: Vector3D) -> np.ndarray:
        """Calculate dot product with all vectors."""
        return np.dot(self.data, other.to_array())
    
    def __len__(self) -> int:
        """Number of vectors in array."""
        return len(self.data)


class Vector3DMath:
    """Static utility class for vector mathematics."""
    
    @staticmethod
    def min_vector(v1: Vector3D, v2: Vector3D) -> Vector3D:
        """Component-wise minimum of two vectors."""
        return Vector3D(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z))
    
    @staticmethod
    def max_vector(v1: Vector3D, v2: Vector3D) -> Vector3D:
        """Component-wise maximum of two vectors."""
        return Vector3D(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z))
    
    @staticmethod
    def random_unit_vector() -> Vector3D:
        """Generate random unit vector."""
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            s = x*x + y*y
            if s < 1:
                z = np.random.uniform(-1, 1)
                t = 2 * math.sqrt(1 - s)
                return Vector3D(x * t, y * t, z * t)
    
    @staticmethod
    def barycentric_coordinates(point: Vector3D, a: Vector3D, b: Vector3D, c: Vector3D) -> Tuple[float, float, float]:
        """Calculate barycentric coordinates of point relative to triangle."""
        v0 = b - a
        v1 = c - a
        v2 = point - a
        
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-9:
            return (0, 0, 0)
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return (u, v, w)


def demonstrate_vector_operations():
    """Demonstrate various vector operations."""
    print("=== Vector Operations Demonstration ===\n")
    
    # Create test vectors
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print()
    
    # Basic operations
    print("Basic Operations:")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"v1 / 2 = {v1 / 2}")
    print()
    
    # Vector properties
    print("Vector Properties:")
    print(f"|v1| = {v1.magnitude():.3f}")
    print(f"v1 normalized = {v1.normalize()}")
    print(f"v1 · v2 = {v1.dot(v2):.3f}")
    print(f"v1 × v2 = {v1.cross(v2)}")
    print(f"Distance(v1, v2) = {v1.distance_to(v2):.3f}")
    print(f"Angle(v1, v2) = {v1.angle_to(v2):.3f} radians")
    print()
    
    # Interpolation
    print("Interpolation:")
    print(f"Lerp(v1, v2, 0.5) = {v1.lerp(v2, 0.5)}")
    print()
    
    # Reflection
    normal = Vector3D(0, 1, 0)
    print("Reflection:")
    print(f"Reflect(v1, {normal}) = {v1.reflect(normal)}")
    print()
    
    # Random vectors
    print("Random Vectors:")
    print(f"Random unit vector = {Vector3DMath.random_unit_vector()}")
    print()
    
    # Barycentric coordinates
    a = Vector3D(0, 0, 0)
    b = Vector3D(1, 0, 0)
    c = Vector3D(0, 1, 0)
    point = Vector3D(0.5, 0.5, 0)
    
    u, v, w = Vector3DMath.barycentric_coordinates(point, a, b, c)
    print("Barycentric Coordinates:")
    print(f"Point {point} in triangle (a={a}, b={b}, c={c})")
    print(f"Barycentric coordinates: u={u:.3f}, v={v:.3f}, w={w:.3f}")


if __name__ == "__main__":
    demonstrate_vector_operations()
