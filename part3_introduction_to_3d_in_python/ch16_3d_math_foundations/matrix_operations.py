"""
Chapter 16: 3D Math Foundations - Matrix Operations
==================================================

This module demonstrates 4x4 matrix operations and transformations
essential for 3D graphics programming.

Key Concepts:
- 4x4 transformation matrices
- Matrix arithmetic and operations
- Transformation matrices (translation, rotation, scaling)
- Matrix decomposition and interpolation
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from vector_operations import Vector3D


@dataclass
class Matrix4x4:
    """4x4 transformation matrix with comprehensive operations."""
    data: List[List[float]]
    
    def __post_init__(self):
        """Initialize matrix data."""
        if self.data is None:
            self.data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        elif len(self.data) != 4 or any(len(row) != 4 for row in self.data):
            raise ValueError("Matrix must be 4x4")
    
    def __mul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Matrix multiplication."""
        result = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix4x4(result)
    
    def transpose(self) -> 'Matrix4x4':
        """Transpose matrix."""
        result = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                result[i][j] = self.data[j][i]
        return Matrix4x4(result)
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transform a 3D point by this matrix."""
        x = (self.data[0][0] * point.x + self.data[0][1] * point.y + 
             self.data[0][2] * point.z + self.data[0][3])
        y = (self.data[1][0] * point.x + self.data[1][1] * point.y + 
             self.data[1][2] * point.z + self.data[1][3])
        z = (self.data[2][0] * point.x + self.data[2][1] * point.y + 
             self.data[2][2] * point.z + self.data[2][3])
        w = (self.data[3][0] * point.x + self.data[3][1] * point.y + 
             self.data[3][2] * point.z + self.data[3][3])
        
        if abs(w) > 1e-9:
            return Vector3D(x / w, y / w, z / w)
        else:
            return Vector3D(x, y, z)
    
    def transform_vector(self, vector: Vector3D) -> Vector3D:
        """Transform a 3D vector by this matrix (ignores translation)."""
        x = (self.data[0][0] * vector.x + self.data[0][1] * vector.y + 
             self.data[0][2] * vector.z)
        y = (self.data[1][0] * vector.x + self.data[1][1] * vector.y + 
             self.data[1][2] * vector.z)
        z = (self.data[2][0] * vector.x + self.data[2][1] * vector.y + 
             self.data[2][2] * vector.z)
        
        return Vector3D(x, y, z)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return np.array(self.data, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Matrix4x4':
        """Create matrix from NumPy array."""
        if array.shape != (4, 4):
            raise ValueError("Array must be 4x4")
        return cls(array.tolist())
    
    @classmethod
    def identity(cls) -> 'Matrix4x4':
        """Create identity matrix."""
        return cls([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create translation matrix."""
        return cls([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    
    @classmethod
    def scaling(cls, x: float, y: float, z: float) -> 'Matrix4x4':
        """Create scaling matrix."""
        return cls([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])
    
    @classmethod
    def rotation_x(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around X axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([[1, 0, 0, 0], [0, cos_a, -sin_a, 0], [0, sin_a, cos_a, 0], [0, 0, 0, 1]])
    
    @classmethod
    def rotation_y(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Y axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([[cos_a, 0, sin_a, 0], [0, 1, 0, 0], [-sin_a, 0, cos_a, 0], [0, 0, 0, 1]])
    
    @classmethod
    def rotation_z(cls, angle: float) -> 'Matrix4x4':
        """Create rotation matrix around Z axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return cls([[cos_a, -sin_a, 0, 0], [sin_a, cos_a, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    @classmethod
    def look_at(cls, eye: Vector3D, target: Vector3D, up: Vector3D) -> 'Matrix4x4':
        """Create look-at matrix."""
        z = (eye - target).normalize()
        x = up.cross(z).normalize()
        y = z.cross(x)
        
        return cls([
            [x.x, x.y, x.z, -x.dot(eye)],
            [y.x, y.y, y.z, -y.dot(eye)],
            [z.x, z.y, z.z, -z.dot(eye)],
            [0, 0, 0, 1]
        ])
    
    @classmethod
    def perspective(cls, fov: float, aspect: float, near: float, far: float) -> 'Matrix4x4':
        """Create perspective projection matrix."""
        f = 1.0 / math.tan(fov * 0.5)
        return cls([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])


class Matrix4x4Array:
    """Optimized array of 4x4 matrices using NumPy."""
    
    def __init__(self, matrices: Optional[List[Matrix4x4]] = None):
        """Initialize matrix array."""
        if matrices is None:
            self.data = np.zeros((0, 4, 4), dtype=np.float32)
        else:
            self.data = np.array([m.to_numpy() for m in matrices], dtype=np.float32)
    
    def add_matrix(self, matrix: Matrix4x4):
        """Add a matrix to the array."""
        self.data = np.vstack([self.data, matrix.to_numpy().reshape(1, 4, 4)])
    
    def get_matrix(self, index: int) -> Matrix4x4:
        """Get matrix at index."""
        if index < 0 or index >= len(self.data):
            raise IndexError("Matrix index out of range")
        return Matrix4x4.from_numpy(self.data[index])
    
    def multiply_all(self, matrix: Matrix4x4) -> 'Matrix4x4Array':
        """Multiply all matrices by given matrix."""
        matrix_np = matrix.to_numpy()
        result_data = np.matmul(self.data, matrix_np)
        result = Matrix4x4Array()
        result.data = result_data
        return result
    
    def __len__(self) -> int:
        """Number of matrices in array."""
        return len(self.data)


class Matrix4x4Math:
    """Static utility class for matrix mathematics."""
    
    @staticmethod
    def decompose_translation_rotation_scale(matrix: Matrix4x4) -> Tuple[Vector3D, Vector3D, Vector3D]:
        """Decompose matrix into translation, rotation, and scale."""
        # Extract translation
        translation = Vector3D(matrix.data[0][3], matrix.data[1][3], matrix.data[2][3])
        
        # Extract scale from rotation matrix
        scale_x = Vector3D(matrix.data[0][0], matrix.data[1][0], matrix.data[2][0]).magnitude()
        scale_y = Vector3D(matrix.data[0][1], matrix.data[1][1], matrix.data[2][1]).magnitude()
        scale_z = Vector3D(matrix.data[0][2], matrix.data[1][2], matrix.data[2][2]).magnitude()
        scale = Vector3D(scale_x, scale_y, scale_z)
        
        # Extract rotation (simplified)
        rotation_matrix = Matrix4x4([
            [matrix.data[0][0] / scale_x, matrix.data[0][1] / scale_y, matrix.data[0][2] / scale_z, 0],
            [matrix.data[1][0] / scale_x, matrix.data[1][1] / scale_y, matrix.data[1][2] / scale_z, 0],
            [matrix.data[2][0] / scale_x, matrix.data[2][1] / scale_y, matrix.data[2][2] / scale_z, 0],
            [0, 0, 0, 1]
        ])
        
        # Convert to Euler angles (simplified)
        if abs(rotation_matrix.data[2][0]) < 0.999999:
            yaw = math.atan2(rotation_matrix.data[2][0], rotation_matrix.data[2][2])
            pitch = math.asin(-rotation_matrix.data[2][0])
            roll = math.atan2(rotation_matrix.data[1][0], rotation_matrix.data[0][0])
        else:
            yaw = math.atan2(-rotation_matrix.data[0][1], rotation_matrix.data[1][1])
            pitch = math.asin(-rotation_matrix.data[2][0])
            roll = 0
        
        rotation = Vector3D(pitch, yaw, roll)
        
        return translation, rotation, scale
    
    @staticmethod
    def interpolate_matrices(m1: Matrix4x4, m2: Matrix4x4, t: float) -> Matrix4x4:
        """Interpolate between two matrices."""
        t = max(0, min(1, t))
        
        # Decompose matrices
        trans1, rot1, scale1 = Matrix4x4Math.decompose_translation_rotation_scale(m1)
        trans2, rot2, scale2 = Matrix4x4Math.decompose_translation_rotation_scale(m2)
        
        # Interpolate components
        trans_interp = trans1.lerp(trans2, t)
        scale_interp = scale1.lerp(scale2, t)
        rot_interp = rot1.lerp(rot2, t)
        
        # Reconstruct matrix
        translation_mat = Matrix4x4.translation(trans_interp.x, trans_interp.y, trans_interp.z)
        rotation_mat = (Matrix4x4.rotation_x(rot_interp.x) * 
                       Matrix4x4.rotation_y(rot_interp.y) * 
                       Matrix4x4.rotation_z(rot_interp.z))
        scale_mat = Matrix4x4.scaling(scale_interp.x, scale_interp.y, scale_interp.z)
        
        return translation_mat * rotation_mat * scale_mat


def demonstrate_matrix_operations():
    """Demonstrate various matrix operations."""
    print("=== Matrix Operations Demonstration ===\n")
    
    # Create test matrices
    identity = Matrix4x4.identity()
    translation = Matrix4x4.translation(1, 2, 3)
    rotation = Matrix4x4.rotation_y(math.pi / 4)
    scaling = Matrix4x4.scaling(2, 1, 0.5)
    
    print("Basic Matrices:")
    print(f"Identity: {identity.data}")
    print(f"Translation: {translation.data}")
    print(f"Rotation Y: {rotation.data}")
    print(f"Scaling: {scaling.data}")
    print()
    
    # Matrix multiplication
    combined = translation * rotation * scaling
    print("Combined Transformation:")
    print(f"Translation * Rotation * Scaling = {combined.data}")
    print()
    
    # Point transformation
    point = Vector3D(1, 0, 0)
    transformed_point = combined.transform_point(point)
    print("Point Transformation:")
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")
    print()
    
    # Special matrices
    look_at = Matrix4x4.look_at(
        Vector3D(0, 0, 5), 
        Vector3D(0, 0, 0), 
        Vector3D(0, 1, 0)
    )
    perspective = Matrix4x4.perspective(math.pi / 4, 16/9, 0.1, 100)
    
    print("Special Matrices:")
    print(f"Look-at matrix: {look_at.data}")
    print(f"Perspective matrix: {perspective.data}")
    print()
    
    # Matrix decomposition
    trans, rot, scale = Matrix4x4Math.decompose_translation_rotation_scale(combined)
    print("Matrix Decomposition:")
    print(f"Translation: {trans}")
    print(f"Rotation (Pitch, Yaw, Roll): {rot}")
    print(f"Scale: {scale}")


if __name__ == "__main__":
    demonstrate_matrix_operations()
