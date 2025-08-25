"""
Chapter 18: Transformations - Transformation Matrices
====================================================

This module demonstrates 3D transformation matrices and operations.

Key Concepts:
- Translation, rotation, and scaling matrices
- Matrix composition and decomposition
- Transformation hierarchies
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion


@dataclass
class Transform:
    """3D transformation with position, rotation, and scale."""
    position: Vector3D = Vector3D(0, 0, 0)
    rotation: Quaternion = Quaternion(1, 0, 0, 0)
    scale: Vector3D = Vector3D(1, 1, 1)
    
    def get_matrix(self) -> Matrix4x4:
        """Get transformation matrix."""
        scale_matrix = Matrix4x4.scaling(self.scale.x, self.scale.y, self.scale.z)
        rotation_matrix = self.rotation.to_matrix()
        translation_matrix = Matrix4x4.translation(self.position.x, self.position.y, self.position.z)
        return translation_matrix * rotation_matrix * scale_matrix
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transform a point by this transformation."""
        matrix = self.get_matrix()
        return matrix.transform_point(point)
    
    def transform_vector(self, vector: Vector3D) -> Vector3D:
        """Transform a vector by this transformation (ignores translation)."""
        scale_matrix = Matrix4x4.scaling(self.scale.x, self.scale.y, self.scale.z)
        rotation_matrix = self.rotation.to_matrix()
        transform_matrix = rotation_matrix * scale_matrix
        return transform_matrix.transform_vector(vector)
    
    def combine(self, other: 'Transform') -> 'Transform':
        """Combine this transformation with another."""
        combined_position = self.position + self.rotation.rotate_vector(other.position * self.scale)
        combined_rotation = self.rotation * other.rotation
        combined_scale = Vector3D(
            self.scale.x * other.scale.x,
            self.scale.y * other.scale.y,
            self.scale.z * other.scale.z
        )
        return Transform(combined_position, combined_rotation, combined_scale)


class TransformationMatrix:
    """Utility class for creating transformation matrices."""
    
    @staticmethod
    def rotation_x(angle: float) -> Matrix4x4:
        """Create rotation matrix around X-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [1, 0, 0, 0],
            [0, cos_a, -sin_a, 0],
            [0, sin_a, cos_a, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> Matrix4x4:
        """Create rotation matrix around Y-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> Matrix4x4:
        """Create rotation matrix around Z-axis."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def rotation_axis(axis: Vector3D, angle: float) -> Matrix4x4:
        """Create rotation matrix around arbitrary axis."""
        axis = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        one_minus_cos = 1 - cos_a
        
        return Matrix4x4([
            [cos_a + axis.x * axis.x * one_minus_cos,
             axis.x * axis.y * one_minus_cos - axis.z * sin_a,
             axis.x * axis.z * one_minus_cos + axis.y * sin_a, 0],
            [axis.y * axis.x * one_minus_cos + axis.z * sin_a,
             cos_a + axis.y * axis.y * one_minus_cos,
             axis.y * axis.z * one_minus_cos - axis.x * sin_a, 0],
            [axis.z * axis.x * one_minus_cos - axis.y * sin_a,
             axis.z * axis.y * one_minus_cos + axis.x * sin_a,
             cos_a + axis.z * axis.z * one_minus_cos, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def reflection(plane_normal: Vector3D) -> Matrix4x4:
        """Create reflection matrix across a plane."""
        normal = plane_normal.normalize()
        return Matrix4x4([
            [1 - 2 * normal.x * normal.x,
             -2 * normal.x * normal.y,
             -2 * normal.x * normal.z, 0],
            [-2 * normal.y * normal.x,
             1 - 2 * normal.y * normal.y,
             -2 * normal.y * normal.z, 0],
            [-2 * normal.z * normal.x,
             -2 * normal.z * normal.y,
             1 - 2 * normal.z * normal.z, 0],
            [0, 0, 0, 1]
        ])


class MatrixDecomposer:
    """Decompose transformation matrices into components."""
    
    @staticmethod
    def decompose(matrix: Matrix4x4) -> Tuple[Vector3D, Quaternion, Vector3D]:
        """Decompose matrix into translation, rotation, and scale."""
        translation = Vector3D(matrix.data[0][3], matrix.data[1][3], matrix.data[2][3])
        
        scale_x = math.sqrt(matrix.data[0][0]**2 + matrix.data[1][0]**2 + matrix.data[2][0]**2)
        scale_y = math.sqrt(matrix.data[0][1]**2 + matrix.data[1][1]**2 + matrix.data[2][1]**2)
        scale_z = math.sqrt(matrix.data[0][2]**2 + matrix.data[1][2]**2 + matrix.data[2][2]**2)
        scale = Vector3D(scale_x, scale_y, scale_z)
        
        rotation_matrix = Matrix4x4([
            [matrix.data[0][0] / scale_x, matrix.data[0][1] / scale_y, matrix.data[0][2] / scale_z, 0],
            [matrix.data[1][0] / scale_x, matrix.data[1][1] / scale_y, matrix.data[1][2] / scale_z, 0],
            [matrix.data[2][0] / scale_x, matrix.data[2][1] / scale_y, matrix.data[2][2] / scale_z, 0],
            [0, 0, 0, 1]
        ])
        
        rotation = Quaternion.from_matrix(rotation_matrix)
        return translation, rotation, scale


def demonstrate_transformation_matrices():
    """Demonstrate transformation matrices and operations."""
    print("=== Transformation Matrices Demonstration ===\n")
    
    # Create basic transformation
    transform = Transform(
        position=Vector3D(1, 2, 3),
        rotation=Quaternion.from_euler(0, math.pi/4, 0),
        scale=Vector3D(2, 1, 0.5)
    )
    
    print(f"Position: {transform.position}")
    print(f"Scale: {transform.scale}")
    
    # Transform points
    point = Vector3D(1, 0, 0)
    transformed_point = transform.transform_point(point)
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")
    
    # Matrix decomposition
    matrix = transform.get_matrix()
    translation, rotation, scale = MatrixDecomposer.decompose(matrix)
    print(f"Decomposed translation: {translation}")
    print(f"Decomposed scale: {scale}")
    
    # Special transformations
    axis_rotation = TransformationMatrix.rotation_axis(Vector3D(1, 1, 1).normalize(), math.pi/3)
    reflection_matrix = TransformationMatrix.reflection(Vector3D(0, 1, 0))
    print("Created axis rotation and reflection matrices")


if __name__ == "__main__":
    demonstrate_transformation_matrices()
