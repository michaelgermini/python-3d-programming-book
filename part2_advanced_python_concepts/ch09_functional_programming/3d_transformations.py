"""
Chapter 9: Functional Programming - 3D Transformations
====================================================

This module demonstrates functional programming applied to 3D transformations,
showing how pure functions and immutability can be used for 3D graphics operations.

Key Concepts:
- Pure transformation functions
- Immutable 3D objects
- Function composition for transformations
- Pipeline processing
- Matrix operations as pure functions
"""

import math
from typing import List, Tuple, Callable, Union
from dataclasses import dataclass
from functools import reduce


@dataclass(frozen=True)
class Vector3D:
    """Immutable 3D vector for functional transformations."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass(frozen=True)
class Matrix4x4:
    """Immutable 4x4 transformation matrix."""
    m11: float; m12: float; m13: float; m14: float
    m21: float; m22: float; m23: float; m24: float
    m31: float; m32: float; m33: float; m34: float
    m41: float; m42: float; m43: float; m44: float
    
    @classmethod
    def identity(cls) -> 'Matrix4x4':
        """Creates identity matrix."""
        return cls(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        )
    
    def __mul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Matrix multiplication."""
        return Matrix4x4(
            self.m11 * other.m11 + self.m12 * other.m21 + self.m13 * other.m31 + self.m14 * other.m41,
            self.m11 * other.m12 + self.m12 * other.m22 + self.m13 * other.m32 + self.m14 * other.m42,
            self.m11 * other.m13 + self.m12 * other.m23 + self.m13 * other.m33 + self.m14 * other.m43,
            self.m11 * other.m14 + self.m12 * other.m24 + self.m13 * other.m34 + self.m14 * other.m44,
            
            self.m21 * other.m11 + self.m22 * other.m21 + self.m23 * other.m31 + self.m24 * other.m41,
            self.m21 * other.m12 + self.m22 * other.m22 + self.m23 * other.m32 + self.m24 * other.m42,
            self.m21 * other.m13 + self.m22 * other.m23 + self.m23 * other.m33 + self.m24 * other.m43,
            self.m21 * other.m14 + self.m22 * other.m24 + self.m23 * other.m34 + self.m24 * other.m44,
            
            self.m31 * other.m11 + self.m32 * other.m21 + self.m33 * other.m31 + self.m34 * other.m41,
            self.m31 * other.m12 + self.m32 * other.m22 + self.m33 * other.m32 + self.m34 * other.m42,
            self.m31 * other.m13 + self.m32 * other.m23 + self.m33 * other.m33 + self.m34 * other.m43,
            self.m31 * other.m14 + self.m32 * other.m24 + self.m33 * other.m34 + self.m34 * other.m44,
            
            self.m41 * other.m11 + self.m42 * other.m21 + self.m43 * other.m31 + self.m44 * other.m41,
            self.m41 * other.m12 + self.m42 * other.m22 + self.m43 * other.m32 + self.m44 * other.m42,
            self.m41 * other.m13 + self.m42 * other.m23 + self.m43 * other.m33 + self.m44 * other.m43,
            self.m41 * other.m14 + self.m42 * other.m24 + self.m43 * other.m34 + self.m44 * other.m44
        )
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transforms a 3D point using this matrix."""
        x = self.m11 * point.x + self.m12 * point.y + self.m13 * point.z + self.m14
        y = self.m21 * point.x + self.m22 * point.y + self.m23 * point.z + self.m24
        z = self.m31 * point.x + self.m32 * point.y + self.m33 * point.z + self.m34
        w = self.m41 * point.x + self.m42 * point.y + self.m43 * point.z + self.m44
        
        if w != 0:
            return Vector3D(x / w, y / w, z / w)
        else:
            return Vector3D(x, y, z)


# Pure Transformation Functions
def create_translation_matrix(offset: Vector3D) -> Matrix4x4:
    """Creates a translation matrix."""
    return Matrix4x4(
        1, 0, 0, offset.x,
        0, 1, 0, offset.y,
        0, 0, 1, offset.z,
        0, 0, 0, 1
    )


def create_scaling_matrix(scale: Vector3D) -> Matrix4x4:
    """Creates a scaling matrix."""
    return Matrix4x4(
        scale.x, 0, 0, 0,
        0, scale.y, 0, 0,
        0, 0, scale.z, 0,
        0, 0, 0, 1
    )


def create_rotation_matrix_x(angle: float) -> Matrix4x4:
    """Creates a rotation matrix around X-axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Matrix4x4(
        1, 0, 0, 0,
        0, cos_a, -sin_a, 0,
        0, sin_a, cos_a, 0,
        0, 0, 0, 1
    )


def create_rotation_matrix_y(angle: float) -> Matrix4x4:
    """Creates a rotation matrix around Y-axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Matrix4x4(
        cos_a, 0, sin_a, 0,
        0, 1, 0, 0,
        -sin_a, 0, cos_a, 0,
        0, 0, 0, 1
    )


def create_rotation_matrix_z(angle: float) -> Matrix4x4:
    """Creates a rotation matrix around Z-axis."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return Matrix4x4(
        cos_a, -sin_a, 0, 0,
        sin_a, cos_a, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    )


def create_look_at_matrix(eye: Vector3D, target: Vector3D, up: Vector3D) -> Matrix4x4:
    """Creates a look-at matrix for camera positioning."""
    forward = (target - eye).normalize()
    right = forward.cross(up).normalize()
    up_corrected = right.cross(forward)
    
    return Matrix4x4(
        right.x, right.y, right.z, -right.dot(eye),
        up_corrected.x, up_corrected.y, up_corrected.z, -up_corrected.dot(eye),
        -forward.x, -forward.y, -forward.z, forward.dot(eye),
        0, 0, 0, 1
    )


def create_perspective_matrix(fov: float, aspect: float, near: float, far: float) -> Matrix4x4:
    """Creates a perspective projection matrix."""
    f = 1.0 / math.tan(fov / 2)
    return Matrix4x4(
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
        0, 0, -1, 0
    )


# Higher-Order Functions for Transformations
def compose_transformations(*matrices: Matrix4x4) -> Matrix4x4:
    """Composes multiple transformation matrices."""
    return reduce(lambda acc, matrix: acc * matrix, matrices, Matrix4x4.identity())


def apply_transformation_to_points(points: List[Vector3D], 
                                 transform: Callable[[Vector3D], Vector3D]) -> List[Vector3D]:
    """Applies a transformation function to a list of points."""
    return [transform(point) for point in points]


def apply_matrix_to_points(points: List[Vector3D], matrix: Matrix4x4) -> List[Vector3D]:
    """Applies a transformation matrix to a list of points."""
    return [matrix.transform_point(point) for point in points]


# Functional Transformation Pipeline
class TransformationPipeline:
    """Functional pipeline for applying multiple transformations."""
    
    def __init__(self):
        self.transformations: List[Matrix4x4] = []
    
    def translate(self, offset: Vector3D) -> 'TransformationPipeline':
        """Adds translation to the pipeline."""
        self.transformations.append(create_translation_matrix(offset))
        return self
    
    def scale(self, scale: Vector3D) -> 'TransformationPipeline':
        """Adds scaling to the pipeline."""
        self.transformations.append(create_scaling_matrix(scale))
        return self
    
    def rotate_x(self, angle: float) -> 'TransformationPipeline':
        """Adds X-axis rotation to the pipeline."""
        self.transformations.append(create_rotation_matrix_x(angle))
        return self
    
    def rotate_y(self, angle: float) -> 'TransformationPipeline':
        """Adds Y-axis rotation to the pipeline."""
        self.transformations.append(create_rotation_matrix_y(angle))
        return self
    
    def rotate_z(self, angle: float) -> 'TransformationPipeline':
        """Adds Z-axis rotation to the pipeline."""
        self.transformations.append(create_rotation_matrix_z(angle))
        return self
    
    def build(self) -> Matrix4x4:
        """Builds the final transformation matrix."""
        return compose_transformations(*self.transformations)
    
    def apply_to_points(self, points: List[Vector3D]) -> List[Vector3D]:
        """Applies the pipeline to a list of points."""
        final_matrix = self.build()
        return apply_matrix_to_points(points, final_matrix)


# Advanced Functional Transformations
def create_orbital_camera_matrix(radius: float, theta: float, phi: float) -> Matrix4x4:
    """Creates a camera matrix for orbital camera movement."""
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    
    eye = Vector3D(x, y, z)
    target = Vector3D(0, 0, 0)
    up = Vector3D(0, 1, 0)
    
    return create_look_at_matrix(eye, target, up)


def create_billboard_matrix(camera_position: Vector3D, object_position: Vector3D) -> Matrix4x4:
    """Creates a billboard matrix that always faces the camera."""
    direction = (camera_position - object_position).normalize()
    
    # Create rotation to align object with camera direction
    up = Vector3D(0, 1, 0)
    right = direction.cross(up).normalize()
    up_corrected = right.cross(direction)
    
    return Matrix4x4(
        right.x, right.y, right.z, object_position.x,
        up_corrected.x, up_corrected.y, up_corrected.z, object_position.y,
        direction.x, direction.y, direction.z, object_position.z,
        0, 0, 0, 1
    )


# Functional Scene Processing
def process_3d_scene_functional(points: List[Vector3D], 
                              transformations: List[Callable[[Vector3D], Vector3D]]) -> List[Vector3D]:
    """Processes a 3D scene using functional transformations."""
    def apply_transformations(point: Vector3D) -> Vector3D:
        result = point
        for transform in transformations:
            result = transform(result)
        return result
    
    return [apply_transformations(point) for point in points]


def create_animation_frame(time: float, 
                          base_points: List[Vector3D],
                          animation_functions: List[Callable[[float], Matrix4x4]]) -> List[Vector3D]:
    """Creates an animation frame using functional animation functions."""
    # Compose all animation matrices for this time
    animation_matrix = compose_transformations(*[func(time) for func in animation_functions])
    
    # Apply to all base points
    return apply_matrix_to_points(base_points, animation_matrix)


# Example Usage and Demonstration
def demonstrate_3d_transformations():
    """Demonstrates 3D transformations using functional programming."""
    print("=== 3D Transformations with Functional Programming ===\n")
    
    # Create sample 3D points
    points = [
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
        Vector3D(0, 0, 1),
        Vector3D(1, 1, 1)
    ]
    
    print("Original points:")
    for i, point in enumerate(points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate basic transformations
    print("\n=== Basic Transformations ===")
    
    # Translation
    translation_matrix = create_translation_matrix(Vector3D(2, 3, 4))
    translated_points = apply_matrix_to_points(points, translation_matrix)
    print("After translation (2, 3, 4):")
    for i, point in enumerate(translated_points):
        print(f"  Point {i}: {point}")
    
    # Scaling
    scaling_matrix = create_scaling_matrix(Vector3D(2, 2, 2))
    scaled_points = apply_matrix_to_points(points, scaling_matrix)
    print("\nAfter scaling (2, 2, 2):")
    for i, point in enumerate(scaled_points):
        print(f"  Point {i}: {point}")
    
    # Rotation
    rotation_matrix = create_rotation_matrix_z(math.pi / 4)  # 45 degrees
    rotated_points = apply_matrix_to_points(points, rotation_matrix)
    print("\nAfter rotation (45° around Z-axis):")
    for i, point in enumerate(rotated_points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate transformation pipeline
    print("\n=== Transformation Pipeline ===")
    
    pipeline = (TransformationPipeline()
               .translate(Vector3D(1, 1, 1))
               .scale(Vector3D(0.5, 0.5, 0.5))
               .rotate_y(math.pi / 6))
    
    pipeline_points = pipeline.apply_to_points(points)
    print("After pipeline (translate + scale + rotate):")
    for i, point in enumerate(pipeline_points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate camera transformations
    print("\n=== Camera Transformations ===")
    
    # Look-at matrix
    eye = Vector3D(5, 5, 5)
    target = Vector3D(0, 0, 0)
    up = Vector3D(0, 1, 0)
    look_at_matrix = create_look_at_matrix(eye, target, up)
    
    # Perspective projection
    perspective_matrix = create_perspective_matrix(math.pi / 4, 16/9, 0.1, 100)
    
    # Combine camera transformations
    camera_matrix = look_at_matrix * perspective_matrix
    
    camera_points = apply_matrix_to_points(points, camera_matrix)
    print("After camera transformation:")
    for i, point in enumerate(camera_points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate orbital camera
    print("\n=== Orbital Camera ===")
    
    orbital_matrix = create_orbital_camera_matrix(10, math.pi / 4, math.pi / 4)
    orbital_points = apply_matrix_to_points(points, orbital_matrix)
    print("After orbital camera transformation:")
    for i, point in enumerate(orbital_points):
        print(f"  Point {i}: {point}")
    
    # Demonstrate functional scene processing
    print("\n=== Functional Scene Processing ===")
    
    def translate_func(offset: Vector3D) -> Callable[[Vector3D], Vector3D]:
        matrix = create_translation_matrix(offset)
        return lambda point: matrix.transform_point(point)
    
    def scale_func(scale: Vector3D) -> Callable[[Vector3D], Vector3D]:
        matrix = create_scaling_matrix(scale)
        return lambda point: matrix.transform_point(point)
    
    transformations = [
        translate_func(Vector3D(-1, -1, -1)),
        scale_func(Vector3D(2, 2, 2)),
        lambda point: create_rotation_matrix_z(math.pi / 6).transform_point(point)
    ]
    
    processed_points = process_3d_scene_functional(points, transformations)
    print("After functional scene processing:")
    for i, point in enumerate(processed_points):
        print(f"  Point {i}: {point}")


if __name__ == "__main__":
    demonstrate_3d_transformations()
