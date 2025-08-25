#!/usr/bin/env python3
"""
Chapter 10: Introduction to 3D in Python
3D Transformations

Demonstrates 3D transformations including translation, rotation, scaling,
and matrix operations for 3D graphics.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Transformations"
__description__ = "Working with 3D transformations and matrices"

# ============================================================================
# 3D VECTOR AND MATRIX CLASSES
# ============================================================================

@dataclass
class Vector3D:
    """3D vector class for representing positions and directions"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self * scalar
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product of two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product of two vectors"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

class Matrix4x4:
    """4x4 transformation matrix for 3D graphics"""
    
    def __init__(self, matrix: Optional[List[List[float]]] = None):
        if matrix is None:
            # Identity matrix
            self.matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        else:
            self.matrix = matrix
    
    def __mul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        """Matrix multiplication"""
        result = [[0.0 for _ in range(4)] for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j]
        
        return Matrix4x4(result)
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transform a 3D point using this matrix"""
        # Convert to homogeneous coordinates
        x = point.x * self.matrix[0][0] + point.y * self.matrix[0][1] + point.z * self.matrix[0][2] + self.matrix[0][3]
        y = point.x * self.matrix[1][0] + point.y * self.matrix[1][1] + point.z * self.matrix[1][2] + self.matrix[1][3]
        z = point.x * self.matrix[2][0] + point.y * self.matrix[2][1] + point.z * self.matrix[2][2] + self.matrix[2][3]
        w = point.x * self.matrix[3][0] + point.y * self.matrix[3][1] + point.z * self.matrix[3][2] + self.matrix[3][3]
        
        # Convert back from homogeneous coordinates
        if w != 0:
            return Vector3D(x / w, y / w, z / w)
        else:
            return Vector3D(x, y, z)
    
    def transform_vector(self, vector: Vector3D) -> Vector3D:
        """Transform a 3D vector using this matrix (ignores translation)"""
        x = vector.x * self.matrix[0][0] + vector.y * self.matrix[0][1] + vector.z * self.matrix[0][2]
        y = vector.x * self.matrix[1][0] + vector.y * self.matrix[1][1] + vector.z * self.matrix[1][2]
        z = vector.x * self.matrix[2][0] + vector.y * self.matrix[2][1] + vector.z * self.matrix[2][2]
        
        return Vector3D(x, y, z)
    
    def transpose(self) -> 'Matrix4x4':
        """Transpose the matrix"""
        result = [[0.0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                result[i][j] = self.matrix[j][i]
        return Matrix4x4(result)
    
    def __str__(self):
        result = "Matrix4x4:\n"
        for row in self.matrix:
            result += f"  [{row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f}]\n"
        return result

# ============================================================================
# TRANSFORMATION MATRICES
# ============================================================================

class TransformMatrix:
    """Utility class for creating transformation matrices"""
    
    @staticmethod
    def translation(tx: float, ty: float, tz: float) -> Matrix4x4:
        """Create a translation matrix"""
        return Matrix4x4([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    @staticmethod
    def scaling(sx: float, sy: float, sz: float) -> Matrix4x4:
        """Create a scaling matrix"""
        return Matrix4x4([
            [sx,  0.0, 0.0, 0.0],
            [0.0, sy,  0.0, 0.0],
            [0.0, 0.0, sz,  0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    @staticmethod
    def rotation_x(angle: float) -> Matrix4x4:
        """Create a rotation matrix around X-axis"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [1.0, 0.0,   0.0,  0.0],
            [0.0, cos_a, -sin_a, 0.0],
            [0.0, sin_a, cos_a,  0.0],
            [0.0, 0.0,   0.0,  1.0]
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> Matrix4x4:
        """Create a rotation matrix around Y-axis"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [cos_a,  0.0, sin_a, 0.0],
            [0.0,    1.0, 0.0,  0.0],
            [-sin_a, 0.0, cos_a, 0.0],
            [0.0,    0.0, 0.0,  1.0]
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> Matrix4x4:
        """Create a rotation matrix around Z-axis"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a,  0.0, 0.0],
            [0.0,   0.0,   1.0, 0.0],
            [0.0,   0.0,   0.0, 1.0]
        ])
    
    @staticmethod
    def rotation_axis(axis: Vector3D, angle: float) -> Matrix4x4:
        """Create a rotation matrix around an arbitrary axis"""
        axis = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        one_minus_cos = 1.0 - cos_a
        
        return Matrix4x4([
            [cos_a + one_minus_cos * axis.x * axis.x,
             one_minus_cos * axis.x * axis.y - sin_a * axis.z,
             one_minus_cos * axis.x * axis.z + sin_a * axis.y,
             0.0],
            [one_minus_cos * axis.x * axis.y + sin_a * axis.z,
             cos_a + one_minus_cos * axis.y * axis.y,
             one_minus_cos * axis.y * axis.z - sin_a * axis.x,
             0.0],
            [one_minus_cos * axis.x * axis.z - sin_a * axis.y,
             one_minus_cos * axis.y * axis.z + sin_a * axis.x,
             cos_a + one_minus_cos * axis.z * axis.z,
             0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    @staticmethod
    def look_at(eye: Vector3D, target: Vector3D, up: Vector3D) -> Matrix4x4:
        """Create a look-at matrix for camera positioning"""
        forward = (target - eye).normalize()
        right = forward.cross(up).normalize()
        up_new = right.cross(forward)
        
        return Matrix4x4([
            [right.x, right.y, right.z, -right.dot(eye)],
            [up_new.x, up_new.y, up_new.z, -up_new.dot(eye)],
            [-forward.x, -forward.y, -forward.z, forward.dot(eye)],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    @staticmethod
    def perspective(fov: float, aspect: float, near: float, far: float) -> Matrix4x4:
        """Create a perspective projection matrix"""
        f = 1.0 / math.tan(fov / 2.0)
        return Matrix4x4([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0.0, 0.0, -1.0, 0.0]
        ])
    
    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Matrix4x4:
        """Create an orthographic projection matrix"""
        return Matrix4x4([
            [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
            [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
            [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0.0, 0.0, 0.0, 1.0]
        ])

# ============================================================================
# TRANSFORMATION UTILITIES
# ============================================================================

class Transform3D:
    """3D transformation utility class"""
    
    def __init__(self):
        self.matrix = Matrix4x4()  # Identity matrix
        self.position = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)  # Euler angles
        self.scale = Vector3D(1, 1, 1)
    
    def set_position(self, position: Vector3D):
        """Set the position"""
        self.position = position
        self._update_matrix()
    
    def set_rotation(self, rotation: Vector3D):
        """Set the rotation (Euler angles in radians)"""
        self.rotation = rotation
        self._update_matrix()
    
    def set_scale(self, scale: Vector3D):
        """Set the scale"""
        self.scale = scale
        self._update_matrix()
    
    def translate(self, offset: Vector3D):
        """Translate by an offset"""
        self.position = self.position + offset
        self._update_matrix()
    
    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        """Rotate by given angles"""
        self.rotation.x += x_angle
        self.rotation.y += y_angle
        self.rotation.z += z_angle
        self._update_matrix()
    
    def scale_by(self, x_scale: float = 1, y_scale: float = 1, z_scale: float = 1):
        """Scale by given factors"""
        self.scale.x *= x_scale
        self.scale.y *= y_scale
        self.scale.z *= z_scale
        self._update_matrix()
    
    def _update_matrix(self):
        """Update the transformation matrix"""
        # Create transformation matrices
        translation_matrix = TransformMatrix.translation(self.position.x, self.position.y, self.position.z)
        rotation_x_matrix = TransformMatrix.rotation_x(self.rotation.x)
        rotation_y_matrix = TransformMatrix.rotation_y(self.rotation.y)
        rotation_z_matrix = TransformMatrix.rotation_z(self.rotation.z)
        scale_matrix = TransformMatrix.scaling(self.scale.x, self.scale.y, self.scale.z)
        
        # Combine transformations: scale * rotation * translation
        self.matrix = scale_matrix * rotation_z_matrix * rotation_y_matrix * rotation_x_matrix * translation_matrix
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        """Transform a point using this transformation"""
        return self.matrix.transform_point(point)
    
    def transform_vector(self, vector: Vector3D) -> Vector3D:
        """Transform a vector using this transformation"""
        return self.matrix.transform_vector(vector)
    
    def get_matrix(self) -> Matrix4x4:
        """Get the current transformation matrix"""
        return self.matrix

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_transformations():
    """Demonstrate basic 3D transformations"""
    print("=== Basic 3D Transformations ===\n")
    
    # Test points
    test_points = [
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
        Vector3D(0, 0, 1),
        Vector3D(1, 1, 1)
    ]
    
    print("Original points:")
    for point in test_points:
        print(f"  {point}")
    
    print("\n1. Translation (move by (2, 3, 4)):")
    translation_matrix = TransformMatrix.translation(2, 3, 4)
    for point in test_points:
        transformed = translation_matrix.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print("\n2. Scaling (scale by 2x in all directions):")
    scale_matrix = TransformMatrix.scaling(2, 2, 2)
    for point in test_points:
        transformed = scale_matrix.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print("\n3. Rotation (45° around Y-axis):")
    rotation_matrix = TransformMatrix.rotation_y(math.pi / 4)
    for point in test_points:
        transformed = rotation_matrix.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print()

def demonstrate_matrix_operations():
    """Demonstrate matrix operations"""
    print("=== Matrix Operations ===\n")
    
    # Create matrices
    translation = TransformMatrix.translation(1, 2, 3)
    rotation = TransformMatrix.rotation_y(math.pi / 4)
    scaling = TransformMatrix.scaling(2, 2, 2)
    
    print("1. Translation Matrix:")
    print(translation)
    
    print("2. Rotation Matrix (45° around Y-axis):")
    print(rotation)
    
    print("3. Scaling Matrix (2x):")
    print(scaling)
    
    print("4. Combined Transformation (scale * rotate * translate):")
    combined = scaling * rotation * translation
    print(combined)
    
    # Test point transformation
    test_point = Vector3D(1, 0, 0)
    print(f"\n5. Transforming point {test_point}:")
    print(f"  Translation: {translation.transform_point(test_point)}")
    print(f"  Rotation: {rotation.transform_point(test_point)}")
    print(f"  Scaling: {scaling.transform_point(test_point)}")
    print(f"  Combined: {combined.transform_point(test_point)}")
    
    print()

def demonstrate_advanced_transformations():
    """Demonstrate advanced transformations"""
    print("=== Advanced Transformations ===\n")
    
    test_point = Vector3D(1, 0, 0)
    
    print("1. Rotation around arbitrary axis:")
    axis = Vector3D(1, 1, 1).normalize()
    rotation_matrix = TransformMatrix.rotation_axis(axis, math.pi / 2)
    transformed = rotation_matrix.transform_point(test_point)
    print(f"  Axis: {axis}")
    print(f"  Point: {test_point} → {transformed}")
    
    print("\n2. Look-at transformation:")
    eye = Vector3D(0, 0, 5)
    target = Vector3D(0, 0, 0)
    up = Vector3D(0, 1, 0)
    look_at_matrix = TransformMatrix.look_at(eye, target, up)
    print(f"  Eye: {eye}, Target: {target}, Up: {up}")
    print(look_at_matrix)
    
    print("\n3. Perspective projection:")
    perspective_matrix = TransformMatrix.perspective(math.pi / 4, 16/9, 0.1, 100)
    print("  FOV: 45°, Aspect: 16:9, Near: 0.1, Far: 100")
    print(perspective_matrix)
    
    print("\n4. Orthographic projection:")
    ortho_matrix = TransformMatrix.orthographic(-2, 2, -2, 2, 0.1, 100)
    print("  Left: -2, Right: 2, Bottom: -2, Top: 2, Near: 0.1, Far: 100")
    print(ortho_matrix)
    
    print()

def demonstrate_transform3d_class():
    """Demonstrate the Transform3D class"""
    print("=== Transform3D Class ===\n")
    
    # Create a transform
    transform = Transform3D()
    test_points = [Vector3D(1, 0, 0), Vector3D(0, 1, 0), Vector3D(0, 0, 1)]
    
    print("Original points:")
    for point in test_points:
        print(f"  {point}")
    
    print("\n1. Setting position to (2, 3, 4):")
    transform.set_position(Vector3D(2, 3, 4))
    for point in test_points:
        transformed = transform.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print("\n2. Adding rotation (45° around Y-axis):")
    transform.set_rotation(Vector3D(0, math.pi / 4, 0))
    for point in test_points:
        transformed = transform.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print("\n3. Adding scaling (2x):")
    transform.set_scale(Vector3D(2, 2, 2))
    for point in test_points:
        transformed = transform.transform_point(point)
        print(f"  {point} → {transformed}")
    
    print("\n4. Current transformation matrix:")
    print(transform.get_matrix())
    
    print()

def demonstrate_transformation_chaining():
    """Demonstrate chaining multiple transformations"""
    print("=== Transformation Chaining ===\n")
    
    # Create multiple transformations
    transform1 = Transform3D()
    transform1.set_position(Vector3D(1, 0, 0))
    transform1.set_rotation(Vector3D(0, math.pi / 4, 0))
    
    transform2 = Transform3D()
    transform2.set_position(Vector3D(0, 1, 0))
    transform2.set_rotation(Vector3D(math.pi / 4, 0, 0))
    
    transform3 = Transform3D()
    transform3.set_scale(Vector3D(2, 1, 1))
    
    # Test point
    test_point = Vector3D(1, 0, 0)
    print(f"Original point: {test_point}")
    
    print("\n1. Transform 1 (translate + rotate Y):")
    result1 = transform1.transform_point(test_point)
    print(f"  {test_point} → {result1}")
    
    print("\n2. Transform 2 (translate + rotate X):")
    result2 = transform2.transform_point(test_point)
    print(f"  {test_point} → {result2}")
    
    print("\n3. Transform 3 (scale X):")
    result3 = transform3.transform_point(test_point)
    print(f"  {test_point} → {result3}")
    
    print("\n4. Combined transformations (1 → 2 → 3):")
    combined_matrix = transform3.get_matrix() * transform2.get_matrix() * transform1.get_matrix()
    combined_transform = Transform3D()
    combined_transform.matrix = combined_matrix
    result_combined = combined_transform.transform_point(test_point)
    print(f"  {test_point} → {result_combined}")
    
    print()

def demonstrate_vector_operations():
    """Demonstrate vector operations"""
    print("=== Vector Operations ===\n")
    
    # Test vectors
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    
    print(f"\n1. Addition: {v1} + {v2} = {v1 + v2}")
    print(f"2. Subtraction: {v1} - {v2} = {v1 - v2}")
    print(f"3. Scalar multiplication: 2 * {v1} = {2 * v1}")
    print(f"4. Dot product: {v1} · {v2} = {v1.dot(v2)}")
    print(f"5. Cross product: {v1} × {v2} = {v1.cross(v2)}")
    print(f"6. Magnitude of {v1}: {v1.magnitude():.3f}")
    print(f"7. Normalized {v1}: {v1.normalize()}")
    
    print()

def demonstrate_transformation_examples():
    """Demonstrate practical transformation examples"""
    print("=== Practical Transformation Examples ===\n")
    
    # Example 1: Camera positioning
    print("1. Camera positioning:")
    camera_pos = Vector3D(0, 0, 5)
    target_pos = Vector3D(0, 0, 0)
    up_vector = Vector3D(0, 1, 0)
    
    view_matrix = TransformMatrix.look_at(camera_pos, target_pos, up_vector)
    print(f"  Camera at {camera_pos}, looking at {target_pos}")
    print(f"  View matrix created successfully")
    
    # Example 2: Object orbiting
    print("\n2. Object orbiting around origin:")
    center = Vector3D(0, 0, 0)
    radius = 3.0
    
    for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]:
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        orbit_point = Vector3D(x, 0, z)
        print(f"  Angle {math.degrees(angle):3.0f}°: {orbit_point}")
    
    # Example 3: Scaling with different factors
    print("\n3. Non-uniform scaling:")
    original = Vector3D(1, 1, 1)
    scale_matrix = TransformMatrix.scaling(2, 1, 0.5)
    scaled = scale_matrix.transform_point(original)
    print(f"  Original: {original}")
    print(f"  Scaled (2x, 1x, 0.5x): {scaled}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate 3D transformations"""
    print("=== 3D Transformations Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate basic transformations
    demonstrate_basic_transformations()
    
    # Demonstrate matrix operations
    demonstrate_matrix_operations()
    
    # Demonstrate advanced transformations
    demonstrate_advanced_transformations()
    
    # Demonstrate Transform3D class
    demonstrate_transform3d_class()
    
    # Demonstrate transformation chaining
    demonstrate_transformation_chaining()
    
    # Demonstrate vector operations
    demonstrate_vector_operations()
    
    # Demonstrate practical examples
    demonstrate_transformation_examples()
    
    print("="*60)
    print("3D Transformations demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Vector operations (addition, dot product, cross product)")
    print("✓ Matrix operations (multiplication, transformation)")
    print("✓ Basic transformations (translation, rotation, scaling)")
    print("✓ Advanced transformations (look-at, perspective, orthographic)")
    print("✓ Transformation chaining and composition")
    print("✓ Practical applications and examples")
    
    print("\nTransformation types covered:")
    print("• Translation: Moving objects in 3D space")
    print("• Rotation: Rotating around X, Y, Z axes and arbitrary axes")
    print("• Scaling: Uniform and non-uniform scaling")
    print("• Projection: Perspective and orthographic projections")
    print("• Look-at: Camera positioning and orientation")
    
    print("\nApplications:")
    print("• Game development: Object positioning and movement")
    print("• 3D modeling: Object transformations and manipulation")
    print("• Computer graphics: Camera systems and rendering")
    print("• Animation: Keyframe interpolation and motion")
    print("• Physics simulation: Object dynamics and constraints")
    
    print("\nNext steps:")
    print("• Explore quaternions for smooth rotations")
    print("• Learn about scene graphs and hierarchies")
    print("• Study advanced rendering techniques")
    print("• Understand animation and interpolation")

if __name__ == "__main__":
    main()
