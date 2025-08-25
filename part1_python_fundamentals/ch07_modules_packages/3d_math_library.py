#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
3D Math Library Example

Demonstrates a complete modular 3D math library with vector,
matrix, and quaternion operations for 3D graphics applications.
"""

import math
import sys
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Math Library"
__description__ = "Complete 3D mathematics library for graphics programming"

# ============================================================================
# CONSTANTS
# ============================================================================

PI = math.pi
TWO_PI = 2.0 * PI
HALF_PI = PI / 2.0
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# ============================================================================
# VECTOR3 CLASS
# ============================================================================

@dataclass
class Vector3:
    """3D Vector class with mathematical operations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        """Add two vectors"""
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """Subtract two vectors"""
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        """Multiply vector by scalar"""
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3':
        """Right multiplication by scalar"""
        return self * scalar
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        """Divide vector by scalar"""
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        """Negate vector"""
        return Vector3(-self.x, -self.y, -self.z)
    
    def __eq__(self, other: 'Vector3') -> bool:
        """Check if vectors are equal"""
        return (abs(self.x - other.x) < 1e-6 and 
                abs(self.y - other.y) < 1e-6 and 
                abs(self.z - other.z) < 1e-6)
    
    def __str__(self) -> str:
        """String representation"""
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def magnitude_squared(self) -> float:
        """Calculate squared magnitude (faster than magnitude)"""
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def normalize(self) -> 'Vector3':
        """Normalize vector to unit length"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag
    
    def dot(self, other: 'Vector3') -> float:
        """Calculate dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        """Calculate cross product"""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def distance_to(self, other: 'Vector3') -> float:
        """Calculate distance to another vector"""
        return (self - other).magnitude()
    
    def lerp(self, other: 'Vector3', t: float) -> 'Vector3':
        """Linear interpolation between vectors"""
        t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]
        return Vector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple"""
        return (self.x, self.y, self.z)
    
    @classmethod
    def from_tuple(cls, tup: Tuple[float, float, float]) -> 'Vector3':
        """Create from tuple"""
        return cls(tup[0], tup[1], tup[2])
    
    @classmethod
    def zero(cls) -> 'Vector3':
        """Create zero vector"""
        return cls(0.0, 0.0, 0.0)
    
    @classmethod
    def one(cls) -> 'Vector3':
        """Create vector with all components 1"""
        return cls(1.0, 1.0, 1.0)
    
    @classmethod
    def up(cls) -> 'Vector3':
        """Create up vector"""
        return cls(0.0, 1.0, 0.0)
    
    @classmethod
    def down(cls) -> 'Vector3':
        """Create down vector"""
        return cls(0.0, -1.0, 0.0)
    
    @classmethod
    def left(cls) -> 'Vector3':
        """Create left vector"""
        return cls(-1.0, 0.0, 0.0)
    
    @classmethod
    def right(cls) -> 'Vector3':
        """Create right vector"""
        return cls(1.0, 0.0, 0.0)
    
    @classmethod
    def forward(cls) -> 'Vector3':
        """Create forward vector"""
        return cls(0.0, 0.0, 1.0)
    
    @classmethod
    def back(cls) -> 'Vector3':
        """Create back vector"""
        return cls(0.0, 0.0, -1.0)

# ============================================================================
# MATRIX4 CLASS
# ============================================================================

class Matrix4:
    """4x4 Matrix class for 3D transformations"""
    
    def __init__(self, data: Optional[List[List[float]]] = None):
        if data is None:
            # Identity matrix
            self.data = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        else:
            self.data = data
    
    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Get matrix element"""
        row, col = key
        return self.data[row][col]
    
    def __setitem__(self, key: Tuple[int, int], value: float):
        """Set matrix element"""
        row, col = key
        self.data[row][col] = value
    
    def __mul__(self, other: 'Matrix4') -> 'Matrix4':
        """Matrix multiplication"""
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                result[i, j] = sum(self[i, k] * other[k, j] for k in range(4))
        return result
    
    def __str__(self) -> str:
        """String representation"""
        lines = []
        for row in self.data:
            lines.append(f"[{row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f}]")
        return "\n".join(lines)
    
    def transpose(self) -> 'Matrix4':
        """Transpose matrix"""
        result = Matrix4()
        for i in range(4):
            for j in range(4):
                result[i, j] = self[j, i]
        return result
    
    def determinant(self) -> float:
        """Calculate determinant (simplified for 4x4)"""
        # This is a simplified determinant calculation
        # In practice, you'd want a more robust implementation
        return 1.0  # Placeholder
    
    def inverse(self) -> 'Matrix4':
        """Calculate matrix inverse"""
        # This is a simplified inverse calculation
        # In practice, you'd want a more robust implementation
        return self.transpose()  # Placeholder for orthogonal matrices
    
    @classmethod
    def identity(cls) -> 'Matrix4':
        """Create identity matrix"""
        return cls()
    
    @classmethod
    def translation(cls, x: float, y: float, z: float) -> 'Matrix4':
        """Create translation matrix"""
        matrix = cls()
        matrix[0, 3] = x
        matrix[1, 3] = y
        matrix[2, 3] = z
        return matrix
    
    @classmethod
    def translation_vec(cls, vec: Vector3) -> 'Matrix4':
        """Create translation matrix from vector"""
        return cls.translation(vec.x, vec.y, vec.z)
    
    @classmethod
    def scaling(cls, x: float, y: float, z: float) -> 'Matrix4':
        """Create scaling matrix"""
        matrix = cls()
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
        return matrix
    
    @classmethod
    def scaling_vec(cls, vec: Vector3) -> 'Matrix4':
        """Create scaling matrix from vector"""
        return cls.scaling(vec.x, vec.y, vec.z)
    
    @classmethod
    def rotation_x(cls, angle_radians: float) -> 'Matrix4':
        """Create rotation matrix around X axis"""
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        
        matrix = cls()
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
        return matrix
    
    @classmethod
    def rotation_y(cls, angle_radians: float) -> 'Matrix4':
        """Create rotation matrix around Y axis"""
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        
        matrix = cls()
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
        return matrix
    
    @classmethod
    def rotation_z(cls, angle_radians: float) -> 'Matrix4':
        """Create rotation matrix around Z axis"""
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        
        matrix = cls()
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
        return matrix
    
    @classmethod
    def rotation_axis(cls, axis: Vector3, angle_radians: float) -> 'Matrix4':
        """Create rotation matrix around arbitrary axis"""
        axis = axis.normalize()
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)
        one_minus_cos = 1.0 - cos_a
        
        matrix = cls()
        matrix[0, 0] = cos_a + axis.x * axis.x * one_minus_cos
        matrix[0, 1] = axis.x * axis.y * one_minus_cos - axis.z * sin_a
        matrix[0, 2] = axis.x * axis.z * one_minus_cos + axis.y * sin_a
        
        matrix[1, 0] = axis.y * axis.x * one_minus_cos + axis.z * sin_a
        matrix[1, 1] = cos_a + axis.y * axis.y * one_minus_cos
        matrix[1, 2] = axis.y * axis.z * one_minus_cos - axis.x * sin_a
        
        matrix[2, 0] = axis.z * axis.x * one_minus_cos - axis.y * sin_a
        matrix[2, 1] = axis.z * axis.y * one_minus_cos + axis.x * sin_a
        matrix[2, 2] = cos_a + axis.z * axis.z * one_minus_cos
        
        return matrix
    
    @classmethod
    def perspective(cls, fov_radians: float, aspect_ratio: float, near: float, far: float) -> 'Matrix4':
        """Create perspective projection matrix"""
        f = 1.0 / math.tan(fov_radians / 2.0)
        
        matrix = cls()
        matrix[0, 0] = f / aspect_ratio
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2.0 * far * near) / (near - far)
        matrix[3, 2] = -1.0
        matrix[3, 3] = 0.0
        return matrix
    
    @classmethod
    def orthographic(cls, left: float, right: float, bottom: float, top: float, near: float, far: float) -> 'Matrix4':
        """Create orthographic projection matrix"""
        matrix = cls()
        matrix[0, 0] = 2.0 / (right - left)
        matrix[1, 1] = 2.0 / (top - bottom)
        matrix[2, 2] = -2.0 / (far - near)
        matrix[0, 3] = -(right + left) / (right - left)
        matrix[1, 3] = -(top + bottom) / (top - bottom)
        matrix[2, 3] = -(far + near) / (far - near)
        return matrix
    
    @classmethod
    def look_at(cls, eye: Vector3, target: Vector3, up: Vector3) -> 'Matrix4':
        """Create look-at matrix"""
        z = (eye - target).normalize()
        x = up.cross(z).normalize()
        y = z.cross(x)
        
        matrix = cls()
        matrix[0, 0] = x.x
        matrix[0, 1] = x.y
        matrix[0, 2] = x.z
        matrix[1, 0] = y.x
        matrix[1, 1] = y.y
        matrix[1, 2] = y.z
        matrix[2, 0] = z.x
        matrix[2, 1] = z.y
        matrix[2, 2] = z.z
        matrix[0, 3] = -x.dot(eye)
        matrix[1, 3] = -y.dot(eye)
        matrix[2, 3] = -z.dot(eye)
        return matrix

# ============================================================================
# QUATERNION CLASS
# ============================================================================

@dataclass
class Quaternion:
    """Quaternion class for 3D rotations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication"""
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"Quaternion({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"
    
    def magnitude(self) -> float:
        """Calculate quaternion magnitude"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
    
    def normalize(self) -> 'Quaternion':
        """Normalize quaternion"""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion()
        return Quaternion(self.x / mag, self.y / mag, self.z / mag, self.w / mag)
    
    def conjugate(self) -> 'Quaternion':
        """Get quaternion conjugate"""
        return Quaternion(-self.x, -self.y, -self.z, self.w)
    
    def inverse(self) -> 'Quaternion':
        """Get quaternion inverse"""
        mag_sq = self.magnitude() ** 2
        if mag_sq == 0:
            return Quaternion()
        conj = self.conjugate()
        return Quaternion(conj.x / mag_sq, conj.y / mag_sq, conj.z / mag_sq, conj.w / mag_sq)
    
    def rotate_vector(self, vec: Vector3) -> Vector3:
        """Rotate a vector by this quaternion"""
        # Convert vector to quaternion
        vec_quat = Quaternion(vec.x, vec.y, vec.z, 0.0)
        
        # Perform rotation: q * v * q^(-1)
        result = self * vec_quat * self.inverse()
        
        return Vector3(result.x, result.y, result.z)
    
    def to_matrix(self) -> Matrix4:
        """Convert quaternion to rotation matrix"""
        # Normalize quaternion
        q = self.normalize()
        
        # Calculate matrix elements
        xx = q.x * q.x
        xy = q.x * q.y
        xz = q.x * q.z
        xw = q.x * q.w
        yy = q.y * q.y
        yz = q.y * q.z
        yw = q.y * q.w
        zz = q.z * q.z
        zw = q.z * q.w
        
        matrix = Matrix4()
        matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
        matrix[0, 1] = 2.0 * (xy - zw)
        matrix[0, 2] = 2.0 * (xz + yw)
        matrix[1, 0] = 2.0 * (xy + zw)
        matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
        matrix[1, 2] = 2.0 * (yz - xw)
        matrix[2, 0] = 2.0 * (xz - yw)
        matrix[2, 1] = 2.0 * (yz + xw)
        matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
        
        return matrix
    
    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle_radians: float) -> 'Quaternion':
        """Create quaternion from axis-angle rotation"""
        axis = axis.normalize()
        half_angle = angle_radians * 0.5
        sin_half = math.sin(half_angle)
        
        return cls(
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half,
            math.cos(half_angle)
        )
    
    @classmethod
    def from_euler(cls, x_rad: float, y_rad: float, z_rad: float) -> 'Quaternion':
        """Create quaternion from Euler angles (XYZ order)"""
        # Convert to half angles
        half_x = x_rad * 0.5
        half_y = y_rad * 0.5
        half_z = z_rad * 0.5
        
        # Calculate trig values
        cos_x = math.cos(half_x)
        sin_x = math.sin(half_x)
        cos_y = math.cos(half_y)
        sin_y = math.sin(half_y)
        cos_z = math.cos(half_z)
        sin_z = math.sin(half_z)
        
        # Combine rotations (XYZ order)
        return cls(
            sin_x * cos_y * cos_z + cos_x * sin_y * sin_z,
            cos_x * sin_y * cos_z - sin_x * cos_y * sin_z,
            cos_x * cos_y * sin_z - sin_x * sin_y * cos_z,
            cos_x * cos_y * cos_z + sin_x * sin_y * sin_z
        )
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        """Create identity quaternion"""
        return cls(0.0, 0.0, 0.0, 1.0)
    
    @classmethod
    def lerp(cls, q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """Linear interpolation between quaternions"""
        t = max(0.0, min(1.0, t))
        
        return cls(
            q1.x + (q2.x - q1.x) * t,
            q1.y + (q2.y - q1.y) * t,
            q1.z + (q2.z - q1.z) * t,
            q1.w + (q2.w - q1.w) * t
        ).normalize()
    
    @classmethod
    def slerp(cls, q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between quaternions"""
        t = max(0.0, min(1.0, t))
        
        # Ensure we take the shortest path
        dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
        if dot < 0:
            q2 = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w)
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            return cls.lerp(q1, q2, t)
        
        # Calculate angle and perform slerp
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        
        w1 = math.sin((1.0 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        return cls(
            q1.x * w1 + q2.x * w2,
            q1.y * w1 + q2.y * w2,
            q1.z * w1 + q2.z * w2,
            q1.w * w1 + q2.w * w2
        ).normalize()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values"""
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t

def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth step interpolation"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * DEG_TO_RAD

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * RAD_TO_DEG

def distance_point_line(point: Vector3, line_start: Vector3, line_end: Vector3) -> float:
    """Calculate distance from point to line"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_length_sq = line_vec.magnitude_squared()
    if line_length_sq == 0:
        return point_vec.magnitude()
    
    t = point_vec.dot(line_vec) / line_length_sq
    t = clamp(t, 0.0, 1.0)
    
    closest_point = line_start + line_vec * t
    return point.distance_to(closest_point)

def distance_point_plane(point: Vector3, plane_point: Vector3, plane_normal: Vector3) -> float:
    """Calculate distance from point to plane"""
    plane_normal = plane_normal.normalize()
    return (point - plane_point).dot(plane_normal)

def reflect_vector(incident: Vector3, normal: Vector3) -> Vector3:
    """Reflect vector off surface with given normal"""
    normal = normal.normalize()
    return incident - normal * (2.0 * incident.dot(normal))

def refract_vector(incident: Vector3, normal: Vector3, ior: float) -> Vector3:
    """Refract vector through surface with given index of refraction"""
    normal = normal.normalize()
    incident = incident.normalize()
    
    cos_i = -incident.dot(normal)
    sin_t_sq = ior * ior * (1.0 - cos_i * cos_i)
    
    if sin_t_sq > 1.0:
        return reflect_vector(incident, normal)  # Total internal reflection
    
    cos_t = math.sqrt(1.0 - sin_t_sq)
    return incident * ior + normal * (ior * cos_i - cos_t)

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_vectors():
    """Demonstrate Vector3 operations"""
    print("=== Vector3 Operations ===\n")
    
    # Create vectors
    v1 = Vector3(1, 2, 3)
    v2 = Vector3(4, 5, 6)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"v1.magnitude() = {v1.magnitude():.3f}")
    print(f"v1.normalize() = {v1.normalize()}")
    print(f"v1.dot(v2) = {v1.dot(v2)}")
    print(f"v1.cross(v2) = {v1.cross(v2)}")
    print(f"v1.distance_to(v2) = {v1.distance_to(v2):.3f}")
    print(f"v1.lerp(v2, 0.5) = {v1.lerp(v2, 0.5)}")
    print()

def demonstrate_matrices():
    """Demonstrate Matrix4 operations"""
    print("=== Matrix4 Operations ===\n")
    
    # Create matrices
    trans = Matrix4.translation(1, 2, 3)
    scale = Matrix4.scaling(2, 2, 2)
    rot_x = Matrix4.rotation_x(PI / 4)
    
    print("Translation matrix:")
    print(trans)
    print("\nScaling matrix:")
    print(scale)
    print("\nRotation X matrix (45°):")
    print(rot_x)
    print("\nCombined transformation:")
    combined = trans * scale * rot_x
    print(combined)
    print()

def demonstrate_quaternions():
    """Demonstrate Quaternion operations"""
    print("=== Quaternion Operations ===\n")
    
    # Create quaternions
    q1 = Quaternion.from_axis_angle(Vector3.up(), PI / 4)
    q2 = Quaternion.from_euler(PI / 6, PI / 4, PI / 3)
    
    print(f"q1 (90° around Y) = {q1}")
    print(f"q2 (Euler XYZ) = {q2}")
    print(f"q1 * q2 = {q1 * q2}")
    print(f"q1.magnitude() = {q1.magnitude():.3f}")
    print(f"q1.normalize() = {q1.normalize()}")
    
    # Rotate a vector
    vec = Vector3(1, 0, 0)
    rotated = q1.rotate_vector(vec)
    print(f"q1.rotate_vector({vec}) = {rotated}")
    
    # Convert to matrix
    matrix = q1.to_matrix()
    print(f"q1.to_matrix():")
    print(matrix)
    print()

def demonstrate_utilities():
    """Demonstrate utility functions"""
    print("=== Utility Functions ===\n")
    
    print(f"clamp(15, 0, 10) = {clamp(15, 0, 10)}")
    print(f"lerp(0, 10, 0.5) = {lerp(0, 10, 0.5)}")
    print(f"smoothstep(0, 10, 5) = {smoothstep(0, 10, 5):.3f}")
    print(f"degrees_to_radians(90) = {degrees_to_radians(90):.3f}")
    print(f"radians_to_degrees(PI/2) = {radians_to_degrees(PI/2):.3f}")
    
    # Point-line distance
    point = Vector3(0, 1, 0)
    line_start = Vector3(-1, 0, 0)
    line_end = Vector3(1, 0, 0)
    dist = distance_point_line(point, line_start, line_end)
    print(f"distance_point_line({point}, {line_start}, {line_end}) = {dist:.3f}")
    
    # Reflection
    incident = Vector3(1, -1, 0).normalize()
    normal = Vector3(0, 1, 0)
    reflected = reflect_vector(incident, normal)
    print(f"reflect_vector({incident}, {normal}) = {reflected}")
    print()

def demonstrate_practical_example():
    """Demonstrate practical 3D graphics example"""
    print("=== Practical 3D Graphics Example ===\n")
    
    # Create a camera
    camera_pos = Vector3(0, 5, -10)
    camera_target = Vector3(0, 0, 0)
    camera_up = Vector3.up()
    
    # Create view matrix
    view_matrix = Matrix4.look_at(camera_pos, camera_target, camera_up)
    print("Camera view matrix:")
    print(view_matrix)
    
    # Create projection matrix
    fov = degrees_to_radians(45)
    aspect = 16.0 / 9.0
    near = 0.1
    far = 100.0
    proj_matrix = Matrix4.perspective(fov, aspect, near, far)
    print("\nProjection matrix:")
    print(proj_matrix)
    
    # Create object transformation
    object_pos = Vector3(2, 1, 0)
    object_rot = Quaternion.from_euler(0, PI / 4, 0)
    object_scale = Vector3(1, 2, 1)
    
    # Build transformation matrix
    trans_matrix = Matrix4.translation_vec(object_pos)
    rot_matrix = object_rot.to_matrix()
    scale_matrix = Matrix4.scaling_vec(object_scale)
    
    model_matrix = trans_matrix * rot_matrix * scale_matrix
    print("\nObject model matrix:")
    print(model_matrix)
    
    # Final transformation
    mvp_matrix = proj_matrix * view_matrix * model_matrix
    print("\nMVP matrix (Model-View-Projection):")
    print(mvp_matrix)
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate 3D math library"""
    print("=== 3D Math Library Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_vectors()
    demonstrate_matrices()
    demonstrate_quaternions()
    demonstrate_utilities()
    demonstrate_practical_example()
    
    print("="*60)
    print("3D Math Library demo completed successfully!")
    print("\nKey features:")
    print("✓ Vector3: 3D vectors with mathematical operations")
    print("✓ Matrix4: 4x4 matrices for transformations")
    print("✓ Quaternion: 3D rotations and interpolation")
    print("✓ Utility functions: Common 3D math operations")
    print("✓ Practical examples: Camera and object transformations")

if __name__ == "__main__":
    main()
