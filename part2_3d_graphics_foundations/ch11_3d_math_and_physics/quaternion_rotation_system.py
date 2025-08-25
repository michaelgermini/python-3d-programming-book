#!/usr/bin/env python3
"""
Chapter 11: 3D Math and Physics
Quaternion Rotation System

Demonstrates quaternion algebra, rotation representation, interpolation
techniques, and conversion utilities for smooth 3D rotations.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Quaternion Rotation System"
__description__ = "Advanced quaternion-based rotation system"

# ============================================================================
# CORE DATA STRUCTURES
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

@dataclass
class Quaternion:
    """Quaternion class for representing 3D rotations"""
    w: float  # Real part
    x: float  # i component
    y: float  # j component
    z: float  # k component
    
    def __post_init__(self):
        """Normalize the quaternion after initialization"""
        self.normalize()
    
    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion addition"""
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion subtraction"""
        return Quaternion(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (Hamilton product)"""
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
    
    def __rmul__(self, scalar: float) -> 'Quaternion':
        """Scalar multiplication"""
        return Quaternion(
            self.w * scalar,
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Quaternion inverse"""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            raise ValueError("Cannot invert zero quaternion")
        return self.conjugate() * (1.0 / norm_sq)
    
    def magnitude(self) -> float:
        """Quaternion magnitude"""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Normalize the quaternion"""
        mag = self.magnitude()
        if mag > 0:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    def normalized(self) -> 'Quaternion':
        """Return a normalized copy of the quaternion"""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag
        )
    
    def dot(self, other: 'Quaternion') -> float:
        """Dot product of two quaternions"""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def __str__(self):
        return f"Quaternion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# ============================================================================
# QUATERNION UTILITIES
# ============================================================================

class QuaternionUtils:
    """Utility class for quaternion operations"""
    
    @staticmethod
    def identity() -> Quaternion:
        """Create identity quaternion (no rotation)"""
        return Quaternion(1, 0, 0, 0)
    
    @staticmethod
    def from_axis_angle(axis: Vector3D, angle: float) -> Quaternion:
        """Create quaternion from axis-angle representation"""
        axis = axis.normalize()
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        
        return Quaternion(
            cos_half,
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half
        )
    
    @staticmethod
    def from_euler(x: float, y: float, z: float, order: str = "xyz") -> Quaternion:
        """Create quaternion from Euler angles (in radians)"""
        # Convert to half angles
        x_half = x / 2.0
        y_half = y / 2.0
        z_half = z / 2.0
        
        # Precompute trigonometric values
        cx = math.cos(x_half)
        sx = math.sin(x_half)
        cy = math.cos(y_half)
        sy = math.sin(y_half)
        cz = math.cos(z_half)
        sz = math.sin(z_half)
        
        if order.lower() == "xyz":
            return Quaternion(
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz
            )
        elif order.lower() == "yxz":
            return Quaternion(
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz
            )
        elif order.lower() == "zxy":
            return Quaternion(
                cx * cy * cz - sx * sy * sz,
                sx * cy * cz + cx * sy * sz,
                cx * sy * cz - sx * cy * sz,
                cx * cy * sz + sx * sy * cz
            )
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
    
    @staticmethod
    def to_euler(q: Quaternion, order: str = "xyz") -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (in radians)"""
        # Normalize quaternion
        q = q.normalized()
        
        if order.lower() == "xyz":
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
    
    @staticmethod
    def to_matrix(q: Quaternion) -> List[List[float]]:
        """Convert quaternion to 3x3 rotation matrix"""
        q = q.normalized()
        
        # Extract components
        w, x, y, z = q.w, q.x, q.y, q.z
        
        # Compute matrix elements
        return [
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ]
    
    @staticmethod
    def from_matrix(matrix: List[List[float]]) -> Quaternion:
        """Create quaternion from 3x3 rotation matrix"""
        trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (matrix[2][1] - matrix[1][2]) / s
            y = (matrix[0][2] - matrix[2][0]) / s
            z = (matrix[1][0] - matrix[0][1]) / s
        elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
            s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
            w = (matrix[2][1] - matrix[1][2]) / s
            x = 0.25 * s
            y = (matrix[0][1] + matrix[1][0]) / s
            z = (matrix[0][2] + matrix[2][0]) / s
        elif matrix[1][1] > matrix[2][2]:
            s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
            w = (matrix[0][2] - matrix[2][0]) / s
            x = (matrix[0][1] + matrix[1][0]) / s
            y = 0.25 * s
            z = (matrix[1][2] + matrix[2][1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
            w = (matrix[1][0] - matrix[0][1]) / s
            x = (matrix[0][2] + matrix[2][0]) / s
            y = (matrix[1][2] + matrix[2][1]) / s
            z = 0.25 * s
        
        return Quaternion(w, x, y, z)
    
    @staticmethod
    def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
        """Spherical Linear Interpolation (SLERP) between two quaternions"""
        # Ensure t is in [0, 1]
        t = max(0.0, min(1.0, t))
        
        # Normalize quaternions
        q1 = q1.normalized()
        q2 = q2.normalized()
        
        # Calculate dot product
        dot = q1.dot(q2)
        
        # If dot product is negative, negate one quaternion to ensure shortest path
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            return Quaternion(
                q1.w + (q2.w - q1.w) * t,
                q1.x + (q2.x - q1.x) * t,
                q1.y + (q2.y - q1.y) * t,
                q1.z + (q2.z - q1.z) * t
            ).normalized()
        
        # Calculate angle
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        
        # Calculate interpolation factors
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        # Interpolate
        return Quaternion(
            w1 * q1.w + w2 * q2.w,
            w1 * q1.x + w2 * q2.x,
            w1 * q1.y + w2 * q2.y,
            w1 * q1.z + w2 * q2.z
        ).normalized()
    
    @staticmethod
    def nlerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
        """Normalized Linear Interpolation (NLERP) between two quaternions"""
        # Ensure t is in [0, 1]
        t = max(0.0, min(1.0, t))
        
        # Normalize quaternions
        q1 = q1.normalized()
        q2 = q2.normalized()
        
        # Linear interpolation
        result = Quaternion(
            q1.w + (q2.w - q1.w) * t,
            q1.x + (q2.x - q1.x) * t,
            q1.y + (q2.y - q1.y) * t,
            q1.z + (q2.z - q1.z) * t
        )
        
        return result.normalized()
    
    @staticmethod
    def rotate_vector(q: Quaternion, v: Vector3D) -> Vector3D:
        """Rotate a vector using a quaternion"""
        # Convert vector to quaternion (w=0)
        v_quat = Quaternion(0, v.x, v.y, v.z)
        
        # Apply rotation: q * v * q^(-1)
        q_inv = q.inverse()
        result = q * v_quat * q_inv
        
        return Vector3D(result.x, result.y, result.z)

# ============================================================================
# ROTATION SYSTEM
# ============================================================================

class RotationSystem:
    """Advanced rotation system using quaternions"""
    
    def __init__(self):
        self.rotation = QuaternionUtils.identity()
        self.position = Vector3D(0, 0, 0)
        self.scale = Vector3D(1, 1, 1)
    
    def set_rotation(self, rotation: Quaternion):
        """Set the rotation"""
        self.rotation = rotation.normalized()
    
    def set_position(self, position: Vector3D):
        """Set the position"""
        self.position = position
    
    def set_scale(self, scale: Vector3D):
        """Set the scale"""
        self.scale = scale
    
    def rotate_by_axis_angle(self, axis: Vector3D, angle: float):
        """Rotate by axis-angle"""
        rotation_quat = QuaternionUtils.from_axis_angle(axis, angle)
        self.rotation = rotation_quat * self.rotation
    
    def rotate_by_euler(self, x: float, y: float, z: float, order: str = "xyz"):
        """Rotate by Euler angles"""
        rotation_quat = QuaternionUtils.from_euler(x, y, z, order)
        self.rotation = rotation_quat * self.rotation
    
    def look_at(self, target: Vector3D, up: Vector3D = Vector3D(0, 1, 0)):
        """Make the object look at a target point"""
        # Calculate forward direction
        forward = (target - self.position).normalize()
        
        # Calculate right direction
        right = forward.cross(up).normalize()
        
        # Recalculate up direction to ensure orthogonality
        up = right.cross(forward)
        
        # Create rotation matrix
        matrix = [
            [right.x, right.y, right.z],
            [up.x, up.y, up.z],
            [-forward.x, -forward.y, -forward.z]
        ]
        
        # Convert to quaternion
        self.rotation = QuaternionUtils.from_matrix(matrix)
    
    def get_forward(self) -> Vector3D:
        """Get the forward direction"""
        return QuaternionUtils.rotate_vector(self.rotation, Vector3D(0, 0, -1))
    
    def get_right(self) -> Vector3D:
        """Get the right direction"""
        return QuaternionUtils.rotate_vector(self.rotation, Vector3D(1, 0, 0))
    
    def get_up(self) -> Vector3D:
        """Get the up direction"""
        return QuaternionUtils.rotate_vector(self.rotation, Vector3D(0, 1, 0))
    
    def get_euler_angles(self, order: str = "xyz") -> Tuple[float, float, float]:
        """Get Euler angles from current rotation"""
        return QuaternionUtils.to_euler(self.rotation, order)
    
    def get_matrix(self) -> List[List[float]]:
        """Get rotation matrix"""
        return QuaternionUtils.to_matrix(self.rotation)
    
    def __str__(self):
        euler = self.get_euler_angles()
        return f"RotationSystem(pos={self.position}, euler=({math.degrees(euler[0]):.1f}°, {math.degrees(euler[1]):.1f}°, {math.degrees(euler[2]):.1f}°))"

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_quaternion_basics():
    """Demonstrate basic quaternion operations"""
    print("=== Quaternion Basics ===\n")
    
    # Create quaternions
    q1 = Quaternion(1, 0, 0, 0)  # Identity
    q2 = Quaternion(0.707, 0.707, 0, 0)  # 90° around X-axis
    q3 = Quaternion(0.707, 0, 0.707, 0)  # 90° around Y-axis
    
    print(f"Identity quaternion: {q1}")
    print(f"90° X-axis rotation: {q2}")
    print(f"90° Y-axis rotation: {q3}")
    
    # Basic operations
    print(f"\nQuaternion addition: {q1} + {q2} = {q1 + q2}")
    print(f"Quaternion multiplication: {q2} * {q3} = {q2 * q3}")
    print(f"Quaternion conjugate: {q2.conjugate()}")
    print(f"Quaternion magnitude: {q2.magnitude():.3f}")
    
    # Vector rotation
    test_vector = Vector3D(1, 0, 0)
    rotated_vector = QuaternionUtils.rotate_vector(q2, test_vector)
    print(f"\nRotating {test_vector} by {q2}: {rotated_vector}")
    
    print()

def demonstrate_quaternion_creation():
    """Demonstrate different ways to create quaternions"""
    print("=== Quaternion Creation ===\n")
    
    # From axis-angle
    axis = Vector3D(0, 1, 0)
    angle = math.pi / 2  # 90 degrees
    q_axis_angle = QuaternionUtils.from_axis_angle(axis, angle)
    print(f"From axis-angle (Y-axis, 90°): {q_axis_angle}")
    
    # From Euler angles
    q_euler = QuaternionUtils.from_euler(0, math.pi/2, 0, "xyz")
    print(f"From Euler angles (0, 90°, 0): {q_euler}")
    
    # From rotation matrix
    matrix = [
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ]
    q_matrix = QuaternionUtils.from_matrix(matrix)
    print(f"From rotation matrix: {q_matrix}")
    
    # Identity quaternion
    q_identity = QuaternionUtils.identity()
    print(f"Identity quaternion: {q_identity}")
    
    print()

def demonstrate_quaternion_conversion():
    """Demonstrate quaternion conversion utilities"""
    print("=== Quaternion Conversion ===\n")
    
    # Create a quaternion
    q = QuaternionUtils.from_euler(math.pi/4, math.pi/3, math.pi/6, "xyz")
    print(f"Original quaternion: {q}")
    
    # Convert to Euler angles
    euler = QuaternionUtils.to_euler(q, "xyz")
    print(f"Euler angles (radians): {euler}")
    print(f"Euler angles (degrees): ({math.degrees(euler[0]):.1f}°, {math.degrees(euler[1]):.1f}°, {math.degrees(euler[2]):.1f}°)")
    
    # Convert to matrix
    matrix = QuaternionUtils.to_matrix(q)
    print(f"Rotation matrix:")
    for row in matrix:
        print(f"  [{row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}]")
    
    # Convert back to quaternion
    q_back = QuaternionUtils.from_matrix(matrix)
    print(f"Quaternion from matrix: {q_back}")
    
    # Test round-trip conversion
    euler_back = QuaternionUtils.to_euler(q_back, "xyz")
    print(f"Round-trip Euler angles: {euler_back}")
    
    print()

def demonstrate_quaternion_interpolation():
    """Demonstrate quaternion interpolation techniques"""
    print("=== Quaternion Interpolation ===\n")
    
    # Create two quaternions
    q1 = QuaternionUtils.from_euler(0, 0, 0, "xyz")  # No rotation
    q2 = QuaternionUtils.from_euler(0, math.pi, 0, "xyz")  # 180° around Y-axis
    
    print(f"Start quaternion: {q1}")
    print(f"End quaternion: {q2}")
    
    # Test different interpolation values
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # SLERP interpolation
        q_slerp = QuaternionUtils.slerp(q1, q2, t)
        euler_slerp = QuaternionUtils.to_euler(q_slerp, "xyz")
        
        # NLERP interpolation
        q_nlerp = QuaternionUtils.nlerp(q1, q2, t)
        euler_nlerp = QuaternionUtils.to_euler(q_nlerp, "xyz")
        
        print(f"t={t:.2f}: SLERP={math.degrees(euler_slerp[1]):6.1f}°, NLERP={math.degrees(euler_nlerp[1]):6.1f}°")
    
    # Demonstrate smooth interpolation
    print("\nSmooth interpolation (SLERP):")
    for i in range(11):
        t = i / 10.0
        q_interp = QuaternionUtils.slerp(q1, q2, t)
        euler = QuaternionUtils.to_euler(q_interp, "xyz")
        print(f"  t={t:.1f}: {math.degrees(euler[1]):6.1f}°")
    
    print()

def demonstrate_rotation_system():
    """Demonstrate the rotation system"""
    print("=== Rotation System ===\n")
    
    # Create rotation system
    rot_system = RotationSystem()
    rot_system.set_position(Vector3D(0, 0, 0))
    
    print(f"Initial state: {rot_system}")
    print(f"Forward: {rot_system.get_forward()}")
    print(f"Right: {rot_system.get_right()}")
    print(f"Up: {rot_system.get_up()}")
    
    # Rotate by axis-angle
    print("\nRotating 90° around Y-axis:")
    rot_system.rotate_by_axis_angle(Vector3D(0, 1, 0), math.pi/2)
    print(f"State: {rot_system}")
    print(f"Forward: {rot_system.get_forward()}")
    print(f"Right: {rot_system.get_right()}")
    print(f"Up: {rot_system.get_up()}")
    
    # Rotate by Euler angles
    print("\nRotating by Euler angles (45°, 30°, 60°):")
    rot_system.rotate_by_euler(math.pi/4, math.pi/6, math.pi/3, "xyz")
    print(f"State: {rot_system}")
    
    # Look at target
    print("\nLooking at target (5, 0, 5):")
    rot_system.look_at(Vector3D(5, 0, 5))
    print(f"State: {rot_system}")
    print(f"Forward: {rot_system.get_forward()}")
    
    print()

def demonstrate_quaternion_operations():
    """Demonstrate advanced quaternion operations"""
    print("=== Advanced Quaternion Operations ===\n")
    
    # Create test quaternions
    q1 = QuaternionUtils.from_axis_angle(Vector3D(1, 0, 0), math.pi/4)  # 45° around X
    q2 = QuaternionUtils.from_axis_angle(Vector3D(0, 1, 0), math.pi/3)  # 60° around Y
    q3 = QuaternionUtils.from_axis_angle(Vector3D(0, 0, 1), math.pi/6)  # 30° around Z
    
    print(f"q1 (45° X-axis): {q1}")
    print(f"q2 (60° Y-axis): {q2}")
    print(f"q3 (30° Z-axis): {q3}")
    
    # Composition of rotations
    q_composed = q1 * q2 * q3
    print(f"\nComposed rotation (q1 * q2 * q3): {q_composed}")
    
    # Inverse operations
    q_inverse = q1.inverse()
    q_identity = q1 * q_inverse
    print(f"\nInverse of q1: {q_inverse}")
    print(f"q1 * q1^(-1) (should be identity): {q_identity}")
    
    # Dot product and angle between quaternions
    dot_product = q1.dot(q2)
    angle = math.acos(abs(dot_product)) * 2  # *2 because quaternions represent half-angles
    print(f"\nDot product between q1 and q2: {dot_product:.3f}")
    print(f"Angle between rotations: {math.degrees(angle):.1f}°")
    
    # Vector rotation chain
    test_vector = Vector3D(1, 0, 0)
    print(f"\nRotating vector {test_vector}:")
    
    v1 = QuaternionUtils.rotate_vector(q1, test_vector)
    print(f"  After q1: {v1}")
    
    v2 = QuaternionUtils.rotate_vector(q2, v1)
    print(f"  After q2: {v2}")
    
    v3 = QuaternionUtils.rotate_vector(q3, v2)
    print(f"  After q3: {v3}")
    
    # Compare with composed rotation
    v_composed = QuaternionUtils.rotate_vector(q_composed, test_vector)
    print(f"  Composed rotation: {v_composed}")
    
    print()

def demonstrate_performance_comparison():
    """Demonstrate performance comparison between interpolation methods"""
    print("=== Performance Comparison ===\n")
    
    import time
    
    # Create test quaternions
    q1 = QuaternionUtils.from_euler(0, 0, 0, "xyz")
    q2 = QuaternionUtils.from_euler(math.pi, math.pi/2, math.pi/4, "xyz")
    
    # Test SLERP performance
    start_time = time.time()
    for i in range(10000):
        t = i / 10000.0
        QuaternionUtils.slerp(q1, q2, t)
    slerp_time = time.time() - start_time
    
    # Test NLERP performance
    start_time = time.time()
    for i in range(10000):
        t = i / 10000.0
        QuaternionUtils.nlerp(q1, q2, t)
    nlerp_time = time.time() - start_time
    
    print(f"SLERP time for 10,000 interpolations: {slerp_time:.4f} seconds")
    print(f"NLERP time for 10,000 interpolations: {nlerp_time:.4f} seconds")
    print(f"Speedup: {slerp_time/nlerp_time:.2f}x")
    
    print("\nNote: NLERP is faster but may not follow the shortest path")
    print("SLERP is more accurate but computationally more expensive")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate quaternion rotation system"""
    print("=== Quaternion Rotation System Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate quaternion basics
    demonstrate_quaternion_basics()
    
    # Demonstrate quaternion creation
    demonstrate_quaternion_creation()
    
    # Demonstrate quaternion conversion
    demonstrate_quaternion_conversion()
    
    # Demonstrate quaternion interpolation
    demonstrate_quaternion_interpolation()
    
    # Demonstrate rotation system
    demonstrate_rotation_system()
    
    # Demonstrate advanced operations
    demonstrate_quaternion_operations()
    
    # Demonstrate performance comparison
    demonstrate_performance_comparison()
    
    print("="*60)
    print("Quaternion Rotation System demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Quaternion algebra and operations")
    print("✓ Rotation representation and composition")
    print("✓ Interpolation techniques (SLERP, NLERP)")
    print("✓ Conversion utilities (quaternion ↔ euler ↔ matrix)")
    print("✓ Advanced rotation system with look-at functionality")
    print("✓ Performance comparison and optimization")
    
    print("\nQuaternion advantages:")
    print("• Smooth interpolation without gimbal lock")
    print("• Efficient rotation composition")
    print("• Compact representation (4 values vs 9 for matrix)")
    print("• Numerically stable for repeated operations")
    print("• Natural for animation and interpolation")
    
    print("\nApplications:")
    print("• Camera systems: Smooth camera movement and tracking")
    print("• Character animation: Joint rotations and skeletal animation")
    print("• Object orientation: 3D object positioning and rotation")
    print("• Physics simulation: Rigid body dynamics and constraints")
    print("• Virtual reality: Head tracking and controller orientation")
    
    print("\nNext steps:")
    print("• Explore physics simulation with quaternion-based rotations")
    print("• Learn about advanced interpolation techniques")
    print("• Study quaternion-based animation systems")
    print("• Understand quaternions in modern graphics APIs")
    print("• Master quaternion optimization techniques")

if __name__ == "__main__":
    main()
