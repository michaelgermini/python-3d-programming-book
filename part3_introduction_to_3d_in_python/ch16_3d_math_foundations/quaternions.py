"""
Chapter 16: 3D Math Foundations - Quaternions
============================================

This module demonstrates quaternion operations for 3D rotations,
providing smooth interpolation and avoiding gimbal lock.

Key Concepts:
- Quaternion representation and operations
- Quaternion-based rotations
- Spherical linear interpolation (SLERP)
- Conversion between quaternions and Euler angles
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from vector_operations import Vector3D


@dataclass
class Quaternion:
    """Quaternion for 3D rotations with comprehensive operations."""
    x: float
    y: float
    z: float
    w: float
    
    def __post_init__(self):
        """Normalize quaternion after initialization."""
        self.normalize()
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication."""
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        )
    
    def magnitude(self) -> float:
        """Calculate quaternion magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def normalize(self):
        """Normalize quaternion to unit length."""
        mag = self.magnitude()
        if mag > 1e-9:
            self.x /= mag
            self.y /= mag
            self.z /= mag
            self.w /= mag
        else:
            self.x = 0
            self.y = 0
            self.z = 0
            self.w = 1
    
    def conjugate(self) -> 'Quaternion':
        """Return quaternion conjugate."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)
    
    def inverse(self) -> 'Quaternion':
        """Return quaternion inverse."""
        mag_sq = self.x**2 + self.y**2 + self.z**2 + self.w**2
        if mag_sq < 1e-9:
            return Quaternion(0, 0, 0, 1)
        return self.conjugate() * (1.0 / mag_sq)
    
    def dot(self, other: 'Quaternion') -> float:
        """Dot product of two quaternions."""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def rotate_vector(self, vector: Vector3D) -> Vector3D:
        """Rotate a 3D vector by this quaternion."""
        v_quat = Quaternion(vector.x, vector.y, vector.z, 0)
        result = self * v_quat * self.inverse()
        return Vector3D(result.x, result.y, result.z)
    
    def to_euler_angles(self) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (pitch, yaw, roll)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (pitch, yaw, roll)
    
    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between two quaternions."""
        t = max(0, min(1, t))
        
        q1 = Quaternion(self.x, self.y, self.z, self.w)
        q2 = Quaternion(other.x, other.y, other.z, other.w)
        q1.normalize()
        q2.normalize()
        
        dot = q1.dot(q2)
        if dot < 0:
            q2 = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w)
            dot = -dot
        
        if dot > 0.9995:
            result = Quaternion(
                q1.x + (q2.x - q1.x) * t,
                q1.y + (q2.y - q1.y) * t,
                q1.z + (q2.z - q1.z) * t,
                q1.w + (q2.w - q1.w) * t
            )
            result.normalize()
            return result
        
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        return Quaternion(
            q1.x * w1 + q2.x * w2,
            q1.y * w1 + q2.y * w2,
            q1.z * w1 + q2.z * w2,
            q1.w * w1 + q2.w * w2
        )
    
    def to_array(self) -> List[float]:
        """Convert to array representation."""
        return [self.x, self.y, self.z, self.w]
    
    @classmethod
    def from_array(cls, arr: List[float]) -> 'Quaternion':
        """Create quaternion from array."""
        if len(arr) != 4:
            raise ValueError("Array must have exactly 4 elements")
        return cls(arr[0], arr[1], arr[2], arr[3])
    
    @classmethod
    def from_euler_angles(cls, pitch: float, yaw: float, roll: float) -> 'Quaternion':
        """Create quaternion from Euler angles."""
        pitch *= 0.5
        yaw *= 0.5
        roll *= 0.5
        
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cr = math.cos(roll)
        sr = math.sin(roll)
        
        return cls(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        )
    
    @classmethod
    def from_axis_angle(cls, axis: Vector3D, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        axis = axis.normalize()
        half_angle = angle * 0.5
        sin_half = math.sin(half_angle)
        
        return cls(
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half,
            math.cos(half_angle)
        )
    
    @classmethod
    def identity(cls) -> 'Quaternion':
        """Create identity quaternion."""
        return cls(0, 0, 0, 1)


class QuaternionArray:
    """Optimized array of quaternions using NumPy."""
    
    def __init__(self, quaternions: Optional[List[Quaternion]] = None):
        """Initialize quaternion array."""
        if quaternions is None:
            self.data = np.zeros((0, 4), dtype=np.float32)
        else:
            self.data = np.array([q.to_array() for q in quaternions], dtype=np.float32)
    
    def add_quaternion(self, quaternion: Quaternion):
        """Add a quaternion to the array."""
        self.data = np.vstack([self.data, quaternion.to_array()])
    
    def get_quaternion(self, index: int) -> Quaternion:
        """Get quaternion at index."""
        if index < 0 or index >= len(self.data):
            raise IndexError("Quaternion index out of range")
        return Quaternion.from_array(self.data[index].tolist())
    
    def normalize_all(self):
        """Normalize all quaternions."""
        magnitudes = np.sqrt(np.sum(self.data**2, axis=1))
        magnitudes = np.where(magnitudes == 0, 1, magnitudes)
        self.data = self.data / magnitudes[:, np.newaxis]
    
    def __len__(self) -> int:
        """Number of quaternions in array."""
        return len(self.data)


class QuaternionMath:
    """Static utility class for quaternion mathematics."""
    
    @staticmethod
    def shortest_arc(from_vector: Vector3D, to_vector: Vector3D) -> Quaternion:
        """Create quaternion for shortest rotation between two vectors."""
        from_vec = from_vector.normalize()
        to_vec = to_vector.normalize()
        
        dot = from_vec.dot(to_vec)
        
        if dot > 0.999999:
            return Quaternion.identity()
        elif dot < -0.999999:
            if abs(from_vec.x) < abs(from_vec.y):
                if abs(from_vec.x) < abs(from_vec.z):
                    axis = Vector3D(1, 0, 0).cross(from_vec)
                else:
                    axis = Vector3D(0, 0, 1).cross(from_vec)
            else:
                if abs(from_vec.y) < abs(from_vec.z):
                    axis = Vector3D(0, 1, 0).cross(from_vec)
                else:
                    axis = Vector3D(0, 0, 1).cross(from_vec)
            
            return Quaternion.from_axis_angle(axis, math.pi)
        else:
            cross = from_vec.cross(to_vec)
            s = math.sqrt((1 + dot) * 2)
            invs = 1 / s
            
            return Quaternion(
                cross.x * invs,
                cross.y * invs,
                cross.z * invs,
                s * 0.5
            )
    
    @staticmethod
    def look_at(forward: Vector3D, up: Vector3D = Vector3D(0, 1, 0)) -> Quaternion:
        """Create quaternion that rotates to look in forward direction."""
        forward = forward.normalize()
        up = up.normalize()
        
        right = up.cross(forward).normalize()
        up = forward.cross(right)
        
        # Create rotation matrix and convert to quaternion
        # Simplified conversion
        trace = right.x + up.y + forward.z
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (up.z - forward.y) / s
            y = (forward.x - right.z) / s
            z = (right.y - up.x) / s
        else:
            if right.x > up.y and right.x > forward.z:
                s = math.sqrt(1.0 + right.x - up.y - forward.z) * 2
                w = (up.z - forward.y) / s
                x = 0.25 * s
                y = (up.x + right.y) / s
                z = (forward.x + right.z) / s
            elif up.y > forward.z:
                s = math.sqrt(1.0 + up.y - right.x - forward.z) * 2
                w = (forward.x - right.z) / s
                x = (up.x + right.y) / s
                y = 0.25 * s
                z = (up.z + forward.y) / s
            else:
                s = math.sqrt(1.0 + forward.z - right.x - up.y) * 2
                w = (right.y - up.x) / s
                x = (forward.x + right.z) / s
                y = (up.z + forward.y) / s
                z = 0.25 * s
        
        return Quaternion(x, y, z, w)


def demonstrate_quaternion_operations():
    """Demonstrate various quaternion operations."""
    print("=== Quaternion Operations Demonstration ===\n")
    
    # Create test quaternions
    identity = Quaternion.identity()
    rotation_x = Quaternion.from_axis_angle(Vector3D(1, 0, 0), math.pi / 2)
    rotation_y = Quaternion.from_axis_angle(Vector3D(0, 1, 0), math.pi / 4)
    rotation_z = Quaternion.from_euler_angles(0, math.pi / 3, 0)
    
    print("Basic Quaternions:")
    print(f"Identity: {identity.to_array()}")
    print(f"Rotation X (90°): {rotation_x.to_array()}")
    print(f"Rotation Y (45°): {rotation_y.to_array()}")
    print(f"Rotation Z (60°): {rotation_z.to_array()}")
    print()
    
    # Quaternion multiplication
    combined = rotation_x * rotation_y * rotation_z
    print("Combined Rotation:")
    print(f"X * Y * Z = {combined.to_array()}")
    print()
    
    # Vector rotation
    vector = Vector3D(1, 0, 0)
    rotated_vector = combined.rotate_vector(vector)
    print("Vector Rotation:")
    print(f"Original vector: {vector}")
    print(f"Rotated vector: {rotated_vector}")
    print()
    
    # Euler angles conversion
    euler = combined.to_euler_angles()
    print("Euler Angles:")
    print(f"Pitch: {euler[0]:.3f} radians ({math.degrees(euler[0]):.1f}°)")
    print(f"Yaw: {euler[1]:.3f} radians ({math.degrees(euler[1]):.1f}°)")
    print(f"Roll: {euler[2]:.3f} radians ({math.degrees(euler[2]):.1f}°)")
    print()
    
    # Quaternion interpolation
    q1 = Quaternion.from_axis_angle(Vector3D(0, 1, 0), 0)
    q2 = Quaternion.from_axis_angle(Vector3D(0, 1, 0), math.pi)
    
    print("Quaternion Interpolation:")
    for t in [0, 0.25, 0.5, 0.75, 1]:
        slerp_result = q1.slerp(q2, t)
        print(f"t={t}: SLERP={slerp_result.to_array()}")
    print()
    
    # Shortest arc
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    shortest_arc = QuaternionMath.shortest_arc(v1, v2)
    print("Shortest Arc:")
    print(f"From {v1} to {v2}: {shortest_arc.to_array()}")
    print(f"Rotated vector: {shortest_arc.rotate_vector(v1)}")
    print()
    
    # Look at
    forward = Vector3D(1, 1, 0).normalize()
    look_at = QuaternionMath.look_at(forward)
    print("Look At:")
    print(f"Forward: {forward}")
    print(f"Quaternion: {look_at.to_array()}")
    print(f"Rotated forward: {look_at.rotate_vector(Vector3D(0, 0, 1))}")


if __name__ == "__main__":
    demonstrate_quaternion_operations()
