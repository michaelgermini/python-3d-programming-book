"""
Chapter 18: Transformations - Coordinate Systems
===============================================

This module demonstrates coordinate systems and coordinate transformations.

Key Concepts:
- Local and world coordinate systems
- Coordinate system transformations
- Multiple coordinate spaces
- Coordinate system conversions
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion
from transformation_matrices import Transform


@dataclass
class CoordinateSystem:
    """Represents a coordinate system with origin and axes."""
    name: str
    origin: Vector3D
    x_axis: Vector3D
    y_axis: Vector3D
    z_axis: Vector3D
    
    def __post_init__(self):
        """Normalize axes to ensure they are unit vectors."""
        self.x_axis = self.x_axis.normalize()
        self.y_axis = self.y_axis.normalize()
        self.z_axis = self.z_axis.normalize()
    
    def get_transform_matrix(self) -> Matrix4x4:
        """Get transformation matrix from world to this coordinate system."""
        return Matrix4x4([
            [self.x_axis.x, self.y_axis.x, self.z_axis.x, self.origin.x],
            [self.x_axis.y, self.y_axis.y, self.z_axis.y, self.origin.y],
            [self.x_axis.z, self.y_axis.z, self.z_axis.z, self.origin.z],
            [0, 0, 0, 1]
        ])
    
    def get_inverse_transform_matrix(self) -> Matrix4x4:
        """Get transformation matrix from this coordinate system to world."""
        # Inverse of rotation part (transpose for orthogonal matrix)
        rotation_inverse = Matrix4x4([
            [self.x_axis.x, self.x_axis.y, self.x_axis.z, 0],
            [self.y_axis.x, self.y_axis.y, self.y_axis.z, 0],
            [self.z_axis.x, self.z_axis.y, self.z_axis.z, 0],
            [0, 0, 0, 1]
        ])
        
        # Translation inverse
        translation_inverse = Matrix4x4.translation(-self.origin.x, -self.origin.y, -self.origin.z)
        
        return rotation_inverse * translation_inverse
    
    def world_to_local(self, world_point: Vector3D) -> Vector3D:
        """Convert world coordinates to local coordinates."""
        matrix = self.get_inverse_transform_matrix()
        return matrix.transform_point(world_point)
    
    def local_to_world(self, local_point: Vector3D) -> Vector3D:
        """Convert local coordinates to world coordinates."""
        matrix = self.get_transform_matrix()
        return matrix.transform_point(local_point)
    
    def world_to_local_vector(self, world_vector: Vector3D) -> Vector3D:
        """Convert world vector to local vector."""
        matrix = self.get_inverse_transform_matrix()
        return matrix.transform_vector(world_vector)
    
    def local_to_world_vector(self, local_vector: Vector3D) -> Vector3D:
        """Convert local vector to world vector."""
        matrix = self.get_transform_matrix()
        return matrix.transform_vector(local_vector)
    
    def is_right_handed(self) -> bool:
        """Check if coordinate system is right-handed."""
        return self.x_axis.cross(self.y_axis).dot(self.z_axis) > 0
    
    def is_orthogonal(self) -> bool:
        """Check if axes are orthogonal."""
        return (abs(self.x_axis.dot(self.y_axis)) < 1e-6 and
                abs(self.y_axis.dot(self.z_axis)) < 1e-6 and
                abs(self.z_axis.dot(self.x_axis)) < 1e-6)


class CoordinateSystemManager:
    """Manages multiple coordinate systems and conversions."""
    
    def __init__(self):
        self.coordinate_systems: Dict[str, CoordinateSystem] = {}
        self.world_system = CoordinateSystem(
            "World",
            Vector3D(0, 0, 0),
            Vector3D(1, 0, 0),
            Vector3D(0, 1, 0),
            Vector3D(0, 0, 1)
        )
        self.coordinate_systems["World"] = self.world_system
    
    def add_coordinate_system(self, system: CoordinateSystem):
        """Add a coordinate system."""
        self.coordinate_systems[system.name] = system
    
    def get_coordinate_system(self, name: str) -> Optional[CoordinateSystem]:
        """Get coordinate system by name."""
        return self.coordinate_systems.get(name)
    
    def convert_point(self, point: Vector3D, from_system: str, to_system: str) -> Vector3D:
        """Convert point from one coordinate system to another."""
        from_sys = self.get_coordinate_system(from_system)
        to_sys = self.get_coordinate_system(to_system)
        
        if not from_sys or not to_sys:
            return point
        
        # Convert to world first, then to target system
        world_point = from_sys.local_to_world(point)
        return to_sys.world_to_local(world_point)
    
    def convert_vector(self, vector: Vector3D, from_system: str, to_system: str) -> Vector3D:
        """Convert vector from one coordinate system to another."""
        from_sys = self.get_coordinate_system(from_system)
        to_sys = self.get_coordinate_system(to_system)
        
        if not from_sys or not to_sys:
            return vector
        
        # Convert to world first, then to target system
        world_vector = from_sys.local_to_world_vector(vector)
        return to_sys.world_to_local_vector(world_vector)


class CameraCoordinateSystem(CoordinateSystem):
    """Camera-specific coordinate system."""
    
    def __init__(self, camera_position: Vector3D, camera_target: Vector3D, camera_up: Vector3D = Vector3D(0, 1, 0)):
        # Z-axis points away from camera (negative view direction)
        z_axis = (camera_position - camera_target).normalize()
        
        # X-axis is right vector
        x_axis = camera_up.cross(z_axis).normalize()
        
        # Y-axis is up vector (recalculated to ensure orthogonality)
        y_axis = z_axis.cross(x_axis).normalize()
        
        super().__init__("Camera", camera_position, x_axis, y_axis, z_axis)
    
    def get_view_matrix(self) -> Matrix4x4:
        """Get view matrix for this camera coordinate system."""
        return self.get_inverse_transform_matrix()


class ObjectCoordinateSystem(CoordinateSystem):
    """Object-specific coordinate system."""
    
    def __init__(self, name: str, transform: Transform):
        # Extract axes from rotation
        rotation_matrix = transform.rotation.to_matrix()
        
        x_axis = Vector3D(rotation_matrix.data[0][0], rotation_matrix.data[1][0], rotation_matrix.data[2][0])
        y_axis = Vector3D(rotation_matrix.data[0][1], rotation_matrix.data[1][1], rotation_matrix.data[2][1])
        z_axis = Vector3D(rotation_matrix.data[0][2], rotation_matrix.data[1][2], rotation_matrix.data[2][2])
        
        super().__init__(name, transform.position, x_axis, y_axis, z_axis)
    
    def update_from_transform(self, transform: Transform):
        """Update coordinate system from transform."""
        self.origin = transform.position
        
        rotation_matrix = transform.rotation.to_matrix()
        self.x_axis = Vector3D(rotation_matrix.data[0][0], rotation_matrix.data[1][0], rotation_matrix.data[2][0])
        self.y_axis = Vector3D(rotation_matrix.data[0][1], rotation_matrix.data[1][1], rotation_matrix.data[2][1])
        self.z_axis = Vector3D(rotation_matrix.data[0][2], rotation_matrix.data[1][2], rotation_matrix.data[2][2])
        
        # Normalize axes
        self.x_axis = self.x_axis.normalize()
        self.y_axis = self.y_axis.normalize()
        self.z_axis = self.z_axis.normalize()


def demonstrate_coordinate_systems():
    """Demonstrate coordinate systems and transformations."""
    print("=== Coordinate Systems Demonstration ===\n")
    
    # Create coordinate system manager
    manager = CoordinateSystemManager()
    
    # Create camera coordinate system
    camera_pos = Vector3D(0, 0, 5)
    camera_target = Vector3D(0, 0, 0)
    camera_system = CameraCoordinateSystem(camera_pos, camera_target)
    manager.add_coordinate_system(camera_system)
    
    print("1. Camera Coordinate System:")
    print(f"Origin: {camera_system.origin}")
    print(f"X-axis: {camera_system.x_axis}")
    print(f"Y-axis: {camera_system.y_axis}")
    print(f"Z-axis: {camera_system.z_axis}")
    print(f"Right-handed: {camera_system.is_right_handed()}")
    print(f"Orthogonal: {camera_system.is_orthogonal()}")
    print()
    
    # Create object coordinate system
    object_transform = Transform(
        position=Vector3D(2, 1, 0),
        rotation=Quaternion.from_euler(0, math.pi/4, 0),
        scale=Vector3D(1, 1, 1)
    )
    object_system = ObjectCoordinateSystem("Object", object_transform)
    manager.add_coordinate_system(object_system)
    
    print("2. Object Coordinate System:")
    print(f"Origin: {object_system.origin}")
    print(f"X-axis: {object_system.x_axis}")
    print(f"Y-axis: {object_system.y_axis}")
    print(f"Z-axis: {object_system.z_axis}")
    print()
    
    # Test coordinate conversions
    print("3. Coordinate Conversions:")
    world_point = Vector3D(1, 0, 0)
    
    # Convert to camera space
    camera_point = manager.convert_point(world_point, "World", "Camera")
    print(f"World point {world_point} in camera space: {camera_point}")
    
    # Convert to object space
    object_point = manager.convert_point(world_point, "World", "Object")
    print(f"World point {world_point} in object space: {object_point}")
    
    # Convert back to world
    world_point_back = manager.convert_point(camera_point, "Camera", "World")
    print(f"Camera point {camera_point} back to world: {world_point_back}")
    print()
    
    # Test vector conversions
    print("4. Vector Conversions:")
    world_vector = Vector3D(0, 1, 0)
    
    camera_vector = manager.convert_vector(world_vector, "World", "Camera")
    print(f"World vector {world_vector} in camera space: {camera_vector}")
    
    object_vector = manager.convert_vector(world_vector, "World", "Object")
    print(f"World vector {world_vector} in object space: {object_vector}")
    print()
    
    # Test view matrix
    print("5. View Matrix:")
    view_matrix = camera_system.get_view_matrix()
    print("View matrix:")
    for row in view_matrix.data:
        print(f"  {row}")
    
    # Test point transformation with view matrix
    test_point = Vector3D(0, 0, -1)  # Point in front of camera
    view_point = view_matrix.transform_point(test_point)
    print(f"Point {test_point} in view space: {view_point}")


if __name__ == "__main__":
    demonstrate_coordinate_systems()
