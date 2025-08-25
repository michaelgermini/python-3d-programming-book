"""
Chapter 17: Camera and Projection Concepts - Camera Systems
==========================================================

This module demonstrates camera systems and controls for 3D graphics applications.

Key Concepts:
- Camera positioning and orientation
- Camera movement and controls
- View matrix calculations
- Camera interpolation and smoothing
- Multiple camera types and behaviors
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion


@dataclass
class Camera:
    """Base camera class with fundamental camera operations."""
    position: Vector3D
    target: Vector3D
    up: Vector3D
    fov: float = math.pi / 4  # 45 degrees
    aspect_ratio: float = 16.0 / 9.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    
    def __post_init__(self):
        """Initialize camera properties."""
        self.forward = (self.target - self.position).normalize()
        self.right = self.up.cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
    
    def get_view_matrix(self) -> Matrix4x4:
        """Calculate view matrix for this camera."""
        return Matrix4x4.look_at(self.position, self.target, self.up)
    
    def get_projection_matrix(self) -> Matrix4x4:
        """Calculate projection matrix for this camera."""
        return Matrix4x4.perspective(self.fov, self.aspect_ratio, self.near_plane, self.far_plane)
    
    def get_view_projection_matrix(self) -> Matrix4x4:
        """Calculate combined view-projection matrix."""
        return self.get_projection_matrix() * self.get_view_matrix()
    
    def look_at(self, target: Vector3D):
        """Make camera look at a specific target."""
        self.target = target
        self.forward = (self.target - self.position).normalize()
        self.right = self.up.cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
    
    def move(self, direction: Vector3D, distance: float):
        """Move camera in a specific direction."""
        self.position = self.position + direction.normalize() * distance
        self.target = self.target + direction.normalize() * distance
    
    def rotate_around_target(self, yaw: float, pitch: float):
        """Rotate camera around its target."""
        # Calculate current distance from target
        distance = self.position.distance_to(self.target)
        
        # Create rotation quaternions
        yaw_quat = Quaternion.from_axis_angle(Vector3D(0, 1, 0), yaw)
        pitch_quat = Quaternion.from_axis_angle(self.right, pitch)
        
        # Apply rotations to forward vector
        rotated_forward = yaw_quat.rotate_vector(self.forward)
        rotated_forward = pitch_quat.rotate_vector(rotated_forward)
        
        # Update camera position and orientation
        self.position = self.target - rotated_forward * distance
        self.forward = rotated_forward
        self.right = self.up.cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
    
    def set_fov(self, fov: float):
        """Set camera field of view."""
        self.fov = max(0.1, min(math.pi - 0.1, fov))
    
    def set_aspect_ratio(self, aspect_ratio: float):
        """Set camera aspect ratio."""
        self.aspect_ratio = max(0.1, aspect_ratio)
    
    def get_frustum_corners(self) -> List[Vector3D]:
        """Get frustum corner points in world space."""
        # Calculate frustum corners in view space
        tan_half_fov = math.tan(self.fov * 0.5)
        half_height = self.near_plane * tan_half_fov
        half_width = half_height * self.aspect_ratio
        
        # Near plane corners
        near_top_left = Vector3D(-half_width, half_height, -self.near_plane)
        near_top_right = Vector3D(half_width, half_height, -self.near_plane)
        near_bottom_left = Vector3D(-half_width, -half_height, -self.near_plane)
        near_bottom_right = Vector3D(half_width, -half_height, -self.near_plane)
        
        # Far plane corners
        far_scale = self.far_plane / self.near_plane
        far_top_left = near_top_left * far_scale
        far_top_right = near_top_right * far_scale
        far_bottom_left = near_bottom_left * far_scale
        far_bottom_right = near_bottom_right * far_scale
        
        # Transform to world space
        view_matrix = self.get_view_matrix()
        view_inverse = view_matrix.transpose()  # Simplified inverse for rotation matrix
        
        corners = [
            near_top_left, near_top_right, near_bottom_left, near_bottom_right,
            far_top_left, far_top_right, far_bottom_left, far_bottom_right
        ]
        
        world_corners = []
        for corner in corners:
            world_corner = view_inverse.transform_point(corner)
            world_corners.append(world_corner)
        
        return world_corners


class FirstPersonCamera(Camera):
    """First-person camera with mouse and keyboard controls."""
    
    def __init__(self, position: Vector3D, target: Vector3D, up: Vector3D = Vector3D(0, 1, 0)):
        super().__init__(position, target, up)
        self.mouse_sensitivity = 0.002
        self.movement_speed = 5.0
        self.yaw = 0.0
        self.pitch = 0.0
        
        # Calculate initial yaw and pitch
        self._update_angles()
    
    def _update_angles(self):
        """Update yaw and pitch from current forward vector."""
        # Calculate yaw (horizontal rotation)
        self.yaw = math.atan2(self.forward.x, self.forward.z)
        
        # Calculate pitch (vertical rotation)
        self.pitch = math.asin(-self.forward.y)
    
    def handle_mouse_movement(self, delta_x: float, delta_y: float):
        """Handle mouse movement for camera rotation."""
        self.yaw += delta_x * self.mouse_sensitivity
        self.pitch += delta_y * self.mouse_sensitivity
        
        # Clamp pitch to prevent gimbal lock
        self.pitch = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.pitch))
        
        # Update forward vector
        self.forward = Vector3D(
            math.sin(self.yaw) * math.cos(self.pitch),
            -math.sin(self.pitch),
            math.cos(self.yaw) * math.cos(self.pitch)
        )
        
        # Update right and up vectors
        self.right = self.up.cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()
        
        # Update target
        self.target = self.position + self.forward
    
    def move_forward(self, delta_time: float):
        """Move camera forward."""
        self.move(self.forward, self.movement_speed * delta_time)
    
    def move_backward(self, delta_time: float):
        """Move camera backward."""
        self.move(-self.forward, self.movement_speed * delta_time)
    
    def move_right(self, delta_time: float):
        """Move camera right."""
        self.move(self.right, self.movement_speed * delta_time)
    
    def move_left(self, delta_time: float):
        """Move camera left."""
        self.move(-self.right, self.movement_speed * delta_time)
    
    def move_up(self, delta_time: float):
        """Move camera up."""
        self.move(self.up, self.movement_speed * delta_time)
    
    def move_down(self, delta_time: float):
        """Move camera down."""
        self.move(-self.up, self.movement_speed * delta_time)


class ThirdPersonCamera(Camera):
    """Third-person camera that follows a target object."""
    
    def __init__(self, target: Vector3D, distance: float = 5.0, height: float = 2.0):
        position = target + Vector3D(0, height, distance)
        super().__init__(position, target, Vector3D(0, 1, 0))
        self.distance = distance
        self.height = height
        self.yaw = 0.0
        self.pitch = 0.0
        self.rotation_speed = 2.0
        self.zoom_speed = 1.0
        self.min_distance = 1.0
        self.max_distance = 20.0
    
    def update_position(self):
        """Update camera position based on target and angles."""
        # Calculate position from target, distance, and angles
        x = self.target.x + self.distance * math.sin(self.yaw) * math.cos(self.pitch)
        y = self.target.y + self.height + self.distance * math.sin(self.pitch)
        z = self.target.z + self.distance * math.cos(self.yaw) * math.cos(self.pitch)
        
        self.position = Vector3D(x, y, z)
        self.look_at(self.target)
    
    def rotate_around_target(self, delta_yaw: float, delta_pitch: float):
        """Rotate camera around target."""
        self.yaw += delta_yaw * self.rotation_speed
        self.pitch += delta_pitch * self.rotation_speed
        
        # Clamp pitch
        self.pitch = max(-math.pi / 3, min(math.pi / 3, self.pitch))
        
        self.update_position()
    
    def zoom(self, zoom_factor: float):
        """Zoom camera in or out."""
        self.distance += zoom_factor * self.zoom_speed
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        self.update_position()
    
    def follow_target(self, new_target: Vector3D, smooth_factor: float = 1.0):
        """Smoothly follow a moving target."""
        self.target = self.target.lerp(new_target, smooth_factor)
        self.update_position()


class OrbitCamera(Camera):
    """Orbit camera that rotates around a fixed point."""
    
    def __init__(self, center: Vector3D, distance: float = 5.0):
        position = center + Vector3D(0, 0, distance)
        super().__init__(position, center, Vector3D(0, 1, 0))
        self.center = center
        self.distance = distance
        self.azimuth = 0.0  # Horizontal angle
        self.elevation = 0.0  # Vertical angle
        self.rotation_speed = 1.0
        self.zoom_speed = 1.0
        self.min_distance = 0.5
        self.max_distance = 50.0
    
    def update_position(self):
        """Update camera position based on spherical coordinates."""
        x = self.center.x + self.distance * math.cos(self.azimuth) * math.cos(self.elevation)
        y = self.center.y + self.distance * math.sin(self.elevation)
        z = self.center.z + self.distance * math.sin(self.azimuth) * math.cos(self.elevation)
        
        self.position = Vector3D(x, y, z)
        self.look_at(self.center)
    
    def rotate(self, delta_azimuth: float, delta_elevation: float):
        """Rotate camera around center."""
        self.azimuth += delta_azimuth * self.rotation_speed
        self.elevation += delta_elevation * self.rotation_speed
        
        # Clamp elevation to prevent flipping
        self.elevation = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, self.elevation))
        
        self.update_position()
    
    def zoom(self, zoom_factor: float):
        """Zoom camera in or out."""
        self.distance += zoom_factor * self.zoom_speed
        self.distance = max(self.min_distance, min(self.max_distance, self.distance))
        self.update_position()
    
    def set_center(self, center: Vector3D):
        """Set the center point to orbit around."""
        self.center = center
        self.update_position()


class CameraController:
    """Controller for managing multiple cameras and transitions."""
    
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.current_camera: Optional[str] = None
        self.transition_time = 0.0
        self.transition_duration = 1.0
        self.transition_start_camera: Optional[Camera] = None
        self.transition_end_camera: Optional[Camera] = None
    
    def add_camera(self, name: str, camera: Camera):
        """Add a camera to the controller."""
        self.cameras[name] = camera
        if self.current_camera is None:
            self.current_camera = name
    
    def get_current_camera(self) -> Optional[Camera]:
        """Get the currently active camera."""
        if self.current_camera and self.current_camera in self.cameras:
            return self.cameras[self.current_camera]
        return None
    
    def switch_camera(self, camera_name: str, transition_duration: float = 1.0):
        """Switch to a different camera with smooth transition."""
        if camera_name not in self.cameras:
            return
        
        current = self.get_current_camera()
        if current is None:
            self.current_camera = camera_name
            return
        
        self.transition_start_camera = Camera(
            current.position, current.target, current.up,
            current.fov, current.aspect_ratio, current.near_plane, current.far_plane
        )
        self.transition_end_camera = self.cameras[camera_name]
        self.transition_time = 0.0
        self.transition_duration = transition_duration
        self.current_camera = camera_name
    
    def update(self, delta_time: float):
        """Update camera controller and handle transitions."""
        if self.transition_start_camera and self.transition_end_camera:
            self.transition_time += delta_time
            t = min(1.0, self.transition_time / self.transition_duration)
            
            # Smooth interpolation
            t = self._smooth_step(t)
            
            # Interpolate camera properties
            current = self.get_current_camera()
            if current:
                current.position = self.transition_start_camera.position.lerp(
                    self.transition_end_camera.position, t
                )
                current.target = self.transition_start_camera.target.lerp(
                    self.transition_end_camera.target, t
                )
                current.fov = self.transition_start_camera.fov + (
                    self.transition_end_camera.fov - self.transition_start_camera.fov
                ) * t
                
                # Update camera orientation
                current.look_at(current.target)
            
            # End transition
            if t >= 1.0:
                self.transition_start_camera = None
                self.transition_end_camera = None
    
    def _smooth_step(self, t: float) -> float:
        """Apply smooth step interpolation."""
        return t * t * (3.0 - 2.0 * t)
    
    def get_camera_list(self) -> List[str]:
        """Get list of available camera names."""
        return list(self.cameras.keys())


def demonstrate_camera_systems():
    """Demonstrate various camera systems and controls."""
    print("=== Camera Systems Demonstration ===\n")
    
    # Create different camera types
    print("1. Basic Camera:")
    basic_camera = Camera(
        Vector3D(0, 0, 5),
        Vector3D(0, 0, 0),
        Vector3D(0, 1, 0)
    )
    print(f"Position: {basic_camera.position}")
    print(f"Target: {basic_camera.target}")
    print(f"Forward: {basic_camera.forward}")
    print()
    
    # First-person camera
    print("2. First-Person Camera:")
    fp_camera = FirstPersonCamera(
        Vector3D(0, 0, 5),
        Vector3D(0, 0, 0)
    )
    print(f"Initial position: {fp_camera.position}")
    print(f"Initial yaw: {fp_camera.yaw:.3f}")
    print(f"Initial pitch: {fp_camera.pitch:.3f}")
    
    # Simulate mouse movement
    fp_camera.handle_mouse_movement(100, 50)
    print(f"After mouse movement - Yaw: {fp_camera.yaw:.3f}, Pitch: {fp_camera.pitch:.3f}")
    print(f"New position: {fp_camera.position}")
    print()
    
    # Third-person camera
    print("3. Third-Person Camera:")
    tp_camera = ThirdPersonCamera(
        Vector3D(0, 0, 0),
        distance=5.0,
        height=2.0
    )
    print(f"Position: {tp_camera.position}")
    print(f"Distance: {tp_camera.distance}")
    print(f"Height: {tp_camera.height}")
    
    # Simulate rotation
    tp_camera.rotate_around_target(0.5, 0.2)
    print(f"After rotation - Position: {tp_camera.position}")
    print()
    
    # Orbit camera
    print("4. Orbit Camera:")
    orbit_camera = OrbitCamera(
        Vector3D(0, 0, 0),
        distance=8.0
    )
    print(f"Position: {orbit_camera.position}")
    print(f"Azimuth: {orbit_camera.azimuth:.3f}")
    print(f"Elevation: {orbit_camera.elevation:.3f}")
    
    # Simulate orbit movement
    orbit_camera.rotate(0.3, 0.1)
    print(f"After orbit - Position: {orbit_camera.position}")
    print()
    
    # Camera controller
    print("5. Camera Controller:")
    controller = CameraController()
    controller.add_camera("basic", basic_camera)
    controller.add_camera("first_person", fp_camera)
    controller.add_camera("third_person", tp_camera)
    controller.add_camera("orbit", orbit_camera)
    
    print(f"Available cameras: {controller.get_camera_list()}")
    print(f"Current camera: {controller.current_camera}")
    
    # Switch cameras
    controller.switch_camera("orbit", 2.0)
    print(f"Switched to: {controller.current_camera}")
    
    # Simulate transition
    controller.update(0.5)  # 0.5 seconds into transition
    current = controller.get_current_camera()
    if current:
        print(f"Transition position: {current.position}")


if __name__ == "__main__":
    demonstrate_camera_systems()
