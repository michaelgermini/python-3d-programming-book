"""
Chapter 17: Camera and Projection Concepts - Camera Controls
===========================================================

This module demonstrates camera controls and input handling for 3D graphics applications.

Key Concepts:
- Camera input handling and controls
- Smooth camera movement and interpolation
- Camera constraints and limits
- Input mapping and configuration
"""

import math
import time
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from vector_operations import Vector3D
from camera_systems import Camera, FirstPersonCamera, ThirdPersonCamera, OrbitCamera


class InputType(Enum):
    """Types of input for camera controls."""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"


@dataclass
class InputEvent:
    """Input event for camera controls."""
    input_type: InputType
    key: str
    value: float
    timestamp: float = field(default_factory=time.time)


class CameraConstraints:
    """Constraints and limits for camera movement."""
    
    def __init__(self):
        self.min_distance = 0.5
        self.max_distance = 100.0
        self.min_pitch = -math.pi / 2 + 0.1
        self.max_pitch = math.pi / 2 - 0.1
        self.min_fov = 0.1
        self.max_fov = math.pi - 0.1
    
    def clamp_distance(self, distance: float) -> float:
        """Clamp camera distance to valid range."""
        return max(self.min_distance, min(self.max_distance, distance))
    
    def clamp_pitch(self, pitch: float) -> float:
        """Clamp camera pitch to valid range."""
        return max(self.min_pitch, min(self.max_pitch, pitch))
    
    def clamp_fov(self, fov: float) -> float:
        """Clamp field of view to valid range."""
        return max(self.min_fov, min(self.max_fov, fov))


class InputMapper:
    """Maps input events to camera actions."""
    
    def __init__(self):
        self.key_mappings: Dict[str, str] = {
            'w': 'move_forward',
            's': 'move_backward',
            'a': 'move_left',
            'd': 'move_right',
            'q': 'move_up',
            'e': 'move_down',
            'mouse_left': 'look',
            'mouse_wheel': 'zoom'
        }
        
        self.action_callbacks: Dict[str, Callable] = {}
        self.pressed_keys: set = set()
        self.mouse_delta: Tuple[float, float] = (0, 0)
    
    def register_action(self, action: str, callback: Callable):
        """Register a callback for an action."""
        self.action_callbacks[action] = callback
    
    def handle_key_press(self, key: str):
        """Handle key press event."""
        self.pressed_keys.add(key.lower())
    
    def handle_key_release(self, key: str):
        """Handle key release event."""
        self.pressed_keys.discard(key.lower())
    
    def handle_mouse_move(self, x: float, y: float):
        """Handle mouse movement."""
        self.mouse_delta = (x, y)
    
    def update(self, delta_time: float):
        """Update input mapper and process actions."""
        for key in self.pressed_keys:
            if key in self.key_mappings:
                action = self.key_mappings[key]
                if action in self.action_callbacks:
                    self.action_callbacks[action](delta_time)


class SmoothCameraController:
    """Smooth camera controller with interpolation."""
    
    def __init__(self, camera: Camera):
        self.camera = camera
        self.target_position = camera.position
        self.target_target = camera.target
        self.position_smoothness = 0.1
        self.target_smoothness = 0.1
    
    def set_target_position(self, position: Vector3D):
        """Set target position for smooth interpolation."""
        self.target_position = position
    
    def set_target_look_at(self, target: Vector3D):
        """Set target look-at point for smooth interpolation."""
        self.target_target = target
    
    def update(self, delta_time: float):
        """Update camera with smooth interpolation."""
        # Smooth position interpolation
        position_diff = self.target_position - self.camera.position
        self.camera.position = self.camera.position + position_diff * self.position_smoothness
        
        # Smooth target interpolation
        target_diff = self.target_target - self.camera.target
        self.camera.target = self.camera.target + target_diff * self.target_smoothness
        
        # Update camera orientation
        self.camera.look_at(self.camera.target)


class AdvancedCameraController:
    """Advanced camera controller with multiple modes."""
    
    def __init__(self, camera: Camera):
        self.camera = camera
        self.input_mapper = InputMapper()
        self.constraints = CameraConstraints()
        self.smooth_controller = SmoothCameraController(camera)
        
        # Camera modes
        self.current_mode = 'free'
        self.follow_target: Optional[Vector3D] = None
        self.follow_distance = 5.0
        
        # Setup input mappings
        self._setup_input_mappings()
    
    def _setup_input_mappings(self):
        """Setup input mappings for camera controls."""
        self.input_mapper.register_action('move_forward', self._move_forward)
        self.input_mapper.register_action('move_backward', self._move_backward)
        self.input_mapper.register_action('move_left', self._move_left)
        self.input_mapper.register_action('move_right', self._move_right)
        self.input_mapper.register_action('move_up', self._move_up)
        self.input_mapper.register_action('move_down', self._move_down)
        self.input_mapper.register_action('look', self._look)
        self.input_mapper.register_action('zoom', self._zoom)
    
    def set_follow_target(self, target: Vector3D, distance: float = 5.0):
        """Set target to follow in follow mode."""
        self.follow_target = target
        self.follow_distance = distance
    
    def update(self, delta_time: float):
        """Update camera controller."""
        self.input_mapper.update(delta_time)
        
        if self.current_mode == 'follow' and self.follow_target:
            self._follow_mode(delta_time)
        
        self.smooth_controller.update(delta_time)
    
    def _follow_mode(self, delta_time: float):
        """Follow camera mode."""
        direction = (self.camera.position - self.follow_target).normalize()
        target_pos = self.follow_target + direction * self.follow_distance
        self.smooth_controller.set_target_position(target_pos)
        self.smooth_controller.set_target_look_at(self.follow_target)
    
    # Input handlers
    def _move_forward(self, delta_time: float):
        """Move camera forward."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_forward(delta_time)
    
    def _move_backward(self, delta_time: float):
        """Move camera backward."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_backward(delta_time)
    
    def _move_left(self, delta_time: float):
        """Move camera left."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_left(delta_time)
    
    def _move_right(self, delta_time: float):
        """Move camera right."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_right(delta_time)
    
    def _move_up(self, delta_time: float):
        """Move camera up."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_up(delta_time)
    
    def _move_down(self, delta_time: float):
        """Move camera down."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.move_down(delta_time)
    
    def _look(self, delta_x: float, delta_y: float):
        """Handle mouse look."""
        if isinstance(self.camera, FirstPersonCamera):
            self.camera.handle_mouse_movement(delta_x, delta_y)
        elif isinstance(self.camera, ThirdPersonCamera):
            self.camera.rotate_around_target(delta_x * 0.01, delta_y * 0.01)
        elif isinstance(self.camera, OrbitCamera):
            self.camera.rotate(delta_x * 0.01, delta_y * 0.01)
    
    def _zoom(self, delta: float):
        """Handle zoom input."""
        if isinstance(self.camera, ThirdPersonCamera):
            self.camera.zoom(delta * 0.1)
        elif isinstance(self.camera, OrbitCamera):
            self.camera.zoom(delta * 0.1)
        else:
            new_fov = self.camera.fov + delta * 0.1
            self.camera.fov = self.constraints.clamp_fov(new_fov)


def demonstrate_camera_controls():
    """Demonstrate camera controls and input handling."""
    print("=== Camera Controls Demonstration ===\n")
    
    # Create camera and controller
    camera = FirstPersonCamera(Vector3D(0, 0, 5), Vector3D(0, 0, 0))
    controller = AdvancedCameraController(camera)
    
    print(f"Camera position: {camera.position}")
    print(f"Current mode: {controller.current_mode}")
    
    # Test constraints
    constraints = controller.constraints
    test_distance = 200.0
    clamped_distance = constraints.clamp_distance(test_distance)
    print(f"Distance {test_distance} clamped to: {clamped_distance}")
    
    # Test follow mode
    follow_target = Vector3D(0, 0, 0)
    controller.set_follow_target(follow_target, distance=8.0)
    controller.current_mode = 'follow'
    
    # Update controller
    controller.update(0.016)
    print(f"After follow update: {camera.position}")


if __name__ == "__main__":
    demonstrate_camera_controls()
