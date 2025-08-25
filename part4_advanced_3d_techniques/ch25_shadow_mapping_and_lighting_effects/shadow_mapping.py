"""
Chapter 25: Shadow Mapping and Lighting Effects - Shadow Mapping
==============================================================

This module demonstrates shadow mapping techniques for realistic lighting.
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ShadowMapType(Enum):
    """Shadow map type enumeration."""
    HARD_SHADOWS = "hard_shadows"
    SOFT_SHADOWS = "soft_shadows"
    CASCADED_SHADOWS = "cascaded_shadows"


@dataclass
class ShadowMapConfig:
    """Configuration for shadow mapping."""
    resolution: int = 2048
    near_plane: float = 0.1
    far_plane: float = 100.0
    bias: float = 0.005
    filter_size: int = 3


class ShadowMap:
    """Represents a shadow map with depth texture."""

    def __init__(self, width: int, height: int, config: ShadowMapConfig):
        self.width = width
        self.height = height
        self.config = config
        self.depth_texture_id = 0
        self.framebuffer_id = 0
        self.light_view_matrix = np.eye(4)
        self.light_projection_matrix = np.eye(4)
        self.setup_shadow_map()

    def setup_shadow_map(self):
        """Setup shadow map texture and framebuffer."""
        # Create depth texture
        self.depth_texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        
        # Allocate texture storage
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, 
                       self.width, self.height, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        
        # Create framebuffer
        self.framebuffer_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer_id)
        
        # Attach depth texture
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, 
                                 gl.GL_TEXTURE_2D, self.depth_texture_id, 0)
        
        # No color attachments
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def bind_for_writing(self):
        """Bind shadow map for writing depth values."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer_id)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    def bind_for_reading(self, texture_unit: int = 0):
        """Bind shadow map for reading in shaders."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)

    def set_light_matrices(self, light_position: np.ndarray, light_direction: np.ndarray):
        """Set light view and projection matrices."""
        # Calculate light view matrix
        target = light_position + light_direction
        up = np.array([0.0, 1.0, 0.0])
        
        z_axis = light_direction / np.linalg.norm(light_direction)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        self.light_view_matrix = np.array([
            [x_axis[0], x_axis[1], x_axis[2], -np.dot(x_axis, light_position)],
            [y_axis[0], y_axis[1], y_axis[2], -np.dot(y_axis, light_position)],
            [z_axis[0], z_axis[1], z_axis[2], -np.dot(z_axis, light_position)],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Simple orthographic projection
        self.light_projection_matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0 / (self.config.far_plane - self.config.near_plane), 
             -(self.config.far_plane + self.config.near_plane) / (self.config.far_plane - self.config.near_plane)],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def get_light_space_matrix(self) -> np.ndarray:
        """Get combined light view-projection matrix."""
        return self.light_projection_matrix @ self.light_view_matrix

    def cleanup(self):
        """Clean up shadow map resources."""
        if self.depth_texture_id:
            gl.glDeleteTextures(1, [self.depth_texture_id])
        if self.framebuffer_id:
            gl.glDeleteFramebuffers(1, [self.framebuffer_id])


class SoftShadowMap(ShadowMap):
    """Implements soft shadows using PCF."""

    def setup_shadow_map(self):
        """Setup shadow map for soft shadows."""
        super().setup_shadow_map()
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL)


class ShadowManager:
    """Manages multiple shadow maps."""

    def __init__(self):
        self.shadow_maps: Dict[str, Any] = {}

    def create_shadow_map(self, name: str, shadow_type: ShadowMapType, 
                         width: int, height: int, config: ShadowMapConfig) -> Any:
        """Create a shadow map of the specified type."""
        if shadow_type == ShadowMapType.HARD_SHADOWS:
            shadow_map = ShadowMap(width, height, config)
        elif shadow_type == ShadowMapType.SOFT_SHADOWS:
            shadow_map = SoftShadowMap(width, height, config)
        else:
            raise ValueError(f"Unsupported shadow map type: {shadow_type}")
        
        self.shadow_maps[name] = shadow_map
        return shadow_map

    def get_shadow_map(self, name: str) -> Optional[Any]:
        """Get a shadow map by name."""
        return self.shadow_maps.get(name)

    def cleanup(self):
        """Clean up all shadow maps."""
        for shadow_map in self.shadow_maps.values():
            shadow_map.cleanup()
        self.shadow_maps.clear()


def demonstrate_shadow_mapping():
    """Demonstrate shadow mapping techniques."""
    print("=== Shadow Mapping and Lighting Effects - Shadow Mapping ===\n")

    manager = ShadowManager()
    config = ShadowMapConfig(resolution=2048, bias=0.005)

    print("1. Creating hard shadow map...")
    hard_shadows = manager.create_shadow_map("hard_shadows", ShadowMapType.HARD_SHADOWS, 
                                           2048, 2048, config)

    print("2. Creating soft shadow map...")
    soft_shadows = manager.create_shadow_map("soft_shadows", ShadowMapType.SOFT_SHADOWS, 
                                           2048, 2048, config)

    print("3. Shadow Map Usage:")
    print("  # Render shadow map")
    print("  hard_shadows.bind_for_writing()")
    print("  # Render scene from light perspective")
    print("  hard_shadows.bind_for_reading()")
    print("  # Render scene with shadow sampling")

    light_position = np.array([10.0, 10.0, 10.0])
    light_direction = np.array([-1.0, -1.0, -1.0])
    hard_shadows.set_light_matrices(light_position, light_direction)
    
    light_space_matrix = hard_shadows.get_light_space_matrix()
    print(f"4. Light space matrix shape: {light_space_matrix.shape}")

    manager.cleanup()
    print("5. Shadow maps cleaned up successfully")


if __name__ == "__main__":
    demonstrate_shadow_mapping()
