"""
Chapter 26: Normal Mapping, Bump Mapping, and PBR - Normal Mapping
===============================================================

This module demonstrates normal mapping techniques for enhanced surface detail.
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class NormalMapType(Enum):
    """Normal map type enumeration."""
    TANGENT_SPACE = "tangent_space"
    OBJECT_SPACE = "object_space"


@dataclass
class NormalMapConfig:
    """Configuration for normal mapping."""
    resolution: int = 1024
    strength: float = 1.0


class TangentSpaceCalculator:
    """Calculates tangent space vectors for normal mapping."""

    def calculate_tangent_space(self, vertices: np.ndarray, normals: np.ndarray, 
                              tex_coords: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate tangent space vectors for a mesh."""
        vertex_count = len(vertices)
        tangents = np.zeros((vertex_count, 3), dtype=np.float32)
        bitangents = np.zeros((vertex_count, 3), dtype=np.float32)
        
        for i in range(0, len(indices), 3):
            idx0, idx1, idx2 = indices[i:i+3]
            v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
            uv0, uv1, uv2 = tex_coords[idx0], tex_coords[idx1], tex_coords[idx2]
            
            edge1, edge2 = v1 - v0, v2 - v0
            delta_uv1, delta_uv2 = uv1 - uv0, uv2 - uv0
            
            f = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1])
            tangent = f * (delta_uv2[1] * edge1 - delta_uv1[1] * edge2)
            bitangent = f * (-delta_uv2[0] * edge1 + delta_uv1[0] * edge2)
            
            for idx in [idx0, idx1, idx2]:
                tangents[idx] += tangent
                bitangents[idx] += bitangent
        
        # Normalize
        for i in range(vertex_count):
            normal = normals[i] / np.linalg.norm(normals[i])
            tangents[i] = tangents[i] / np.linalg.norm(tangents[i])
            tangents[i] = tangents[i] - np.dot(tangents[i], normal) * normal
            tangents[i] = tangents[i] / np.linalg.norm(tangents[i])
            bitangents[i] = np.cross(normal, tangents[i])
        
        return tangents, bitangents, normals

    def create_tbn_matrix(self, tangent: np.ndarray, bitangent: np.ndarray, 
                         normal: np.ndarray) -> np.ndarray:
        """Create TBN matrix."""
        return np.array([
            [tangent[0], bitangent[0], normal[0]],
            [tangent[1], bitangent[1], normal[1]],
            [tangent[2], bitangent[2], normal[2]]
        ])


class NormalMap:
    """Represents a normal map texture."""

    def __init__(self, width: int, height: int, config: NormalMapConfig):
        self.width = width
        self.height = height
        self.config = config
        self.texture_id = 0
        self.normal_data = np.zeros((height, width, 3), dtype=np.float32)
        self.setup_normal_map()

    def setup_normal_map(self):
        """Setup normal map texture."""
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self.width, self.height, 
                       0, gl.GL_RGB, gl.GL_FLOAT, None)

    def generate_procedural(self, pattern: str = "checkerboard"):
        """Generate procedural normal map."""
        if pattern == "checkerboard":
            for y in range(self.height):
                for x in range(self.width):
                    check_size = 32
                    check_x = (x // check_size) % 2
                    check_y = (y // check_size) % 2
                    
                    if check_x == check_y:
                        normal = np.array([0.0, 0.0, 1.0])
                    else:
                        normal = np.array([0.0, 0.0, 0.8])
                    
                    self.normal_data[y, x] = normal * 0.5 + 0.5
            
            self.update_texture()

    def update_texture(self):
        """Update the OpenGL texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                          gl.GL_RGB, gl.GL_FLOAT, self.normal_data)

    def bind(self, texture_unit: int = 0):
        """Bind normal map texture."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

    def sample_normal(self, u: float, v: float) -> np.ndarray:
        """Sample normal from texture coordinates."""
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        normal = self.normal_data[y, x]
        return normal * 2.0 - 1.0

    def cleanup(self):
        """Clean up normal map resources."""
        if self.texture_id:
            gl.glDeleteTextures(1, [self.texture_id])


class NormalMappingSystem:
    """Manages normal mapping for a rendering system."""

    def __init__(self):
        self.normal_maps: Dict[str, NormalMap] = {}
        self.tangent_calculator = TangentSpaceCalculator()
        self.active_normal_map: Optional[NormalMap] = None

    def create_normal_map(self, name: str, width: int, height: int, 
                         config: NormalMapConfig) -> NormalMap:
        """Create a normal map."""
        normal_map = NormalMap(width, height, config)
        self.normal_maps[name] = normal_map
        return normal_map

    def bind_normal_map(self, name: str, texture_unit: int = 0):
        """Bind a normal map for rendering."""
        if name in self.normal_maps:
            normal_map = self.normal_maps[name]
            normal_map.bind(texture_unit)
            self.active_normal_map = normal_map

    def calculate_lighting_with_normal_map(self, position: np.ndarray, normal: np.ndarray,
                                         tangent: np.ndarray, bitangent: np.ndarray,
                                         light_direction: np.ndarray, 
                                         tex_coords: np.ndarray) -> float:
        """Calculate lighting using normal mapping."""
        if not self.active_normal_map:
            return max(np.dot(normal, light_direction), 0.0)
        
        u, v = tex_coords[0], tex_coords[1]
        sampled_normal = self.active_normal_map.sample_normal(u, v)
        
        tbn_matrix = self.tangent_calculator.create_tbn_matrix(tangent, bitangent, normal)
        light_tangent = tbn_matrix @ light_direction
        
        return max(np.dot(sampled_normal, light_tangent), 0.0)

    def cleanup(self):
        """Clean up all normal maps."""
        for normal_map in self.normal_maps.values():
            normal_map.cleanup()
        self.normal_maps.clear()


def demonstrate_normal_mapping():
    """Demonstrate normal mapping techniques."""
    print("=== Normal Mapping, Bump Mapping, and PBR - Normal Mapping ===\n")

    system = NormalMappingSystem()
    config = NormalMapConfig(resolution=512, strength=1.0)

    print("1. Creating normal maps...")
    checkerboard = system.create_normal_map("checkerboard", 512, 512, config)
    checkerboard.generate_procedural("checkerboard")
    print("   Created checkerboard normal map")

    print("\n2. Tangent space calculations...")
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    tex_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    
    tangents, bitangents, calculated_normals = system.tangent_calculator.calculate_tangent_space(
        vertices, normals, tex_coords, indices)
    print(f"   Calculated {len(tangents)} tangent vectors")

    print("\n3. Normal mapping lighting example...")
    system.bind_normal_map("checkerboard")
    
    position = np.array([0.5, 0.5, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    tangent = tangents[0]
    bitangent = bitangents[0]
    light_direction = np.array([1.0, 1.0, 1.0])
    tex_coords = np.array([0.5, 0.5])
    
    lighting = system.calculate_lighting_with_normal_map(
        position, normal, tangent, bitangent, light_direction, tex_coords)
    print(f"   Normal mapping lighting result: {lighting:.3f}")

    print("\n4. Features demonstrated:")
    print("   - Tangent space calculations")
    print("   - Procedural normal map generation")
    print("   - Normal map sampling")
    print("   - Lighting with normal mapping")

    system.cleanup()
    print("\n5. Normal mapping system cleaned up successfully")


if __name__ == "__main__":
    demonstrate_normal_mapping()
