"""
Chapter 26: Normal Mapping, Bump Mapping, and PBR - Bump Mapping
=============================================================

This module demonstrates bump mapping techniques for surface detail.
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class BumpMapType(Enum):
    """Bump map type enumeration."""
    HEIGHT_MAP = "height_map"
    DISPLACEMENT = "displacement"
    PARALLAX = "parallax"


@dataclass
class BumpMapConfig:
    """Configuration for bump mapping."""
    resolution: int = 1024
    height_scale: float = 0.1
    parallax_scale: float = 0.05
    min_samples: int = 8
    max_samples: int = 32


class BumpMap:
    """Represents a bump map texture."""

    def __init__(self, width: int, height: int, config: BumpMapConfig):
        self.width = width
        self.height = height
        self.config = config
        self.texture_id = 0
        self.height_data = np.zeros((height, width), dtype=np.float32)
        self.setup_bump_map()

    def setup_bump_map(self):
        """Setup bump map texture."""
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, self.width, self.height, 
                       0, gl.GL_RED, gl.GL_FLOAT, None)

    def generate_procedural(self, pattern: str = "noise"):
        """Generate procedural bump map."""
        if pattern == "noise":
            self._generate_noise_bump_map()
        elif pattern == "ridges":
            self._generate_ridges_bump_map()
        elif pattern == "craters":
            self._generate_craters_bump_map()
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    def _generate_noise_bump_map(self):
        """Generate noise-based bump map."""
        for y in range(self.height):
            for x in range(self.width):
                # Simple noise generation
                noise = (np.sin(x * 0.1) + np.cos(y * 0.1) + 
                        np.sin((x + y) * 0.05)) / 3.0
                self.height_data[y, x] = noise * 0.5 + 0.5
        
        self.update_texture()

    def _generate_ridges_bump_map(self):
        """Generate ridges pattern bump map."""
        for y in range(self.height):
            for x in range(self.width):
                # Create ridge pattern
                ridge_freq = 0.02
                ridge_height = 0.8
                
                ridge_x = np.sin(x * ridge_freq) * ridge_height
                ridge_y = np.cos(y * ridge_freq) * ridge_height
                
                height = (ridge_x + ridge_y) * 0.5 + 0.5
                self.height_data[y, x] = height
        
        self.update_texture()

    def _generate_craters_bump_map(self):
        """Generate craters pattern bump map."""
        # Create base height
        self.height_data.fill(0.5)
        
        # Add random craters
        num_craters = 20
        for _ in range(num_craters):
            center_x = np.random.randint(0, self.width)
            center_y = np.random.randint(0, self.height)
            radius = np.random.randint(10, 50)
            depth = np.random.uniform(0.3, 0.8)
            
            for y in range(max(0, center_y - radius), min(self.height, center_y + radius)):
                for x in range(max(0, center_x - radius), min(self.width, center_x + radius)):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        # Create crater shape
                        crater_height = depth * (1.0 - (dist / radius)**2)
                        self.height_data[y, x] = max(0.0, self.height_data[y, x] - crater_height)
        
        self.update_texture()

    def update_texture(self):
        """Update the OpenGL texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                          gl.GL_RED, gl.GL_FLOAT, self.height_data)

    def bind(self, texture_unit: int = 0):
        """Bind bump map texture."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

    def sample_height(self, u: float, v: float) -> float:
        """Sample height from texture coordinates."""
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        return self.height_data[y, x]

    def calculate_normal_offset(self, u: float, v: float, normal: np.ndarray) -> np.ndarray:
        """Calculate normal offset from bump map."""
        # Sample neighboring heights for gradient calculation
        du = 1.0 / self.width
        dv = 1.0 / self.height
        
        h_center = self.sample_height(u, v)
        h_right = self.sample_height(u + du, v)
        h_up = self.sample_height(u, v + dv)
        
        # Calculate gradients
        grad_x = (h_right - h_center) / du
        grad_y = (h_up - h_center) / dv
        
        # Create tangent and bitangent vectors
        tangent = np.array([1.0, 0.0, grad_x * self.config.height_scale])
        bitangent = np.array([0.0, 1.0, grad_y * self.config.height_scale])
        
        # Calculate perturbed normal
        perturbed_normal = np.cross(tangent, bitangent)
        perturbed_normal = perturbed_normal / np.linalg.norm(perturbed_normal)
        
        return perturbed_normal

    def cleanup(self):
        """Clean up bump map resources."""
        if self.texture_id:
            gl.glDeleteTextures(1, [self.texture_id])


class ParallaxMapping:
    """Implements parallax mapping for enhanced bump mapping."""

    def __init__(self, bump_map: BumpMap, config: BumpMapConfig):
        self.bump_map = bump_map
        self.config = config

    def calculate_parallax_offset(self, tex_coords: np.ndarray, view_dir: np.ndarray,
                                tbn_matrix: np.ndarray) -> np.ndarray:
        """Calculate parallax offset for texture coordinates."""
        # Transform view direction to tangent space
        view_tangent = tbn_matrix @ view_dir
        
        # Calculate parallax offset
        height = self.bump_map.sample_height(tex_coords[0], tex_coords[1])
        offset = view_tangent[:2] * height * self.config.parallax_scale / view_tangent[2]
        
        return tex_coords[:2] - offset

    def calculate_parallax_occlusion_mapping(self, tex_coords: np.ndarray, view_dir: np.ndarray,
                                           tbn_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate parallax occlusion mapping."""
        # Transform view direction to tangent space
        view_tangent = tbn_matrix @ view_dir
        
        # Ray marching parameters
        ray_step = view_tangent[:2] * self.config.parallax_scale / view_tangent[2]
        ray_step = ray_step / self.config.max_samples
        
        current_tex_coords = tex_coords[:2].copy()
        current_height = 1.0
        current_ray_height = 0.0
        
        # Ray marching
        for i in range(self.config.max_samples):
            current_ray_height += 1.0 / self.config.max_samples
            current_tex_coords += ray_step
            
            # Sample height at current position
            sample_height = self.bump_map.sample_height(current_tex_coords[0], current_tex_coords[1])
            
            if current_ray_height >= sample_height:
                # Hit point found, interpolate
                prev_tex_coords = current_tex_coords - ray_step
                prev_height = self.bump_map.sample_height(prev_tex_coords[0], prev_tex_coords[1])
                
                # Linear interpolation
                t = (prev_height - (current_ray_height - 1.0 / self.config.max_samples)) / (prev_height - sample_height)
                final_tex_coords = prev_tex_coords + t * ray_step
                
                return final_tex_coords, 1.0 - t
        
        # No hit found
        return tex_coords[:2], 0.0


class BumpMappingSystem:
    """Manages bump mapping for a rendering system."""

    def __init__(self):
        self.bump_maps: Dict[str, BumpMap] = {}
        self.parallax_mappers: Dict[str, ParallaxMapping] = {}
        self.active_bump_map: Optional[BumpMap] = None
        self.active_parallax_mapper: Optional[ParallaxMapping] = None

    def create_bump_map(self, name: str, width: int, height: int, 
                       config: BumpMapConfig) -> BumpMap:
        """Create a bump map."""
        bump_map = BumpMap(width, height, config)
        self.bump_maps[name] = bump_map
        
        # Create parallax mapper
        parallax_mapper = ParallaxMapping(bump_map, config)
        self.parallax_mappers[name] = parallax_mapper
        
        return bump_map

    def bind_bump_map(self, name: str, texture_unit: int = 0):
        """Bind a bump map for rendering."""
        if name in self.bump_maps:
            bump_map = self.bump_maps[name]
            bump_map.bind(texture_unit)
            self.active_bump_map = bump_map
            self.active_parallax_mapper = self.parallax_mappers[name]

    def calculate_bump_lighting(self, position: np.ndarray, normal: np.ndarray,
                              light_direction: np.ndarray, tex_coords: np.ndarray) -> float:
        """Calculate lighting with bump mapping."""
        if not self.active_bump_map:
            return max(np.dot(normal, light_direction), 0.0)
        
        # Calculate perturbed normal
        perturbed_normal = self.active_bump_map.calculate_normal_offset(
            tex_coords[0], tex_coords[1], normal)
        
        # Calculate lighting with perturbed normal
        return max(np.dot(perturbed_normal, light_direction), 0.0)

    def calculate_parallax_lighting(self, position: np.ndarray, normal: np.ndarray,
                                  light_direction: np.ndarray, tex_coords: np.ndarray,
                                  view_direction: np.ndarray, tbn_matrix: np.ndarray) -> float:
        """Calculate lighting with parallax mapping."""
        if not self.active_parallax_mapper:
            return self.calculate_bump_lighting(position, normal, light_direction, tex_coords)
        
        # Calculate parallax offset
        offset_tex_coords, occlusion = self.active_parallax_mapper.calculate_parallax_occlusion_mapping(
            tex_coords, view_direction, tbn_matrix)
        
        # Calculate lighting with offset texture coordinates
        lighting = self.calculate_bump_lighting(position, normal, light_direction, offset_tex_coords)
        
        # Apply occlusion
        return lighting * occlusion

    def cleanup(self):
        """Clean up all bump maps."""
        for bump_map in self.bump_maps.values():
            bump_map.cleanup()
        self.bump_maps.clear()
        self.parallax_mappers.clear()


def demonstrate_bump_mapping():
    """Demonstrate bump mapping techniques."""
    print("=== Normal Mapping, Bump Mapping, and PBR - Bump Mapping ===\n")

    system = BumpMappingSystem()
    config = BumpMapConfig(resolution=512, height_scale=0.1, parallax_scale=0.05)

    print("1. Creating bump maps...")
    
    noise_bump = system.create_bump_map("noise", 512, 512, config)
    noise_bump.generate_procedural("noise")
    print("   Created noise bump map")

    ridges_bump = system.create_bump_map("ridges", 512, 512, config)
    ridges_bump.generate_procedural("ridges")
    print("   Created ridges bump map")

    craters_bump = system.create_bump_map("craters", 512, 512, config)
    craters_bump.generate_procedural("craters")
    print("   Created craters bump map")

    print("\n2. Bump mapping lighting example...")
    
    system.bind_bump_map("noise")
    
    position = np.array([0.5, 0.5, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    light_direction = np.array([1.0, 1.0, 1.0])
    tex_coords = np.array([0.5, 0.5])
    
    lighting = system.calculate_bump_lighting(position, normal, light_direction, tex_coords)
    print(f"   Bump mapping lighting result: {lighting:.3f}")

    print("\n3. Parallax mapping example...")
    
    view_direction = np.array([0.0, 0.0, -1.0])
    tbn_matrix = np.eye(3)  # Identity matrix for simplicity
    
    parallax_lighting = system.calculate_parallax_lighting(
        position, normal, light_direction, tex_coords, view_direction, tbn_matrix)
    print(f"   Parallax mapping lighting result: {parallax_lighting:.3f}")

    print("\n4. Height sampling example...")
    
    # Sample heights from different positions
    sample_positions = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    
    for u, v in sample_positions:
        height = noise_bump.sample_height(u, v)
        print(f"   Height at ({u}, {v}): {height:.3f}")

    print("\n5. Features demonstrated:")
    print("   - Procedural bump map generation")
    print("   - Height map sampling")
    print("   - Normal perturbation")
    print("   - Parallax mapping")
    print("   - Parallax occlusion mapping")

    system.cleanup()
    print("\n6. Bump mapping system cleaned up successfully")


if __name__ == "__main__":
    demonstrate_bump_mapping()
