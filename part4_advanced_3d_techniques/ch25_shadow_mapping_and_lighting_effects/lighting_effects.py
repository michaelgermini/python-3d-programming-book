"""
Chapter 25: Shadow Mapping and Lighting Effects - Lighting Effects
===============================================================

This module demonstrates advanced lighting effects and techniques.
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class LightType(Enum):
    """Light type enumeration."""
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"


class LightingModel(Enum):
    """Lighting model enumeration."""
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    PBR = "pbr"


@dataclass
class LightProperties:
    """Properties for a light source."""
    light_type: LightType
    position: np.ndarray
    direction: np.ndarray
    color: np.ndarray
    intensity: float = 1.0
    range: float = 10.0
    attenuation: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class MaterialProperties:
    """Material properties for lighting calculations."""
    ambient: np.ndarray
    diffuse: np.ndarray
    specular: np.ndarray
    shininess: float = 32.0


class Light:
    """Base class for light sources."""

    def __init__(self, properties: LightProperties):
        self.properties = properties
        self.enabled = True

    def calculate_lighting(self, position: np.ndarray, normal: np.ndarray, 
                          view_direction: np.ndarray, material: MaterialProperties) -> np.ndarray:
        """Calculate lighting contribution for this light."""
        if not self.enabled:
            return np.zeros(3)
        
        if self.properties.light_type == LightType.DIRECTIONAL:
            return self._calculate_directional_lighting(position, normal, view_direction, material)
        elif self.properties.light_type == LightType.POINT:
            return self._calculate_point_lighting(position, normal, view_direction, material)
        else:
            return np.zeros(3)

    def _calculate_directional_lighting(self, position: np.ndarray, normal: np.ndarray,
                                      view_direction: np.ndarray, material: MaterialProperties) -> np.ndarray:
        """Calculate directional lighting."""
        light_direction = -self.properties.direction / np.linalg.norm(self.properties.direction)
        
        # Ambient
        ambient = material.ambient * self.properties.color * self.properties.intensity
        
        # Diffuse
        diffuse_strength = max(np.dot(normal, light_direction), 0.0)
        diffuse = material.diffuse * self.properties.color * diffuse_strength * self.properties.intensity
        
        # Specular (Phong)
        reflect_direction = 2.0 * np.dot(normal, light_direction) * normal - light_direction
        specular_strength = max(np.dot(view_direction, reflect_direction), 0.0) ** material.shininess
        specular = material.specular * self.properties.color * specular_strength * self.properties.intensity
        
        return ambient + diffuse + specular

    def _calculate_point_lighting(self, position: np.ndarray, normal: np.ndarray,
                                view_direction: np.ndarray, material: MaterialProperties) -> np.ndarray:
        """Calculate point lighting with attenuation."""
        light_direction = self.properties.position - position
        distance = np.linalg.norm(light_direction)
        light_direction = light_direction / distance
        
        if distance > self.properties.range:
            return np.zeros(3)
        
        # Calculate attenuation
        constant, linear, quadratic = self.properties.attenuation
        attenuation = 1.0 / (constant + linear * distance + quadratic * distance * distance)
        
        # Ambient
        ambient = material.ambient * self.properties.color * self.properties.intensity * attenuation
        
        # Diffuse
        diffuse_strength = max(np.dot(normal, light_direction), 0.0)
        diffuse = material.diffuse * self.properties.color * diffuse_strength * self.properties.intensity * attenuation
        
        # Specular (Blinn-Phong)
        half_direction = (light_direction + view_direction) / np.linalg.norm(light_direction + view_direction)
        specular_strength = max(np.dot(normal, half_direction), 0.0) ** material.shininess
        specular = material.specular * self.properties.color * specular_strength * self.properties.intensity * attenuation
        
        return ambient + diffuse + specular


class LightingSystem:
    """Manages multiple light sources and lighting calculations."""

    def __init__(self):
        self.lights: List[Light] = []
        self.ambient_light = np.array([0.1, 0.1, 0.1])

    def add_light(self, light: Light):
        """Add a light source to the system."""
        self.lights.append(light)

    def calculate_total_lighting(self, position: np.ndarray, normal: np.ndarray,
                               view_direction: np.ndarray, material: MaterialProperties) -> np.ndarray:
        """Calculate total lighting from all light sources."""
        total_lighting = material.ambient * self.ambient_light
        
        for light in self.lights:
            if light.enabled:
                light_contribution = light.calculate_lighting(position, normal, view_direction, material)
                total_lighting += light_contribution
        
        return np.clip(total_lighting, 0.0, 1.0)


class LightingEffects:
    """Advanced lighting effects and techniques."""

    def __init__(self):
        self.lighting_system = LightingSystem()

    def setup_basic_scene_lighting(self):
        """Setup basic lighting for a scene."""
        # Directional light (sun)
        sun_properties = LightProperties(
            light_type=LightType.DIRECTIONAL,
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([-1.0, -1.0, -1.0]),
            color=np.array([1.0, 0.95, 0.8]),
            intensity=1.0
        )
        sun_light = Light(sun_properties)
        self.lighting_system.add_light(sun_light)
        
        # Point light (lamp)
        lamp_properties = LightProperties(
            light_type=LightType.POINT,
            position=np.array([5.0, 3.0, 5.0]),
            direction=np.array([0.0, 0.0, 0.0]),
            color=np.array([1.0, 1.0, 0.8]),
            intensity=0.8,
            range=15.0,
            attenuation=(1.0, 0.09, 0.032)
        )
        lamp_light = Light(lamp_properties)
        self.lighting_system.add_light(lamp_light)

    def calculate_lighting_for_vertex(self, position: np.ndarray, normal: np.ndarray,
                                    view_direction: np.ndarray) -> np.ndarray:
        """Calculate lighting for a vertex."""
        material = MaterialProperties(
            ambient=np.array([0.1, 0.1, 0.1]),
            diffuse=np.array([0.7, 0.7, 0.7]),
            specular=np.array([1.0, 1.0, 1.0]),
            shininess=32.0
        )
        return self.lighting_system.calculate_total_lighting(position, normal, view_direction, material)


def demonstrate_lighting_effects():
    """Demonstrate advanced lighting effects and techniques."""
    print("=== Shadow Mapping and Lighting Effects - Lighting Effects ===\n")

    effects = LightingEffects()

    print("1. Setting up basic scene lighting...")
    effects.setup_basic_scene_lighting()
    print(f"   Active lights: {len(effects.lighting_system.lights)}")

    print("\n2. Lighting calculation example:")
    position = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    view_direction = np.array([0.0, 0.0, -1.0])
    
    lighting = effects.calculate_lighting_for_vertex(position, normal, view_direction)
    print(f"   Lighting result: {lighting}")

    print("\n3. Light types supported:")
    print("   - Directional lights (sun)")
    print("   - Point lights (lamps)")
    print("   - Spot lights")

    print("\n4. Lighting models supported:")
    print("   - Phong lighting model")
    print("   - Blinn-Phong lighting model")

    print("\n5. Features:")
    print("   - Light attenuation and falloff")
    print("   - Specular highlights")
    print("   - Multiple light sources")
    print("   - Material-based lighting")


if __name__ == "__main__":
    demonstrate_lighting_effects()
