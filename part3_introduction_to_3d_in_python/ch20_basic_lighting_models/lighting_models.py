"""
Chapter 20: Basic Lighting Models - Lighting Models
==================================================

This module demonstrates basic lighting models and calculations for 3D graphics.

Key Concepts:
- Ambient, diffuse, and specular lighting
- Light types (directional, point, spot)
- Material properties and reflectance
- Lighting calculations and shading
"""

import math
from typing import List, Optional, Dict
from dataclasses import dataclass
from vector_operations import Vector3D


@dataclass
class Material:
    """Material properties for lighting calculations."""
    ambient: Vector3D = None
    diffuse: Vector3D = None
    specular: Vector3D = None
    shininess: float = 32.0
    emission: Vector3D = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.ambient is None:
            self.ambient = Vector3D(0.2, 0.2, 0.2)
        if self.diffuse is None:
            self.diffuse = Vector3D(0.8, 0.8, 0.8)
        if self.specular is None:
            self.specular = Vector3D(1.0, 1.0, 1.0)
        if self.emission is None:
            self.emission = Vector3D(0.0, 0.0, 0.0)


class Light:
    """Base class for different types of lights."""
    
    def __init__(self, position: Vector3D, color: Vector3D, intensity: float = 1.0):
        self.position = position
        self.color = color
        self.intensity = intensity
        self.enabled = True
    
    def get_light_vector(self, point: Vector3D) -> Vector3D:
        """Get light direction vector from light to point."""
        return (self.position - point).normalized()
    
    def get_attenuation(self, distance: float) -> float:
        """Get light attenuation based on distance."""
        return 1.0


class DirectionalLight(Light):
    """Directional light (like the sun)."""
    
    def __init__(self, direction: Vector3D, color: Vector3D, intensity: float = 1.0):
        super().__init__(direction, color, intensity)
    
    def get_light_vector(self, point: Vector3D) -> Vector3D:
        """Get light direction (constant for directional light)."""
        return self.position.normalized()


class PointLight(Light):
    """Point light (like a light bulb)."""
    
    def __init__(self, position: Vector3D, color: Vector3D, intensity: float = 1.0,
                 constant: float = 1.0, linear: float = 0.09, quadratic: float = 0.032):
        super().__init__(position, color, intensity)
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
    
    def get_attenuation(self, distance: float) -> float:
        """Calculate attenuation based on distance."""
        attenuation = self.constant + self.linear * distance + self.quadratic * distance * distance
        return 1.0 / max(attenuation, 1.0)


class SpotLight(Light):
    """Spot light (like a flashlight)."""
    
    def __init__(self, position: Vector3D, direction: Vector3D, color: Vector3D,
                 intensity: float = 1.0, cutoff_angle: float = math.pi / 6):
        super().__init__(position, color, intensity)
        self.direction = direction.normalized()
        self.cutoff_angle = cutoff_angle
        self.cutoff_cos = math.cos(cutoff_angle)
    
    def get_attenuation(self, distance: float) -> float:
        """Calculate spot light attenuation."""
        attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance)
        light_dir = self.get_light_vector(Vector3D(0, 0, 0))
        cos_theta = light_dir.dot(self.direction)
        intensity = 1.0 if cos_theta > self.cutoff_cos else 0.0
        return attenuation * intensity


class LightingCalculator:
    """Calculates lighting for surfaces."""
    
    def __init__(self):
        self.ambient_light = Vector3D(0.1, 0.1, 0.1)
        self.lights: List[Light] = []
    
    def add_light(self, light: Light):
        """Add a light to the scene."""
        self.lights.append(light)
    
    def calculate_lighting(self, point: Vector3D, normal: Vector3D, material: Material,
                          view_direction: Vector3D) -> Vector3D:
        """Calculate total lighting at a point."""
        result = material.emission + self.ambient_light * material.ambient
        
        for light in self.lights:
            if not light.enabled:
                continue
            
            light_direction = light.get_light_vector(point)
            distance = (light.position - point).magnitude()
            attenuation = light.get_attenuation(distance)
            
            # Diffuse lighting
            diffuse_strength = max(0.0, normal.dot(light_direction))
            diffuse = light.color * material.diffuse * diffuse_strength
            
            # Specular lighting
            reflect_direction = (2.0 * normal.dot(light_direction) * normal - light_direction).normalized()
            specular_strength = max(0.0, view_direction.dot(reflect_direction))
            specular = light.color * material.specular * (specular_strength ** material.shininess)
            
            result = result + (diffuse + specular) * attenuation * light.intensity
        
        return Vector3D(
            max(0.0, min(1.0, result.x)),
            max(0.0, min(1.0, result.y)),
            max(0.0, min(1.0, result.z))
        )


def demonstrate_lighting_models():
    """Demonstrate lighting models and calculations."""
    print("=== Basic Lighting Models Demonstration ===\n")
    
    # Create materials
    red_material = Material(
        ambient=Vector3D(0.2, 0.0, 0.0),
        diffuse=Vector3D(0.8, 0.0, 0.0),
        specular=Vector3D(1.0, 1.0, 1.0)
    )
    
    # Create lights
    directional_light = DirectionalLight(
        direction=Vector3D(1, 1, 1),
        color=Vector3D(1, 1, 1),
        intensity=0.8
    )
    
    point_light = PointLight(
        position=Vector3D(5, 5, 5),
        color=Vector3D(1, 0.8, 0.6),
        intensity=1.0
    )
    
    # Create calculator
    calculator = LightingCalculator()
    calculator.add_light(directional_light)
    calculator.add_light(point_light)
    
    # Test lighting
    point = Vector3D(0, 0, 0)
    normal = Vector3D(0, 1, 0)
    view_direction = Vector3D(0, 0, -1).normalized()
    
    lighting = calculator.calculate_lighting(point, normal, red_material, view_direction)
    print(f"Lighting at origin: {lighting}")
    
    # Test attenuation
    for distance in [0, 1, 2, 5]:
        att = point_light.get_attenuation(distance)
        print(f"Distance {distance}: attenuation = {att:.3f}")


if __name__ == "__main__":
    demonstrate_lighting_models()
