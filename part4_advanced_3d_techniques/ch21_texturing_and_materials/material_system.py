"""
Chapter 21: Texturing and Materials - Material System
===================================================

This module demonstrates material systems and properties for 3D graphics.

Key Concepts:
- Material properties and parameters
- Texture mapping and UV coordinates
- Material shader integration
- Material management and optimization
"""

import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from vector_operations import Vector3D


class MaterialType(Enum):
    """Material type enumeration."""
    LAMBERT = "lambert"
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    PBR = "pbr"
    EMISSIVE = "emissive"
    TRANSPARENT = "transparent"


class TextureType(Enum):
    """Texture type enumeration."""
    DIFFUSE = "diffuse"
    NORMAL = "normal"
    SPECULAR = "specular"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    EMISSIVE = "emissive"
    AO = "ao"
    HEIGHT = "height"


@dataclass
class MaterialProperties:
    """Base material properties."""
    # Basic properties
    ambient: Vector3D = field(default_factory=lambda: Vector3D(0.1, 0.1, 0.1))
    diffuse: Vector3D = field(default_factory=lambda: Vector3D(0.7, 0.7, 0.7))
    specular: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    emission: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    
    # Surface properties
    shininess: float = 32.0
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    
    # PBR properties
    albedo: Vector3D = field(default_factory=lambda: Vector3D(0.7, 0.7, 0.7))
    ao: float = 1.0
    
    # Advanced properties
    fresnel: float = 0.04
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.1


@dataclass
class TextureMapping:
    """Texture mapping information."""
    texture_name: str
    texture_type: TextureType
    uv_scale: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    uv_offset: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    uv_rotation: float = 0.0
    enabled: bool = True


class Material:
    """Represents a material with properties and textures."""
    
    def __init__(self, name: str, material_type: MaterialType = MaterialType.PHONG):
        self.name = name
        self.material_type = material_type
        self.properties = MaterialProperties()
        self.textures: Dict[TextureType, TextureMapping] = {}
        self.shader_program = None
        self.enabled = True
        
    def set_property(self, property_name: str, value):
        """Set a material property."""
        if hasattr(self.properties, property_name):
            setattr(self.properties, property_name, value)
        else:
            raise ValueError(f"Unknown property: {property_name}")
            
    def get_property(self, property_name: str):
        """Get a material property."""
        if hasattr(self.properties, property_name):
            return getattr(self.properties, property_name)
        else:
            raise ValueError(f"Unknown property: {property_name}")
            
    def add_texture(self, texture_type: TextureType, texture_name: str, 
                    uv_scale: Vector3D = None, uv_offset: Vector3D = None):
        """Add a texture to the material."""
        mapping = TextureMapping(
            texture_name=texture_name,
            texture_type=texture_type,
            uv_scale=uv_scale or Vector3D(1.0, 1.0, 1.0),
            uv_offset=uv_offset or Vector3D(0.0, 0.0, 0.0)
        )
        self.textures[texture_type] = mapping
        
    def remove_texture(self, texture_type: TextureType):
        """Remove a texture from the material."""
        if texture_type in self.textures:
            del self.textures[texture_type]
            
    def has_texture(self, texture_type: TextureType) -> bool:
        """Check if material has a specific texture."""
        return texture_type in self.textures and self.textures[texture_type].enabled
        
    def get_texture(self, texture_type: TextureType) -> Optional[TextureMapping]:
        """Get texture mapping for a specific type."""
        return self.textures.get(texture_type)
        
    def set_shader_program(self, shader_program):
        """Set the shader program for this material."""
        self.shader_program = shader_program
        
    def bind_textures(self, texture_manager):
        """Bind all textures for this material."""
        for texture_type, mapping in self.textures.items():
            if mapping.enabled:
                unit = self._get_texture_unit(texture_type)
                texture_manager.bind_texture(mapping.texture_name, unit)
                
    def _get_texture_unit(self, texture_type: TextureType) -> int:
        """Get OpenGL texture unit for texture type."""
        texture_units = {
            TextureType.DIFFUSE: 0,
            TextureType.NORMAL: 1,
            TextureType.SPECULAR: 2,
            TextureType.ROUGHNESS: 3,
            TextureType.METALLIC: 4,
            TextureType.EMISSIVE: 5,
            TextureType.AO: 6,
            TextureType.HEIGHT: 7
        }
        return texture_units.get(texture_type, 0)


class MaterialPreset:
    """Predefined material presets."""
    
    @staticmethod
    def create_metal(name: str = "metal", color: Vector3D = None) -> Material:
        """Create a metallic material."""
        material = Material(name, MaterialType.PBR)
        material.properties.albedo = color or Vector3D(0.8, 0.8, 0.8)
        material.properties.metallic = 1.0
        material.properties.roughness = 0.2
        material.properties.ao = 0.8
        return material
        
    @staticmethod
    def create_plastic(name: str = "plastic", color: Vector3D = None) -> Material:
        """Create a plastic material."""
        material = Material(name, MaterialType.PBR)
        material.properties.albedo = color or Vector3D(0.2, 0.2, 0.2)
        material.properties.metallic = 0.0
        material.properties.roughness = 0.3
        material.properties.ao = 0.9
        return material
        
    @staticmethod
    def create_wood(name: str = "wood") -> Material:
        """Create a wood material."""
        material = Material(name, MaterialType.PBR)
        material.properties.albedo = Vector3D(0.6, 0.4, 0.2)
        material.properties.metallic = 0.0
        material.properties.roughness = 0.8
        material.properties.ao = 0.9
        return material
        
    @staticmethod
    def create_glass(name: str = "glass") -> Material:
        """Create a glass material."""
        material = Material(name, MaterialType.TRANSPARENT)
        material.properties.albedo = Vector3D(0.9, 0.9, 0.9)
        material.properties.metallic = 0.0
        material.properties.roughness = 0.0
        material.properties.opacity = 0.3
        material.properties.fresnel = 0.04
        return material
        
    @staticmethod
    def create_emissive(name: str = "emissive", color: Vector3D = None) -> Material:
        """Create an emissive material."""
        material = Material(name, MaterialType.EMISSIVE)
        material.properties.emission = color or Vector3D(1.0, 1.0, 1.0)
        material.properties.albedo = Vector3D(0.1, 0.1, 0.1)
        material.properties.metallic = 0.0
        material.properties.roughness = 0.5
        return material


class MaterialManager:
    """Manages materials and their resources."""
    
    def __init__(self):
        self.materials: Dict[str, Material] = {}
        self.active_material: Optional[Material] = None
        self.material_counter = 0
        
    def create_material(self, name: str = None, material_type: MaterialType = MaterialType.PHONG) -> Material:
        """Create a new material."""
        if name is None:
            name = f"material_{self.material_counter}"
            self.material_counter += 1
            
        material = Material(name, material_type)
        self.materials[name] = material
        return material
        
    def get_material(self, name: str) -> Optional[Material]:
        """Get material by name."""
        return self.materials.get(name)
        
    def remove_material(self, name: str):
        """Remove material by name."""
        if name in self.materials:
            del self.materials[name]
            
    def set_active_material(self, name: str) -> bool:
        """Set active material by name."""
        material = self.get_material(name)
        if material:
            self.active_material = material
            return True
        return False
        
    def get_active_material(self) -> Optional[Material]:
        """Get currently active material."""
        return self.active_material
        
    def create_preset_material(self, preset_type: str, name: str = None, **kwargs) -> Optional[Material]:
        """Create material from preset."""
        preset_methods = {
            "metal": MaterialPreset.create_metal,
            "plastic": MaterialPreset.create_plastic,
            "wood": MaterialPreset.create_wood,
            "glass": MaterialPreset.create_glass,
            "emissive": MaterialPreset.create_emissive
        }
        
        if preset_type in preset_methods:
            material = preset_methods[preset_type](name, **kwargs)
            self.materials[material.name] = material
            return material
        else:
            print(f"Unknown preset type: {preset_type}")
            return None
            
    def bind_material(self, material_name: str, texture_manager) -> bool:
        """Bind material and its textures."""
        material = self.get_material(material_name)
        if material:
            material.bind_textures(texture_manager)
            self.active_material = material
            return True
        return False
        
    def get_material_list(self) -> List[str]:
        """Get list of all material names."""
        return list(self.materials.keys())
        
    def cleanup(self):
        """Clean up all materials."""
        self.materials.clear()
        self.active_material = None


class UVMapper:
    """Handles UV coordinate mapping and transformations."""
    
    @staticmethod
    def generate_planar_uv(vertices: List[Vector3D], scale: Vector3D = None) -> List[Vector3D]:
        """Generate planar UV coordinates."""
        if scale is None:
            scale = Vector3D(1.0, 1.0, 1.0)
            
        uvs = []
        for vertex in vertices:
            u = (vertex.x * scale.x + 0.5) % 1.0
            v = (vertex.y * scale.y + 0.5) % 1.0
            uvs.append(Vector3D(u, v, 0.0))
            
        return uvs
        
    @staticmethod
    def generate_spherical_uv(vertices: List[Vector3D], center: Vector3D = None) -> List[Vector3D]:
        """Generate spherical UV coordinates."""
        if center is None:
            center = Vector3D(0.0, 0.0, 0.0)
            
        uvs = []
        for vertex in vertices:
            # Calculate spherical coordinates
            direction = (vertex - center).normalized()
            
            # Convert to UV coordinates
            u = 0.5 + math.atan2(direction.z, direction.x) / (2 * math.pi)
            v = 0.5 + math.asin(direction.y) / math.pi
            
            uvs.append(Vector3D(u, v, 0.0))
            
        return uvs
        
    @staticmethod
    def transform_uv(uvs: List[Vector3D], scale: Vector3D, offset: Vector3D, rotation: float) -> List[Vector3D]:
        """Transform UV coordinates."""
        transformed = []
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        
        for uv in uvs:
            # Apply rotation
            x = uv.x * cos_r - uv.y * sin_r
            y = uv.x * sin_r + uv.y * cos_r
            
            # Apply scale and offset
            x = x * scale.x + offset.x
            y = y * scale.y + offset.y
            
            transformed.append(Vector3D(x, y, uv.z))
            
        return transformed


def demonstrate_material_system():
    """Demonstrate material system functionality."""
    print("=== Material System Demonstration ===\n")
    
    # Create material manager
    manager = MaterialManager()
    
    # Create preset materials
    print("1. Creating preset materials:")
    
    metal = manager.create_preset_material("metal", "steel", color=Vector3D(0.7, 0.7, 0.8))
    print(f"   - Created {metal.name} material (metallic: {metal.properties.metallic})")
    
    plastic = manager.create_preset_material("plastic", "red_plastic", color=Vector3D(0.8, 0.2, 0.2))
    print(f"   - Created {plastic.name} material (roughness: {plastic.properties.roughness})")
    
    wood = manager.create_preset_material("wood", "oak")
    print(f"   - Created {wood.name} material (albedo: {wood.properties.albedo})")
    
    glass = manager.create_preset_material("glass", "crystal")
    print(f"   - Created {glass.name} material (opacity: {glass.properties.opacity})")
    
    # Create custom material
    print("\n2. Creating custom material:")
    
    custom = manager.create_material("custom_gold", MaterialType.PBR)
    custom.properties.albedo = Vector3D(1.0, 0.8, 0.0)
    custom.properties.metallic = 1.0
    custom.properties.roughness = 0.1
    custom.properties.ao = 0.7
    print(f"   - Created {custom.name} material")
    print(f"     Albedo: {custom.properties.albedo}")
    print(f"     Metallic: {custom.properties.metallic}")
    print(f"     Roughness: {custom.properties.roughness}")
    
    # Add textures to material
    print("\n3. Adding textures to material:")
    
    custom.add_texture(TextureType.DIFFUSE, "gold_diffuse")
    custom.add_texture(TextureType.NORMAL, "gold_normal")
    custom.add_texture(TextureType.ROUGHNESS, "gold_roughness", 
                      uv_scale=Vector3D(2.0, 2.0, 1.0))
    
    print(f"   - Added textures to {custom.name}:")
    for texture_type, mapping in custom.textures.items():
        print(f"     {texture_type.value}: {mapping.texture_name}")
        
    # Test material operations
    print("\n4. Testing material operations:")
    
    # Set active material
    manager.set_active_material("custom_gold")
    active = manager.get_active_material()
    print(f"   - Active material: {active.name}")
    
    # Modify material properties
    custom.set_property("roughness", 0.05)
    new_roughness = custom.get_property("roughness")
    print(f"   - Updated roughness to: {new_roughness}")
    
    # Test UV mapping
    print("\n5. Testing UV mapping:")
    
    # Sample vertices
    vertices = [
        Vector3D(-1, -1, 0),
        Vector3D(1, -1, 0),
        Vector3D(1, 1, 0),
        Vector3D(-1, 1, 0)
    ]
    
    # Generate planar UVs
    planar_uvs = UVMapper.generate_planar_uv(vertices, Vector3D(1.0, 1.0, 1.0))
    print(f"   - Generated planar UVs for {len(vertices)} vertices")
    
    # Transform UVs
    transformed_uvs = UVMapper.transform_uv(
        planar_uvs, 
        Vector3D(2.0, 2.0, 1.0), 
        Vector3D(0.1, 0.1, 0.0), 
        math.pi / 4
    )
    print(f"   - Transformed UVs with scale, offset, and rotation")
    
    # List all materials
    print("\n6. Available materials:")
    material_list = manager.get_material_list()
    for material_name in material_list:
        material = manager.get_material(material_name)
        print(f"   - {material_name} ({material.material_type.value})")
        
    # Cleanup
    print("\n7. Cleaning up materials...")
    manager.cleanup()
    print("   - All materials cleaned up")


if __name__ == "__main__":
    demonstrate_material_system()
