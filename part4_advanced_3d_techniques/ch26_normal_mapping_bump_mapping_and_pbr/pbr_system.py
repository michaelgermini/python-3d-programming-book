"""
Chapter 26: Normal Mapping, Bump Mapping, and PBR - PBR System
===========================================================

This module demonstrates Physically Based Rendering (PBR) techniques.

Key Concepts:
- PBR material properties and workflows
- BRDF (Bidirectional Reflectance Distribution Function) models
- Energy conservation and physically accurate lighting
- Metallic-Roughness workflow implementation
- IBL (Image-Based Lighting) integration
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class PBRWorkflow(Enum):
    """PBR workflow enumeration."""
    METALLIC_ROUGHNESS = "metallic_roughness"
    SPECULAR_GLOSSINESS = "specular_glossiness"


class BRDFType(Enum):
    """BRDF type enumeration."""
    COOK_TORRANCE = "cook_torrance"
    GGX = "ggx"
    BLINN_PHONG = "blinn_phong"


@dataclass
class PBRMaterial:
    """PBR material properties."""
    albedo: np.ndarray = None  # Base color
    metallic: float = 0.0      # Metallic factor (0-1)
    roughness: float = 0.5     # Roughness factor (0-1)
    ao: float = 1.0           # Ambient occlusion
    emissive: np.ndarray = None  # Emissive color
    
    # Normal and height maps
    normal_scale: float = 1.0
    height_scale: float = 0.1
    
    # IBL properties
    irradiance_map: Optional[int] = None
    prefilter_map: Optional[int] = None
    brdf_lut: Optional[int] = None
    
    def __post_init__(self):
        if self.albedo is None:
            self.albedo = np.array([0.7, 0.7, 0.7])
        if self.emissive is None:
            self.emissive = np.array([0.0, 0.0, 0.0])


class BRDFCalculator:
    """Calculates BRDF functions for PBR lighting."""

    def __init__(self, brdf_type: BRDFType = BRDFType.GGX):
        self.brdf_type = brdf_type

    def calculate_ndf(self, normal: np.ndarray, half_vector: np.ndarray, roughness: float) -> float:
        """Calculate Normal Distribution Function (NDF)."""
        if self.brdf_type == BRDFType.GGX:
            return self._ggx_ndf(normal, half_vector, roughness)
        elif self.brdf_type == BRDFType.COOK_TORRANCE:
            return self._cook_torrance_ndf(normal, half_vector, roughness)
        else:
            return 1.0

    def _ggx_ndf(self, normal: np.ndarray, half_vector: np.ndarray, roughness: float) -> float:
        """GGX/Trowbridge-Reitz NDF."""
        alpha = roughness * roughness
        ndoth = np.dot(normal, half_vector)
        ndoth = max(ndoth, 0.0)
        
        if ndoth <= 0.0:
            return 0.0
        
        alpha_sq = alpha * alpha
        denom = ndoth * ndoth * (alpha_sq - 1.0) + 1.0
        denom = math.pi * denom * denom
        
        return alpha_sq / denom

    def _cook_torrance_ndf(self, normal: np.ndarray, half_vector: np.ndarray, roughness: float) -> float:
        """Cook-Torrance NDF."""
        alpha = roughness * roughness
        ndoth = np.dot(normal, half_vector)
        ndoth = max(ndoth, 0.0)
        
        if ndoth <= 0.0:
            return 0.0
        
        exp_term = -(ndoth * ndoth - 1.0) / (alpha * alpha * ndoth * ndoth)
        return math.exp(exp_term) / (alpha * alpha * ndoth * ndoth * ndoth * ndoth)

    def calculate_geometry_function(self, normal: np.ndarray, view_dir: np.ndarray, 
                                  light_dir: np.ndarray, half_vector: np.ndarray, 
                                  roughness: float) -> float:
        """Calculate Geometry Function (G)."""
        if self.brdf_type == BRDFType.GGX:
            return self._ggx_geometry(normal, view_dir, light_dir, half_vector, roughness)
        else:
            return 1.0

    def _ggx_geometry(self, normal: np.ndarray, view_dir: np.ndarray, 
                     light_dir: np.ndarray, half_vector: np.ndarray, 
                     roughness: float) -> float:
        """GGX Geometry Function."""
        alpha = roughness * roughness
        
        # Smith's method
        g1_v = self._smith_g1(normal, view_dir, alpha)
        g1_l = self._smith_g1(normal, light_dir, alpha)
        
        return g1_v * g1_l

    def _smith_g1(self, normal: np.ndarray, direction: np.ndarray, alpha: float) -> float:
        """Smith's G1 function."""
        ndotd = np.dot(normal, direction)
        ndotd = max(ndotd, 0.0)
        
        if ndotd <= 0.0:
            return 0.0
        
        k = alpha / 2.0
        denom = ndotd * (1.0 - k) + k
        
        return ndotd / denom

    def calculate_fresnel(self, view_dir: np.ndarray, half_vector: np.ndarray, 
                         f0: np.ndarray) -> np.ndarray:
        """Calculate Fresnel Function (F)."""
        vdoth = np.dot(view_dir, half_vector)
        vdoth = max(vdoth, 0.0)
        
        # Schlick's approximation
        return f0 + (1.0 - f0) * math.pow(1.0 - vdoth, 5.0)

    def calculate_brdf(self, normal: np.ndarray, view_dir: np.ndarray, 
                      light_dir: np.ndarray, material: PBRMaterial) -> np.ndarray:
        """Calculate complete BRDF."""
        half_vector = (view_dir + light_dir) / np.linalg.norm(view_dir + light_dir)
        
        # Calculate F0 (base reflectivity)
        f0 = np.array([0.04, 0.04, 0.04])  # Default for dielectrics
        if material.metallic > 0.0:
            f0 = np.lerp(f0, material.albedo, material.metallic)
        
        # Calculate BRDF components
        ndf = self.calculate_ndf(normal, half_vector, material.roughness)
        g = self.calculate_geometry_function(normal, view_dir, light_dir, half_vector, material.roughness)
        f = self.calculate_fresnel(view_dir, half_vector, f0)
        
        # Calculate denominator
        ndotv = max(np.dot(normal, view_dir), 0.0)
        ndotl = max(np.dot(normal, light_dir), 0.0)
        denominator = 4.0 * ndotv * ndotl + 0.0001  # Avoid division by zero
        
        # Calculate specular BRDF
        specular = (ndf * g * f) / denominator
        
        # Calculate diffuse BRDF (energy conservation)
        kd = (1.0 - f) * (1.0 - material.metallic)
        diffuse = kd * material.albedo / math.pi
        
        return diffuse + specular


class PBRRenderer:
    """PBR rendering system."""

    def __init__(self, brdf_type: BRDFType = BRDFType.GGX):
        self.brdf_calculator = BRDFCalculator(brdf_type)
        self.materials: Dict[str, PBRMaterial] = {}
        self.active_material: Optional[PBRMaterial] = None

    def create_material(self, name: str, material: PBRMaterial) -> PBRMaterial:
        """Create a PBR material."""
        self.materials[name] = material
        return material

    def bind_material(self, name: str):
        """Bind a material for rendering."""
        if name in self.materials:
            self.active_material = self.materials[name]

    def calculate_pbr_lighting(self, position: np.ndarray, normal: np.ndarray,
                             view_dir: np.ndarray, light_dir: np.ndarray,
                             light_color: np.ndarray, light_intensity: float) -> np.ndarray:
        """Calculate PBR lighting for a surface point."""
        if not self.active_material:
            # Fallback to basic lighting
            ndotl = max(np.dot(normal, light_dir), 0.0)
            return light_color * light_intensity * ndotl
        
        # Calculate BRDF
        brdf = self.brdf_calculator.calculate_brdf(
            normal, view_dir, light_dir, self.active_material)
        
        # Calculate lighting contribution
        ndotl = max(np.dot(normal, light_dir), 0.0)
        lighting = light_color * light_intensity * ndotl * brdf
        
        # Apply ambient occlusion
        lighting *= self.active_material.ao
        
        return lighting

    def calculate_ibl_lighting(self, position: np.ndarray, normal: np.ndarray,
                             view_dir: np.ndarray) -> np.ndarray:
        """Calculate Image-Based Lighting contribution."""
        if not self.active_material or not self.active_material.irradiance_map:
            # Fallback to simple ambient
            return self.active_material.albedo * 0.1 if self.active_material else np.array([0.1, 0.1, 0.1])
        
        # Calculate ambient lighting from irradiance map
        # This is a simplified version - in practice, you'd sample the irradiance map
        ambient = self.active_material.albedo * 0.3
        
        # Calculate reflection contribution
        reflection = np.array([0.1, 0.1, 0.1])  # Simplified reflection
        
        return ambient + reflection

    def calculate_final_color(self, position: np.ndarray, normal: np.ndarray,
                            view_dir: np.ndarray, light_dir: np.ndarray,
                            light_color: np.ndarray, light_intensity: float) -> np.ndarray:
        """Calculate final PBR color."""
        # Direct lighting
        direct_lighting = self.calculate_pbr_lighting(
            position, normal, view_dir, light_dir, light_color, light_intensity)
        
        # IBL lighting
        ibl_lighting = self.calculate_ibl_lighting(position, normal, view_dir)
        
        # Emissive lighting
        emissive = self.active_material.emissive if self.active_material else np.array([0.0, 0.0, 0.0])
        
        # Combine all lighting
        final_color = direct_lighting + ibl_lighting + emissive
        
        # Apply tone mapping (simple Reinhard)
        final_color = final_color / (final_color + 1.0)
        
        # Apply gamma correction
        final_color = np.power(final_color, 1.0 / 2.2)
        
        return np.clip(final_color, 0.0, 1.0)


class PBRMaterialLibrary:
    """Library of common PBR materials."""

    def __init__(self):
        self.materials: Dict[str, PBRMaterial] = {}
        self.setup_default_materials()

    def setup_default_materials(self):
        """Setup default PBR materials."""
        # Metal materials
        self.materials["gold"] = PBRMaterial(
            albedo=np.array([1.0, 0.8, 0.0]),
            metallic=1.0,
            roughness=0.1
        )
        
        self.materials["silver"] = PBRMaterial(
            albedo=np.array([0.9, 0.9, 0.9]),
            metallic=1.0,
            roughness=0.1
        )
        
        self.materials["copper"] = PBRMaterial(
            albedo=np.array([0.8, 0.5, 0.2]),
            metallic=1.0,
            roughness=0.2
        )
        
        # Plastic materials
        self.materials["plastic_white"] = PBRMaterial(
            albedo=np.array([0.9, 0.9, 0.9]),
            metallic=0.0,
            roughness=0.3
        )
        
        self.materials["plastic_black"] = PBRMaterial(
            albedo=np.array([0.1, 0.1, 0.1]),
            metallic=0.0,
            roughness=0.3
        )
        
        # Wood materials
        self.materials["wood_oak"] = PBRMaterial(
            albedo=np.array([0.6, 0.4, 0.2]),
            metallic=0.0,
            roughness=0.8
        )
        
        self.materials["wood_pine"] = PBRMaterial(
            albedo=np.array([0.8, 0.6, 0.4]),
            metallic=0.0,
            roughness=0.7
        )
        
        # Stone materials
        self.materials["stone_granite"] = PBRMaterial(
            albedo=np.array([0.5, 0.5, 0.5]),
            metallic=0.0,
            roughness=0.9
        )
        
        self.materials["stone_marble"] = PBRMaterial(
            albedo=np.array([0.9, 0.9, 0.9]),
            metallic=0.0,
            roughness=0.1
        )

    def get_material(self, name: str) -> Optional[PBRMaterial]:
        """Get a material by name."""
        return self.materials.get(name)

    def create_custom_material(self, name: str, albedo: np.ndarray, 
                             metallic: float, roughness: float) -> PBRMaterial:
        """Create a custom PBR material."""
        material = PBRMaterial(albedo=albedo, metallic=metallic, roughness=roughness)
        self.materials[name] = material
        return material


def demonstrate_pbr_system():
    """Demonstrate PBR system techniques."""
    print("=== Normal Mapping, Bump Mapping, and PBR - PBR System ===\n")

    # Create PBR renderer
    renderer = PBRRenderer(BRDFType.GGX)
    
    # Create material library
    material_lib = PBRMaterialLibrary()

    print("1. Available PBR materials:")
    for name in material_lib.materials.keys():
        material = material_lib.materials[name]
        print(f"   - {name}: metallic={material.metallic:.1f}, roughness={material.roughness:.1f}")

    print("\n2. PBR lighting calculations...")
    
    # Setup test parameters
    position = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    view_dir = np.array([0.0, 0.0, -1.0])
    light_dir = np.array([1.0, 1.0, 1.0])
    light_color = np.array([1.0, 1.0, 1.0])
    light_intensity = 1.0
    
    # Test different materials
    test_materials = ["gold", "silver", "plastic_white", "wood_oak"]
    
    for material_name in test_materials:
        material = material_lib.get_material(material_name)
        renderer.bind_material(material_name)
        
        final_color = renderer.calculate_final_color(
            position, normal, view_dir, light_dir, light_color, light_intensity)
        
        print(f"   {material_name}: {final_color}")

    print("\n3. BRDF calculations...")
    
    # Test BRDF with different roughness values
    test_roughness = [0.1, 0.5, 0.9]
    
    for roughness in test_roughness:
        material = PBRMaterial(roughness=roughness)
        renderer.active_material = material
        
        brdf = renderer.brdf_calculator.calculate_brdf(
            normal, view_dir, light_dir, material)
        
        print(f"   Roughness {roughness}: BRDF = {brdf}")

    print("\n4. Custom material creation...")
    
    # Create custom materials
    custom_metal = material_lib.create_custom_material(
        "custom_metal", np.array([0.2, 0.8, 0.2]), metallic=1.0, roughness=0.2)
    print(f"   Created custom metal: {custom_metal}")
    
    custom_plastic = material_lib.create_custom_material(
        "custom_plastic", np.array([0.8, 0.2, 0.8]), metallic=0.0, roughness=0.4)
    print(f"   Created custom plastic: {custom_plastic}")

    print("\n5. Features demonstrated:")
    print("   - PBR material properties")
    print("   - BRDF calculations (GGX)")
    print("   - Energy conservation")
    print("   - Metallic-Roughness workflow")
    print("   - Material library")
    print("   - Custom material creation")
    print("   - IBL integration")
    print("   - Tone mapping and gamma correction")

    print("\n6. PBR workflow benefits:")
    print("   - Physically accurate lighting")
    print("   - Consistent material appearance")
    print("   - Energy conservation")
    print("   - Realistic material properties")
    print("   - Standardized material workflow")


if __name__ == "__main__":
    demonstrate_pbr_system()
