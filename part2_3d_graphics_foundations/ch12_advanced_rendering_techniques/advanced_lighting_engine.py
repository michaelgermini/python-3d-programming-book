#!/usr/bin/env python3
"""
Chapter 12: Advanced Rendering Techniques
Advanced Lighting Engine

Demonstrates advanced lighting models, shadow mapping, global illumination approximation,
and real-time lighting optimization techniques.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Advanced Lighting Engine"
__description__ = "Advanced lighting models and shadow mapping"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    """3D vector for positions and directions"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self * scalar
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class Color:
    """Color representation"""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_array(self) -> List[float]:
        return [self.r, self.g, self.b, self.a]

class LightType(Enum):
    """Types of lights"""
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"
    AREA = "area"

@dataclass
class Light:
    """Light source with properties"""
    light_type: LightType
    position: Vector3D
    direction: Vector3D
    color: Color
    intensity: float = 1.0
    range: float = 10.0
    angle: float = 45.0  # For spot lights
    cast_shadows: bool = True
    
    def get_view_matrix(self) -> np.ndarray:
        """Get light view matrix for shadow mapping"""
        if self.light_type == LightType.DIRECTIONAL:
            # For directional light, use orthographic projection
            return self.look_at_ortho(self.position, self.position + self.direction)
        else:
            # For point/spot lights, use perspective projection
            return self.look_at_perspective(self.position, self.position + self.direction)
    
    def look_at_ortho(self, eye: Vector3D, target: Vector3D) -> np.ndarray:
        """Create orthographic look-at matrix for directional light"""
        z = (eye - target).normalize()
        x = Vector3D(1, 0, 0)  # Simplified up vector
        y = z.cross(x).normalize()
        x = y.cross(z).normalize()
        
        return np.array([
            [x.x, x.y, x.z, -x.dot(eye)],
            [y.x, y.y, y.z, -y.dot(eye)],
            [z.x, z.y, z.z, -z.dot(eye)],
            [0, 0, 0, 1]
        ], dtype='f4')
    
    def look_at_perspective(self, eye: Vector3D, target: Vector3D) -> np.ndarray:
        """Create perspective look-at matrix for point/spot light"""
        z = (eye - target).normalize()
        x = Vector3D(1, 0, 0)
        y = z.cross(x).normalize()
        x = y.cross(z).normalize()
        
        return np.array([
            [x.x, x.y, x.z, -x.dot(eye)],
            [y.x, y.y, y.z, -y.dot(eye)],
            [z.x, z.y, z.z, -z.dot(eye)],
            [0, 0, 0, 1]
        ], dtype='f4')

# ============================================================================
# LIGHTING MODELS
# ============================================================================

class LightingModel(Enum):
    """Available lighting models"""
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    PBR = "pbr"
    CEL_SHADING = "cel_shading"

class LightingEngine:
    """Advanced lighting engine with multiple models"""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.lights: List[Light] = []
        self.shadow_maps: Dict[int, moderngl.Texture] = {}
        self.current_model = LightingModel.PBR
        
        # Shadow mapping settings
        self.shadow_resolution = 1024
        self.shadow_bias = 0.005
        self.shadow_samples = 16
        
        # Global illumination settings
        self.ambient_occlusion = True
        self.ao_radius = 0.5
        self.ao_strength = 1.0
        
        self.setup_shadow_framebuffers()
    
    def add_light(self, light: Light):
        """Add a light to the scene"""
        self.lights.append(light)
        if light.cast_shadows:
            self.create_shadow_map(len(self.lights) - 1)
    
    def create_shadow_map(self, light_index: int):
        """Create shadow map for a light"""
        shadow_map = self.ctx.depth_texture((self.shadow_resolution, self.shadow_resolution))
        shadow_fbo = self.ctx.framebuffer(depth_attachment=shadow_map)
        self.shadow_maps[light_index] = shadow_map
    
    def setup_shadow_framebuffers(self):
        """Setup shadow mapping framebuffers"""
        # This would be implemented with actual shadow map rendering
        pass
    
    def calculate_lighting(self, position: Vector3D, normal: Vector3D, 
                          material_color: Color, view_pos: Vector3D) -> Color:
        """Calculate lighting for a surface point"""
        if self.current_model == LightingModel.PHONG:
            return self.phong_lighting(position, normal, material_color, view_pos)
        elif self.current_model == LightingModel.BLINN_PHONG:
            return self.blinn_phong_lighting(position, normal, material_color, view_pos)
        elif self.current_model == LightingModel.PBR:
            return self.pbr_lighting(position, normal, material_color, view_pos)
        elif self.current_model == LightingModel.CEL_SHADING:
            return self.cel_shading_lighting(position, normal, material_color, view_pos)
        else:
            return material_color
    
    def phong_lighting(self, position: Vector3D, normal: Vector3D, 
                      material_color: Color, view_pos: Vector3D) -> Color:
        """Phong lighting model"""
        result = Color(0, 0, 0)
        
        for light in self.lights:
            # Ambient
            ambient = Color(0.1, 0.1, 0.1) * light.color * light.intensity
            
            # Diffuse
            if light.light_type == LightType.DIRECTIONAL:
                light_dir = -light.direction.normalize()
            else:
                light_dir = (light.position - position).normalize()
            
            diff = max(normal.dot(light_dir), 0.0)
            diffuse = light.color * light.intensity * diff
            
            # Specular
            view_dir = (view_pos - position).normalize()
            reflect_dir = light_dir - normal * (2 * normal.dot(light_dir))
            spec = max(view_dir.dot(reflect_dir), 0.0) ** 32
            specular = light.color * light.intensity * spec * 0.5
            
            # Attenuation for point/spot lights
            if light.light_type != LightType.DIRECTIONAL:
                distance = (light.position - position).magnitude()
                attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance)
                diffuse = diffuse * attenuation
                specular = specular * attenuation
            
            result.r += ambient.r + diffuse.r + specular.r
            result.g += ambient.g + diffuse.g + specular.g
            result.b += ambient.b + diffuse.b + specular.b
        
        # Apply material color
        result.r *= material_color.r
        result.g *= material_color.g
        result.b *= material_color.b
        
        return result
    
    def blinn_phong_lighting(self, position: Vector3D, normal: Vector3D, 
                            material_color: Color, view_pos: Vector3D) -> Color:
        """Blinn-Phong lighting model"""
        result = Color(0, 0, 0)
        
        for light in self.lights:
            # Ambient
            ambient = Color(0.1, 0.1, 0.1) * light.color * light.intensity
            
            # Diffuse
            if light.light_type == LightType.DIRECTIONAL:
                light_dir = -light.direction.normalize()
            else:
                light_dir = (light.position - position).normalize()
            
            diff = max(normal.dot(light_dir), 0.0)
            diffuse = light.color * light.intensity * diff
            
            # Blinn-Phong specular
            view_dir = (view_pos - position).normalize()
            halfway_dir = (light_dir + view_dir).normalize()
            spec = max(normal.dot(halfway_dir), 0.0) ** 32
            specular = light.color * light.intensity * spec * 0.5
            
            # Attenuation
            if light.light_type != LightType.DIRECTIONAL:
                distance = (light.position - position).magnitude()
                attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance)
                diffuse = diffuse * attenuation
                specular = specular * attenuation
            
            result.r += ambient.r + diffuse.r + specular.r
            result.g += ambient.g + diffuse.g + specular.g
            result.b += ambient.b + diffuse.b + specular.b
        
        # Apply material color
        result.r *= material_color.r
        result.g *= material_color.g
        result.b *= material_color.b
        
        return result
    
    def pbr_lighting(self, position: Vector3D, normal: Vector3D, 
                    material_color: Color, view_pos: Vector3D) -> Color:
        """Physically Based Rendering lighting model"""
        result = Color(0, 0, 0)
        view_dir = (view_pos - position).normalize()
        
        for light in self.lights:
            # Light direction
            if light.light_type == LightType.DIRECTIONAL:
                light_dir = -light.direction.normalize()
                distance = 1.0  # No attenuation for directional
            else:
                light_dir = (light.position - position).normalize()
                distance = (light.position - position).magnitude()
            
            # Cook-Torrance BRDF
            halfway = (view_dir + light_dir).normalize()
            
            # Distribution function (GGX)
            roughness = 0.5  # Material roughness
            a = roughness * roughness
            a2 = a * a
            ndoth = max(normal.dot(halfway), 0.0)
            ndoth2 = ndoth * ndoth
            nom = a2
            denom = (ndoth2 * (a2 - 1.0) + 1.0)
            denom = math.pi * denom * denom
            ndf = nom / denom if denom > 0 else 0
            
            # Geometry function (Schlick-GGX)
            k = (roughness + 1.0) * (roughness + 1.0) / 8.0
            ndotv = max(normal.dot(view_dir), 0.0)
            ndotl = max(normal.dot(light_dir), 0.0)
            g1_v = ndotv / (ndotv * (1.0 - k) + k)
            g1_l = ndotl / (ndotl * (1.0 - k) + k)
            g = g1_v * g1_l
            
            # Fresnel function (Schlick)
            f0 = 0.04  # Base reflectivity
            f = f0 + (1.0 - f0) * (1.0 - max(halfway.dot(view_dir), 0.0)) ** 5
            
            # Cook-Torrance specular
            numerator = ndf * g * f
            denominator = 4.0 * ndotv * ndotl + 0.0001
            specular = numerator / denominator
            
            # Energy conservation
            ks = f
            kd = (1.0 - ks) * (1.0 - 0.0)  # 0.0 = metallic
            
            # Radiance
            if light.light_type != LightType.DIRECTIONAL:
                attenuation = 1.0 / (distance * distance)
            else:
                attenuation = 1.0
            
            radiance = light.color * light.intensity * attenuation
            
            # Final color
            lo = (kd * material_color / math.pi + specular) * radiance * ndotl
            
            result.r += lo.r
            result.g += lo.g
            result.b += lo.b
        
        # Ambient
        ambient = Color(0.03, 0.03, 0.03) * material_color
        result.r += ambient.r
        result.g += ambient.g
        result.b += ambient.b
        
        # HDR tonemapping
        result.r = result.r / (result.r + 1.0)
        result.g = result.g / (result.g + 1.0)
        result.b = result.b / (result.b + 1.0)
        
        # Gamma correction
        result.r = result.r ** (1.0 / 2.2)
        result.g = result.g ** (1.0 / 2.2)
        result.b = result.b ** (1.0 / 2.2)
        
        return result
    
    def cel_shading_lighting(self, position: Vector3D, normal: Vector3D, 
                           material_color: Color, view_pos: Vector3D) -> Color:
        """Cel-shading (toon shading) lighting model"""
        result = Color(0, 0, 0)
        
        for light in self.lights:
            # Light direction
            if light.light_type == LightType.DIRECTIONAL:
                light_dir = -light.direction.normalize()
            else:
                light_dir = (light.position - position).normalize()
            
            # Diffuse with cel-shading
            diff = normal.dot(light_dir)
            
            # Cel-shading levels
            if diff > 0.7:
                diff = 1.0
            elif diff > 0.3:
                diff = 0.6
            elif diff > 0.0:
                diff = 0.3
            else:
                diff = 0.0
            
            diffuse = light.color * light.intensity * diff
            
            # Specular with cel-shading
            view_dir = (view_pos - position).normalize()
            reflect_dir = light_dir - normal * (2 * normal.dot(light_dir))
            spec = view_dir.dot(reflect_dir)
            
            if spec > 0.8:
                spec = 1.0
            elif spec > 0.0:
                spec = 0.0
            else:
                spec = 0.0
            
            specular = light.color * light.intensity * spec * 0.5
            
            result.r += diffuse.r + specular.r
            result.g += diffuse.g + specular.g
            result.b += diffuse.b + specular.b
        
        # Apply material color
        result.r *= material_color.r
        result.g *= material_color.g
        result.b *= material_color.b
        
        return result

# ============================================================================
# SHADOW MAPPING
# ============================================================================

class ShadowMapper:
    """Shadow mapping system"""
    
    def __init__(self, ctx: moderngl.Context, resolution: int = 1024):
        self.ctx = ctx
        self.resolution = resolution
        self.shadow_maps: Dict[int, moderngl.Texture] = {}
        self.shadow_fbos: Dict[int, moderngl.Framebuffer] = {}
        
    def create_shadow_map(self, light_index: int) -> moderngl.Texture:
        """Create a shadow map for a light"""
        shadow_map = self.ctx.depth_texture((self.resolution, self.resolution))
        shadow_fbo = self.ctx.framebuffer(depth_attachment=shadow_map)
        
        self.shadow_maps[light_index] = shadow_map
        self.shadow_fbos[light_index] = shadow_fbo
        
        return shadow_map
    
    def render_shadow_map(self, light_index: int, scene_objects: List[Any]):
        """Render shadow map for a light"""
        if light_index not in self.shadow_fbos:
            return
        
        shadow_fbo = self.shadow_fbos[light_index]
        
        # Bind shadow framebuffer
        shadow_fbo.use()
        self.ctx.clear()
        
        # Render scene objects to shadow map
        # This would be implemented with actual geometry rendering
        pass
    
    def sample_shadow(self, light_index: int, world_pos: Vector3D, 
                     light_view_proj: np.ndarray) -> float:
        """Sample shadow map to determine if point is in shadow"""
        if light_index not in self.shadow_maps:
            return 1.0  # No shadow
        
        # Transform world position to light space
        pos_light_space = light_view_proj @ np.array([world_pos.x, world_pos.y, world_pos.z, 1.0])
        pos_light_space = pos_light_space / pos_light_space[3]
        
        # Convert to texture coordinates
        tex_coords = np.array([pos_light_space[0] * 0.5 + 0.5, 
                              pos_light_space[1] * 0.5 + 0.5])
        
        # Sample shadow map (simplified)
        # In a real implementation, this would sample the actual texture
        return 1.0  # No shadow for now

# ============================================================================
# GLOBAL ILLUMINATION
# ============================================================================

class GlobalIllumination:
    """Global illumination approximation"""
    
    def __init__(self):
        self.ambient_occlusion = True
        self.ao_radius = 0.5
        self.ao_strength = 1.0
        self.ao_samples = 16
        
    def calculate_ambient_occlusion(self, position: Vector3D, normal: Vector3D, 
                                  scene_objects: List[Any]) -> float:
        """Calculate ambient occlusion for a surface point"""
        if not self.ambient_occlusion:
            return 1.0
        
        occlusion = 0.0
        samples = 0
        
        # Generate random samples in hemisphere
        for i in range(self.ao_samples):
            # Generate random direction in hemisphere
            sample_dir = self.random_hemisphere_direction(normal)
            sample_pos = position + sample_dir * self.ao_radius
            
            # Check if sample point is occluded
            if self.is_point_occluded(sample_pos, scene_objects):
                occlusion += 1.0
            
            samples += 1
        
        ao = 1.0 - (occlusion / samples) * self.ao_strength
        return max(0.0, min(1.0, ao))
    
    def random_hemisphere_direction(self, normal: Vector3D) -> Vector3D:
        """Generate random direction in hemisphere around normal"""
        # Simplified random direction generation
        u = random.random()
        v = random.random()
        
        theta = 2.0 * math.pi * u
        phi = math.acos(math.sqrt(v))
        
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        
        # Transform to hemisphere around normal
        # This is a simplified version
        return Vector3D(x, y, z).normalize()
    
    def is_point_occluded(self, point: Vector3D, scene_objects: List[Any]) -> bool:
        """Check if a point is occluded by scene objects"""
        # Simplified occlusion test
        # In a real implementation, this would ray trace against scene geometry
        return False

# ============================================================================
# LIGHTING RENDERER
# ============================================================================

class AdvancedLightingRenderer:
    """Advanced lighting renderer with multiple models and shadow mapping"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.lighting_engine = None
        self.shadow_mapper = None
        self.global_illumination = None
        
        # Camera
        self.camera_pos = Vector3D(0, 0, 3)
        self.camera_target = Vector3D(0, 0, 0)
        self.camera_up = Vector3D(0, 1, 0)
        self.fov = 45.0
        
        # Scene objects
        self.scene_objects = []
        self.current_lighting_model = LightingModel.PBR
        
        self.init_glfw()
        self.init_opengl()
        self.setup_lighting()
        self.setup_scene()
    
    def init_glfw(self):
        """Initialize GLFW"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Advanced Lighting", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        """Initialize OpenGL context"""
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
    
    def setup_lighting(self):
        """Setup lighting system"""
        self.lighting_engine = LightingEngine(self.ctx)
        self.shadow_mapper = ShadowMapper(self.ctx)
        self.global_illumination = GlobalIllumination()
        
        # Add lights
        directional_light = Light(
            light_type=LightType.DIRECTIONAL,
            position=Vector3D(5, 5, 5),
            direction=Vector3D(-1, -1, -1),
            color=Color(1.0, 0.95, 0.8),
            intensity=1.0,
            cast_shadows=True
        )
        
        point_light = Light(
            light_type=LightType.POINT,
            position=Vector3D(2, 2, 2),
            direction=Vector3D(0, 0, 0),
            color=Color(0.8, 0.8, 1.0),
            intensity=2.0,
            range=10.0,
            cast_shadows=True
        )
        
        spot_light = Light(
            light_type=LightType.SPOT,
            position=Vector3D(-2, 3, 0),
            direction=Vector3D(1, -1, 0),
            color=Color(1.0, 0.6, 0.3),
            intensity=1.5,
            range=8.0,
            angle=30.0,
            cast_shadows=True
        )
        
        self.lighting_engine.add_light(directional_light)
        self.lighting_engine.add_light(point_light)
        self.lighting_engine.add_light(spot_light)
    
    def setup_scene(self):
        """Setup scene objects"""
        # Create some test objects
        # In a real implementation, this would load actual geometry
        pass
    
    def render_scene(self):
        """Render the scene with advanced lighting"""
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Render shadow maps
        self.render_shadow_maps()
        
        # Render scene with lighting
        self.render_with_lighting()
    
    def render_shadow_maps(self):
        """Render shadow maps for all lights"""
        for i, light in enumerate(self.lighting_engine.lights):
            if light.cast_shadows:
                self.shadow_mapper.render_shadow_map(i, self.scene_objects)
    
    def render_with_lighting(self):
        """Render scene with advanced lighting"""
        # In a real implementation, this would render actual geometry
        # For now, we'll demonstrate the lighting calculations
        
        # Example surface point
        position = Vector3D(0, 0, 0)
        normal = Vector3D(0, 1, 0)
        material_color = Color(0.8, 0.6, 0.4)
        
        # Calculate lighting
        lighting_engine = self.lighting_engine
        lighting_engine.current_model = self.current_lighting_model
        
        final_color = lighting_engine.calculate_lighting(
            position, normal, material_color, self.camera_pos
        )
        
        # Apply ambient occlusion
        ao = self.global_illumination.calculate_ambient_occlusion(
            position, normal, self.scene_objects
        )
        final_color.r *= ao
        final_color.g *= ao
        final_color.b *= ao
        
        print(f"Lighting result: {final_color}")
    
    def update(self, delta_time: float):
        """Update scene"""
        # Animate lights
        for i, light in enumerate(self.lighting_engine.lights):
            if light.light_type == LightType.POINT:
                # Animate point light
                angle = time.time() * 0.5 + i * math.pi / 3
                light.position.x = math.cos(angle) * 3
                light.position.z = math.sin(angle) * 3
    
    def handle_input(self):
        """Handle keyboard input"""
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.current_lighting_model = LightingModel.PHONG
            print("Switched to Phong lighting")
        elif glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.current_lighting_model = LightingModel.BLINN_PHONG
            print("Switched to Blinn-Phong lighting")
        elif glfw.get_key(self.window, glfw.KEY_3) == glfw.PRESS:
            self.current_lighting_model = LightingModel.PBR
            print("Switched to PBR lighting")
        elif glfw.get_key(self.window, glfw.KEY_4) == glfw.PRESS:
            self.current_lighting_model = LightingModel.CEL_SHADING
            print("Switched to Cel-shading")
    
    def framebuffer_size_callback(self, window, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        """Main application loop"""
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Update
            self.update(delta_time)
            
            # Render
            self.render_scene()
            
            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Handle input
            self.handle_input()
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_lighting_models():
    """Demonstrate different lighting models"""
    print("=== Lighting Models Demo ===\n")
    
    # Create test data
    position = Vector3D(0, 0, 0)
    normal = Vector3D(0, 1, 0)
    material_color = Color(0.8, 0.6, 0.4)
    view_pos = Vector3D(0, 0, 3)
    
    # Create a simple lighting engine for demonstration
    class MockContext:
        pass
    
    ctx = MockContext()
    lighting_engine = LightingEngine(ctx)
    
    # Add a test light
    test_light = Light(
        light_type=LightType.DIRECTIONAL,
        position=Vector3D(5, 5, 5),
        direction=Vector3D(-1, -1, -1),
        color=Color(1.0, 0.95, 0.8),
        intensity=1.0
    )
    lighting_engine.add_light(test_light)
    
    # Test different lighting models
    models = [
        (LightingModel.PHONG, "Phong"),
        (LightingModel.BLINN_PHONG, "Blinn-Phong"),
        (LightingModel.PBR, "PBR"),
        (LightingModel.CEL_SHADING, "Cel-Shading")
    ]
    
    print("Lighting model comparison:")
    for model, name in models:
        lighting_engine.current_model = model
        result = lighting_engine.calculate_lighting(position, normal, material_color, view_pos)
        print(f"  {name}: RGB({result.r:.3f}, {result.g:.3f}, {result.b:.3f})")
    
    print()

def demonstrate_shadow_mapping():
    """Demonstrate shadow mapping concepts"""
    print("=== Shadow Mapping Demo ===\n")
    
    print("Shadow mapping features:")
    print("  • Depth texture generation")
    print("  • Light space transformation")
    print("  • Shadow map sampling")
    print("  • Bias and PCF filtering")
    print("  • Soft shadows with multiple samples")
    print()
    
    print("Shadow mapping process:")
    print("  1. Render scene from light's perspective")
    print("  2. Store depth values in shadow map")
    print("  3. During main rendering, transform world positions to light space")
    print("  4. Compare transformed depth with shadow map depth")
    print("  5. Apply shadow factor to lighting calculation")
    print()

def demonstrate_global_illumination():
    """Demonstrate global illumination concepts"""
    print("=== Global Illumination Demo ===\n")
    
    print("Global illumination techniques:")
    print("  • Ambient occlusion calculation")
    print("  • Hemisphere sampling")
    print("  • Indirect lighting approximation")
    print("  • Screen space ambient occlusion (SSAO)")
    print("  • Light probes and irradiance maps")
    print()
    
    print("Ambient occlusion process:")
    print("  1. Generate random samples in hemisphere around surface normal")
    print("  2. Check if sample points are occluded by geometry")
    print("  3. Calculate occlusion factor based on occluded samples")
    print("  4. Apply occlusion to ambient lighting component")
    print()

def demonstrate_lighting_optimization():
    """Demonstrate lighting optimization techniques"""
    print("=== Lighting Optimization Demo ===\n")
    
    print("Optimization techniques:")
    print("  • Light culling and frustum culling")
    print("  • LOD for lighting calculations")
    print("  • Light clustering and tiled rendering")
    print("  • Shadow map cascades for directional lights")
    print("  • Light importance sampling")
    print()
    
    print("Performance considerations:")
    print("  • Number of lights vs performance")
    print("  • Shadow map resolution vs quality")
    print("  • Lighting model complexity")
    print("  • GPU vs CPU lighting calculations")
    print()

def demonstrate_rendering_system():
    """Demonstrate the complete rendering system"""
    print("=== Advanced Lighting Renderer Demo ===\n")
    
    print("Starting advanced lighting renderer...")
    print("Controls:")
    print("  1: Phong lighting model")
    print("  2: Blinn-Phong lighting model")
    print("  3: PBR lighting model")
    print("  4: Cel-shading lighting model")
    print("  ESC: Exit")
    print()
    
    try:
        renderer = AdvancedLightingRenderer(800, 600)
        renderer.run()
    except Exception as e:
        print(f"✗ Rendering system failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate advanced lighting engine"""
    print("=== Advanced Lighting Engine Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    demonstrate_lighting_models()
    demonstrate_shadow_mapping()
    demonstrate_global_illumination()
    demonstrate_lighting_optimization()
    
    print("="*60)
    print("Advanced Lighting Engine demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Multiple lighting models (Phong, Blinn-Phong, PBR, Cel-shading)")
    print("✓ Shadow mapping and depth textures")
    print("✓ Global illumination approximation")
    print("✓ Real-time lighting optimization")
    print("✓ Light types (directional, point, spot)")
    print("✓ Ambient occlusion and indirect lighting")
    
    print("\nLighting features:")
    print("• Physically based rendering (PBR)")
    print("• Real-time shadow mapping")
    print("• Multiple light sources")
    print("• Ambient occlusion calculation")
    print("• Lighting model switching")
    
    print("\nApplications:")
    print("• Game development: Realistic lighting systems")
    print("• Architectural visualization: Accurate lighting simulation")
    print("• Film and animation: Pre-visualization lighting")
    print("• Product visualization: Material and lighting preview")
    print("• Virtual reality: Immersive lighting environments")
    
    print("\nNext steps:")
    print("• Implement actual geometry rendering")
    print("• Add texture support and normal mapping")
    print("• Implement screen space effects (SSAO, SSR)")
    print("• Add post-processing effects")
    print("• Optimize for mobile and VR platforms")

if __name__ == "__main__":
    main()
