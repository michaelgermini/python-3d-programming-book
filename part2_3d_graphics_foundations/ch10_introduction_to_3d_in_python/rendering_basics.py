#!/usr/bin/env python3
"""
Chapter 10: Introduction to 3D in Python
Rendering Basics

Demonstrates fundamental rendering concepts including scene setup, camera
positioning, basic lighting, and rendering pipeline concepts for 3D graphics.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Rendering Basics"
__description__ = "Fundamental rendering concepts and pipeline"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    """3D vector class for representing positions and directions"""
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
        """Dot product of two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product of two vectors"""
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
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class Color:
    """Color class for rendering"""
    r: float  # Red component (0.0 to 1.0)
    g: float  # Green component (0.0 to 1.0)
    b: float  # Blue component (0.0 to 1.0)
    a: float = 1.0  # Alpha component (0.0 to 1.0)
    
    def __add__(self, other: 'Color') -> 'Color':
        return Color(
            min(1.0, self.r + other.r),
            min(1.0, self.g + other.g),
            min(1.0, self.b + other.b),
            min(1.0, self.a + other.a)
        )
    
    def __mul__(self, scalar: float) -> 'Color':
        return Color(
            max(0.0, min(1.0, self.r * scalar)),
            max(0.0, min(1.0, self.g * scalar)),
            max(0.0, min(1.0, self.b * scalar)),
            self.a
        )
    
    def __rmul__(self, scalar: float) -> 'Color':
        return self * scalar
    
    def clamp(self) -> 'Color':
        """Clamp color values to valid range"""
        return Color(
            max(0.0, min(1.0, self.r)),
            max(0.0, min(1.0, self.g)),
            max(0.0, min(1.0, self.b)),
            max(0.0, min(1.0, self.a))
        )
    
    def __str__(self):
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"

class LightType(Enum):
    """Types of lights"""
    AMBIENT = "ambient"
    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"

# ============================================================================
# RENDERING COMPONENTS
# ============================================================================

class Camera:
    """Camera class for viewing 3D scenes"""
    
    def __init__(self, position: Vector3D, target: Vector3D, up: Vector3D = Vector3D(0, 1, 0)):
        self.position = position
        self.target = target
        self.up = up
        self.fov = math.pi / 4  # 45 degrees
        self.aspect_ratio = 16.0 / 9.0
        self.near_plane = 0.1
        self.far_plane = 100.0
    
    def get_view_direction(self) -> Vector3D:
        """Get the view direction vector"""
        return (self.target - self.position).normalize()
    
    def get_right_vector(self) -> Vector3D:
        """Get the right vector of the camera"""
        view_dir = self.get_view_direction()
        return view_dir.cross(self.up).normalize()
    
    def get_up_vector(self) -> Vector3D:
        """Get the up vector of the camera"""
        view_dir = self.get_view_direction()
        right = view_dir.cross(self.up).normalize()
        return right.cross(view_dir).normalize()
    
    def set_fov(self, fov: float):
        """Set the field of view in radians"""
        self.fov = fov
    
    def set_aspect_ratio(self, aspect_ratio: float):
        """Set the aspect ratio"""
        self.aspect_ratio = aspect_ratio
    
    def set_clipping_planes(self, near: float, far: float):
        """Set the near and far clipping planes"""
        self.near_plane = near
        self.far_plane = far
    
    def look_at(self, target: Vector3D):
        """Make the camera look at a specific target"""
        self.target = target
    
    def move_to(self, position: Vector3D):
        """Move the camera to a new position"""
        self.position = position
    
    def __str__(self):
        return f"Camera(pos={self.position}, target={self.target}, fov={math.degrees(self.fov):.1f}°)"

class Light:
    """Base light class"""
    
    def __init__(self, light_type: LightType, color: Color = Color(1, 1, 1), intensity: float = 1.0):
        self.light_type = light_type
        self.color = color
        self.intensity = intensity
    
    def get_illumination(self, point: Vector3D, normal: Vector3D) -> Color:
        """Get illumination at a point (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.light_type.value.capitalize()}Light(color={self.color}, intensity={self.intensity})"

class AmbientLight(Light):
    """Ambient light that provides uniform illumination"""
    
    def __init__(self, color: Color = Color(0.1, 0.1, 0.1), intensity: float = 1.0):
        super().__init__(LightType.AMBIENT, color, intensity)
    
    def get_illumination(self, point: Vector3D, normal: Vector3D) -> Color:
        """Get ambient illumination"""
        return self.color * self.intensity

class DirectionalLight(Light):
    """Directional light (like the sun)"""
    
    def __init__(self, direction: Vector3D, color: Color = Color(1, 1, 1), intensity: float = 1.0):
        super().__init__(LightType.DIRECTIONAL, color, intensity)
        self.direction = direction.normalize()
    
    def get_illumination(self, point: Vector3D, normal: Vector3D) -> Color:
        """Get directional illumination"""
        # Calculate diffuse lighting
        light_dir = -self.direction  # Light direction towards the surface
        diffuse_factor = max(0.0, normal.dot(light_dir))
        return self.color * self.intensity * diffuse_factor

class PointLight(Light):
    """Point light that emits light in all directions"""
    
    def __init__(self, position: Vector3D, color: Color = Color(1, 1, 1), intensity: float = 1.0, 
                 attenuation: float = 1.0):
        super().__init__(LightType.POINT, color, intensity)
        self.position = position
        self.attenuation = attenuation
    
    def get_illumination(self, point: Vector3D, normal: Vector3D) -> Color:
        """Get point light illumination"""
        # Calculate light direction
        light_dir = (self.position - point).normalize()
        
        # Calculate distance for attenuation
        distance = (self.position - point).magnitude()
        attenuation_factor = 1.0 / (1.0 + self.attenuation * distance * distance)
        
        # Calculate diffuse lighting
        diffuse_factor = max(0.0, normal.dot(light_dir))
        
        return self.color * self.intensity * diffuse_factor * attenuation_factor

class Material:
    """Material class for defining surface properties"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.diffuse_color = Color(0.8, 0.8, 0.8)
        self.ambient_color = Color(0.2, 0.2, 0.2)
        self.specular_color = Color(1.0, 1.0, 1.0)
        self.shininess = 32.0
        self.reflectivity = 0.0
        self.transparency = 0.0
    
    def set_diffuse(self, color: Color):
        """Set the diffuse color"""
        self.diffuse_color = color
    
    def set_ambient(self, color: Color):
        """Set the ambient color"""
        self.ambient_color = color
    
    def set_specular(self, color: Color, shininess: float):
        """Set the specular color and shininess"""
        self.specular_color = color
        self.shininess = shininess
    
    def set_reflectivity(self, reflectivity: float):
        """Set the reflectivity (0.0 to 1.0)"""
        self.reflectivity = max(0.0, min(1.0, reflectivity))
    
    def set_transparency(self, transparency: float):
        """Set the transparency (0.0 to 1.0)"""
        self.transparency = max(0.0, min(1.0, transparency))
    
    def __str__(self):
        return f"Material('{self.name}', diffuse={self.diffuse_color})"

class RenderableObject:
    """Base class for objects that can be rendered"""
    
    def __init__(self, position: Vector3D, material: Material = None):
        self.position = position
        self.material = material or Material()
        self.visible = True
    
    def get_surface_normal(self, point: Vector3D) -> Vector3D:
        """Get the surface normal at a point (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_color_at(self, point: Vector3D) -> Color:
        """Get the color at a point"""
        return self.material.diffuse_color
    
    def is_visible(self) -> bool:
        """Check if the object is visible"""
        return self.visible
    
    def set_visibility(self, visible: bool):
        """Set the visibility of the object"""
        self.visible = visible
    
    def __str__(self):
        return f"{self.__class__.__name__}(pos={self.position}, material='{self.material.name}')"

class Sphere(RenderableObject):
    """Sphere object for rendering"""
    
    def __init__(self, position: Vector3D, radius: float, material: Material = None):
        super().__init__(position, material)
        self.radius = radius
    
    def get_surface_normal(self, point: Vector3D) -> Vector3D:
        """Get the surface normal at a point on the sphere"""
        return (point - self.position).normalize()
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if a point is inside the sphere"""
        distance = (point - self.position).magnitude()
        return distance <= self.radius

class Cube(RenderableObject):
    """Cube object for rendering"""
    
    def __init__(self, position: Vector3D, size: float, material: Material = None):
        super().__init__(position, material)
        self.size = size
    
    def get_surface_normal(self, point: Vector3D) -> Vector3D:
        """Get the surface normal at a point on the cube (simplified)"""
        # Simplified: return the normal of the closest face
        relative_point = point - self.position
        half_size = self.size / 2
        
        # Find the closest face
        abs_x = abs(relative_point.x)
        abs_y = abs(relative_point.y)
        abs_z = abs(relative_point.z)
        
        if abs_x >= abs_y and abs_x >= abs_z:
            return Vector3D(1 if relative_point.x > 0 else -1, 0, 0)
        elif abs_y >= abs_z:
            return Vector3D(0, 1 if relative_point.y > 0 else -1, 0)
        else:
            return Vector3D(0, 0, 1 if relative_point.z > 0 else -1)

# ============================================================================
# RENDERING PIPELINE
# ============================================================================

class Renderer:
    """Basic renderer for 3D scenes"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.camera = None
        self.objects = []
        self.lights = []
        self.background_color = Color(0.1, 0.1, 0.2)
    
    def set_camera(self, camera: Camera):
        """Set the camera for rendering"""
        self.camera = camera
    
    def add_object(self, obj: RenderableObject):
        """Add a renderable object to the scene"""
        self.objects.append(obj)
    
    def add_light(self, light: Light):
        """Add a light to the scene"""
        self.lights.append(light)
    
    def set_background_color(self, color: Color):
        """Set the background color"""
        self.background_color = color
    
    def render_scene(self) -> List[List[Color]]:
        """Render the scene and return a 2D array of colors"""
        if not self.camera:
            raise ValueError("No camera set for rendering")
        
        # Initialize the image buffer
        image = [[self.background_color for _ in range(self.width)] for _ in range(self.height)]
        
        # Simple ray casting for each pixel
        for y in range(self.height):
            for x in range(self.width):
                # Convert screen coordinates to world coordinates
                world_point = self._screen_to_world(x, y)
                
                # Find the closest object
                closest_object = None
                closest_distance = float('inf')
                
                for obj in self.objects:
                    if not obj.is_visible():
                        continue
                    
                    # Simplified intersection test
                    if isinstance(obj, Sphere):
                        distance = (world_point - obj.position).magnitude()
                        if distance <= obj.radius and distance < closest_distance:
                            closest_distance = distance
                            closest_object = obj
                
                # Calculate lighting for the closest object
                if closest_object:
                    # Get the surface point and normal
                    if isinstance(closest_object, Sphere):
                        surface_point = closest_object.position + (world_point - closest_object.position).normalize() * closest_object.radius
                        normal = closest_object.get_surface_normal(surface_point)
                    else:
                        surface_point = world_point
                        normal = closest_object.get_surface_normal(surface_point)
                    
                    # Calculate lighting
                    final_color = self._calculate_lighting(surface_point, normal, closest_object)
                    image[y][x] = final_color
        
        return image
    
    def _screen_to_world(self, x: int, y: int) -> Vector3D:
        """Convert screen coordinates to world coordinates (simplified)"""
        # Normalize screen coordinates to [-1, 1]
        nx = (2.0 * x / self.width) - 1.0
        ny = 1.0 - (2.0 * y / self.height)
        
        # Apply camera transformations (simplified)
        camera_pos = self.camera.position
        view_dir = self.camera.get_view_direction()
        right = self.camera.get_right_vector()
        up = self.camera.get_up_vector()
        
        # Calculate world position
        world_point = camera_pos + view_dir + (right * nx) + (up * ny)
        return world_point
    
    def _calculate_lighting(self, point: Vector3D, normal: Vector3D, obj: RenderableObject) -> Color:
        """Calculate lighting at a point"""
        final_color = Color(0, 0, 0)
        
        for light in self.lights:
            illumination = light.get_illumination(point, normal)
            object_color = obj.get_color_at(point)
            
            # Combine light and object color
            lit_color = Color(
                illumination.r * object_color.r,
                illumination.g * object_color.g,
                illumination.b * object_color.b
            )
            
            final_color = final_color + lit_color
        
        return final_color.clamp()
    
    def save_image(self, filename: str, image: List[List[Color]]):
        """Save the rendered image to a file (simplified text representation)"""
        with open(filename, 'w') as f:
            f.write(f"P3\n{self.width} {self.height}\n255\n")
            for row in image:
                for color in row:
                    r = int(color.r * 255)
                    g = int(color.g * 255)
                    b = int(color.b * 255)
                    f.write(f"{r} {g} {b}\n")

# ============================================================================
# SCENE MANAGEMENT
# ============================================================================

class Scene:
    """Scene class for managing 3D objects and rendering"""
    
    def __init__(self, name: str = "default_scene"):
        self.name = name
        self.renderer = Renderer()
        self.objects = []
        self.lights = []
        self.camera = None
    
    def add_object(self, obj: RenderableObject):
        """Add an object to the scene"""
        self.objects.append(obj)
        self.renderer.add_object(obj)
    
    def add_light(self, light: Light):
        """Add a light to the scene"""
        self.lights.append(light)
        self.renderer.add_light(light)
    
    def set_camera(self, camera: Camera):
        """Set the camera for the scene"""
        self.camera = camera
        self.renderer.set_camera(camera)
    
    def render(self, width: int = 800, height: int = 600) -> List[List[Color]]:
        """Render the scene"""
        self.renderer.width = width
        self.renderer.height = height
        return self.renderer.render_scene()
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get information about the scene"""
        return {
            'name': self.name,
            'object_count': len(self.objects),
            'light_count': len(self.lights),
            'has_camera': self.camera is not None,
            'objects': [str(obj) for obj in self.objects],
            'lights': [str(light) for light in self.lights],
            'camera': str(self.camera) if self.camera else "None"
        }

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_camera_setup():
    """Demonstrate camera setup and positioning"""
    print("=== Camera Setup ===\n")
    
    # Create different camera configurations
    cameras = [
        Camera(Vector3D(0, 0, 5), Vector3D(0, 0, 0), Vector3D(0, 1, 0)),
        Camera(Vector3D(5, 5, 5), Vector3D(0, 0, 0), Vector3D(0, 1, 0)),
        Camera(Vector3D(0, 10, 0), Vector3D(0, 0, 0), Vector3D(0, 0, -1))
    ]
    
    for i, camera in enumerate(cameras):
        print(f"Camera {i+1}:")
        print(f"  Position: {camera.position}")
        print(f"  Target: {camera.target}")
        print(f"  View Direction: {camera.get_view_direction()}")
        print(f"  Right Vector: {camera.get_right_vector()}")
        print(f"  Up Vector: {camera.get_up_vector()}")
        print(f"  FOV: {math.degrees(camera.fov):.1f}°")
        print()
    
    # Demonstrate camera movement
    camera = Camera(Vector3D(0, 0, 5), Vector3D(0, 0, 0))
    print("Camera movement demonstration:")
    print(f"  Initial position: {camera.position}")
    
    camera.move_to(Vector3D(2, 3, 4))
    print(f"  After move_to(2, 3, 4): {camera.position}")
    
    camera.look_at(Vector3D(1, 1, 1))
    print(f"  After look_at(1, 1, 1): {camera.target}")
    print()

def demonstrate_lighting_system():
    """Demonstrate different types of lights"""
    print("=== Lighting System ===\n")
    
    # Create different types of lights
    ambient_light = AmbientLight(Color(0.1, 0.1, 0.1), 1.0)
    directional_light = DirectionalLight(Vector3D(1, 1, 1), Color(1, 1, 1), 0.8)
    point_light = PointLight(Vector3D(0, 5, 0), Color(1, 0.5, 0.5), 1.0, 0.1)
    
    lights = [ambient_light, directional_light, point_light]
    
    # Test point and normal
    test_point = Vector3D(0, 0, 0)
    test_normal = Vector3D(0, 1, 0)
    
    print("Lighting at point (0, 0, 0) with normal (0, 1, 0):")
    for light in lights:
        illumination = light.get_illumination(test_point, test_normal)
        print(f"  {light}: {illumination}")
    
    print("\nLighting at point (2, 0, 0) with normal (0, 1, 0):")
    test_point2 = Vector3D(2, 0, 0)
    for light in lights:
        illumination = light.get_illumination(test_point2, test_normal)
        print(f"  {light}: {illumination}")
    
    print()

def demonstrate_materials():
    """Demonstrate material properties"""
    print("=== Materials ===\n")
    
    # Create different materials
    materials = [
        Material("red_plastic"),
        Material("blue_metal"),
        Material("green_glass"),
        Material("white_plastic")
    ]
    
    # Set material properties
    materials[0].set_diffuse(Color(0.8, 0.2, 0.2))
    materials[0].set_specular(Color(1.0, 1.0, 1.0), 32.0)
    
    materials[1].set_diffuse(Color(0.2, 0.2, 0.8))
    materials[1].set_specular(Color(0.8, 0.8, 0.8), 128.0)
    materials[1].set_reflectivity(0.3)
    
    materials[2].set_diffuse(Color(0.2, 0.8, 0.2))
    materials[2].set_transparency(0.7)
    
    materials[3].set_diffuse(Color(0.9, 0.9, 0.9))
    materials[3].set_ambient(Color(0.3, 0.3, 0.3))
    
    for material in materials:
        print(f"Material: {material}")
        print(f"  Diffuse: {material.diffuse_color}")
        print(f"  Ambient: {material.ambient_color}")
        print(f"  Specular: {material.specular_color} (shininess: {material.shininess})")
        print(f"  Reflectivity: {material.reflectivity}")
        print(f"  Transparency: {material.transparency}")
        print()
    
    print()

def demonstrate_scene_setup():
    """Demonstrate scene setup and object management"""
    print("=== Scene Setup ===\n")
    
    # Create a scene
    scene = Scene("demo_scene")
    
    # Create camera
    camera = Camera(Vector3D(0, 0, 5), Vector3D(0, 0, 0))
    scene.set_camera(camera)
    
    # Create materials
    red_material = Material("red")
    red_material.set_diffuse(Color(0.8, 0.2, 0.2))
    
    blue_material = Material("blue")
    blue_material.set_diffuse(Color(0.2, 0.2, 0.8))
    
    green_material = Material("green")
    green_material.set_diffuse(Color(0.2, 0.8, 0.2))
    
    # Create objects
    sphere1 = Sphere(Vector3D(-2, 0, 0), 1.0, red_material)
    sphere2 = Sphere(Vector3D(2, 0, 0), 1.5, blue_material)
    cube1 = Cube(Vector3D(0, 0, 0), 1.0, green_material)
    
    # Add objects to scene
    scene.add_object(sphere1)
    scene.add_object(sphere2)
    scene.add_object(cube1)
    
    # Create lights
    ambient_light = AmbientLight(Color(0.1, 0.1, 0.1))
    directional_light = DirectionalLight(Vector3D(1, 1, 1), Color(1, 1, 1), 0.8)
    point_light = PointLight(Vector3D(0, 5, 0), Color(1, 0.5, 0.5), 1.0, 0.1)
    
    # Add lights to scene
    scene.add_light(ambient_light)
    scene.add_light(directional_light)
    scene.add_light(point_light)
    
    # Display scene information
    scene_info = scene.get_scene_info()
    print("Scene Information:")
    for key, value in scene_info.items():
        print(f"  {key}: {value}")
    
    print()

def demonstrate_rendering_pipeline():
    """Demonstrate the rendering pipeline"""
    print("=== Rendering Pipeline ===\n")
    
    # Create a simple scene
    scene = Scene("pipeline_demo")
    
    # Setup camera
    camera = Camera(Vector3D(0, 0, 5), Vector3D(0, 0, 0))
    scene.set_camera(camera)
    
    # Create a simple object
    material = Material("test")
    material.set_diffuse(Color(0.8, 0.6, 0.4))
    sphere = Sphere(Vector3D(0, 0, 0), 2.0, material)
    scene.add_object(sphere)
    
    # Create lighting
    ambient_light = AmbientLight(Color(0.2, 0.2, 0.2))
    directional_light = DirectionalLight(Vector3D(1, 1, 1), Color(1, 1, 1), 0.8)
    scene.add_light(ambient_light)
    scene.add_light(directional_light)
    
    print("Rendering pipeline steps:")
    print("1. Scene setup ✓")
    print("2. Camera positioning ✓")
    print("3. Object placement ✓")
    print("4. Lighting setup ✓")
    print("5. Ray casting (simplified) ✓")
    print("6. Lighting calculations ✓")
    print("7. Color composition ✓")
    
    print("\nScene ready for rendering!")
    print("Note: This is a simplified rendering pipeline for demonstration purposes.")
    print("Real 3D rendering engines use more sophisticated algorithms.")
    
    print()

def demonstrate_lighting_calculations():
    """Demonstrate lighting calculations"""
    print("=== Lighting Calculations ===\n")
    
    # Create test materials and lights
    material = Material("test")
    material.set_diffuse(Color(0.8, 0.6, 0.4))
    material.set_ambient(Color(0.2, 0.2, 0.2))
    
    ambient_light = AmbientLight(Color(0.1, 0.1, 0.1), 1.0)
    directional_light = DirectionalLight(Vector3D(1, 1, 1), Color(1, 1, 1), 0.8)
    point_light = PointLight(Vector3D(0, 5, 0), Color(1, 0.5, 0.5), 1.0, 0.1)
    
    lights = [ambient_light, directional_light, point_light]
    
    # Test different surface orientations
    test_points = [
        Vector3D(0, 0, 0),
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
        Vector3D(0, 0, 1)
    ]
    
    test_normals = [
        Vector3D(0, 1, 0),  # Facing up
        Vector3D(1, 0, 0),  # Facing right
        Vector3D(0, 0, 1),  # Facing forward
        Vector3D(0.707, 0.707, 0)  # Diagonal
    ]
    
    print("Lighting calculations for different surface orientations:")
    for i, (point, normal) in enumerate(zip(test_points, test_normals)):
        print(f"\nSurface {i+1} at {point} with normal {normal}:")
        
        total_illumination = Color(0, 0, 0)
        for light in lights:
            illumination = light.get_illumination(point, normal)
            total_illumination = total_illumination + illumination
            print(f"  {light.light_type.value}: {illumination}")
        
        # Apply material
        final_color = Color(
            total_illumination.r * material.diffuse_color.r,
            total_illumination.g * material.diffuse_color.g,
            total_illumination.b * material.diffuse_color.b
        )
        print(f"  Final color: {final_color}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate rendering basics"""
    print("=== Rendering Basics Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate camera setup
    demonstrate_camera_setup()
    
    # Demonstrate lighting system
    demonstrate_lighting_system()
    
    # Demonstrate materials
    demonstrate_materials()
    
    # Demonstrate scene setup
    demonstrate_scene_setup()
    
    # Demonstrate rendering pipeline
    demonstrate_rendering_pipeline()
    
    # Demonstrate lighting calculations
    demonstrate_lighting_calculations()
    
    print("="*60)
    print("Rendering Basics demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Camera positioning and orientation")
    print("✓ Different types of lights (ambient, directional, point)")
    print("✓ Material properties and surface characteristics")
    print("✓ Scene management and object organization")
    print("✓ Basic rendering pipeline concepts")
    print("✓ Lighting calculations and color composition")
    
    print("\nRendering components covered:")
    print("• Camera: Position, target, field of view, aspect ratio")
    print("• Lights: Ambient, directional, point lights with attenuation")
    print("• Materials: Diffuse, ambient, specular, reflectivity, transparency")
    print("• Objects: Spheres, cubes with surface normals")
    print("• Scene: Object and light management")
    print("• Renderer: Basic ray casting and lighting calculations")
    
    print("\nApplications:")
    print("• Game development: 3D scene rendering and lighting")
    print("• Computer graphics: Realistic image synthesis")
    print("• Visualization: Scientific and data visualization")
    print("• Animation: Frame-by-frame rendering")
    print("• Virtual reality: Real-time 3D environments")
    
    print("\nNext steps:")
    print("• Explore advanced lighting models (PBR, ray tracing)")
    print("• Learn about texture mapping and UV coordinates")
    print("• Study shader programming and GPU rendering")
    print("• Understand optimization techniques and LOD systems")
    print("• Master advanced rendering algorithms")

if __name__ == "__main__":
    main()
