"""
Chapter 28: Simple Ray Tracing and Path Tracing - Ray Tracing
============================================================

This module demonstrates basic ray tracing implementation with shapes and materials.

Key Concepts:
- Ray tracing fundamentals and mathematics
- Shape intersection algorithms
- Material systems and BRDF models
- Camera and ray generation
- Basic rendering pipeline
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math


class MaterialType(Enum):
    """Material type enumeration."""
    LAMBERTIAN = "lambertian"
    METAL = "metal"
    DIELECTRIC = "dielectric"
    EMISSIVE = "emissive"


@dataclass
class Ray:
    """Represents a ray in 3D space."""
    origin: np.ndarray
    direction: np.ndarray
    
    def __post_init__(self):
        if self.origin is None:
            self.origin = np.array([0.0, 0.0, 0.0])
        if self.direction is None:
            self.direction = np.array([0.0, 0.0, 1.0])
        # Normalize direction
        self.direction = self.direction / np.linalg.norm(self.direction)
    
    def at(self, t: float) -> np.ndarray:
        """Get point along ray at parameter t."""
        return self.origin + t * self.direction


@dataclass
class HitRecord:
    """Record of a ray-object intersection."""
    point: np.ndarray
    normal: np.ndarray
    t: float
    front_face: bool
    material: 'Material' = None
    
    def __post_init__(self):
        if self.point is None:
            self.point = np.array([0.0, 0.0, 0.0])
        if self.normal is None:
            self.normal = np.array([0.0, 1.0, 0.0])
    
    def set_face_normal(self, ray: Ray, outward_normal: np.ndarray):
        """Set the face normal based on ray direction."""
        self.front_face = np.dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


@dataclass
class Material:
    """Base material class."""
    material_type: MaterialType
    albedo: np.ndarray = None
    roughness: float = 0.5
    metallic: float = 0.0
    emission: np.ndarray = None
    
    def __post_init__(self):
        if self.albedo is None:
            self.albedo = np.array([0.5, 0.5, 0.5])
        if self.emission is None:
            self.emission = np.array([0.0, 0.0, 0.0])
    
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Ray, np.ndarray]:
        """Scatter ray based on material properties."""
        raise NotImplementedError


class LambertianMaterial(Material):
    """Lambertian (diffuse) material."""
    
    def __init__(self, albedo: np.ndarray):
        super().__init__(MaterialType.LAMBERTIAN, albedo=albedo)
    
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Ray, np.ndarray]:
        """Scatter ray diffusely."""
        scatter_direction = hit_record.normal + self._random_unit_vector()
        
        # Catch degenerate scatter direction
        if np.allclose(scatter_direction, np.zeros(3)):
            scatter_direction = hit_record.normal
        
        scattered_ray = Ray(hit_record.point, scatter_direction)
        attenuation = self.albedo
        return True, scattered_ray, attenuation
    
    def _random_unit_vector(self) -> np.ndarray:
        """Generate random unit vector."""
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) < 1:
                return p / np.linalg.norm(p)


class MetalMaterial(Material):
    """Metallic material with reflection."""
    
    def __init__(self, albedo: np.ndarray, roughness: float = 0.0):
        super().__init__(MaterialType.METAL, albedo=albedo, roughness=roughness)
    
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Ray, np.ndarray]:
        """Scatter ray with reflection."""
        reflected = self._reflect(ray.direction, hit_record.normal)
        reflected = reflected + self.roughness * self._random_unit_vector()
        reflected = reflected / np.linalg.norm(reflected)
        
        scattered_ray = Ray(hit_record.point, reflected)
        attenuation = self.albedo
        return np.dot(scattered_ray.direction, hit_record.normal) > 0, scattered_ray, attenuation
    
    def _reflect(self, v: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Reflect vector v around normal n."""
        return v - 2 * np.dot(v, n) * n
    
    def _random_unit_vector(self) -> np.ndarray:
        """Generate random unit vector."""
        while True:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) < 1:
                return p / np.linalg.norm(p)


class DielectricMaterial(Material):
    """Dielectric material with refraction."""
    
    def __init__(self, index_of_refraction: float):
        super().__init__(MaterialType.DIELECTRIC, albedo=np.array([1.0, 1.0, 1.0]))
        self.ir = index_of_refraction
    
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Ray, np.ndarray]:
        """Scatter ray with refraction."""
        refraction_ratio = 1.0 / self.ir if hit_record.front_face else self.ir
        
        unit_direction = ray.direction / np.linalg.norm(ray.direction)
        cos_theta = min(np.dot(-unit_direction, hit_record.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        cannot_refract = refraction_ratio * sin_theta > 1.0
        
        if cannot_refract or self._reflectance(cos_theta, refraction_ratio) > np.random.random():
            direction = self._reflect(unit_direction, hit_record.normal)
        else:
            direction = self._refract(unit_direction, hit_record.normal, refraction_ratio)
        
        scattered_ray = Ray(hit_record.point, direction)
        attenuation = self.albedo
        return True, scattered_ray, attenuation
    
    def _reflect(self, v: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Reflect vector v around normal n."""
        return v - 2 * np.dot(v, n) * n
    
    def _refract(self, uv: np.ndarray, n: np.ndarray, etai_over_etat: float) -> np.ndarray:
        """Refract vector uv through surface with normal n."""
        cos_theta = min(np.dot(-uv, n), 1.0)
        r_out_perp = etai_over_etat * (uv + cos_theta * n)
        r_out_parallel = -math.sqrt(abs(1.0 - np.linalg.norm(r_out_perp) ** 2)) * n
        return r_out_perp + r_out_parallel
    
    def _reflectance(self, cosine: float, ref_idx: float) -> float:
        """Use Schlick's approximation for reflectance."""
        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0 * r0
        return r0 + (1 - r0) * pow((1 - cosine), 5)


class EmissiveMaterial(Material):
    """Emissive material that emits light."""
    
    def __init__(self, emission: np.ndarray):
        super().__init__(MaterialType.EMISSIVE, emission=emission)
    
    def scatter(self, ray: Ray, hit_record: HitRecord) -> Tuple[bool, Ray, np.ndarray]:
        """Emissive materials don't scatter rays."""
        return False, None, self.emission


class Hittable:
    """Base class for objects that can be hit by rays."""
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits this object."""
        raise NotImplementedError


class Sphere(Hittable):
    """Sphere shape."""
    
    def __init__(self, center: np.ndarray, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check ray-sphere intersection."""
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        half_b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        
        if discriminant < 0:
            return None
        
        sqrtd = math.sqrt(discriminant)
        root = (-half_b - sqrtd) / a
        
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return None
        
        t = root
        point = ray.at(t)
        outward_normal = (point - self.center) / self.radius
        
        hit_record = HitRecord(point=point, normal=outward_normal, t=t, front_face=True, material=self.material)
        hit_record.set_face_normal(ray, outward_normal)
        
        return hit_record


class Plane(Hittable):
    """Infinite plane shape."""
    
    def __init__(self, point: np.ndarray, normal: np.ndarray, material: Material):
        self.point = point
        self.normal = normal / np.linalg.norm(normal)
        self.material = material
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check ray-plane intersection."""
        denom = np.dot(ray.direction, self.normal)
        
        if abs(denom) < 1e-6:  # Ray is parallel to plane
            return None
        
        t = np.dot(self.point - ray.origin, self.normal) / denom
        
        if t < t_min or t > t_max:
            return None
        
        point = ray.at(t)
        hit_record = HitRecord(point=point, normal=self.normal, t=t, front_face=True, material=self.material)
        hit_record.set_face_normal(ray, self.normal)
        
        return hit_record


class Triangle(Hittable):
    """Triangle shape."""
    
    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, material: Material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        
        # Precompute triangle properties
        self.edge1 = v1 - v0
        self.edge2 = v2 - v0
        self.normal = np.cross(self.edge1, self.edge2)
        self.normal = self.normal / np.linalg.norm(self.normal)
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check ray-triangle intersection using Möller–Trumbore algorithm."""
        h = np.cross(ray.direction, self.edge2)
        a = np.dot(self.edge1, h)
        
        if abs(a) < 1e-6:  # Ray is parallel to triangle
            return None
        
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return None
        
        q = np.cross(s, self.edge1)
        v = f * np.dot(ray.direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return None
        
        t = f * np.dot(self.edge2, q)
        
        if t < t_min or t > t_max:
            return None
        
        point = ray.at(t)
        hit_record = HitRecord(point=point, normal=self.normal, t=t, front_face=True, material=self.material)
        hit_record.set_face_normal(ray, self.normal)
        
        return hit_record


class HittableList(Hittable):
    """List of hittable objects."""
    
    def __init__(self):
        self.objects: List[Hittable] = []
    
    def add(self, obj: Hittable):
        """Add object to list."""
        self.objects.append(obj)
    
    def clear(self):
        """Clear all objects."""
        self.objects.clear()
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check ray intersection with all objects."""
        hit_anything = None
        closest_so_far = t_max
        
        for obj in self.objects:
            hit_record = obj.hit(ray, t_min, closest_so_far)
            if hit_record:
                closest_so_far = hit_record.t
                hit_anything = hit_record
        
        return hit_anything


class Camera:
    """Camera for ray generation."""
    
    def __init__(self, look_from: np.ndarray, look_at: np.ndarray, up: np.ndarray,
                 vfov: float, aspect_ratio: float, aperture: float = 0.0, focus_dist: float = 1.0):
        self.look_from = look_from
        self.look_at = look_at
        self.up = up
        self.vfov = vfov
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        self.focus_dist = focus_dist
        
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup camera coordinate system."""
        self.origin = self.look_from
        self.w = (self.look_from - self.look_at) / np.linalg.norm(self.look_from - self.look_at)
        self.u = np.cross(self.up, self.w) / np.linalg.norm(np.cross(self.up, self.w))
        self.v = np.cross(self.w, self.u)
        
        # Viewport dimensions
        theta = math.radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = viewport_height * (self.aspect_ratio)
        
        # Viewport vectors
        self.viewport_u = viewport_width * self.u
        self.viewport_v = viewport_height * (-self.v)
        
        # Store viewport dimensions for later use
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Camera basis vectors
        self.u = self.u
        self.v = self.v
        self.w = self.w
    
    def set_image_dimensions(self, width: int, height: int):
        """Set image dimensions and recalculate camera parameters."""
        self.image_width = width
        self.image_height = height
        
        # Calculate pixel deltas
        self.pixel_delta_u = self.viewport_u / self.image_width
        self.pixel_delta_v = self.viewport_v / self.image_height
        
        # Viewport upper left corner
        viewport_upper_left = self.origin - (self.focus_dist * self.w) - self.viewport_u / 2 - self.viewport_v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
    
    def get_ray(self, i: int, j: int) -> Ray:
        """Get ray for pixel (i, j)."""
        pixel_center = self.pixel00_loc + (i * self.pixel_delta_u) + (j * self.pixel_delta_v)
        pixel_sample = pixel_center + self._pixel_sample_square()
        
        ray_origin = self.origin if self.aperture <= 0 else self._defocus_disk_sample()
        ray_direction = pixel_sample - ray_origin
        
        return Ray(ray_origin, ray_direction)
    
    def _pixel_sample_square(self) -> np.ndarray:
        """Get random sample within pixel square."""
        px = -0.5 + np.random.random()
        py = -0.5 + np.random.random()
        return (px * self.pixel_delta_u) + (py * self.pixel_delta_v)
    
    def _defocus_disk_sample(self) -> np.ndarray:
        """Get random sample within defocus disk."""
        p = self._random_in_unit_disk()
        return self.origin + (p[0] * self.u) + (p[1] * self.v)
    
    def _random_in_unit_disk(self) -> np.ndarray:
        """Generate random point in unit disk."""
        while True:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1:
                return p


class RayTracer:
    """Basic ray tracer."""
    
    def __init__(self, world: HittableList, camera: Camera):
        self.world = world
        self.camera = camera
        self.max_depth = 50
        self.samples_per_pixel = 100
    
    def ray_color(self, ray: Ray, depth: int) -> np.ndarray:
        """Calculate color for a ray."""
        if depth <= 0:
            return np.array([0.0, 0.0, 0.0])
        
        hit_record = self.world.hit(ray, 0.001, float('inf'))
        
        if hit_record:
            scattered, scattered_ray, attenuation = hit_record.material.scatter(ray, hit_record)
            
            if scattered:
                return attenuation * self.ray_color(scattered_ray, depth - 1)
            else:
                return hit_record.material.emission
        
        # Background gradient
        unit_direction = ray.direction / np.linalg.norm(ray.direction)
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    
    def render(self, width: int, height: int) -> np.ndarray:
        """Render the scene."""
        self.camera.set_image_dimensions(width, height)
        
        image = np.zeros((height, width, 3))
        
        for j in range(height):
            for i in range(width):
                pixel_color = np.array([0.0, 0.0, 0.0])
                
                for _ in range(self.samples_per_pixel):
                    ray = self.camera.get_ray(i, j)
                    pixel_color += self.ray_color(ray, self.max_depth)
                
                # Average and gamma correct
                pixel_color /= self.samples_per_pixel
                pixel_color = np.sqrt(pixel_color)  # Gamma correction
                
                image[j, i] = np.clip(pixel_color, 0.0, 1.0)
        
        return image


def demonstrate_ray_tracing():
    """Demonstrate ray tracing functionality."""
    print("=== Simple Ray Tracing and Path Tracing - Ray Tracing ===\n")

    # Create materials
    print("1. Creating materials...")
    
    lambertian_red = LambertianMaterial(np.array([0.7, 0.3, 0.3]))
    lambertian_blue = LambertianMaterial(np.array([0.3, 0.3, 0.7]))
    metal_gold = MetalMaterial(np.array([0.8, 0.6, 0.2]), roughness=0.1)
    metal_silver = MetalMaterial(np.array([0.8, 0.8, 0.8]), roughness=0.3)
    glass = DielectricMaterial(1.5)
    light = EmissiveMaterial(np.array([1.0, 1.0, 0.8]))
    
    print("   - Lambertian materials (red, blue)")
    print("   - Metal materials (gold, silver)")
    print("   - Dielectric material (glass)")
    print("   - Emissive material (light)")

    # Create world
    print("\n2. Creating scene objects...")
    
    world = HittableList()
    
    # Add spheres
    world.add(Sphere(np.array([0.0, 0.0, -1.0]), 0.5, lambertian_red))
    world.add(Sphere(np.array([0.0, -100.5, -1.0]), 100, lambertian_blue))
    world.add(Sphere(np.array([1.0, 0.0, -1.0]), 0.5, metal_gold))
    world.add(Sphere(np.array([-1.0, 0.0, -1.0]), 0.5, glass))
    world.add(Sphere(np.array([0.0, 1.0, -1.0]), 0.3, light))
    
    print("   - Ground plane (large sphere)")
    print("   - Red diffuse sphere")
    print("   - Gold metal sphere")
    print("   - Glass sphere")
    print("   - Light sphere")

    # Create camera
    print("\n3. Setting up camera...")
    
    camera = Camera(
        look_from=np.array([0.0, 1.0, 3.0]),
        look_at=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, 1.0, 0.0]),
        vfov=60.0,
        aspect_ratio=16.0 / 9.0,
        aperture=0.1,
        focus_dist=3.0
    )
    
    print("   - Position: (0, 1, 3)")
    print("   - Looking at: (0, 0, -1)")
    print("   - FOV: 60 degrees")
    print("   - Aperture: 0.1 (depth of field)")

    # Create ray tracer
    print("\n4. Creating ray tracer...")
    
    tracer = RayTracer(world, camera)
    tracer.max_depth = 10
    tracer.samples_per_pixel = 50
    
    print(f"   - Max ray depth: {tracer.max_depth}")
    print(f"   - Samples per pixel: {tracer.samples_per_pixel}")

    # Test ray generation
    print("\n5. Testing ray generation...")
    
    camera.set_image_dimensions(800, 600)
    test_ray = camera.get_ray(400, 300)
    print(f"   - Test ray origin: {test_ray.origin}")
    print(f"   - Test ray direction: {test_ray.direction}")

    # Test intersection
    print("\n6. Testing object intersection...")
    
    test_sphere = Sphere(np.array([0.0, 0.0, -1.0]), 0.5, lambertian_red)
    test_ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    
    hit = test_sphere.hit(test_ray, 0.0, 10.0)
    if hit:
        print(f"   - Sphere hit at t={hit.t:.3f}")
        print(f"   - Hit point: {hit.point}")
        print(f"   - Normal: {hit.normal}")
    else:
        print("   - No intersection")

    # Test material scattering
    print("\n7. Testing material scattering...")
    
    if hit:
        scattered, scattered_ray, attenuation = hit.material.scatter(test_ray, hit)
        print(f"   - Material scattered: {scattered}")
        if scattered:
            print(f"   - Scattered ray direction: {scattered_ray.direction}")
            print(f"   - Attenuation: {attenuation}")

    print("\n8. Features demonstrated:")
    print("   - Ray generation and intersection")
    print("   - Multiple material types (Lambertian, Metal, Dielectric, Emissive)")
    print("   - Sphere, plane, and triangle shapes")
    print("   - Camera with depth of field")
    print("   - Basic ray tracing pipeline")
    print("   - Material scattering and reflection")


if __name__ == "__main__":
    demonstrate_ray_tracing()
