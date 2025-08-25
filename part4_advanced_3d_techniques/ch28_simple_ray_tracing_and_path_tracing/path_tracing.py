"""
Chapter 28: Simple Ray Tracing and Path Tracing - Path Tracing
=============================================================

This module demonstrates path tracing implementation with global illumination.

Key Concepts:
- Path tracing fundamentals and Monte Carlo integration
- Global illumination and indirect lighting
- BRDF models and material systems
- Importance sampling and Russian roulette
- Convergence analysis and denoising
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math
import random


class BRDFType(Enum):
    """BRDF type enumeration."""
    LAMBERT = "lambert"
    PHONG = "phong"
    COOK_TORRANCE = "cook_torrance"
    GGX = "ggx"


@dataclass
class BRDF:
    """Bidirectional Reflectance Distribution Function."""
    brdf_type: BRDFType
    albedo: np.ndarray
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5
    
    def __post_init__(self):
        if self.albedo is None:
            self.albedo = np.array([0.5, 0.5, 0.5])


class LambertBRDF(BRDF):
    """Lambertian BRDF (perfectly diffuse reflection)."""
    
    def __init__(self, albedo: np.ndarray):
        super().__init__(BRDFType.LAMBERT, albedo=albedo)
    
    def evaluate(self, wi: np.ndarray, wo: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Evaluate Lambert BRDF."""
        return self.albedo / np.pi
    
    def sample(self, normal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sample Lambert BRDF direction."""
        # Cosine-weighted hemisphere sampling
        u1 = random.random()
        u2 = random.random()
        
        z = math.sqrt(1.0 - u1)
        phi = 2.0 * math.pi * u2
        x = math.cos(phi) * math.sqrt(u1)
        y = math.sin(phi) * math.sqrt(u1)
        
        # Transform to normal space
        tangent = self._get_tangent(normal)
        bitangent = np.cross(normal, tangent)
        
        direction = x * tangent + y * bitangent + z * normal
        pdf = z / math.pi  # Cosine-weighted PDF
        
        return direction, pdf
    
    def _get_tangent(self, normal: np.ndarray) -> np.ndarray:
        """Get tangent vector for normal."""
        if abs(normal[0]) > abs(normal[1]):
            tangent = np.array([normal[2], 0, -normal[0]])
        else:
            tangent = np.array([0, -normal[2], normal[1]])
        return tangent / np.linalg.norm(tangent)


class PhongBRDF(BRDF):
    """Phong BRDF (specular reflection)."""
    
    def __init__(self, albedo: np.ndarray, shininess: float = 32.0):
        super().__init__(BRDFType.PHONG, albedo=albedo)
        self.shininess = shininess
    
    def evaluate(self, wi: np.ndarray, wo: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Evaluate Phong BRDF."""
        # Calculate reflection vector
        r = 2.0 * np.dot(wi, normal) * normal - wi
        cos_alpha = max(0.0, np.dot(wo, r))
        
        specular = math.pow(cos_alpha, self.shininess)
        return self.albedo * specular
    
    def sample(self, normal: np.ndarray, wi: np.ndarray) -> Tuple[np.ndarray, float]:
        """Sample Phong BRDF direction."""
        # Sample around reflection direction
        r = 2.0 * np.dot(wi, normal) * normal - wi
        
        # Spherical coordinates around reflection
        u1 = random.random()
        u2 = random.random()
        
        cos_theta = math.pow(u1, 1.0 / (self.shininess + 1.0))
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        phi = 2.0 * math.pi * u2
        
        # Local coordinate system around reflection
        tangent = self._get_tangent(r)
        bitangent = np.cross(r, tangent)
        
        # Sample direction
        direction = (sin_theta * math.cos(phi) * tangent + 
                    sin_theta * math.sin(phi) * bitangent + 
                    cos_theta * r)
        
        pdf = (self.shininess + 1.0) * math.pow(cos_theta, self.shininess) / (2.0 * math.pi)
        
        return direction, pdf
    
    def _get_tangent(self, normal: np.ndarray) -> np.ndarray:
        """Get tangent vector for normal."""
        if abs(normal[0]) > abs(normal[1]):
            tangent = np.array([normal[2], 0, -normal[0]])
        else:
            tangent = np.array([0, -normal[2], normal[1]])
        return tangent / np.linalg.norm(tangent)


class CookTorranceBRDF(BRDF):
    """Cook-Torrance BRDF (physically based)."""
    
    def __init__(self, albedo: np.ndarray, roughness: float = 0.5, metallic: float = 0.0):
        super().__init__(BRDFType.COOK_TORRANCE, albedo=albedo, roughness=roughness, metallic=metallic)
    
    def evaluate(self, wi: np.ndarray, wo: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Evaluate Cook-Torrance BRDF."""
        half = (wi + wo) / np.linalg.norm(wi + wo)
        
        # Distribution function (D)
        alpha = self.roughness * self.roughness
        cos_theta_h = np.dot(normal, half)
        d = self._distribution_ggx(cos_theta_h, alpha)
        
        # Geometry function (G)
        g = self._geometry_smith(np.dot(normal, wi), np.dot(normal, wo), alpha)
        
        # Fresnel function (F)
        f0 = np.array([0.04, 0.04, 0.04]) * (1.0 - self.metallic) + self.albedo * self.metallic
        f = self._fresnel_schlick(np.dot(wi, half), f0)
        
        # BRDF
        numerator = d * g * f
        denominator = 4.0 * np.dot(normal, wi) * np.dot(normal, wo) + 0.0001
        
        specular = numerator / denominator
        
        # Diffuse component
        kd = (1.0 - self.metallic) * (1.0 - f)
        diffuse = kd * self.albedo / math.pi
        
        return diffuse + specular
    
    def _distribution_ggx(self, cos_theta_h: float, alpha: float) -> float:
        """GGX distribution function."""
        denom = cos_theta_h * cos_theta_h * (alpha * alpha - 1.0) + 1.0
        return alpha * alpha / (math.pi * denom * denom)
    
    def _geometry_smith(self, cos_theta_i: float, cos_theta_o: float, alpha: float) -> float:
        """Smith geometry function."""
        def geometry_schlick_ggx(cos_theta: float, k: float) -> float:
            return cos_theta / (cos_theta * (1.0 - k) + k)
        
        k = alpha * alpha / 2.0
        return geometry_schlick_ggx(cos_theta_i, k) * geometry_schlick_ggx(cos_theta_o, k)
    
    def _fresnel_schlick(self, cos_theta: float, f0: np.ndarray) -> np.ndarray:
        """Schlick Fresnel approximation."""
        return f0 + (1.0 - f0) * math.pow(1.0 - cos_theta, 5.0)


class PathTracer:
    """Path tracer with global illumination."""
    
    def __init__(self, world, camera):
        self.world = world
        self.camera = camera
        self.max_depth = 10
        self.samples_per_pixel = 100
        self.russian_roulette_threshold = 3
    
    def trace_path(self, ray: 'Ray', depth: int = 0) -> np.ndarray:
        """Trace a path and return the color."""
        if depth >= self.max_depth:
            return np.array([0.0, 0.0, 0.0])
        
        hit_record = self.world.hit(ray, 0.001, float('inf'))
        
        if not hit_record:
            return self._background_color(ray)
        
        # Direct lighting
        direct_light = self._direct_lighting(hit_record)
        
        # Indirect lighting (recursive)
        indirect_light = self._indirect_lighting(ray, hit_record, depth)
        
        return direct_light + indirect_light
    
    def _direct_lighting(self, hit_record: 'HitRecord') -> np.ndarray:
        """Calculate direct lighting contribution."""
        # Sample light sources
        light_contribution = np.array([0.0, 0.0, 0.0])
        
        # For simplicity, assume one light source
        light_pos = np.array([0.0, 5.0, 0.0])
        light_intensity = np.array([1.0, 1.0, 1.0])
        
        # Sample light direction
        light_dir = light_pos - hit_record.point
        light_distance = np.linalg.norm(light_dir)
        light_dir = light_dir / light_distance
        
        # Check visibility
        shadow_ray = Ray(hit_record.point + 0.001 * hit_record.normal, light_dir)
        shadow_hit = self.world.hit(shadow_ray, 0.001, light_distance)
        
        if not shadow_hit:
            # Calculate BRDF
            if hasattr(hit_record.material, 'brdf'):
                brdf_value = hit_record.material.brdf.evaluate(-shadow_ray.direction, -shadow_ray.direction, hit_record.normal)
            else:
                brdf_value = hit_record.material.albedo / math.pi
            
            # Cosine term
            cos_theta = max(0.0, np.dot(hit_record.normal, light_dir))
            
            # Attenuation
            attenuation = 1.0 / (light_distance * light_distance)
            
            light_contribution = light_intensity * brdf_value * cos_theta * attenuation
        
        return light_contribution
    
    def _indirect_lighting(self, ray: 'Ray', hit_record: 'HitRecord', depth: int) -> np.ndarray:
        """Calculate indirect lighting contribution."""
        # Russian roulette
        if depth > self.russian_roulette_threshold:
            survival_prob = min(1.0, np.max(hit_record.material.albedo))
            if random.random() > survival_prob:
                return np.array([0.0, 0.0, 0.0])
        
        # Sample BRDF
        if hasattr(hit_record.material, 'brdf'):
            brdf = hit_record.material.brdf
            wi = -ray.direction
            wo, pdf = brdf.sample(hit_record.normal, wi)
            
            if pdf > 0.0:
                scattered_ray = Ray(hit_record.point + 0.001 * hit_record.normal, wo)
                indirect_color = self.trace_path(scattered_ray, depth + 1)
                
                brdf_value = brdf.evaluate(wi, wo, hit_record.normal)
                cos_theta = max(0.0, np.dot(hit_record.normal, wo))
                
                return brdf_value * indirect_color * cos_theta / pdf
        
        # Fallback to simple material scattering
        scattered, scattered_ray, attenuation = hit_record.material.scatter(ray, hit_record)
        if scattered:
            return attenuation * self.trace_path(scattered_ray, depth + 1)
        
        return np.array([0.0, 0.0, 0.0])
    
    def _background_color(self, ray: 'Ray') -> np.ndarray:
        """Calculate background color."""
        unit_direction = ray.direction / np.linalg.norm(ray.direction)
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    
    def render(self, width: int, height: int) -> np.ndarray:
        """Render the scene using path tracing."""
        self.camera.set_image_dimensions(width, height)
        
        image = np.zeros((height, width, 3))
        
        for j in range(height):
            for i in range(width):
                pixel_color = np.array([0.0, 0.0, 0.0])
                
                for _ in range(self.samples_per_pixel):
                    ray = self.camera.get_ray(i, j)
                    pixel_color += self.trace_path(ray)
                
                # Average and gamma correct
                pixel_color /= self.samples_per_pixel
                pixel_color = np.sqrt(pixel_color)  # Gamma correction
                
                image[j, i] = np.clip(pixel_color, 0.0, 1.0)
        
        return image


class GlobalIllumination:
    """Global illumination system."""
    
    def __init__(self, world, camera):
        self.world = world
        self.camera = camera
        self.path_tracer = PathTracer(world, camera)
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def render_with_gi(self, width: int, height: int, max_samples: int = 1000) -> np.ndarray:
        """Render with global illumination and convergence analysis."""
        image = np.zeros((height, width, 3))
        sample_count = 0
        
        while sample_count < max_samples:
            # Progressive rendering
            for j in range(height):
                for i in range(width):
                    ray = self.camera.get_ray(i, j)
                    sample_color = self.path_tracer.trace_path(ray)
                    
                    # Accumulate samples
                    image[j, i] = (image[j, i] * sample_count + sample_color) / (sample_count + 1)
            
            sample_count += 1
            
            # Check convergence
            if sample_count % 10 == 0:
                convergence = self.convergence_analyzer.analyze_convergence(image, sample_count)
                if convergence < 0.01:  # Convergence threshold
                    break
        
        # Final gamma correction
        image = np.sqrt(image)
        return np.clip(image, 0.0, 1.0)


class ConvergenceAnalyzer:
    """Analyze convergence of path tracing."""
    
    def __init__(self):
        self.previous_image = None
    
    def analyze_convergence(self, current_image: np.ndarray, sample_count: int) -> float:
        """Analyze convergence between current and previous image."""
        if self.previous_image is None:
            self.previous_image = current_image.copy()
            return float('inf')
        
        # Calculate mean squared error
        mse = np.mean((current_image - self.previous_image) ** 2)
        self.previous_image = current_image.copy()
        
        return mse
    
    def estimate_remaining_samples(self, current_convergence: float, target_convergence: float) -> int:
        """Estimate remaining samples needed for target convergence."""
        if current_convergence <= target_convergence:
            return 0
        
        # Simple linear estimation
        return int(current_convergence / target_convergence * 10)


class Denoiser:
    """Simple denoiser for path traced images."""
    
    def __init__(self):
        self.kernel_size = 3
        self.sigma = 1.0
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply simple Gaussian denoising."""
        from scipy.ndimage import gaussian_filter
        
        denoised = np.zeros_like(image)
        
        for channel in range(3):
            denoised[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=self.sigma)
        
        return denoised
    
    def bilateral_denoise(self, image: np.ndarray, color_sigma: float = 0.1, spatial_sigma: float = 1.0) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving denoising."""
        # Simplified bilateral filter implementation
        denoised = image.copy()
        height, width = image.shape[:2]
        
        for i in range(self.kernel_size//2, height - self.kernel_size//2):
            for j in range(self.kernel_size//2, width - self.kernel_size//2):
                for c in range(3):
                    # Spatial and color weights
                    total_weight = 0.0
                    weighted_sum = 0.0
                    
                    for di in range(-self.kernel_size//2, self.kernel_size//2 + 1):
                        for dj in range(-self.kernel_size//2, self.kernel_size//2 + 1):
                            # Spatial weight
                            spatial_dist = math.sqrt(di*di + dj*dj)
                            spatial_weight = math.exp(-spatial_dist / (2 * spatial_sigma * spatial_sigma))
                            
                            # Color weight
                            color_diff = abs(image[i, j, c] - image[i+di, j+dj, c])
                            color_weight = math.exp(-color_diff / (2 * color_sigma * color_sigma))
                            
                            weight = spatial_weight * color_weight
                            weighted_sum += weight * image[i+di, j+dj, c]
                            total_weight += weight
                    
                    if total_weight > 0:
                        denoised[i, j, c] = weighted_sum / total_weight
        
        return denoised


def demonstrate_path_tracing():
    """Demonstrate path tracing functionality."""
    print("=== Simple Ray Tracing and Path Tracing - Path Tracing ===\n")

    # Create BRDFs
    print("1. Creating BRDF models...")
    
    lambert_brdf = LambertBRDF(np.array([0.7, 0.3, 0.3]))
    phong_brdf = PhongBRDF(np.array([0.8, 0.8, 0.8]), shininess=64.0)
    cook_torrance_brdf = CookTorranceBRDF(np.array([0.8, 0.6, 0.2]), roughness=0.1, metallic=1.0)
    
    print("   - Lambert BRDF (diffuse reflection)")
    print("   - Phong BRDF (specular reflection)")
    print("   - Cook-Torrance BRDF (physically based)")

    # Test BRDF evaluation
    print("\n2. Testing BRDF evaluation...")
    
    normal = np.array([0.0, 1.0, 0.0])
    wi = np.array([0.0, 1.0, 0.0])
    wo = np.array([0.0, 1.0, 0.0])
    
    lambert_value = lambert_brdf.evaluate(wi, wo, normal)
    phong_value = phong_brdf.evaluate(wi, wo, normal)
    cook_torrance_value = cook_torrance_brdf.evaluate(wi, wo, normal)
    
    print(f"   - Lambert BRDF value: {lambert_value}")
    print(f"   - Phong BRDF value: {phong_value}")
    print(f"   - Cook-Torrance BRDF value: {cook_torrance_value}")

    # Test BRDF sampling
    print("\n3. Testing BRDF sampling...")
    
    lambert_dir, lambert_pdf = lambert_brdf.sample(normal)
    phong_dir, phong_pdf = phong_brdf.sample(normal, wi)
    
    print(f"   - Lambert sampled direction: {lambert_dir}")
    print(f"   - Lambert PDF: {lambert_pdf:.4f}")
    print(f"   - Phong sampled direction: {phong_dir}")
    print(f"   - Phong PDF: {phong_pdf:.4f}")

    # Create convergence analyzer
    print("\n4. Testing convergence analysis...")
    
    analyzer = ConvergenceAnalyzer()
    
    # Simulate progressive rendering
    test_image = np.random.random((100, 100, 3))
    convergence = analyzer.analyze_convergence(test_image, 1)
    print(f"   - Initial convergence: {convergence:.6f}")
    
    # Simulate more samples
    test_image2 = test_image + np.random.normal(0, 0.1, test_image.shape)
    convergence2 = analyzer.analyze_convergence(test_image2, 2)
    print(f"   - Convergence after 2 samples: {convergence2:.6f}")

    # Create denoiser
    print("\n5. Testing denoising...")
    
    denoiser = Denoiser()
    
    # Create noisy image
    clean_image = np.random.random((50, 50, 3))
    noisy_image = clean_image + np.random.normal(0, 0.1, clean_image.shape)
    
    denoised_image = denoiser.denoise(noisy_image)
    noise_reduction = np.mean(np.abs(noisy_image - clean_image)) - np.mean(np.abs(denoised_image - clean_image))
    
    print(f"   - Noise reduction: {noise_reduction:.4f}")

    print("\n6. Features demonstrated:")
    print("   - Multiple BRDF models (Lambert, Phong, Cook-Torrance)")
    print("   - BRDF evaluation and importance sampling")
    print("   - Global illumination with direct and indirect lighting")
    print("   - Russian roulette for path termination")
    print("   - Progressive rendering with convergence analysis")
    print("   - Image denoising techniques")
    print("   - Monte Carlo integration for lighting")


if __name__ == "__main__":
    demonstrate_path_tracing()
