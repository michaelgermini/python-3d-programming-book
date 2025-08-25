"""
Chapter 20: Basic Lighting Models - Lighting Optimization
======================================================

This module demonstrates lighting optimization techniques for real-time rendering.

Key Concepts:
- Light culling and frustum culling
- Light clustering and tiled lighting
- Light LOD and distance-based optimization
- Batch lighting calculations
"""

import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from vector_operations import Vector3D
from lighting_models import Light, PointLight, SpotLight, Material, LightingCalculator


@dataclass
class LightCluster:
    """A cluster of lights for optimization."""
    bounds_min: Vector3D
    bounds_max: Vector3D
    lights: List[Light]
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if cluster contains a point."""
        return (self.bounds_min.x <= point.x <= self.bounds_max.x and
                self.bounds_min.y <= point.y <= self.bounds_max.y and
                self.bounds_min.z <= point.z <= self.bounds_max.z)
    
    def get_center(self) -> Vector3D:
        """Get center of cluster bounds."""
        return (self.bounds_min + self.bounds_max) * 0.5
    
    def get_radius(self) -> float:
        """Get radius of cluster bounds."""
        size = self.bounds_max - self.bounds_min
        return size.magnitude() * 0.5


class LightCuller:
    """Culls lights based on visibility and distance."""
    
    def __init__(self, max_lights_per_cluster: int = 8):
        self.max_lights_per_cluster = max_lights_per_cluster
    
    def cull_lights_by_distance(self, point: Vector3D, lights: List[Light], 
                               max_distance: float) -> List[Light]:
        """Cull lights beyond maximum distance."""
        visible_lights = []
        
        for light in lights:
            if isinstance(light, PointLight) or isinstance(light, SpotLight):
                distance = (light.position - point).magnitude()
                if distance <= max_distance:
                    visible_lights.append(light)
            else:
                # Directional lights are always visible
                visible_lights.append(light)
        
        return visible_lights
    
    def cull_lights_by_frustum(self, frustum_planes: List[Tuple[Vector3D, float]], 
                              lights: List[Light]) -> List[Light]:
        """Cull lights outside view frustum."""
        visible_lights = []
        
        for light in lights:
            if isinstance(light, PointLight) or isinstance(light, SpotLight):
                # Check if light sphere intersects frustum
                if self._sphere_intersects_frustum(light.position, light.range, frustum_planes):
                    visible_lights.append(light)
            else:
                # Directional lights are always visible
                visible_lights.append(light)
        
        return visible_lights
    
    def _sphere_intersects_frustum(self, center: Vector3D, radius: float,
                                 frustum_planes: List[Tuple[Vector3D, float]]) -> bool:
        """Check if sphere intersects frustum."""
        for normal, distance in frustum_planes:
            if center.dot(normal) + distance < -radius:
                return False
        return True


class LightClustering:
    """Organizes lights into spatial clusters for efficient lookup."""
    
    def __init__(self, world_bounds_min: Vector3D, world_bounds_max: Vector3D,
                 cluster_size: Vector3D):
        self.world_bounds_min = world_bounds_min
        self.world_bounds_max = world_bounds_max
        self.cluster_size = cluster_size
        self.clusters: Dict[Tuple[int, int, int], LightCluster] = {}
        
        # Calculate grid dimensions
        world_size = world_bounds_max - world_bounds_min
        self.grid_dimensions = (
            int(math.ceil(world_size.x / cluster_size.x)),
            int(math.ceil(world_size.y / cluster_size.y)),
            int(math.ceil(world_size.z / cluster_size.z))
        )
    
    def get_cluster_key(self, position: Vector3D) -> Tuple[int, int, int]:
        """Get cluster key for a position."""
        relative_pos = position - self.world_bounds_min
        x = int(relative_pos.x / self.cluster_size.x)
        y = int(relative_pos.y / self.cluster_size.y)
        z = int(relative_pos.z / self.cluster_size.z)
        
        # Clamp to grid bounds
        x = max(0, min(x, self.grid_dimensions[0] - 1))
        y = max(0, min(y, self.grid_dimensions[1] - 1))
        z = max(0, min(z, self.grid_dimensions[2] - 1))
        
        return (x, y, z)
    
    def add_light(self, light: Light):
        """Add a light to appropriate clusters."""
        if isinstance(light, PointLight) or isinstance(light, SpotLight):
            # Calculate affected clusters based on light range
            affected_clusters = self._get_affected_clusters(light.position, light.range)
            
            for cluster_key in affected_clusters:
                if cluster_key not in self.clusters:
                    self._create_cluster(cluster_key)
                
                cluster = self.clusters[cluster_key]
                if len(cluster.lights) < 8:  # Max lights per cluster
                    cluster.lights.append(light)
    
    def _get_affected_clusters(self, position: Vector3D, radius: float) -> List[Tuple[int, int, int]]:
        """Get all clusters affected by a light."""
        affected = []
        
        # Calculate bounds of affected area
        min_pos = position - Vector3D(radius, radius, radius)
        max_pos = position + Vector3D(radius, radius, radius)
        
        # Get cluster keys for bounds
        min_key = self.get_cluster_key(min_pos)
        max_key = self.get_cluster_key(max_pos)
        
        # Add all clusters in range
        for x in range(min_key[0], max_key[0] + 1):
            for y in range(min_key[1], max_key[1] + 1):
                for z in range(min_key[2], max_key[2] + 1):
                    affected.append((x, y, z))
        
        return affected
    
    def _create_cluster(self, key: Tuple[int, int, int]):
        """Create a new cluster."""
        min_pos = Vector3D(
            self.world_bounds_min.x + key[0] * self.cluster_size.x,
            self.world_bounds_min.y + key[1] * self.cluster_size.y,
            self.world_bounds_min.z + key[2] * self.cluster_size.z
        )
        max_pos = min_pos + self.cluster_size
        
        self.clusters[key] = LightCluster(min_pos, max_pos, [])
    
    def get_lights_for_point(self, point: Vector3D) -> List[Light]:
        """Get lights affecting a specific point."""
        cluster_key = self.get_cluster_key(point)
        
        if cluster_key in self.clusters:
            return self.clusters[cluster_key].lights
        return []


class LightLOD:
    """Level-of-detail system for lights."""
    
    def __init__(self):
        self.lod_levels: Dict[str, List[Tuple[float, Light]]] = {}
    
    def add_light_lod(self, light_name: str, distance: float, light: Light):
        """Add a LOD level for a light."""
        if light_name not in self.lod_levels:
            self.lod_levels[light_name] = []
        
        self.lod_levels[light_name].append((distance, light))
        # Sort by distance (closest first)
        self.lod_levels[light_name].sort(key=lambda x: x[0])
    
    def get_appropriate_light(self, light_name: str, distance: float) -> Optional[Light]:
        """Get appropriate LOD level for a given distance."""
        if light_name not in self.lod_levels:
            return None
        
        levels = self.lod_levels[light_name]
        
        # Find the highest quality light within range
        for lod_distance, light in reversed(levels):
            if distance <= lod_distance:
                return light
        
        return None


class BatchLightingCalculator:
    """Calculates lighting for multiple points efficiently."""
    
    def __init__(self, calculator: LightingCalculator):
        self.calculator = calculator
        self.light_culler = LightCuller()
        self.light_clustering = None
        self.light_lod = LightLOD()
    
    def set_light_clustering(self, clustering: LightClustering):
        """Set light clustering for optimization."""
        self.light_clustering = clustering
    
    def calculate_batch_lighting(self, points: List[Vector3D], normals: List[Vector3D],
                               material: Material, view_direction: Vector3D,
                               max_lights_per_point: int = 4) -> List[Vector3D]:
        """Calculate lighting for multiple points efficiently."""
        results = []
        
        for i, point in enumerate(points):
            # Get relevant lights for this point
            if self.light_clustering:
                lights = self.light_clustering.get_lights_for_point(point)
            else:
                lights = self.calculator.lights
            
            # Cull lights by distance
            lights = self.light_culler.cull_lights_by_distance(point, lights, 50.0)
            
            # Limit number of lights per point
            if len(lights) > max_lights_per_point:
                # Sort by distance and take closest
                lights.sort(key=lambda l: (l.position - point).magnitude())
                lights = lights[:max_lights_per_point]
            
            # Calculate lighting with subset of lights
            lighting = self._calculate_lighting_with_lights(
                point, normals[i], material, view_direction, lights
            )
            results.append(lighting)
        
        return results
    
    def _calculate_lighting_with_lights(self, point: Vector3D, normal: Vector3D,
                                      material: Material, view_direction: Vector3D,
                                      lights: List[Light]) -> Vector3D:
        """Calculate lighting using a specific set of lights."""
        # Start with ambient and emission
        result = material.emission + self.calculator.ambient_light * material.ambient
        
        for light in lights:
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


def demonstrate_lighting_optimization():
    """Demonstrate lighting optimization techniques."""
    print("=== Lighting Optimization Demonstration ===\n")
    
    # Create lighting calculator
    from lighting_models import DirectionalLight, PointLight, Material
    
    calculator = LightingCalculator()
    directional_light = DirectionalLight(
        direction=Vector3D(1, 1, 1),
        color=Vector3D(1, 1, 1),
        intensity=0.8
    )
    calculator.add_light(directional_light)
    
    # Create point lights
    point_lights = []
    for i in range(10):
        light = PointLight(
            position=Vector3D((i - 5) * 2, 0, 0),
            color=Vector3D(1, 0.8, 0.6),
            intensity=1.0
        )
        point_lights.append(light)
        calculator.add_light(light)
    
    # Create material
    material = Material(
        ambient=Vector3D(0.2, 0.2, 0.2),
        diffuse=Vector3D(0.8, 0.8, 0.8),
        specular=Vector3D(1.0, 1.0, 1.0)
    )
    
    print("1. Light Culling:")
    culler = LightCuller()
    test_point = Vector3D(0, 0, 0)
    
    visible_lights = culler.cull_lights_by_distance(test_point, point_lights, 10.0)
    print(f"Lights within 10 units: {len(visible_lights)}")
    
    visible_lights = culler.cull_lights_by_distance(test_point, point_lights, 5.0)
    print(f"Lights within 5 units: {len(visible_lights)}")
    
    print("\n2. Light Clustering:")
    clustering = LightClustering(
        Vector3D(-20, -20, -20),
        Vector3D(20, 20, 20),
        Vector3D(5, 5, 5)
    )
    
    for light in point_lights:
        clustering.add_light(light)
    
    print(f"Total clusters created: {len(clustering.clusters)}")
    
    # Test cluster lookup
    test_points = [
        Vector3D(0, 0, 0),
        Vector3D(5, 0, 0),
        Vector3D(-5, 0, 0)
    ]
    
    for point in test_points:
        lights = clustering.get_lights_for_point(point)
        print(f"Point {point}: {len(lights)} lights")
    
    print("\n3. Batch Lighting Calculation:")
    batch_calculator = BatchLightingCalculator(calculator)
    batch_calculator.set_light_clustering(clustering)
    
    points = [Vector3D(0, 0, 0), Vector3D(2, 0, 0), Vector3D(-2, 0, 0)]
    normals = [Vector3D(0, 1, 0)] * len(points)
    view_direction = Vector3D(0, 0, -1).normalized()
    
    batch_results = batch_calculator.calculate_batch_lighting(
        points, normals, material, view_direction
    )
    
    for i, result in enumerate(batch_results):
        print(f"Point {i} lighting: {result}")
    
    print("\n4. Light LOD:")
    lod_system = LightLOD()
    
    # Create LOD levels for a light
    base_light = PointLight(Vector3D(0, 0, 0), Vector3D(1, 1, 1), 1.0)
    medium_light = PointLight(Vector3D(0, 0, 0), Vector3D(1, 1, 1), 0.7)
    low_light = PointLight(Vector3D(0, 0, 0), Vector3D(1, 1, 1), 0.4)
    
    lod_system.add_light_lod("test_light", 5.0, base_light)
    lod_system.add_light_lod("test_light", 15.0, medium_light)
    lod_system.add_light_lod("test_light", 30.0, low_light)
    
    test_distances = [2.0, 10.0, 20.0, 40.0]
    for distance in test_distances:
        light = lod_system.get_appropriate_light("test_light", distance)
        intensity = light.intensity if light else 0.0
        print(f"Distance {distance}: light intensity = {intensity}")


if __name__ == "__main__":
    demonstrate_lighting_optimization()
