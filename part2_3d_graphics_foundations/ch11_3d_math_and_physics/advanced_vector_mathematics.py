#!/usr/bin/env python3
"""
Chapter 11: 3D Math and Physics
Advanced Vector Mathematics

Demonstrates advanced vector operations, geometric algorithms, spatial queries,
and mathematical optimization techniques for 3D graphics.
"""

import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Advanced Vector Mathematics"
__description__ = "Advanced vector operations and geometric algorithms"

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
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def distance_to(self, other: 'Vector3D') -> float:
        return (self - other).magnitude()
    
    def angle_to(self, other: 'Vector3D') -> float:
        dot_product = self.dot(other)
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot_product / mag_product))
        return math.acos(cos_angle)
    
    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        other_mag_sq = other.magnitude_squared()
        if other_mag_sq == 0:
            return Vector3D(0, 0, 0)
        projection_length = self.dot(other) / other_mag_sq
        return other * projection_length
    
    def reflect(self, normal: 'Vector3D') -> 'Vector3D':
        normal = normal.normalize()
        return self - normal * (2 * self.dot(normal))
    
    def rotate_around_axis(self, axis: 'Vector3D', angle: float) -> 'Vector3D':
        axis = axis.normalize()
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        return (self * cos_angle + 
                axis.cross(self) * sin_angle + 
                axis * (axis.dot(self) * (1 - cos_angle)))
    
    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        t = max(0.0, min(1.0, t))
        return self + (other - self) * t
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class Ray:
    """Ray with origin and direction"""
    origin: Vector3D
    direction: Vector3D
    
    def __post_init__(self):
        self.direction = self.direction.normalize()
    
    def point_at(self, t: float) -> Vector3D:
        return self.origin + self.direction * t

@dataclass
class Sphere:
    """Sphere defined by center and radius"""
    center: Vector3D
    radius: float
    
    def contains_point(self, point: Vector3D) -> bool:
        return point.distance_squared_to(self.center) <= self.radius**2

# ============================================================================
# GEOMETRIC ALGORITHMS
# ============================================================================

class GeometricAlgorithms:
    """Collection of geometric algorithms"""
    
    @staticmethod
    def ray_sphere_intersection(ray: Ray, sphere: Sphere) -> Optional[Tuple[float, float]]:
        """Find intersection points between ray and sphere"""
        oc = ray.origin - sphere.center
        
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - sphere.radius**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        return (t1, t2)
    
    @staticmethod
    def point_line_distance(point: Vector3D, line_start: Vector3D, line_end: Vector3D) -> float:
        """Calculate distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_sq = line_vec.magnitude_squared()
        if line_length_sq == 0:
            return point_vec.magnitude()
        
        t = point_vec.dot(line_vec) / line_length_sq
        t = max(0.0, min(1.0, t))
        
        closest_point = line_start + line_vec * t
        return point.distance_to(closest_point)
    
    @staticmethod
    def triangle_area(v1: Vector3D, v2: Vector3D, v3: Vector3D) -> float:
        """Calculate area of triangle formed by three points"""
        edge1 = v2 - v1
        edge2 = v3 - v1
        cross_product = edge1.cross(edge2)
        return cross_product.magnitude() / 2.0
    
    @staticmethod
    def triangle_normal(v1: Vector3D, v2: Vector3D, v3: Vector3D) -> Vector3D:
        """Calculate normal vector of triangle"""
        edge1 = v2 - v1
        edge2 = v3 - v1
        return edge1.cross(edge2).normalize()

# ============================================================================
# SPATIAL QUERIES
# ============================================================================

class SpatialQueries:
    """Spatial query algorithms"""
    
    @staticmethod
    def find_nearest_point(query_point: Vector3D, points: List[Vector3D]) -> Tuple[Vector3D, float]:
        """Find nearest point in list to query point"""
        if not points:
            raise ValueError("Points list cannot be empty")
        
        nearest_point = points[0]
        min_distance = query_point.distance_squared_to(nearest_point)
        
        for point in points[1:]:
            distance_sq = query_point.distance_squared_to(point)
            if distance_sq < min_distance:
                min_distance = distance_sq
                nearest_point = point
        
        return nearest_point, math.sqrt(min_distance)
    
    @staticmethod
    def find_points_in_sphere(query_point: Vector3D, radius: float, points: List[Vector3D]) -> List[Vector3D]:
        """Find all points within radius of query point"""
        radius_sq = radius**2
        result = []
        
        for point in points:
            if query_point.distance_squared_to(point) <= radius_sq:
                result.append(point)
        
        return result

# ============================================================================
# MATHEMATICAL OPTIMIZATION
# ============================================================================

class MathematicalOptimization:
    """Mathematical optimization techniques"""
    
    @staticmethod
    def fast_distance_squared(v1: Vector3D, v2: Vector3D) -> float:
        """Fast squared distance calculation (avoids square root)"""
        dx = v1.x - v2.x
        dy = v1.y - v2.y
        dz = v1.z - v2.z
        return dx*dx + dy*dy + dz*dz
    
    @staticmethod
    def fast_normalize(v: Vector3D) -> Vector3D:
        """Fast vector normalization"""
        mag_sq = v.magnitude_squared()
        if mag_sq == 0:
            return Vector3D(0, 0, 0)
        
        inv_mag = 1.0 / math.sqrt(mag_sq)
        return Vector3D(v.x * inv_mag, v.y * inv_mag, v.z * inv_mag)
    
    @staticmethod
    def smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Smoothstep function for smooth interpolation"""
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_advanced_vector_operations():
    """Demonstrate advanced vector operations"""
    print("=== Advanced Vector Operations ===\n")
    
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    v3 = Vector3D(1, 1, 0)
    
    print("Vector operations:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  v3 = {v3}")
    
    print(f"\nAdvanced operations:")
    print(f"  Angle between v1 and v2: {math.degrees(v1.angle_to(v2)):.1f}°")
    print(f"  Projection of v3 onto v1: {v3.project_onto(v1)}")
    print(f"  Reflection of v3 off v1: {v3.reflect(v1)}")
    
    axis = Vector3D(0, 0, 1)
    angle = math.pi / 2
    rotated = v1.rotate_around_axis(axis, angle)
    print(f"  v1 rotated 90° around Z-axis: {rotated}")
    
    print(f"\nInterpolation:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lerp_result = v1.lerp(v2, t)
        print(f"  t={t:.2f}: LERP={lerp_result}")
    
    print()

def demonstrate_geometric_algorithms():
    """Demonstrate geometric algorithms"""
    print("=== Geometric Algorithms ===\n")
    
    ray = Ray(Vector3D(0, 0, 0), Vector3D(1, 0, 0))
    sphere = Sphere(Vector3D(5, 0, 0), 2.0)
    
    intersection = GeometricAlgorithms.ray_sphere_intersection(ray, sphere)
    if intersection:
        t1, t2 = intersection
        point1 = ray.point_at(t1)
        point2 = ray.point_at(t2)
        print(f"Ray-sphere intersection:")
        print(f"  Ray: {ray}")
        print(f"  Sphere: {sphere}")
        print(f"  Intersection points: {point1}, {point2}")
    
    point = Vector3D(2, 2, 0)
    line_start = Vector3D(0, 0, 0)
    line_end = Vector3D(4, 0, 0)
    distance = GeometricAlgorithms.point_line_distance(point, line_start, line_end)
    print(f"\nPoint-line distance:")
    print(f"  Point: {point}")
    print(f"  Line: {line_start} to {line_end}")
    print(f"  Distance: {distance:.3f}")
    
    v1 = Vector3D(0, 0, 0)
    v2 = Vector3D(1, 0, 0)
    v3 = Vector3D(0, 1, 0)
    
    area = GeometricAlgorithms.triangle_area(v1, v2, v3)
    normal = GeometricAlgorithms.triangle_normal(v1, v2, v3)
    print(f"\nTriangle operations:")
    print(f"  Triangle: {v1}, {v2}, {v3}")
    print(f"  Area: {area:.3f}")
    print(f"  Normal: {normal}")
    
    print()

def demonstrate_spatial_queries():
    """Demonstrate spatial queries"""
    print("=== Spatial Queries ===\n")
    
    random.seed(42)
    points = []
    for i in range(20):
        points.append(Vector3D(
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        ))
    
    print(f"Generated {len(points)} random points")
    
    query_point = Vector3D(0, 0, 0)
    nearest, distance = SpatialQueries.find_nearest_point(query_point, points)
    print(f"\nNearest point to {query_point}:")
    print(f"  Point: {nearest}")
    print(f"  Distance: {distance:.3f}")
    
    radius = 5.0
    sphere_points = SpatialQueries.find_points_in_sphere(query_point, radius, points)
    print(f"\nPoints within radius {radius} of {query_point}:")
    print(f"  Found {len(sphere_points)} points")
    
    print()

def demonstrate_mathematical_optimization():
    """Demonstrate mathematical optimization"""
    print("=== Mathematical Optimization ===\n")
    
    v1 = Vector3D(3, 4, 5)
    v2 = Vector3D(1, 2, 3)
    
    print(f"Vector operations comparison:")
    print(f"  Vector: {v1}")
    
    exact_mag = v1.magnitude()
    fast_norm = MathematicalOptimization.fast_normalize(v1)
    print(f"  Exact magnitude: {exact_mag:.6f}")
    print(f"  Fast normalize: {fast_norm}")
    
    print(f"\nInterpolation functions:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        smooth = MathematicalOptimization.smoothstep(0.0, 1.0, t) * 10.0
        print(f"  t={t:.2f}: Smoothstep={smooth:.2f}")
    
    print()

def demonstrate_performance_comparison():
    """Demonstrate performance comparison"""
    print("=== Performance Comparison ===\n")
    
    v1 = Vector3D(1.0, 2.0, 3.0)
    v2 = Vector3D(4.0, 5.0, 6.0)
    num_iterations = 100000
    
    # Test magnitude calculation
    start_time = time.time()
    for _ in range(num_iterations):
        mag = v1.magnitude()
    magnitude_time = time.time() - start_time
    
    # Test fast distance squared
    start_time = time.time()
    for _ in range(num_iterations):
        dist_sq = MathematicalOptimization.fast_distance_squared(v1, v2)
    distance_sq_time = time.time() - start_time
    
    print(f"Operations per second (higher is better):")
    print(f"  Magnitude: {num_iterations/magnitude_time:.0f}")
    print(f"  Distance squared: {num_iterations/distance_sq_time:.0f}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate advanced vector mathematics"""
    print("=== Advanced Vector Mathematics Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    demonstrate_advanced_vector_operations()
    demonstrate_geometric_algorithms()
    demonstrate_spatial_queries()
    demonstrate_mathematical_optimization()
    demonstrate_performance_comparison()
    
    print("="*60)
    print("Advanced Vector Mathematics demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Advanced vector operations (projection, reflection, rotation)")
    print("✓ Geometric algorithms (intersections, distances, areas)")
    print("✓ Spatial queries (nearest neighbor, range queries)")
    print("✓ Mathematical optimization (fast approximations)")
    print("✓ Performance benchmarking and analysis")
    
    print("\nApplications:")
    print("• Computer graphics: ray tracing, collision detection")
    print("• Game development: physics, AI, spatial queries")
    print("• Scientific visualization: geometric algorithms")
    print("• Robotics: path planning, obstacle avoidance")
    print("• Virtual reality: spatial tracking and interaction")

if __name__ == "__main__":
    main()
