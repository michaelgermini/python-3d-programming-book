"""
Chapter 20: Basic Lighting Models - Shading Techniques
====================================================

This module demonstrates shading techniques and normal calculations for 3D lighting.

Key Concepts:
- Flat shading and Gouraud shading
- Normal calculation and interpolation
- Per-vertex and per-fragment lighting
- Normal mapping and bump mapping
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from vector_operations import Vector3D
from lighting_models import Material, Light, LightingCalculator


@dataclass
class Vertex:
    """Vertex with position and normal."""
    position: Vector3D
    normal: Vector3D
    color: Vector3D = None
    
    def __post_init__(self):
        """Initialize default color."""
        if self.color is None:
            self.color = Vector3D(1.0, 1.0, 1.0)


@dataclass
class Triangle:
    """Triangle with three vertices."""
    vertices: List[Vertex]
    
    def __post_init__(self):
        """Ensure we have exactly 3 vertices."""
        if len(self.vertices) != 3:
            raise ValueError("Triangle must have exactly 3 vertices")
    
    def calculate_face_normal(self) -> Vector3D:
        """Calculate face normal using cross product."""
        v0 = self.vertices[0].position
        v1 = self.vertices[1].position
        v2 = self.vertices[2].position
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normal = edge1.cross(edge2)
        return normal.normalized()
    
    def calculate_centroid(self) -> Vector3D:
        """Calculate centroid of triangle."""
        v0 = self.vertices[0].position
        v1 = self.vertices[1].position
        v2 = self.vertices[2].position
        
        return (v0 + v1 + v2) * (1.0 / 3.0)


class FlatShading:
    """Flat shading - one color per face."""
    
    def __init__(self, calculator: LightingCalculator):
        self.calculator = calculator
    
    def shade_triangle(self, triangle: Triangle, material: Material,
                      view_direction: Vector3D) -> Vector3D:
        """Shade triangle using flat shading."""
        # Calculate face normal
        face_normal = triangle.calculate_face_normal()
        
        # Calculate lighting at centroid
        centroid = triangle.calculate_centroid()
        
        return self.calculator.calculate_lighting(
            centroid, face_normal, material, view_direction
        )


class GouraudShading:
    """Gouraud shading - interpolate colors from vertices."""
    
    def __init__(self, calculator: LightingCalculator):
        self.calculator = calculator
    
    def shade_triangle(self, triangle: Triangle, material: Material,
                      view_direction: Vector3D) -> List[Vector3D]:
        """Shade triangle using Gouraud shading."""
        vertex_colors = []
        
        # Calculate lighting for each vertex
        for vertex in triangle.vertices:
            color = self.calculator.calculate_lighting(
                vertex.position, vertex.normal, material, view_direction
            )
            vertex_colors.append(color)
        
        return vertex_colors
    
    def interpolate_color(self, barycentric_coords: Vector3D, 
                         vertex_colors: List[Vector3D]) -> Vector3D:
        """Interpolate color using barycentric coordinates."""
        if len(vertex_colors) != 3:
            raise ValueError("Need exactly 3 vertex colors")
        
        interpolated = (vertex_colors[0] * barycentric_coords.x +
                       vertex_colors[1] * barycentric_coords.y +
                       vertex_colors[2] * barycentric_coords.z)
        
        return interpolated


class PhongShading:
    """Phong shading - interpolate normals and calculate per-fragment lighting."""
    
    def __init__(self, calculator: LightingCalculator):
        self.calculator = calculator
    
    def interpolate_normal(self, barycentric_coords: Vector3D, 
                          triangle: Triangle) -> Vector3D:
        """Interpolate normal using barycentric coordinates."""
        interpolated = (triangle.vertices[0].normal * barycentric_coords.x +
                       triangle.vertices[1].normal * barycentric_coords.y +
                       triangle.vertices[2].normal * barycentric_coords.z)
        
        return interpolated.normalized()
    
    def shade_point(self, point: Vector3D, barycentric_coords: Vector3D,
                   triangle: Triangle, material: Material,
                   view_direction: Vector3D) -> Vector3D:
        """Shade a point using Phong shading."""
        # Interpolate normal
        interpolated_normal = self.interpolate_normal(barycentric_coords, triangle)
        
        # Calculate lighting at the point
        return self.calculator.calculate_lighting(
            point, interpolated_normal, material, view_direction
        )


class NormalCalculator:
    """Calculates and manages surface normals."""
    
    @staticmethod
    def calculate_vertex_normals(triangles: List[Triangle]) -> List[Vector3D]:
        """Calculate vertex normals by averaging face normals."""
        vertex_normals = {}
        
        # Initialize vertex normals
        for triangle in triangles:
            for vertex in triangle.vertices:
                pos_key = (vertex.position.x, vertex.position.y, vertex.position.z)
                if pos_key not in vertex_normals:
                    vertex_normals[pos_key] = Vector3D(0, 0, 0)
        
        # Sum face normals for each vertex
        for triangle in triangles:
            face_normal = triangle.calculate_face_normal()
            for vertex in triangle.vertices:
                pos_key = (vertex.position.x, vertex.position.y, vertex.position.z)
                vertex_normals[pos_key] = vertex_normals[pos_key] + face_normal
        
        # Normalize vertex normals
        for pos_key in vertex_normals:
            vertex_normals[pos_key] = vertex_normals[pos_key].normalized()
        
        return vertex_normals
    
    @staticmethod
    def calculate_triangle_normals(triangles: List[Triangle]) -> List[Vector3D]:
        """Calculate face normals for all triangles."""
        return [triangle.calculate_face_normal() for triangle in triangles]


class BarycentricCalculator:
    """Calculates barycentric coordinates for triangle interpolation."""
    
    @staticmethod
    def calculate_barycentric(point: Vector3D, triangle: Triangle) -> Vector3D:
        """Calculate barycentric coordinates of point relative to triangle."""
        v0 = triangle.vertices[0].position
        v1 = triangle.vertices[1].position
        v2 = triangle.vertices[2].position
        
        # Calculate vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        
        # Calculate dot products
        d00 = v0v1.dot(v0v1)
        d01 = v0v1.dot(v0v2)
        d11 = v0v2.dot(v0v2)
        d20 = v0p.dot(v0v1)
        d21 = v0p.dot(v0v2)
        
        # Calculate barycentric coordinates
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-6:
            # Degenerate triangle
            return Vector3D(1.0/3.0, 1.0/3.0, 1.0/3.0)
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return Vector3D(u, v, w)
    
    @staticmethod
    def is_inside_triangle(barycentric: Vector3D) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        return (barycentric.x >= 0 and barycentric.y >= 0 and barycentric.z >= 0 and
                abs(barycentric.x + barycentric.y + barycentric.z - 1.0) < 1e-6)


class ShadingManager:
    """Manages different shading techniques."""
    
    def __init__(self, calculator: LightingCalculator):
        self.calculator = calculator
        self.flat_shader = FlatShading(calculator)
        self.gouraud_shader = GouraudShading(calculator)
        self.phong_shader = PhongShading(calculator)
        self.normal_calculator = NormalCalculator()
        self.barycentric_calculator = BarycentricCalculator()
    
    def shade_triangle_flat(self, triangle: Triangle, material: Material,
                           view_direction: Vector3D) -> Vector3D:
        """Shade triangle using flat shading."""
        return self.flat_shader.shade_triangle(triangle, material, view_direction)
    
    def shade_triangle_gouraud(self, triangle: Triangle, material: Material,
                              view_direction: Vector3D) -> List[Vector3D]:
        """Shade triangle using Gouraud shading."""
        return self.gouraud_shader.shade_triangle(triangle, material, view_direction)
    
    def shade_point_phong(self, point: Vector3D, triangle: Triangle, material: Material,
                          view_direction: Vector3D) -> Vector3D:
        """Shade point using Phong shading."""
        barycentric = self.barycentric_calculator.calculate_barycentric(point, triangle)
        return self.phong_shader.shade_point(point, barycentric, triangle, material, view_direction)
    
    def calculate_vertex_normals(self, triangles: List[Triangle]) -> List[Vector3D]:
        """Calculate vertex normals for a mesh."""
        return self.normal_calculator.calculate_vertex_normals(triangles)


def demonstrate_shading_techniques():
    """Demonstrate shading techniques and normal calculations."""
    print("=== Shading Techniques Demonstration ===\n")
    
    # Create lighting calculator
    from lighting_models import DirectionalLight, Material
    
    calculator = LightingCalculator()
    directional_light = DirectionalLight(
        direction=Vector3D(1, 1, 1),
        color=Vector3D(1, 1, 1),
        intensity=0.8
    )
    calculator.add_light(directional_light)
    
    # Create material
    material = Material(
        ambient=Vector3D(0.2, 0.2, 0.2),
        diffuse=Vector3D(0.8, 0.8, 0.8),
        specular=Vector3D(1.0, 1.0, 1.0)
    )
    
    # Create triangle
    v0 = Vertex(Vector3D(0, 0, 0), Vector3D(0, 1, 0))
    v1 = Vertex(Vector3D(1, 0, 0), Vector3D(0, 1, 0))
    v2 = Vertex(Vector3D(0.5, 1, 0), Vector3D(0, 1, 0))
    
    triangle = Triangle([v0, v1, v2])
    
    # Create shading manager
    shading_manager = ShadingManager(calculator)
    
    # Test different shading techniques
    view_direction = Vector3D(0, 0, -1).normalized()
    
    print("1. Flat Shading:")
    flat_color = shading_manager.shade_triangle_flat(triangle, material, view_direction)
    print(f"Flat shading color: {flat_color}")
    
    print("\n2. Gouraud Shading:")
    gouraud_colors = shading_manager.shade_triangle_gouraud(triangle, material, view_direction)
    for i, color in enumerate(gouraud_colors):
        print(f"Vertex {i} color: {color}")
    
    print("\n3. Phong Shading:")
    test_point = Vector3D(0.5, 0.5, 0)
    phong_color = shading_manager.shade_point_phong(test_point, triangle, material, view_direction)
    print(f"Point {test_point} color: {phong_color}")
    
    print("\n4. Normal Calculations:")
    face_normal = triangle.calculate_face_normal()
    print(f"Face normal: {face_normal}")
    
    centroid = triangle.calculate_centroid()
    print(f"Centroid: {centroid}")
    
    print("\n5. Barycentric Coordinates:")
    barycentric = shading_manager.barycentric_calculator.calculate_barycentric(test_point, triangle)
    print(f"Barycentric coordinates: {barycentric}")
    print(f"Is inside triangle: {shading_manager.barycentric_calculator.is_inside_triangle(barycentric)}")


if __name__ == "__main__":
    demonstrate_shading_techniques()
