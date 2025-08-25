"""
Chapter 17: Camera and Projection Concepts - Projection Systems
==============================================================

This module demonstrates projection systems and matrices for 3D graphics applications.

Key Concepts:
- Perspective projection matrices
- Orthographic projection matrices
- Frustum culling and clipping
- Projection matrix calculations
- Multiple projection types and uses
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from vector_operations import Vector3D
from matrix_operations import Matrix4x4


@dataclass
class Frustum:
    """View frustum for culling and clipping operations."""
    near_plane: float
    far_plane: float
    fov: float
    aspect_ratio: float
    near_width: float = 0.0
    near_height: float = 0.0
    far_width: float = 0.0
    far_height: float = 0.0
    
    def __post_init__(self):
        """Calculate frustum dimensions."""
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate frustum plane dimensions."""
        tan_half_fov = math.tan(self.fov * 0.5)
        
        # Near plane dimensions
        self.near_height = self.near_plane * tan_half_fov
        self.near_width = self.near_height * self.aspect_ratio
        
        # Far plane dimensions
        self.far_height = self.far_plane * tan_half_fov
        self.far_width = self.far_height * self.aspect_ratio
    
    def get_near_corners(self) -> List[Vector3D]:
        """Get near plane corner points."""
        half_width = self.near_width * 0.5
        half_height = self.near_height * 0.5
        
        return [
            Vector3D(-half_width, -half_height, -self.near_plane),
            Vector3D(half_width, -half_height, -self.near_plane),
            Vector3D(half_width, half_height, -self.near_plane),
            Vector3D(-half_width, half_height, -self.near_plane)
        ]
    
    def get_far_corners(self) -> List[Vector3D]:
        """Get far plane corner points."""
        half_width = self.far_width * 0.5
        half_height = self.far_height * 0.5
        
        return [
            Vector3D(-half_width, -half_height, -self.far_plane),
            Vector3D(half_width, -half_height, -self.far_plane),
            Vector3D(half_width, half_height, -self.far_plane),
            Vector3D(-half_width, half_height, -self.far_plane)
        ]
    
    def get_all_corners(self) -> List[Vector3D]:
        """Get all frustum corner points."""
        return self.get_near_corners() + self.get_far_corners()


class ProjectionMatrix:
    """Base class for projection matrix calculations."""
    
    @staticmethod
    def perspective(fov: float, aspect_ratio: float, near: float, far: float) -> Matrix4x4:
        """Create perspective projection matrix."""
        f = 1.0 / math.tan(fov * 0.5)
        
        return Matrix4x4([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])
    
    @staticmethod
    def perspective_infinite_far(fov: float, aspect_ratio: float, near: float) -> Matrix4x4:
        """Create perspective projection matrix with infinite far plane."""
        f = 1.0 / math.tan(fov * 0.5)
        
        return Matrix4x4([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -1, -2 * near],
            [0, 0, -1, 0]
        ])
    
    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float, 
                    near: float, far: float) -> Matrix4x4:
        """Create orthographic projection matrix."""
        width = right - left
        height = top - bottom
        depth = far - near
        
        return Matrix4x4([
            [2 / width, 0, 0, -(right + left) / width],
            [0, 2 / height, 0, -(top + bottom) / height],
            [0, 0, -2 / depth, -(far + near) / depth],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def orthographic_centered(width: float, height: float, near: float, far: float) -> Matrix4x4:
        """Create centered orthographic projection matrix."""
        half_width = width * 0.5
        half_height = height * 0.5
        
        return ProjectionMatrix.orthographic(
            -half_width, half_width,
            -half_height, half_height,
            near, far
        )
    
    @staticmethod
    def frustum(left: float, right: float, bottom: float, top: float, 
                near: float, far: float) -> Matrix4x4:
        """Create frustum projection matrix."""
        width = right - left
        height = top - bottom
        depth = far - near
        
        return Matrix4x4([
            [2 * near / width, 0, (right + left) / width, 0],
            [0, 2 * near / height, (top + bottom) / height, 0],
            [0, 0, -(far + near) / depth, -2 * far * near / depth],
            [0, 0, -1, 0]
        ])


class ProjectionSystem:
    """Comprehensive projection system with multiple projection types."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.fov = math.pi / 4  # 45 degrees
        self.near_plane = 0.1
        self.far_plane = 1000.0
        
        # Projection matrices
        self.perspective_matrix = None
        self.orthographic_matrix = None
        self.frustum_matrix = None
        
        # Update matrices
        self._update_matrices()
    
    def _update_matrices(self):
        """Update all projection matrices."""
        # Perspective projection
        self.perspective_matrix = ProjectionMatrix.perspective(
            self.fov, self.aspect_ratio, self.near_plane, self.far_plane
        )
        
        # Orthographic projection
        self.orthographic_matrix = ProjectionMatrix.orthographic_centered(
            self.width, self.height, self.near_plane, self.far_plane
        )
        
        # Frustum projection
        tan_half_fov = math.tan(self.fov * 0.5)
        near_height = self.near_plane * tan_half_fov
        near_width = near_height * self.aspect_ratio
        
        self.frustum_matrix = ProjectionMatrix.frustum(
            -near_width, near_width,
            -near_height, near_height,
            self.near_plane, self.far_plane
        )
    
    def set_fov(self, fov: float):
        """Set field of view."""
        self.fov = max(0.1, min(math.pi - 0.1, fov))
        self._update_matrices()
    
    def set_aspect_ratio(self, width: int, height: int):
        """Set aspect ratio from window dimensions."""
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self._update_matrices()
    
    def set_clipping_planes(self, near: float, far: float):
        """Set near and far clipping planes."""
        self.near_plane = near
        self.far_plane = far
        self._update_matrices()
    
    def get_perspective_matrix(self) -> Matrix4x4:
        """Get perspective projection matrix."""
        return self.perspective_matrix
    
    def get_orthographic_matrix(self) -> Matrix4x4:
        """Get orthographic projection matrix."""
        return self.orthographic_matrix
    
    def get_frustum_matrix(self) -> Matrix4x4:
        """Get frustum projection matrix."""
        return self.frustum_matrix
    
    def get_frustum(self) -> Frustum:
        """Get view frustum."""
        return Frustum(
            self.near_plane, self.far_plane,
            self.fov, self.aspect_ratio
        )


class FrustumCuller:
    """Frustum culling system for performance optimization."""
    
    def __init__(self, frustum: Frustum):
        self.frustum = frustum
        self.planes = self._calculate_planes()
    
    def _calculate_planes(self) -> List[Tuple[Vector3D, float]]:
        """Calculate frustum planes in world space."""
        # This is a simplified implementation
        # In practice, you'd transform the frustum corners and calculate planes
        planes = []
        
        # Near plane
        planes.append((Vector3D(0, 0, -1), -self.frustum.near_plane))
        
        # Far plane
        planes.append((Vector3D(0, 0, 1), self.frustum.far_plane))
        
        # Left plane
        left_normal = Vector3D(1, 0, 0)
        planes.append((left_normal, 0))
        
        # Right plane
        right_normal = Vector3D(-1, 0, 0)
        planes.append((right_normal, 0))
        
        # Bottom plane
        bottom_normal = Vector3D(0, 1, 0)
        planes.append((bottom_normal, 0))
        
        # Top plane
        top_normal = Vector3D(0, -1, 0)
        planes.append((top_normal, 0))
        
        return planes
    
    def is_point_visible(self, point: Vector3D) -> bool:
        """Check if a point is visible within the frustum."""
        for normal, distance in self.planes:
            if point.dot(normal) + distance < 0:
                return False
        return True
    
    def is_sphere_visible(self, center: Vector3D, radius: float) -> bool:
        """Check if a sphere is visible within the frustum."""
        for normal, distance in self.planes:
            if center.dot(normal) + distance < -radius:
                return False
        return True
    
    def is_box_visible(self, min_point: Vector3D, max_point: Vector3D) -> bool:
        """Check if an axis-aligned bounding box is visible."""
        for normal, distance in self.planes:
            # Find the most negative vertex
            test_point = Vector3D(
                min_point.x if normal.x >= 0 else max_point.x,
                min_point.y if normal.y >= 0 else max_point.y,
                min_point.z if normal.z >= 0 else max_point.z
            )
            
            if test_point.dot(normal) + distance < 0:
                return False
        return True


class ProjectionAnalyzer:
    """Analyzer for projection matrices and their properties."""
    
    @staticmethod
    def analyze_perspective_matrix(matrix: Matrix4x4) -> Dict[str, Any]:
        """Analyze a perspective projection matrix."""
        # Extract parameters from matrix
        fov_y = 2 * math.atan(1 / matrix.data[1][1])
        aspect_ratio = matrix.data[1][1] / matrix.data[0][0]
        
        # Calculate near and far planes
        near_plane = matrix.data[2][3] / (matrix.data[2][2] - 1)
        far_plane = matrix.data[2][3] / (matrix.data[2][2] + 1)
        
        return {
            "fov_y": fov_y,
            "fov_y_degrees": math.degrees(fov_y),
            "aspect_ratio": aspect_ratio,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "projection_type": "perspective"
        }
    
    @staticmethod
    def analyze_orthographic_matrix(matrix: Matrix4x4) -> Dict[str, Any]:
        """Analyze an orthographic projection matrix."""
        # Extract parameters from matrix
        width = 2 / matrix.data[0][0]
        height = 2 / matrix.data[1][1]
        depth = 2 / abs(matrix.data[2][2])
        
        near_plane = matrix.data[2][3] / matrix.data[2][2] + depth / 2
        far_plane = matrix.data[2][3] / matrix.data[2][2] - depth / 2
        
        return {
            "width": width,
            "height": height,
            "depth": depth,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "aspect_ratio": width / height,
            "projection_type": "orthographic"
        }
    
    @staticmethod
    def compare_projections(perspective: Matrix4x4, orthographic: Matrix4x4) -> Dict[str, Any]:
        """Compare perspective and orthographic projections."""
        persp_analysis = ProjectionAnalyzer.analyze_perspective_matrix(perspective)
        ortho_analysis = ProjectionAnalyzer.analyze_orthographic_matrix(orthographic)
        
        return {
            "perspective": persp_analysis,
            "orthographic": ortho_analysis,
            "comparison": {
                "perspective_fov": persp_analysis["fov_y_degrees"],
                "orthographic_size": ortho_analysis["width"],
                "near_plane_diff": abs(persp_analysis["near_plane"] - ortho_analysis["near_plane"]),
                "far_plane_diff": abs(persp_analysis["far_plane"] - ortho_analysis["far_plane"])
            }
        }


def demonstrate_projection_systems():
    """Demonstrate various projection systems and matrices."""
    print("=== Projection Systems Demonstration ===\n")
    
    # Create projection system
    print("1. Projection System:")
    projection_system = ProjectionSystem(1920, 1080)
    print(f"Aspect ratio: {projection_system.aspect_ratio:.3f}")
    print(f"FOV: {math.degrees(projection_system.fov):.1f}°")
    print(f"Near plane: {projection_system.near_plane}")
    print(f"Far plane: {projection_system.far_plane}")
    print()
    
    # Different projection matrices
    print("2. Projection Matrices:")
    perspective = projection_system.get_perspective_matrix()
    orthographic = projection_system.get_orthographic_matrix()
    frustum = projection_system.get_frustum_matrix()
    
    print("Perspective Matrix:")
    for row in perspective.data:
        print(f"  {row}")
    print()
    
    print("Orthographic Matrix:")
    for row in orthographic.data:
        print(f"  {row}")
    print()
    
    # Frustum analysis
    print("3. Frustum Analysis:")
    frustum_obj = projection_system.get_frustum()
    print(f"Near plane: {frustum_obj.near_width:.3f} x {frustum_obj.near_height:.3f}")
    print(f"Far plane: {frustum_obj.far_width:.3f} x {frustum_obj.far_height:.3f}")
    
    near_corners = frustum_obj.get_near_corners()
    far_corners = frustum_obj.get_far_corners()
    print(f"Near corners: {len(near_corners)} points")
    print(f"Far corners: {len(far_corners)} points")
    print()
    
    # Frustum culling
    print("4. Frustum Culling:")
    culler = FrustumCuller(frustum_obj)
    
    # Test points
    test_points = [
        Vector3D(0, 0, -1),  # Inside
        Vector3D(0, 0, -10),  # Inside
        Vector3D(100, 100, -1),  # Outside
        Vector3D(0, 0, 1),  # Behind camera
    ]
    
    for i, point in enumerate(test_points):
        visible = culler.is_point_visible(point)
        print(f"Point {i+1} {point}: {'Visible' if visible else 'Not visible'}")
    
    # Test sphere
    sphere_center = Vector3D(0, 0, -5)
    sphere_radius = 1.0
    sphere_visible = culler.is_sphere_visible(sphere_center, sphere_radius)
    print(f"Sphere at {sphere_center} (r={sphere_radius}): {'Visible' if sphere_visible else 'Not visible'}")
    print()
    
    # Projection analysis
    print("5. Projection Analysis:")
    analyzer = ProjectionAnalyzer()
    
    persp_analysis = analyzer.analyze_perspective_matrix(perspective)
    ortho_analysis = analyzer.analyze_orthographic_matrix(orthographic)
    
    print("Perspective Analysis:")
    for key, value in persp_analysis.items():
        print(f"  {key}: {value}")
    
    print("\nOrthographic Analysis:")
    for key, value in ortho_analysis.items():
        print(f"  {key}: {value}")
    
    # Comparison
    comparison = analyzer.compare_projections(perspective, orthographic)
    print("\nComparison:")
    for key, value in comparison["comparison"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demonstrate_projection_systems()
