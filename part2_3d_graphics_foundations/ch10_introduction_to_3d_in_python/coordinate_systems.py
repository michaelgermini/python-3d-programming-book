#!/usr/bin/env python3
"""
Chapter 10: Introduction to 3D in Python
Coordinate Systems

Demonstrates different coordinate systems (Cartesian, spherical, cylindrical)
and how to work with them in 3D graphics, including coordinate transformations
and conversions.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Coordinate Systems"
__description__ = "Working with different coordinate systems in 3D graphics"

# ============================================================================
# COORDINATE SYSTEM CLASSES
# ============================================================================

@dataclass
class Cartesian3D:
    """3D Cartesian coordinates (x, y, z)"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Cartesian3D') -> 'Cartesian3D':
        return Cartesian3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Cartesian3D') -> 'Cartesian3D':
        return Cartesian3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Cartesian3D':
        return Cartesian3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Cartesian3D':
        mag = self.magnitude()
        if mag == 0:
            return Cartesian3D(0, 0, 0)
        return Cartesian3D(self.x / mag, self.y / mag, self.z / mag)
    
    def distance_to(self, other: 'Cartesian3D') -> float:
        """Calculate distance to another point"""
        return (self - other).magnitude()
    
    def to_spherical(self) -> 'Spherical3D':
        """Convert to spherical coordinates"""
        r = self.magnitude()
        if r == 0:
            return Spherical3D(0, 0, 0)
        
        theta = math.atan2(self.y, self.x)  # Azimuthal angle (longitude)
        phi = math.acos(self.z / r)  # Polar angle (latitude)
        
        return Spherical3D(r, theta, phi)
    
    def to_cylindrical(self) -> 'Cylindrical3D':
        """Convert to cylindrical coordinates"""
        rho = math.sqrt(self.x**2 + self.y**2)  # Radial distance
        phi = math.atan2(self.y, self.x)  # Azimuthal angle
        z = self.z  # Height
        
        return Cylindrical3D(rho, phi, z)
    
    def __str__(self):
        return f"Cartesian3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class Spherical3D:
    """3D spherical coordinates (r, theta, phi)"""
    r: float      # Radial distance from origin
    theta: float  # Azimuthal angle (longitude) in radians
    phi: float    # Polar angle (latitude) in radians
    
    def to_cartesian(self) -> Cartesian3D:
        """Convert to Cartesian coordinates"""
        x = self.r * math.sin(self.phi) * math.cos(self.theta)
        y = self.r * math.sin(self.phi) * math.sin(self.theta)
        z = self.r * math.cos(self.phi)
        
        return Cartesian3D(x, y, z)
    
    def to_cylindrical(self) -> 'Cylindrical3D':
        """Convert to cylindrical coordinates"""
        rho = self.r * math.sin(self.phi)
        phi = self.theta
        z = self.r * math.cos(self.phi)
        
        return Cylindrical3D(rho, phi, z)
    
    def __str__(self):
        return f"Spherical3D(r={self.r:.3f}, θ={math.degrees(self.theta):.1f}°, φ={math.degrees(self.phi):.1f}°)"

@dataclass
class Cylindrical3D:
    """3D cylindrical coordinates (rho, phi, z)"""
    rho: float    # Radial distance from z-axis
    phi: float    # Azimuthal angle in radians
    z: float      # Height along z-axis
    
    def to_cartesian(self) -> Cartesian3D:
        """Convert to Cartesian coordinates"""
        x = self.rho * math.cos(self.phi)
        y = self.rho * math.sin(self.phi)
        z = self.z
        
        return Cartesian3D(x, y, z)
    
    def to_spherical(self) -> Spherical3D:
        """Convert to spherical coordinates"""
        r = math.sqrt(self.rho**2 + self.z**2)
        theta = self.phi
        phi = math.atan2(self.rho, self.z)
        
        return Spherical3D(r, theta, phi)
    
    def __str__(self):
        return f"Cylindrical3D(ρ={self.rho:.3f}, φ={math.degrees(self.phi):.1f}°, z={self.z:.3f})"

# ============================================================================
# COORDINATE SYSTEM UTILITIES
# ============================================================================

class CoordinateConverter:
    """Utility class for coordinate system conversions"""
    
    @staticmethod
    def cartesian_to_spherical(cart: Cartesian3D) -> Spherical3D:
        """Convert Cartesian to spherical coordinates"""
        return cart.to_spherical()
    
    @staticmethod
    def cartesian_to_cylindrical(cart: Cartesian3D) -> Cylindrical3D:
        """Convert Cartesian to cylindrical coordinates"""
        return cart.to_cylindrical()
    
    @staticmethod
    def spherical_to_cartesian(sph: Spherical3D) -> Cartesian3D:
        """Convert spherical to Cartesian coordinates"""
        return sph.to_cartesian()
    
    @staticmethod
    def spherical_to_cylindrical(sph: Spherical3D) -> Cylindrical3D:
        """Convert spherical to cylindrical coordinates"""
        return sph.to_cylindrical()
    
    @staticmethod
    def cylindrical_to_cartesian(cyl: Cylindrical3D) -> Cartesian3D:
        """Convert cylindrical to Cartesian coordinates"""
        return cyl.to_cartesian()
    
    @staticmethod
    def cylindrical_to_spherical(cyl: Cylindrical3D) -> Spherical3D:
        """Convert cylindrical to spherical coordinates"""
        return cyl.to_spherical()
    
    @staticmethod
    def round_trip_test(original: Cartesian3D) -> Dict[str, float]:
        """Test coordinate conversion round trips"""
        # Cartesian -> Spherical -> Cartesian
        sph = original.to_spherical()
        cart_from_sph = sph.to_cartesian()
        error_sph = original.distance_to(cart_from_sph)
        
        # Cartesian -> Cylindrical -> Cartesian
        cyl = original.to_cylindrical()
        cart_from_cyl = cyl.to_cartesian()
        error_cyl = original.distance_to(cart_from_cyl)
        
        # Spherical -> Cylindrical -> Spherical
        cyl_from_sph = sph.to_cylindrical()
        sph_from_cyl = cyl_from_sph.to_spherical()
        error_sph_cyl = abs(sph.r - sph_from_cyl.r) + abs(sph.theta - sph_from_cyl.theta) + abs(sph.phi - sph_from_cyl.phi)
        
        return {
            'cartesian_spherical_error': error_sph,
            'cartesian_cylindrical_error': error_cyl,
            'spherical_cylindrical_error': error_sph_cyl
        }

class CoordinateSystemExamples:
    """Examples of different coordinate systems and their applications"""
    
    @staticmethod
    def create_unit_sphere_points(num_points: int = 100) -> List[Cartesian3D]:
        """Create points on a unit sphere using spherical coordinates"""
        points = []
        for i in range(num_points):
            # Generate random spherical coordinates
            theta = random.uniform(0, 2 * math.pi)  # Longitude
            phi = math.acos(2 * random.random() - 1)  # Latitude (uniform distribution)
            r = 1.0  # Unit sphere
            
            # Convert to Cartesian
            sph = Spherical3D(r, theta, phi)
            cart = sph.to_cartesian()
            points.append(cart)
        
        return points
    
    @staticmethod
    def create_cylinder_points(radius: float = 1.0, height: float = 2.0, num_points: int = 100) -> List[Cartesian3D]:
        """Create points on a cylinder using cylindrical coordinates"""
        points = []
        for i in range(num_points):
            # Generate random cylindrical coordinates
            phi = random.uniform(0, 2 * math.pi)  # Angle around cylinder
            z = random.uniform(-height/2, height/2)  # Height
            rho = radius  # Fixed radius
            
            # Convert to Cartesian
            cyl = Cylindrical3D(rho, phi, z)
            cart = cyl.to_cartesian()
            points.append(cart)
        
        return points
    
    @staticmethod
    def create_spiral_points(radius: float = 1.0, height: float = 4.0, turns: int = 3, num_points: int = 100) -> List[Cartesian3D]:
        """Create points along a spiral using cylindrical coordinates"""
        points = []
        for i in range(num_points):
            # Generate spiral coordinates
            t = i / (num_points - 1)  # Parameter from 0 to 1
            phi = 2 * math.pi * turns * t  # Multiple turns
            z = height * (t - 0.5)  # Height from -height/2 to height/2
            rho = radius  # Fixed radius
            
            # Convert to Cartesian
            cyl = Cylindrical3D(rho, phi, z)
            cart = cyl.to_cartesian()
            points.append(cart)
        
        return points

# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

class CoordinateTransform:
    """Coordinate transformation utilities"""
    
    def __init__(self):
        self.origin = Cartesian3D(0, 0, 0)
        self.scale = Cartesian3D(1, 1, 1)
        self.rotation = Cartesian3D(0, 0, 0)  # Euler angles
    
    def set_origin(self, origin: Cartesian3D):
        """Set the origin of the coordinate system"""
        self.origin = origin
    
    def set_scale(self, scale: Cartesian3D):
        """Set the scale factors"""
        self.scale = scale
    
    def set_rotation(self, rotation: Cartesian3D):
        """Set the rotation angles (Euler angles in radians)"""
        self.rotation = rotation
    
    def transform_point(self, point: Cartesian3D) -> Cartesian3D:
        """Transform a point using the current transformation"""
        # Apply scale
        scaled = Cartesian3D(
            point.x * self.scale.x,
            point.y * self.scale.y,
            point.z * self.scale.z
        )
        
        # Apply rotation (simplified - only around Y axis)
        rotated = Cartesian3D(
            scaled.x * math.cos(self.rotation.y) - scaled.z * math.sin(self.rotation.y),
            scaled.y,
            scaled.x * math.sin(self.rotation.y) + scaled.z * math.cos(self.rotation.y)
        )
        
        # Apply translation
        transformed = rotated + self.origin
        
        return transformed
    
    def transform_points(self, points: List[Cartesian3D]) -> List[Cartesian3D]:
        """Transform a list of points"""
        return [self.transform_point(point) for point in points]

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_coordinate_systems():
    """Demonstrate different coordinate systems"""
    print("=== Coordinate Systems Demonstration ===\n")
    
    # Create some test points
    test_points = [
        Cartesian3D(1, 0, 0),      # Unit vector along X
        Cartesian3D(0, 1, 0),      # Unit vector along Y
        Cartesian3D(0, 0, 1),      # Unit vector along Z
        Cartesian3D(1, 1, 1),      # Diagonal point
        Cartesian3D(2, 3, 4),      # Arbitrary point
    ]
    
    print("1. Coordinate System Conversions:")
    print("-" * 50)
    
    for i, point in enumerate(test_points):
        print(f"\nPoint {i+1}: {point}")
        
        # Convert to spherical
        spherical = point.to_spherical()
        print(f"  Spherical: {spherical}")
        
        # Convert to cylindrical
        cylindrical = point.to_cylindrical()
        print(f"  Cylindrical: {cylindrical}")
        
        # Round trip test
        errors = CoordinateConverter.round_trip_test(point)
        print(f"  Round trip errors:")
        print(f"    Cartesian ↔ Spherical: {errors['cartesian_spherical_error']:.6f}")
        print(f"    Cartesian ↔ Cylindrical: {errors['cartesian_cylindrical_error']:.6f}")
        print(f"    Spherical ↔ Cylindrical: {errors['spherical_cylindrical_error']:.6f}")
    
    print()

def demonstrate_coordinate_applications():
    """Demonstrate practical applications of coordinate systems"""
    print("2. Coordinate System Applications:")
    print("-" * 50)
    
    # Unit sphere points
    print("\nUnit Sphere Points (using spherical coordinates):")
    sphere_points = CoordinateSystemExamples.create_unit_sphere_points(10)
    for i, point in enumerate(sphere_points):
        spherical = point.to_spherical()
        print(f"  Point {i+1}: {point} → {spherical}")
    
    # Cylinder points
    print("\nCylinder Points (using cylindrical coordinates):")
    cylinder_points = CoordinateSystemExamples.create_cylinder_points(radius=1.0, height=2.0, num_points=8)
    for i, point in enumerate(cylinder_points):
        cylindrical = point.to_cylindrical()
        print(f"  Point {i+1}: {point} → {cylindrical}")
    
    # Spiral points
    print("\nSpiral Points (using cylindrical coordinates):")
    spiral_points = CoordinateSystemExamples.create_spiral_points(radius=1.0, height=4.0, turns=2, num_points=8)
    for i, point in enumerate(spiral_points):
        cylindrical = point.to_cylindrical()
        print(f"  Point {i+1}: {point} → {cylindrical}")
    
    print()

def demonstrate_coordinate_transformations():
    """Demonstrate coordinate transformations"""
    print("3. Coordinate Transformations:")
    print("-" * 50)
    
    # Create a transform
    transform = CoordinateTransform()
    
    # Test points
    test_points = [
        Cartesian3D(1, 0, 0),
        Cartesian3D(0, 1, 0),
        Cartesian3D(0, 0, 1),
        Cartesian3D(1, 1, 1)
    ]
    
    print("\nOriginal points:")
    for point in test_points:
        print(f"  {point}")
    
    # Apply translation
    print("\nAfter translation (origin = (2, 3, 4)):")
    transform.set_origin(Cartesian3D(2, 3, 4))
    translated_points = transform.transform_points(test_points)
    for point in translated_points:
        print(f"  {point}")
    
    # Apply scaling
    print("\nAfter scaling (scale = (2, 2, 2)):")
    transform.set_scale(Cartesian3D(2, 2, 2))
    scaled_points = transform.transform_points(test_points)
    for point in scaled_points:
        print(f"  {point}")
    
    # Apply rotation
    print("\nAfter rotation (45° around Y-axis):")
    transform.set_rotation(Cartesian3D(0, math.pi/4, 0))
    rotated_points = transform.transform_points(test_points)
    for point in rotated_points:
        print(f"  {point}")
    
    print()

def demonstrate_coordinate_properties():
    """Demonstrate properties of different coordinate systems"""
    print("4. Coordinate System Properties:")
    print("-" * 50)
    
    # Test distances and angles
    point1 = Cartesian3D(1, 0, 0)
    point2 = Cartesian3D(0, 1, 0)
    point3 = Cartesian3D(0, 0, 1)
    
    print(f"\nDistance calculations:")
    print(f"  Distance from {point1} to {point2}: {point1.distance_to(point2):.3f}")
    print(f"  Distance from {point1} to {point3}: {point1.distance_to(point3):.3f}")
    print(f"  Distance from {point2} to {point3}: {point2.distance_to(point3):.3f}")
    
    # Test coordinate system advantages
    print(f"\nCoordinate system advantages:")
    
    # Spherical coordinates for radial symmetry
    radial_point = Cartesian3D(3, 4, 0)
    spherical = radial_point.to_spherical()
    print(f"  Radial point {radial_point}:")
    print(f"    Distance from origin: {spherical.r:.3f}")
    print(f"    Azimuthal angle: {math.degrees(spherical.theta):.1f}°")
    print(f"    Polar angle: {math.degrees(spherical.phi):.1f}°")
    
    # Cylindrical coordinates for cylindrical symmetry
    cylindrical_point = Cartesian3D(2, 2, 5)
    cylindrical = cylindrical_point.to_cylindrical()
    print(f"  Cylindrical point {cylindrical_point}:")
    print(f"    Radial distance: {cylindrical.rho:.3f}")
    print(f"    Azimuthal angle: {math.degrees(cylindrical.phi):.1f}°")
    print(f"    Height: {cylindrical.z:.3f}")
    
    print()

def demonstrate_coordinate_errors():
    """Demonstrate coordinate conversion errors and precision"""
    print("5. Coordinate Conversion Precision:")
    print("-" * 50)
    
    # Test with various points
    test_cases = [
        ("Origin", Cartesian3D(0, 0, 0)),
        ("Small values", Cartesian3D(0.001, 0.001, 0.001)),
        ("Large values", Cartesian3D(1000, 2000, 3000)),
        ("Mixed values", Cartesian3D(1.5, -2.7, 0.3)),
    ]
    
    for name, point in test_cases:
        print(f"\n{name}: {point}")
        
        # Convert and back
        spherical = point.to_spherical()
        back_to_cart = spherical.to_cartesian()
        error = point.distance_to(back_to_cart)
        
        print(f"  Spherical: {spherical}")
        print(f"  Back to Cartesian: {back_to_cart}")
        print(f"  Error: {error:.10f}")
        
        # Test cylindrical
        cylindrical = point.to_cylindrical()
        back_to_cart_cyl = cylindrical.to_cartesian()
        error_cyl = point.distance_to(back_to_cart_cyl)
        
        print(f"  Cylindrical: {cylindrical}")
        print(f"  Back to Cartesian: {back_to_cart_cyl}")
        print(f"  Error: {error_cyl:.10f}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate coordinate systems"""
    print("=== Coordinate Systems Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate coordinate systems
    demonstrate_coordinate_systems()
    
    # Demonstrate applications
    demonstrate_coordinate_applications()
    
    # Demonstrate transformations
    demonstrate_coordinate_transformations()
    
    # Demonstrate properties
    demonstrate_coordinate_properties()
    
    # Demonstrate precision
    demonstrate_coordinate_errors()
    
    print("="*60)
    print("Coordinate Systems demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Cartesian coordinates (x, y, z)")
    print("✓ Spherical coordinates (r, θ, φ)")
    print("✓ Cylindrical coordinates (ρ, φ, z)")
    print("✓ Coordinate system conversions")
    print("✓ Coordinate transformations")
    print("✓ Precision and error analysis")
    print("✓ Practical applications")
    
    print("\nCoordinate system characteristics:")
    print("• Cartesian: Simple, intuitive, good for rectangular objects")
    print("• Spherical: Natural for radial symmetry, astronomy, physics")
    print("• Cylindrical: Ideal for cylindrical objects, engineering")
    
    print("\nApplications:")
    print("• Game development: Camera systems, object positioning")
    print("• Scientific visualization: Data plotting, simulations")
    print("• Engineering: CAD systems, mechanical design")
    print("• Physics: Particle systems, force calculations")
    print("• Astronomy: Celestial coordinates, orbital mechanics")
    
    print("\nNext steps:")
    print("• Explore 3D transformations and matrices")
    print("• Learn about scene graphs and object hierarchies")
    print("• Study camera positioning and viewing")
    print("• Understand lighting and materials")

if __name__ == "__main__":
    main()
