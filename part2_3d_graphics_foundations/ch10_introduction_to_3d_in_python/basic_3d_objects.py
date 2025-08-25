#!/usr/bin/env python3
"""
Chapter 10: Introduction to 3D in Python
Basic 3D Objects

Demonstrates how to create and manipulate basic 3D objects like cubes,
spheres, cylinders, and other geometric shapes.
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Basic 3D Objects"
__description__ = "Creating and manipulating basic 3D objects"

# ============================================================================
# 3D OBJECT CLASSES
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
    """Color class for 3D objects"""
    r: float  # Red component (0.0 to 1.0)
    g: float  # Green component (0.0 to 1.0)
    b: float  # Blue component (0.0 to 1.0)
    a: float = 1.0  # Alpha component (0.0 to 1.0)
    
    @classmethod
    def red(cls) -> 'Color':
        return cls(1.0, 0.0, 0.0)
    
    @classmethod
    def green(cls) -> 'Color':
        return cls(0.0, 1.0, 0.0)
    
    @classmethod
    def blue(cls) -> 'Color':
        return cls(0.0, 0.0, 1.0)
    
    @classmethod
    def yellow(cls) -> 'Color':
        return cls(1.0, 1.0, 0.0)
    
    @classmethod
    def cyan(cls) -> 'Color':
        return cls(0.0, 1.0, 1.0)
    
    @classmethod
    def magenta(cls) -> 'Color':
        return cls(1.0, 0.0, 1.0)
    
    @classmethod
    def white(cls) -> 'Color':
        return cls(1.0, 1.0, 1.0)
    
    @classmethod
    def black(cls) -> 'Color':
        return cls(0.0, 0.0, 0.0)
    
    @classmethod
    def random(cls) -> 'Color':
        return cls(random.random(), random.random(), random.random())
    
    def __str__(self):
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"

class Base3DObject:
    """Base class for all 3D objects"""
    
    def __init__(self, position: Vector3D, color: Color = None, name: str = ""):
        self.position = position
        self.color = color or Color.random()
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self.rotation = Vector3D(0, 0, 0)  # Euler angles in radians
        self.scale = Vector3D(1, 1, 1)
        self.visible = True
    
    def move_to(self, new_position: Vector3D):
        """Move object to a new position"""
        self.position = new_position
    
    def translate(self, offset: Vector3D):
        """Translate object by an offset"""
        self.position = self.position + offset
    
    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        """Rotate object by given angles (in radians)"""
        self.rotation.x += x_angle
        self.rotation.y += y_angle
        self.rotation.z += z_angle
    
    def scale_by(self, x_scale: float = 1, y_scale: float = 1, z_scale: float = 1):
        """Scale object by given factors"""
        self.scale.x *= x_scale
        self.scale.y *= y_scale
        self.scale.z *= z_scale
    
    def set_color(self, color: Color):
        """Set object color"""
        self.color = color
    
    def set_visibility(self, visible: bool):
        """Set object visibility"""
        self.visible = visible
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box of the object (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_volume(self) -> float:
        """Get volume of the object (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_surface_area(self) -> float:
        """Get surface area of the object (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}', pos={self.position})"

class Cube(Base3DObject):
    """3D cube object"""
    
    def __init__(self, position: Vector3D, size: float = 1.0, color: Color = None, name: str = ""):
        super().__init__(position, color, name)
        self.size = size
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box of the cube"""
        half_size = self.size / 2
        min_point = Vector3D(
            self.position.x - half_size,
            self.position.y - half_size,
            self.position.z - half_size
        )
        max_point = Vector3D(
            self.position.x + half_size,
            self.position.y + half_size,
            self.position.z + half_size
        )
        return min_point, max_point
    
    def get_volume(self) -> float:
        """Get volume of the cube"""
        return self.size ** 3
    
    def get_surface_area(self) -> float:
        """Get surface area of the cube"""
        return 6 * (self.size ** 2)

class Sphere(Base3DObject):
    """3D sphere object"""
    
    def __init__(self, position: Vector3D, radius: float = 1.0, color: Color = None, name: str = ""):
        super().__init__(position, color, name)
        self.radius = radius
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box of the sphere"""
        min_point = Vector3D(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.position.z - self.radius
        )
        max_point = Vector3D(
            self.position.x + self.radius,
            self.position.y + self.radius,
            self.position.z + self.radius
        )
        return min_point, max_point
    
    def get_volume(self) -> float:
        """Get volume of the sphere"""
        return (4/3) * math.pi * (self.radius ** 3)
    
    def get_surface_area(self) -> float:
        """Get surface area of the sphere"""
        return 4 * math.pi * (self.radius ** 2)
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if a point is inside the sphere"""
        distance = (point - self.position).magnitude()
        return distance <= self.radius

class Cylinder(Base3DObject):
    """3D cylinder object"""
    
    def __init__(self, position: Vector3D, radius: float = 1.0, height: float = 2.0, 
                 color: Color = None, name: str = ""):
        super().__init__(position, color, name)
        self.radius = radius
        self.height = height
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box of the cylinder"""
        half_height = self.height / 2
        min_point = Vector3D(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.position.z - half_height
        )
        max_point = Vector3D(
            self.position.x + self.radius,
            self.position.y + self.radius,
            self.position.z + half_height
        )
        return min_point, max_point
    
    def get_volume(self) -> float:
        """Get volume of the cylinder"""
        return math.pi * (self.radius ** 2) * self.height
    
    def get_surface_area(self) -> float:
        """Get surface area of the cylinder"""
        lateral_area = 2 * math.pi * self.radius * self.height
        base_area = 2 * math.pi * (self.radius ** 2)
        return lateral_area + base_area

class Cone(Base3DObject):
    """3D cone object"""
    
    def __init__(self, position: Vector3D, radius: float = 1.0, height: float = 2.0, 
                 color: Color = None, name: str = ""):
        super().__init__(position, color, name)
        self.radius = radius
        self.height = height
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        """Get bounding box of the cone"""
        min_point = Vector3D(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.position.z
        )
        max_point = Vector3D(
            self.position.x + self.radius,
            self.position.y + self.radius,
            self.position.z + self.height
        )
        return min_point, max_point
    
    def get_volume(self) -> float:
        """Get volume of the cone"""
        return (1/3) * math.pi * (self.radius ** 2) * self.height
    
    def get_surface_area(self) -> float:
        """Get surface area of the cone"""
        slant_height = math.sqrt(self.radius**2 + self.height**2)
        lateral_area = math.pi * self.radius * slant_height
        base_area = math.pi * (self.radius ** 2)
        return lateral_area + base_area

# ============================================================================
# OBJECT FACTORY
# ============================================================================

class ObjectFactory:
    """Factory class for creating 3D objects"""
    
    @staticmethod
    def create_cube(position: Vector3D, size: float = 1.0, color: Color = None) -> Cube:
        """Create a cube"""
        return Cube(position, size, color)
    
    @staticmethod
    def create_sphere(position: Vector3D, radius: float = 1.0, color: Color = None) -> Sphere:
        """Create a sphere"""
        return Sphere(position, radius, color)
    
    @staticmethod
    def create_cylinder(position: Vector3D, radius: float = 1.0, height: float = 2.0, 
                       color: Color = None) -> Cylinder:
        """Create a cylinder"""
        return Cylinder(position, radius, height, color)
    
    @staticmethod
    def create_cone(position: Vector3D, radius: float = 1.0, height: float = 2.0, 
                   color: Color = None) -> Cone:
        """Create a cone"""
        return Cone(position, radius, height, color)
    
    @staticmethod
    def create_random_object(position: Vector3D) -> Base3DObject:
        """Create a random 3D object"""
        object_types = [
            (ObjectFactory.create_cube, {'size': random.uniform(0.5, 2.0)}),
            (ObjectFactory.create_sphere, {'radius': random.uniform(0.5, 1.5)}),
            (ObjectFactory.create_cylinder, {'radius': random.uniform(0.3, 1.0), 'height': random.uniform(1.0, 3.0)}),
            (ObjectFactory.create_cone, {'radius': random.uniform(0.3, 1.0), 'height': random.uniform(1.0, 3.0)})
        ]
        
        obj_type, params = random.choice(object_types)
        return obj_type(position, **params, color=Color.random())

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_objects():
    """Demonstrate creation and manipulation of basic 3D objects"""
    print("=== Basic 3D Objects Demonstration ===\n")
    
    # Create various 3D objects
    print("1. Creating basic 3D objects:")
    
    # Cube
    cube = ObjectFactory.create_cube(
        Vector3D(0, 0, 0), 
        size=2.0, 
        color=Color.red()
    )
    print(f"   Created cube at {cube.position}")
    
    # Sphere
    sphere = ObjectFactory.create_sphere(
        Vector3D(3, 0, 0), 
        radius=1.5, 
        color=Color.blue()
    )
    print(f"   Created sphere at {sphere.position}")
    
    # Cylinder
    cylinder = ObjectFactory.create_cylinder(
        Vector3D(-3, 0, 0), 
        radius=1.0, 
        height=3.0, 
        color=Color.green()
    )
    print(f"   Created cylinder at {cylinder.position}")
    
    # Cone
    cone = ObjectFactory.create_cone(
        Vector3D(0, 3, 0), 
        radius=1.0, 
        height=2.5, 
        color=Color.yellow()
    )
    print(f"   Created cone at {cone.position}")
    
    print()
    
    # Demonstrate object properties
    print("2. Object properties:")
    objects = [cube, sphere, cylinder, cone]
    for obj in objects:
        print(f"   {obj.__class__.__name__}:")
        print(f"     Volume: {obj.get_volume():.3f}")
        print(f"     Surface Area: {obj.get_surface_area():.3f}")
        min_point, max_point = obj.get_bounding_box()
        print(f"     Bounding Box: {min_point} to {max_point}")
        print()
    
    # Demonstrate transformations
    print("3. Applying transformations:")
    
    # Move objects
    cube.translate(Vector3D(1, 1, 1))
    print(f"   Moved cube to {cube.position}")
    
    sphere.rotate(0, math.pi/4, 0)  # Rotate 45 degrees around Y-axis
    print(f"   Rotated sphere by 45° around Y-axis")
    
    cylinder.scale_by(1.5, 1.5, 1.5)
    print(f"   Scaled cylinder by 1.5x")
    
    cone.set_color(Color.cyan())
    print(f"   Changed cone color to cyan")
    
    print()
    
    return objects

def demonstrate_random_objects():
    """Demonstrate creation of random 3D objects"""
    print("=== Random 3D Objects Demonstration ===\n")
    
    objects = []
    
    # Create random objects in a grid
    print("Creating random 3D objects in a grid pattern:")
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            position = Vector3D(i * 3, j * 3, 0)
            obj = ObjectFactory.create_random_object(position)
            objects.append(obj)
            print(f"   Created {obj.__class__.__name__} at {position}")
    
    print(f"\nTotal objects created: {len(objects)}")
    
    # Apply random transformations
    print("\nApplying random transformations:")
    for obj in objects:
        # Random rotation
        obj.rotate(
            random.uniform(0, math.pi/2),
            random.uniform(0, math.pi/2),
            random.uniform(0, math.pi/2)
        )
        
        # Random scale
        scale_factor = random.uniform(0.5, 2.0)
        obj.scale_by(scale_factor, scale_factor, scale_factor)
        
        print(f"   Transformed {obj.name}")
    
    print()
    
    # Print object statistics
    print("Object statistics:")
    object_types = {}
    for obj in objects:
        obj_type = obj.__class__.__name__
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
    
    for obj_type, count in object_types.items():
        print(f"   {obj_type}: {count}")
    
    print()
    
    return objects

def demonstrate_object_interactions():
    """Demonstrate interactions between 3D objects"""
    print("=== Object Interactions Demonstration ===\n")
    
    # Create objects for interaction testing
    sphere1 = Sphere(Vector3D(0, 0, 0), radius=2.0, color=Color.red())
    sphere2 = Sphere(Vector3D(3, 0, 0), radius=1.5, color=Color.blue())
    sphere3 = Sphere(Vector3D(6, 0, 0), radius=1.0, color=Color.green())
    
    print("Testing sphere interactions:")
    
    # Test point containment
    test_point = Vector3D(1, 0, 0)
    print(f"   Point {test_point} in sphere1: {sphere1.contains_point(test_point)}")
    print(f"   Point {test_point} in sphere2: {sphere2.contains_point(test_point)}")
    print(f"   Point {test_point} in sphere3: {sphere3.contains_point(test_point)}")
    
    # Test collision detection (simplified)
    def check_sphere_collision(sphere1: Sphere, sphere2: Sphere) -> bool:
        distance = (sphere1.position - sphere2.position).magnitude()
        return distance < (sphere1.radius + sphere2.radius)
    
    print(f"\nCollision detection:")
    print(f"   sphere1 and sphere2: {check_sphere_collision(sphere1, sphere2)}")
    print(f"   sphere2 and sphere3: {check_sphere_collision(sphere2, sphere3)}")
    print(f"   sphere1 and sphere3: {check_sphere_collision(sphere1, sphere3)}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate basic 3D objects"""
    print("=== Basic 3D Objects Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate basic objects
    print("1. Basic 3D Objects:")
    objects1 = demonstrate_basic_objects()
    
    # Demonstrate random objects
    print("2. Random 3D Objects:")
    objects2 = demonstrate_random_objects()
    
    # Demonstrate object interactions
    print("3. Object Interactions:")
    demonstrate_object_interactions()
    
    print("="*60)
    print("Basic 3D Objects demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Creating basic 3D objects (cube, sphere, cylinder, cone)")
    print("✓ Object properties (volume, surface area, bounding box)")
    print("✓ Transformations (translation, rotation, scaling)")
    print("✓ Object interactions and collision detection")
    print("✓ Random object generation")
    print("✓ Color and material properties")
    
    print("\nObject types covered:")
    print("• Cube: 6 faces, 8 vertices, 12 edges")
    print("• Sphere: Perfectly round, infinite surface detail")
    print("• Cylinder: Circular base and top, rectangular side")
    print("• Cone: Circular base, pointed top")
    
    print("\nNext steps:")
    print("• Explore coordinate systems and transformations")
    print("• Learn about scene organization and hierarchies")
    print("• Study camera positioning and viewing")
    print("• Understand lighting and materials")

if __name__ == "__main__":
    main()
