#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Basic Modules Example

Demonstrates basic module creation, usage, and the __name__ and __main__ concepts
with 3D graphics applications.
"""

import math
import sys
from typing import List, Tuple, Dict, Any

# ============================================================================
# MODULE-LEVEL VARIABLES AND CONSTANTS
# ============================================================================

# Module metadata
__version__ = "1.0.0"
__author__ = "3D Graphics Library"
__description__ = "Basic 3D graphics utilities module"

# Constants
PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# Default values
DEFAULT_POSITION = (0.0, 0.0, 0.0)
DEFAULT_ROTATION = (0.0, 0.0, 0.0)
DEFAULT_SCALE = (1.0, 1.0, 1.0)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians"""
    return degrees * DEG_TO_RAD

def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees"""
    return radians * RAD_TO_DEG

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max"""
    return max(min_val, min(max_val, value))

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b"""
    return a + (b - a) * clamp(t, 0.0, 1.0)

def distance_3d(point1: Tuple[float, float, float], 
                point2: Tuple[float, float, float]) -> float:
    """Calculate distance between two 3D points"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    dz = point2[2] - point1[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

def vector_add(vec1: Tuple[float, float, float], 
               vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Add two 3D vectors"""
    return (vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2])

def vector_subtract(vec1: Tuple[float, float, float], 
                   vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Subtract two 3D vectors"""
    return (vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])

def vector_scale(vec: Tuple[float, float, float], 
                scalar: float) -> Tuple[float, float, float]:
    """Scale a 3D vector by a scalar"""
    return (vec[0] * scalar, vec[1] * scalar, vec[2] * scalar)

def vector_magnitude(vec: Tuple[float, float, float]) -> float:
    """Calculate magnitude of a 3D vector"""
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

def vector_normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector"""
    mag = vector_magnitude(vec)
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return vector_scale(vec, 1.0 / mag)

def vector_dot(vec1: Tuple[float, float, float], 
               vec2: Tuple[float, float, float]) -> float:
    """Calculate dot product of two 3D vectors"""
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def vector_cross(vec1: Tuple[float, float, float], 
                vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate cross product of two 3D vectors"""
    return (
        vec1[1]*vec2[2] - vec1[2]*vec2[1],
        vec1[2]*vec2[0] - vec1[0]*vec2[2],
        vec1[0]*vec2[1] - vec1[1]*vec2[0]
    )

# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def create_transform(position: Tuple[float, float, float] = None,
                    rotation: Tuple[float, float, float] = None,
                    scale: Tuple[float, float, float] = None) -> Dict[str, Any]:
    """Create a transform dictionary"""
    return {
        'position': position or DEFAULT_POSITION,
        'rotation': rotation or DEFAULT_ROTATION,
        'scale': scale or DEFAULT_SCALE
    }

def translate_transform(transform: Dict[str, Any], 
                      translation: Tuple[float, float, float]) -> Dict[str, Any]:
    """Translate a transform by the given vector"""
    new_transform = transform.copy()
    new_transform['position'] = vector_add(transform['position'], translation)
    return new_transform

def rotate_transform(transform: Dict[str, Any], 
                    rotation: Tuple[float, float, float]) -> Dict[str, Any]:
    """Rotate a transform by the given angles (in degrees)"""
    new_transform = transform.copy()
    current_rotation = list(transform['rotation'])
    for i in range(3):
        current_rotation[i] += rotation[i]
    new_transform['rotation'] = tuple(current_rotation)
    return new_transform

def scale_transform(transform: Dict[str, Any], 
                   scale: Tuple[float, float, float]) -> Dict[str, Any]:
    """Scale a transform by the given factors"""
    new_transform = transform.copy()
    new_transform['scale'] = (
        transform['scale'][0] * scale[0],
        transform['scale'][1] * scale[1],
        transform['scale'][2] * scale[2]
    )
    return new_transform

# ============================================================================
# GEOMETRY FUNCTIONS
# ============================================================================

def create_cube_vertices(size: float = 1.0) -> List[Tuple[float, float, float]]:
    """Create vertices for a cube"""
    half_size = size / 2.0
    return [
        # Front face
        (-half_size, -half_size,  half_size),
        ( half_size, -half_size,  half_size),
        ( half_size,  half_size,  half_size),
        (-half_size,  half_size,  half_size),
        # Back face
        (-half_size, -half_size, -half_size),
        ( half_size, -half_size, -half_size),
        ( half_size,  half_size, -half_size),
        (-half_size,  half_size, -half_size),
    ]

def create_sphere_vertices(radius: float = 1.0, 
                          segments: int = 16) -> List[Tuple[float, float, float]]:
    """Create vertices for a sphere (simplified)"""
    vertices = []
    for i in range(segments + 1):
        lat = PI * i / segments
        for j in range(segments + 1):
            lon = 2 * PI * j / segments
            x = radius * math.sin(lat) * math.cos(lon)
            y = radius * math.cos(lat)
            z = radius * math.sin(lat) * math.sin(lon)
            vertices.append((x, y, z))
    return vertices

def calculate_bounding_box(vertices: List[Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
    """Calculate bounding box for a set of vertices"""
    if not vertices:
        return {'min': (0, 0, 0), 'max': (0, 0, 0)}
    
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for vertex in vertices:
        min_x = min(min_x, vertex[0])
        min_y = min(min_y, vertex[1])
        min_z = min(min_z, vertex[2])
        max_x = max(max_x, vertex[0])
        max_y = max(max_y, vertex[1])
        max_z = max(max_z, vertex[2])
    
    return {
        'min': (min_x, min_y, min_z),
        'max': (max_x, max_y, max_z)
    }

# ============================================================================
# MODULE INFORMATION FUNCTIONS
# ============================================================================

def get_module_info() -> Dict[str, Any]:
    """Get information about this module"""
    return {
        'name': __name__,
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'file': __file__,
        'functions': [
            'degrees_to_radians',
            'radians_to_degrees',
            'clamp',
            'lerp',
            'distance_3d',
            'vector_add',
            'vector_subtract',
            'vector_scale',
            'vector_magnitude',
            'vector_normalize',
            'vector_dot',
            'vector_cross',
            'create_transform',
            'translate_transform',
            'rotate_transform',
            'scale_transform',
            'create_cube_vertices',
            'create_sphere_vertices',
            'calculate_bounding_box',
            'get_module_info'
        ]
    }

def list_available_functions() -> List[str]:
    """List all available functions in this module"""
    return [
        name for name in dir() 
        if callable(getattr(sys.modules[__name__], name)) 
        and not name.startswith('_')
    ]

# ============================================================================
# MODULE TESTING FUNCTIONS
# ============================================================================

def test_vector_operations():
    """Test vector operations"""
    print("Testing vector operations:")
    
    # Test vector addition
    vec1 = (1, 2, 3)
    vec2 = (4, 5, 6)
    result = vector_add(vec1, vec2)
    print(f"   Vector addition: {vec1} + {vec2} = {result}")
    
    # Test vector scaling
    scaled = vector_scale(vec1, 2.0)
    print(f"   Vector scaling: {vec1} * 2 = {scaled}")
    
    # Test vector magnitude
    mag = vector_magnitude(vec1)
    print(f"   Vector magnitude: |{vec1}| = {mag:.2f}")
    
    # Test vector normalization
    normalized = vector_normalize(vec1)
    print(f"   Vector normalization: {vec1} -> {normalized}")

def test_transform_operations():
    """Test transform operations"""
    print("\nTesting transform operations:")
    
    # Create transform
    transform = create_transform()
    print(f"   Created transform: {transform}")
    
    # Translate transform
    translated = translate_transform(transform, (1, 2, 3))
    print(f"   Translated transform: {translated}")
    
    # Rotate transform
    rotated = rotate_transform(transform, (45, 90, 0))
    print(f"   Rotated transform: {rotated}")
    
    # Scale transform
    scaled = scale_transform(transform, (2, 2, 2))
    print(f"   Scaled transform: {scaled}")

def test_geometry_operations():
    """Test geometry operations"""
    print("\nTesting geometry operations:")
    
    # Create cube vertices
    cube_vertices = create_cube_vertices(2.0)
    print(f"   Created cube with {len(cube_vertices)} vertices")
    
    # Calculate bounding box
    bbox = calculate_bounding_box(cube_vertices)
    print(f"   Cube bounding box: {bbox}")
    
    # Create sphere vertices
    sphere_vertices = create_sphere_vertices(1.0, 8)
    print(f"   Created sphere with {len(sphere_vertices)} vertices")
    
    # Calculate sphere bounding box
    sphere_bbox = calculate_bounding_box(sphere_vertices)
    print(f"   Sphere bounding box: {sphere_bbox}")

def test_utility_functions():
    """Test utility functions"""
    print("\nTesting utility functions:")
    
    # Test angle conversions
    degrees = 90
    radians = degrees_to_radians(degrees)
    back_to_degrees = radians_to_degrees(radians)
    print(f"   Angle conversion: {degrees}° -> {radians:.2f} rad -> {back_to_degrees:.2f}°")
    
    # Test clamping
    value = 15
    clamped = clamp(value, 0, 10)
    print(f"   Clamping: {value} -> {clamped}")
    
    # Test linear interpolation
    a, b = 0, 10
    t = 0.5
    interpolated = lerp(a, b, t)
    print(f"   Linear interpolation: lerp({a}, {b}, {t}) = {interpolated}")
    
    # Test distance calculation
    point1 = (0, 0, 0)
    point2 = (3, 4, 0)
    dist = distance_3d(point1, point2)
    print(f"   Distance: {point1} to {point2} = {dist:.2f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate module functionality"""
    print("=== Basic 3D Graphics Module Demo ===\n")
    
    # Show module information
    info = get_module_info()
    print(f"Module: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print(f"File: {info['file']}")
    
    # Show available functions
    print(f"\nAvailable functions: {len(info['functions'])}")
    for i, func in enumerate(info['functions'], 1):
        print(f"   {i:2d}. {func}")
    
    # Run tests
    print("\n" + "="*50)
    test_utility_functions()
    test_vector_operations()
    test_transform_operations()
    test_geometry_operations()
    
    print("\n" + "="*50)
    print("Module demo completed successfully!")

# ============================================================================
# MODULE EXECUTION CHECK
# ============================================================================

if __name__ == "__main__":
    # This code only runs when the module is executed directly
    print(f"Running {__name__} as main module")
    main()
else:
    # This code runs when the module is imported
    print(f"Module {__name__} imported successfully")
    print(f"Available functions: {list_available_functions()}")

# ============================================================================
# MODULE DOCUMENTATION
# ============================================================================

"""
This module provides basic 3D graphics utilities including:

Vector Operations:
- vector_add, vector_subtract, vector_scale
- vector_magnitude, vector_normalize
- vector_dot, vector_cross

Transform Operations:
- create_transform, translate_transform
- rotate_transform, scale_transform

Geometry Functions:
- create_cube_vertices, create_sphere_vertices
- calculate_bounding_box

Utility Functions:
- degrees_to_radians, radians_to_degrees
- clamp, lerp, distance_3d

Usage:
    import basic_modules as bm
    
    # Use vector operations
    vec1 = (1, 2, 3)
    vec2 = (4, 5, 6)
    result = bm.vector_add(vec1, vec2)
    
    # Create and manipulate transforms
    transform = bm.create_transform()
    translated = bm.translate_transform(transform, (1, 2, 3))
    
    # Create geometry
    cube_vertices = bm.create_cube_vertices(2.0)
    bbox = bm.calculate_bounding_box(cube_vertices)
"""
