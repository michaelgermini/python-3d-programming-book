#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Import Methods Example

Demonstrates different import techniques and best practices
with 3D graphics applications.
"""

import sys
import os
from typing import Dict, List, Any, Optional

# ============================================================================
# SIMULATED 3D GRAPHICS MODULES
# ============================================================================

# These simulate modules that would be imported
# In a real project, these would be separate files

class Vector3:
    """3D Vector class"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __str__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

class Matrix4:
    """4x4 Matrix class"""
    def __init__(self):
        self.data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    
    def __str__(self):
        return f"Matrix4({self.data})"

class Mesh:
    """3D Mesh class"""
    def __init__(self, name: str = "mesh"):
        self.name = name
        self.vertices = []
        self.faces = []
    
    def __str__(self):
        return f"Mesh({self.name}, {len(self.vertices)} vertices)"

class Shader:
    """Shader class"""
    def __init__(self, name: str = "shader"):
        self.name = name
        self.program_id = 0
    
    def __str__(self):
        return f"Shader({self.name}, program_id={self.program_id})"

# ============================================================================
# MODULE SIMULATION
# ============================================================================

def simulate_math_module():
    """Simulate a math module with vector and matrix operations"""
    print("=== Math Module Functions ===")
    
    def vector_add(v1: Vector3, v2: Vector3) -> Vector3:
        return v1 + v2
    
    def vector_scale(v: Vector3, scalar: float) -> Vector3:
        return Vector3(v.x * scalar, v.y * scalar, v.z * scalar)
    
    def create_rotation_matrix(angle: float, axis: Vector3) -> Matrix4:
        return Matrix4()  # Simplified
    
    def create_translation_matrix(translation: Vector3) -> Matrix4:
        return Matrix4()  # Simplified
    
    return {
        'Vector3': Vector3,
        'Matrix4': Matrix4,
        'vector_add': vector_add,
        'vector_scale': vector_scale,
        'create_rotation_matrix': create_rotation_matrix,
        'create_translation_matrix': create_translation_matrix
    }

def simulate_geometry_module():
    """Simulate a geometry module with mesh operations"""
    print("=== Geometry Module Functions ===")
    
    def create_cube(size: float = 1.0) -> Mesh:
        mesh = Mesh("cube")
        # Simplified cube creation
        return mesh
    
    def create_sphere(radius: float = 1.0, segments: int = 16) -> Mesh:
        mesh = Mesh("sphere")
        # Simplified sphere creation
        return mesh
    
    def load_mesh(filename: str) -> Mesh:
        mesh = Mesh(f"loaded_{filename}")
        # Simplified mesh loading
        return mesh
    
    def save_mesh(mesh: Mesh, filename: str) -> bool:
        print(f"Saving mesh {mesh.name} to {filename}")
        return True
    
    return {
        'Mesh': Mesh,
        'create_cube': create_cube,
        'create_sphere': create_sphere,
        'load_mesh': load_mesh,
        'save_mesh': save_mesh
    }

def simulate_rendering_module():
    """Simulate a rendering module with shader operations"""
    print("=== Rendering Module Functions ===")
    
    def create_shader(vertex_source: str, fragment_source: str) -> Shader:
        shader = Shader("custom_shader")
        shader.program_id = 12345  # Simulated program ID
        return shader
    
    def load_shader(filename: str) -> Shader:
        shader = Shader(f"loaded_{filename}")
        shader.program_id = 67890  # Simulated program ID
        return shader
    
    def use_shader(shader: Shader) -> bool:
        print(f"Using shader: {shader.name}")
        return True
    
    def set_uniform(shader: Shader, name: str, value: Any) -> bool:
        print(f"Setting uniform {name} = {value} in shader {shader.name}")
        return True
    
    return {
        'Shader': Shader,
        'create_shader': create_shader,
        'load_shader': load_shader,
        'use_shader': use_shader,
        'set_uniform': set_uniform
    }

# ============================================================================
# IMPORT METHOD EXAMPLES
# ============================================================================

def demonstrate_basic_imports():
    """Demonstrate basic import methods"""
    print("\n=== Basic Import Methods ===\n")
    
    # Simulate importing modules
    math_module = simulate_math_module()
    geometry_module = simulate_geometry_module()
    rendering_module = simulate_rendering_module()
    
    print("1. Import entire module:")
    print("   import math3d")
    print("   import geometry")
    print("   import rendering")
    print()
    
    # Demonstrate usage
    print("   Usage:")
    vec1 = math_module['Vector3'](1, 2, 3)
    vec2 = math_module['Vector3'](4, 5, 6)
    result = math_module['vector_add'](vec1, vec2)
    print(f"   vec1 = {vec1}")
    print(f"   vec2 = {vec2}")
    print(f"   result = {result}")
    print()

def demonstrate_from_imports():
    """Demonstrate from...import statements"""
    print("2. Import specific items:")
    print("   from math3d import Vector3, Matrix4")
    print("   from geometry import create_cube, create_sphere")
    print("   from rendering import Shader, create_shader")
    print()
    
    # Simulate the imports
    math_module = simulate_math_module()
    geometry_module = simulate_geometry_module()
    rendering_module = simulate_rendering_module()
    
    # Extract specific items
    Vector3 = math_module['Vector3']
    Matrix4 = math_module['Matrix4']
    create_cube = geometry_module['create_cube']
    create_sphere = geometry_module['create_sphere']
    Shader = rendering_module['Shader']
    create_shader = rendering_module['create_shader']
    
    print("   Usage:")
    vec = Vector3(1, 2, 3)
    mat = Matrix4()
    cube = create_cube(2.0)
    sphere = create_sphere(1.5)
    shader = create_shader("vertex.glsl", "fragment.glsl")
    
    print(f"   vec = {vec}")
    print(f"   mat = {mat}")
    print(f"   cube = {cube}")
    print(f"   sphere = {sphere}")
    print(f"   shader = {shader}")
    print()

def demonstrate_import_as():
    """Demonstrate import with aliases"""
    print("3. Import with aliases:")
    print("   import math3d as math")
    print("   import geometry as geom")
    print("   import rendering as render")
    print()
    
    # Simulate the imports with aliases
    math = simulate_math_module()
    geom = simulate_geometry_module()
    render = simulate_rendering_module()
    
    print("   Usage:")
    vec = math['Vector3'](1, 2, 3)
    cube = geom['create_cube'](2.0)
    shader = render['create_shader']("vertex.glsl", "fragment.glsl")
    
    print(f"   vec = {vec}")
    print(f"   cube = {cube}")
    print(f"   shader = {shader}")
    print()

def demonstrate_from_import_as():
    """Demonstrate from...import...as statements"""
    print("4. Import specific items with aliases:")
    print("   from math3d import Vector3 as Vec3")
    print("   from geometry import create_cube as make_cube")
    print("   from rendering import Shader as ShaderProgram")
    print()
    
    # Simulate the imports with aliases
    math_module = simulate_math_module()
    geometry_module = simulate_geometry_module()
    rendering_module = simulate_rendering_module()
    
    # Extract with aliases
    Vec3 = math_module['Vector3']
    make_cube = geometry_module['create_cube']
    ShaderProgram = rendering_module['Shader']
    
    print("   Usage:")
    vec = Vec3(1, 2, 3)
    cube = make_cube(2.0)
    shader = ShaderProgram("custom_shader")
    
    print(f"   vec = {vec}")
    print(f"   cube = {cube}")
    print(f"   shader = {shader}")
    print()

def demonstrate_relative_imports():
    """Demonstrate relative imports"""
    print("5. Relative imports (within package):")
    print("   from .math import Vector3")
    print("   from ..geometry import create_cube")
    print("   from ...rendering import Shader")
    print()
    
    print("   Note: Relative imports only work within packages")
    print("   and when the module is run as part of a package.")
    print()

def demonstrate_conditional_imports():
    """Demonstrate conditional imports"""
    print("6. Conditional imports:")
    print("   try:")
    print("       import numpy as np")
    print("       HAS_NUMPY = True")
    print("   except ImportError:")
    print("       HAS_NUMPY = False")
    print()
    
    # Simulate conditional import
    try:
        import numpy as np
        HAS_NUMPY = True
        print("   NumPy is available")
        print(f"   NumPy version: {np.__version__}")
    except ImportError:
        HAS_NUMPY = False
        print("   NumPy is not available")
    
    print()

def demonstrate_lazy_imports():
    """Demonstrate lazy imports"""
    print("7. Lazy imports (import inside functions):")
    print("   def process_mesh(filename):")
    print("       import geometry  # Import only when needed")
    print("       return geometry.load_mesh(filename)")
    print()
    
    def process_mesh(filename: str):
        # Import only when function is called
        geometry_module = simulate_geometry_module()
        return geometry_module['load_mesh'](filename)
    
    def render_scene():
        # Import only when function is called
        rendering_module = simulate_rendering_module()
        shader = rendering_module['create_shader']("vertex.glsl", "fragment.glsl")
        return shader
    
    print("   Usage:")
    mesh = process_mesh("model.obj")
    shader = render_scene()
    print(f"   mesh = {mesh}")
    print(f"   shader = {shader}")
    print()

# ============================================================================
# IMPORT BEST PRACTICES
# ============================================================================

def demonstrate_import_best_practices():
    """Demonstrate import best practices"""
    print("\n=== Import Best Practices ===\n")
    
    print("1. Import order (PEP 8):")
    print("   # Standard library imports")
    print("   import os")
    print("   import sys")
    print("   from typing import List, Dict")
    print()
    print("   # Third-party imports")
    print("   import numpy as np")
    print("   import matplotlib.pyplot as plt")
    print()
    print("   # Local application imports")
    print("   from .math import Vector3")
    print("   from .geometry import create_cube")
    print()
    
    print("2. Avoid wildcard imports:")
    print("   # Bad:")
    print("   from math3d import *")
    print()
    print("   # Good:")
    print("   from math3d import Vector3, Matrix4, vector_add")
    print()
    
    print("3. Use aliases for long module names:")
    print("   # Instead of:")
    print("   import very_long_module_name as vlmn")
    print()
    print("   # Use:")
    print("   import very_long_module_name as vlm")
    print()
    
    print("4. Group related imports:")
    print("   # Math-related imports")
    print("   from math3d import Vector3, Matrix4, Quaternion")
    print("   from geometry import create_cube, create_sphere")
    print()
    print("   # Rendering-related imports")
    print("   from rendering import Shader, Renderer")
    print("   from materials import Material, Texture")
    print()

# ============================================================================
# IMPORT PROBLEMS AND SOLUTIONS
# ============================================================================

def demonstrate_import_problems():
    """Demonstrate common import problems and solutions"""
    print("\n=== Common Import Problems and Solutions ===\n")
    
    print("1. Circular imports:")
    print("   # Problem: module_a imports module_b, module_b imports module_a")
    print("   # Solution: Use lazy imports or restructure code")
    print()
    
    print("2. Name conflicts:")
    print("   # Problem:")
    print("   from math3d import Vector3")
    print("   from physics import Vector3  # Name conflict!")
    print()
    print("   # Solution:")
    print("   from math3d import Vector3 as MathVector3")
    print("   from physics import Vector3 as PhysicsVector3")
    print()
    
    print("3. Import errors:")
    print("   # Problem: Module not found")
    print("   # Solution: Check PYTHONPATH, use try/except")
    print("   try:")
    print("       import optional_module")
    print("   except ImportError:")
    print("       optional_module = None")
    print()
    
    print("4. Performance issues:")
    print("   # Problem: Importing large modules at startup")
    print("   # Solution: Use lazy imports")
    print("   def expensive_operation():")
    print("       import heavy_module  # Import only when needed")
    print("       return heavy_module.process()")
    print()

# ============================================================================
# PRACTICAL 3D GRAPHICS EXAMPLE
# ============================================================================

def demonstrate_practical_usage():
    """Demonstrate practical import usage in 3D graphics"""
    print("\n=== Practical 3D Graphics Import Example ===\n")
    
    # Simulate importing the modules
    math_module = simulate_math_module()
    geometry_module = simulate_geometry_module()
    rendering_module = simulate_rendering_module()
    
    # Extract commonly used items
    Vector3 = math_module['Vector3']
    vector_add = math_module['vector_add']
    create_cube = geometry_module['create_cube']
    create_sphere = geometry_module['create_sphere']
    Shader = rendering_module['Shader']
    create_shader = rendering_module['create_shader']
    use_shader = rendering_module['use_shader']
    
    print("Creating a 3D scene with proper imports:")
    print()
    
    # Create 3D objects
    print("1. Creating 3D objects:")
    cube = create_cube(size=2.0)
    sphere = create_sphere(radius=1.0)
    print(f"   Created: {cube}")
    print(f"   Created: {sphere}")
    print()
    
    # Create vectors for positioning
    print("2. Working with vectors:")
    position1 = Vector3(0, 0, 0)
    position2 = Vector3(5, 0, 0)
    offset = Vector3(1, 2, 3)
    final_position = vector_add(position1, offset)
    print(f"   Position 1: {position1}")
    print(f"   Position 2: {position2}")
    print(f"   Offset: {offset}")
    print(f"   Final position: {final_position}")
    print()
    
    # Create and use shaders
    print("3. Working with shaders:")
    vertex_shader = create_shader("vertex.glsl", "fragment.glsl")
    use_shader(vertex_shader)
    print(f"   Created and using shader: {vertex_shader}")
    print()
    
    print("Scene setup complete!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate import methods"""
    print("=== Python Import Methods Demo ===\n")
    
    # Demonstrate different import methods
    demonstrate_basic_imports()
    demonstrate_from_imports()
    demonstrate_import_as()
    demonstrate_from_import_as()
    demonstrate_relative_imports()
    demonstrate_conditional_imports()
    demonstrate_lazy_imports()
    
    # Show best practices
    demonstrate_import_best_practices()
    
    # Show common problems and solutions
    demonstrate_import_problems()
    
    # Show practical usage
    demonstrate_practical_usage()
    
    print("\n" + "="*60)
    print("Import methods demo completed successfully!")
    print("\nKey takeaways:")
    print("✓ Choose the right import method for your use case")
    print("✓ Follow PEP 8 import order guidelines")
    print("✓ Avoid wildcard imports (*)")
    print("✓ Use aliases to avoid name conflicts")
    print("✓ Consider lazy imports for performance")
    print("✓ Handle import errors gracefully")
    print("✓ Group related imports together")

if __name__ == "__main__":
    main()
