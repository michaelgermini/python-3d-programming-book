#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Namespace Management Example

Demonstrates managing namespaces and avoiding conflicts
with 3D graphics applications.
"""

import sys
import os
from typing import Dict, List, Any, Optional

# ============================================================================
# SIMULATED MODULES WITH NAMESPACE CONFLICTS
# ============================================================================

# These simulate different modules that might have naming conflicts
# In a real project, these would be separate files

class Vector3:
    """3D Vector class from math module"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __str__(self):
        return f"MathVector3({self.x}, {self.y}, {self.z})"

class PhysicsVector3:
    """3D Vector class from physics module"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = Vector3(0, 0, 0)
    
    def __str__(self):
        return f"PhysicsVector3({self.x}, {self.y}, {self.z})"

class Transform:
    """Transform class from scene module"""
    def __init__(self, position=None, rotation=None, scale=None):
        self.position = position or Vector3()
        self.rotation = rotation or Vector3()
        self.scale = scale or Vector3(1, 1, 1)
    
    def __str__(self):
        return f"SceneTransform(pos={self.position}, rot={self.rotation}, scale={self.scale})"

class PhysicsTransform:
    """Transform class from physics module"""
    def __init__(self, position=None, rotation=None, scale=None):
        self.position = position or PhysicsVector3()
        self.rotation = rotation or PhysicsVector3()
        self.scale = scale or PhysicsVector3(1, 1, 1)
        self.velocity = PhysicsVector3()
        self.angular_velocity = PhysicsVector3()
    
    def __str__(self):
        return f"PhysicsTransform(pos={self.position}, vel={self.velocity})"

# ============================================================================
# MODULE SIMULATION
# ============================================================================

def simulate_math_module():
    """Simulate math module with vector operations"""
    print("=== Math Module Namespace ===")
    
    def vector_add(v1: Vector3, v2: Vector3) -> Vector3:
        return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)
    
    def vector_scale(v: Vector3, scalar: float) -> Vector3:
        return Vector3(v.x * scalar, v.y * scalar, v.z * scalar)
    
    def create_rotation_matrix(angle: float, axis: Vector3):
        return f"RotationMatrix({angle}°, {axis})"
    
    return {
        'Vector3': Vector3,
        'vector_add': vector_add,
        'vector_scale': vector_scale,
        'create_rotation_matrix': create_rotation_matrix,
        '__name__': 'math3d'
    }

def simulate_physics_module():
    """Simulate physics module with physics-specific classes"""
    print("=== Physics Module Namespace ===")
    
    def apply_force(body: PhysicsVector3, force: PhysicsVector3):
        body.velocity.x += force.x
        body.velocity.y += force.y
        body.velocity.z += force.z
        return f"Applied force {force} to {body}"
    
    def update_physics(transform: PhysicsTransform, dt: float):
        transform.position.x += transform.velocity.x * dt
        transform.position.y += transform.velocity.y * dt
        transform.position.z += transform.velocity.z * dt
        return f"Updated physics for {transform}"
    
    def create_rigidbody(mass: float, position: PhysicsVector3):
        return f"RigidBody(mass={mass}, pos={position})"
    
    return {
        'Vector3': PhysicsVector3,  # Same name as math module!
        'Transform': PhysicsTransform,  # Same name as scene module!
        'apply_force': apply_force,
        'update_physics': update_physics,
        'create_rigidbody': create_rigidbody,
        '__name__': 'physics'
    }

def simulate_scene_module():
    """Simulate scene module with scene management"""
    print("=== Scene Module Namespace ===")
    
    def create_object(name: str, transform: Transform):
        return f"SceneObject({name}, transform={transform})"
    
    def add_to_scene(scene: str, obj: str):
        return f"Added {obj} to scene {scene}"
    
    def render_scene(scene: str):
        return f"Rendering scene {scene}"
    
    def create_camera(position: Vector3, target: Vector3):
        return f"Camera(pos={position}, target={target})"
    
    return {
        'Vector3': Vector3,  # Same as math module
        'Transform': Transform,  # Same as physics module!
        'create_object': create_object,
        'add_to_scene': add_to_scene,
        'render_scene': render_scene,
        'create_camera': create_camera,
        '__name__': 'scene'
    }

# ============================================================================
# NAMESPACE CONFLICT EXAMPLES
# ============================================================================

def demonstrate_namespace_conflicts():
    """Demonstrate namespace conflicts and their problems"""
    print("\n=== Namespace Conflicts ===\n")
    
    # Simulate importing modules
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    print("1. Problem: Same class names in different modules")
    print("   Math module has: Vector3")
    print("   Physics module has: Vector3")
    print("   Scene module has: Vector3")
    print()
    
    print("2. Problem: Same function names in different modules")
    print("   Math module has: create_rotation_matrix")
    print("   Physics module has: create_rigidbody")
    print("   Scene module has: create_camera")
    print()
    
    print("3. What happens when we import everything:")
    print("   from math3d import *")
    print("   from physics import *")
    print("   from scene import *")
    print("   # Vector3 gets overwritten multiple times!")
    print()

def demonstrate_bad_imports():
    """Demonstrate what happens with bad import practices"""
    print("\n=== Bad Import Practices ===\n")
    
    # Simulate the modules
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    print("1. Wildcard imports (BAD):")
    print("   from math3d import *")
    print("   from physics import *")
    print("   from scene import *")
    print()
    
    # Simulate what happens
    print("   Result: Vector3 gets overwritten!")
    Vector3_math = math_module['Vector3']
    Vector3_physics = physics_module['Vector3']
    Vector3_scene = scene_module['Vector3']
    
    print(f"   Math Vector3: {Vector3_math}")
    print(f"   Physics Vector3: {Vector3_physics}")
    print(f"   Scene Vector3: {Vector3_scene}")
    print("   After wildcard imports, only the last one remains!")
    print()

# ============================================================================
# NAMESPACE MANAGEMENT SOLUTIONS
# ============================================================================

def demonstrate_module_imports():
    """Demonstrate using module imports to avoid conflicts"""
    print("\n=== Solution 1: Module Imports ===\n")
    
    # Simulate importing modules
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    print("1. Import entire modules:")
    print("   import math3d")
    print("   import physics")
    print("   import scene")
    print()
    
    print("2. Access with module prefix:")
    print("   math3d.Vector3(1, 2, 3)")
    print("   physics.Vector3(1, 2, 3)")
    print("   scene.Vector3(1, 2, 3)")
    print()
    
    # Demonstrate usage
    print("3. Usage example:")
    math_vec = math_module['Vector3'](1, 2, 3)
    physics_vec = physics_module['Vector3'](4, 5, 6)
    scene_vec = scene_module['Vector3'](7, 8, 9)
    
    print(f"   math3d.Vector3: {math_vec}")
    print(f"   physics.Vector3: {physics_vec}")
    print(f"   scene.Vector3: {scene_vec}")
    print()
    
    print("4. Benefits:")
    print("   ✓ No name conflicts")
    print("   ✓ Clear which module each class comes from")
    print("   ✓ Easy to understand code")
    print()

def demonstrate_import_as():
    """Demonstrate using import aliases to avoid conflicts"""
    print("\n=== Solution 2: Import Aliases ===\n")
    
    # Simulate importing with aliases
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    print("1. Import with aliases:")
    print("   import math3d as math")
    print("   import physics as phys")
    print("   import scene as scn")
    print()
    
    print("2. Access with aliases:")
    print("   math.Vector3(1, 2, 3)")
    print("   phys.Vector3(1, 2, 3)")
    print("   scn.Vector3(1, 2, 3)")
    print()
    
    # Demonstrate usage
    print("3. Usage example:")
    math_vec = math_module['Vector3'](1, 2, 3)
    physics_vec = physics_module['Vector3'](4, 5, 6)
    scene_vec = scene_module['Vector3'](7, 8, 9)
    
    print(f"   math.Vector3: {math_vec}")
    print(f"   phys.Vector3: {physics_vec}")
    print(f"   scn.Vector3: {scene_vec}")
    print()
    
    print("4. Benefits:")
    print("   ✓ Shorter module names")
    print("   ✓ No name conflicts")
    print("   ✓ Still clear which module each class comes from")
    print()

def demonstrate_selective_imports():
    """Demonstrate selective imports with aliases"""
    print("\n=== Solution 3: Selective Imports with Aliases ===\n")
    
    # Simulate selective imports
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    print("1. Selective imports with aliases:")
    print("   from math3d import Vector3 as MathVector3")
    print("   from physics import Vector3 as PhysicsVector3")
    print("   from scene import Vector3 as SceneVector3")
    print()
    
    print("2. Access with unique names:")
    print("   MathVector3(1, 2, 3)")
    print("   PhysicsVector3(1, 2, 3)")
    print("   SceneVector3(1, 2, 3)")
    print()
    
    # Demonstrate usage
    print("3. Usage example:")
    math_vec = math_module['Vector3'](1, 2, 3)
    physics_vec = physics_module['Vector3'](4, 5, 6)
    scene_vec = scene_module['Vector3'](7, 8, 9)
    
    print(f"   MathVector3: {math_vec}")
    print(f"   PhysicsVector3: {physics_vec}")
    print(f"   SceneVector3: {scene_vec}")
    print()
    
    print("4. Benefits:")
    print("   ✓ No name conflicts")
    print("   ✓ Clear, descriptive names")
    print("   ✓ Direct access without module prefix")
    print()

def demonstrate_namespace_organization():
    """Demonstrate organizing namespaces properly"""
    print("\n=== Solution 4: Proper Namespace Organization ===\n")
    
    print("1. Organize imports by functionality:")
    print("   # Math-related imports")
    print("   from math3d import Vector3 as MathVector3, Matrix4")
    print("   from math3d import vector_add, vector_scale")
    print()
    print("   # Physics-related imports")
    print("   from physics import Vector3 as PhysicsVector3, Transform as PhysicsTransform")
    print("   from physics import apply_force, update_physics")
    print()
    print("   # Scene-related imports")
    print("   from scene import Vector3 as SceneVector3, Transform as SceneTransform")
    print("   from scene import create_object, add_to_scene")
    print()
    
    print("2. Use descriptive aliases:")
    print("   MathVector3 - for mathematical operations")
    print("   PhysicsVector3 - for physics simulation")
    print("   SceneVector3 - for scene positioning")
    print()
    
    print("3. Group related functionality:")
    print("   # Math operations")
    print("   pos = MathVector3(1, 2, 3)")
    print("   vel = MathVector3(0, 1, 0)")
    print("   result = vector_add(pos, vel)")
    print()
    print("   # Physics simulation")
    print("   body = PhysicsVector3(0, 0, 0)")
    print("   force = PhysicsVector3(0, -9.81, 0)")
    print("   apply_force(body, force)")
    print()
    print("   # Scene management")
    print("   scene_pos = SceneVector3(0, 0, 0)")
    print("   obj = create_object('cube', SceneTransform(scene_pos))")
    print()

# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

def demonstrate_practical_namespace_usage():
    """Demonstrate practical namespace usage in 3D graphics"""
    print("\n=== Practical Namespace Usage Example ===\n")
    
    # Simulate the modules
    math_module = simulate_math_module()
    physics_module = simulate_physics_module()
    scene_module = simulate_scene_module()
    
    # Extract with proper aliases
    MathVector3 = math_module['Vector3']
    vector_add = math_module['vector_add']
    vector_scale = math_module['vector_scale']
    
    PhysicsVector3 = physics_module['Vector3']
    PhysicsTransform = physics_module['Transform']
    apply_force = physics_module['apply_force']
    update_physics = physics_module['update_physics']
    
    SceneVector3 = scene_module['Vector3']
    SceneTransform = scene_module['Transform']
    create_object = scene_module['create_object']
    add_to_scene = scene_module['add_to_scene']
    create_camera = scene_module['create_camera']
    
    print("Creating a 3D scene with proper namespace management:")
    print()
    
    # 1. Math operations
    print("1. Math operations:")
    position = MathVector3(0, 0, 0)
    velocity = MathVector3(0, 1, 0)
    acceleration = MathVector3(0, -9.81, 0)
    
    # Update position using math
    new_position = vector_add(position, velocity)
    print(f"   Position: {position}")
    print(f"   Velocity: {velocity}")
    print(f"   New position: {new_position}")
    print()
    
    # 2. Physics simulation
    print("2. Physics simulation:")
    physics_body = PhysicsVector3(0, 10, 0)  # Start at height 10
    gravity = PhysicsVector3(0, -9.81, 0)
    
    # Apply physics
    apply_force(physics_body, gravity)
    print(f"   Physics body: {physics_body}")
    print(f"   Applied gravity: {gravity}")
    print()
    
    # 3. Scene management
    print("3. Scene management:")
    scene_position = SceneVector3(0, 0, 0)
    camera_position = SceneVector3(0, 5, -10)
    camera_target = SceneVector3(0, 0, 0)
    
    # Create scene objects
    transform = SceneTransform(scene_position)
    cube = create_object("cube", transform)
    camera = create_camera(camera_position, camera_target)
    
    print(f"   Scene position: {scene_position}")
    print(f"   Created object: {cube}")
    print(f"   Created camera: {camera}")
    print()
    
    print("Scene setup complete with proper namespace separation!")

# ============================================================================
# NAMESPACE BEST PRACTICES
# ============================================================================

def demonstrate_namespace_best_practices():
    """Demonstrate namespace management best practices"""
    print("\n=== Namespace Management Best Practices ===\n")
    
    print("1. Always use explicit imports:")
    print("   # Good:")
    print("   from math3d import Vector3 as MathVector3")
    print("   from physics import Vector3 as PhysicsVector3")
    print()
    print("   # Bad:")
    print("   from math3d import *")
    print("   from physics import *")
    print()
    
    print("2. Use descriptive aliases:")
    print("   # Good:")
    print("   from math3d import Vector3 as MathVector3")
    print("   from physics import Vector3 as PhysicsVector3")
    print()
    print("   # Bad:")
    print("   from math3d import Vector3 as V1")
    print("   from physics import Vector3 as V2")
    print()
    
    print("3. Group related imports:")
    print("   # Math-related imports")
    print("   from math3d import Vector3, Matrix4, Quaternion")
    print("   from math3d import vector_add, vector_scale, vector_dot")
    print()
    print("   # Physics-related imports")
    print("   from physics import RigidBody, Collider, PhysicsWorld")
    print("   from physics import apply_force, update_physics")
    print()
    
    print("4. Use module imports for large modules:")
    print("   # Good for large modules:")
    print("   import numpy as np")
    print("   import matplotlib.pyplot as plt")
    print("   import pygame as pg")
    print()
    
    print("5. Document your import strategy:")
    print("   # Import strategy:")
    print("   # - Math operations: MathVector3, MathMatrix4")
    print("   # - Physics simulation: PhysicsVector3, PhysicsTransform")
    print("   # - Scene management: SceneVector3, SceneTransform")
    print()

# ============================================================================
# COMMON NAMESPACE PROBLEMS
# ============================================================================

def demonstrate_common_problems():
    """Demonstrate common namespace problems and solutions"""
    print("\n=== Common Namespace Problems and Solutions ===\n")
    
    print("1. Problem: Import order matters")
    print("   from module_a import Vector3")
    print("   from module_b import Vector3  # Overwrites module_a.Vector3!")
    print()
    print("   Solution: Use aliases")
    print("   from module_a import Vector3 as Vector3A")
    print("   from module_b import Vector3 as Vector3B")
    print()
    
    print("2. Problem: Wildcard imports")
    print("   from module_a import *")
    print("   from module_b import *  # Many conflicts!")
    print()
    print("   Solution: Explicit imports")
    print("   from module_a import Vector3, Matrix4")
    print("   from module_b import PhysicsVector3, PhysicsMatrix4")
    print()
    
    print("3. Problem: Long module names")
    print("   from very_long_module_name import Vector3")
    print("   # Code becomes hard to read")
    print()
    print("   Solution: Use aliases")
    print("   from very_long_module_name import Vector3 as VLMVector3")
    print("   # Or import the module")
    print("   import very_long_module_name as vlm")
    print()
    
    print("4. Problem: Circular imports")
    print("   # module_a.py")
    print("   from module_b import ClassB")
    print("   # module_b.py")
    print("   from module_a import ClassA  # Circular import!")
    print()
    print("   Solution: Restructure or use lazy imports")
    print("   def function_that_needs_class_b():")
    print("       from module_b import ClassB  # Import when needed")
    print("       return ClassB()")
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate namespace management"""
    print("=== Python Namespace Management Demo ===\n")
    
    # Demonstrate namespace conflicts
    demonstrate_namespace_conflicts()
    demonstrate_bad_imports()
    
    # Demonstrate solutions
    demonstrate_module_imports()
    demonstrate_import_as()
    demonstrate_selective_imports()
    demonstrate_namespace_organization()
    
    # Show practical usage
    demonstrate_practical_namespace_usage()
    
    # Show best practices
    demonstrate_namespace_best_practices()
    
    # Show common problems
    demonstrate_common_problems()
    
    print("\n" + "="*60)
    print("Namespace management demo completed successfully!")
    print("\nKey takeaways:")
    print("✓ Always use explicit imports, never wildcard imports (*)")
    print("✓ Use descriptive aliases to avoid name conflicts")
    print("✓ Group related imports together")
    print("✓ Use module imports for large modules")
    print("✓ Document your import strategy")
    print("✓ Be aware of import order and its effects")
    print("✓ Use lazy imports to avoid circular dependencies")

if __name__ == "__main__":
    main()
