#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Package Structure Example

Demonstrates package creation, organization, and __init__.py files
with 3D graphics applications.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# ============================================================================
# SIMULATED PACKAGE STRUCTURE
# ============================================================================

# This example simulates a 3D graphics package structure
# In a real project, these would be separate files in directories

class PackageStructure:
    """Simulates a 3D graphics package structure"""
    
    def __init__(self):
        self.packages = {}
        self.modules = {}
        self.init_files = {}
    
    def create_package(self, name: str, description: str = ""):
        """Create a new package"""
        self.packages[name] = {
            'name': name,
            'description': description,
            'modules': [],
            'subpackages': [],
            'version': '1.0.0'
        }
        print(f"   Created package: {name}")
    
    def create_module(self, package_name: str, module_name: str, functions: List[str]):
        """Create a module within a package"""
        if package_name not in self.packages:
            self.create_package(package_name)
        
        module_info = {
            'name': module_name,
            'package': package_name,
            'functions': functions,
            'file': f"{package_name}/{module_name}.py"
        }
        
        self.modules[f"{package_name}.{module_name}"] = module_info
        self.packages[package_name]['modules'].append(module_name)
        print(f"   Created module: {package_name}.{module_name}")
    
    def create_init_file(self, package_name: str, imports: List[str] = None, exports: List[str] = None):
        """Create an __init__.py file for a package"""
        init_content = {
            'package': package_name,
            'imports': imports or [],
            'exports': exports or [],
            'version': '1.0.0',
            'description': self.packages.get(package_name, {}).get('description', '')
        }
        
        self.init_files[package_name] = init_content
        print(f"   Created __init__.py for package: {package_name}")

# ============================================================================
# 3D GRAPHICS PACKAGE STRUCTURE
# ============================================================================

def create_3d_graphics_package():
    """Create a complete 3D graphics package structure"""
    print("=== Creating 3D Graphics Package Structure ===\n")
    
    pkg = PackageStructure()
    
    # Main package
    pkg.create_package("graphics3d", "Complete 3D graphics library")
    
    # Math subpackage
    pkg.create_package("graphics3d.math", "3D mathematics utilities")
    pkg.create_module("graphics3d.math", "vectors", [
        "Vector3", "vector_add", "vector_subtract", "vector_scale",
        "vector_magnitude", "vector_normalize", "vector_dot", "vector_cross"
    ])
    pkg.create_module("graphics3d.math", "matrices", [
        "Matrix4", "matrix_multiply", "matrix_inverse", "matrix_transpose",
        "create_rotation_matrix", "create_translation_matrix", "create_scale_matrix"
    ])
    pkg.create_module("graphics3d.math", "quaternions", [
        "Quaternion", "quaternion_multiply", "quaternion_rotate",
        "quaternion_from_axis_angle", "quaternion_to_matrix"
    ])
    
    # Geometry subpackage
    pkg.create_package("graphics3d.geometry", "3D geometry primitives")
    pkg.create_module("graphics3d.geometry", "primitives", [
        "create_cube", "create_sphere", "create_cylinder", "create_cone",
        "create_plane", "create_torus"
    ])
    pkg.create_module("graphics3d.geometry", "mesh", [
        "Mesh", "load_mesh", "save_mesh", "calculate_normals",
        "calculate_bounding_box", "optimize_mesh"
    ])
    
    # Rendering subpackage
    pkg.create_package("graphics3d.rendering", "3D rendering engine")
    pkg.create_module("graphics3d.rendering", "renderer", [
        "Renderer", "setup_renderer", "begin_frame", "end_frame",
        "clear_screen", "set_viewport"
    ])
    pkg.create_module("graphics3d.rendering", "shaders", [
        "Shader", "load_shader", "compile_shader", "use_shader",
        "set_uniform", "set_attribute"
    ])
    pkg.create_module("graphics3d.rendering", "materials", [
        "Material", "create_material", "set_texture", "set_properties",
        "apply_material"
    ])
    
    # Scene subpackage
    pkg.create_package("graphics3d.scene", "Scene management")
    pkg.create_module("graphics3d.scene", "scene", [
        "Scene", "add_object", "remove_object", "update_scene",
        "render_scene", "clear_scene"
    ])
    pkg.create_module("graphics3d.scene", "camera", [
        "Camera", "set_position", "set_target", "set_fov",
        "get_view_matrix", "get_projection_matrix"
    ])
    pkg.create_module("graphics3d.scene", "lighting", [
        "Light", "DirectionalLight", "PointLight", "SpotLight",
        "set_light_properties", "update_lighting"
    ])
    
    # Animation subpackage
    pkg.create_package("graphics3d.animation", "Animation system")
    pkg.create_module("graphics3d.animation", "animator", [
        "Animator", "add_animation", "play_animation", "stop_animation",
        "update_animation", "set_animation_speed"
    ])
    pkg.create_module("graphics3d.animation", "keyframes", [
        "Keyframe", "create_keyframe", "interpolate_keyframes",
        "ease_in", "ease_out", "ease_in_out"
    ])
    
    # Physics subpackage
    pkg.create_package("graphics3d.physics", "Physics simulation")
    pkg.create_module("graphics3d.physics", "rigidbody", [
        "RigidBody", "set_mass", "set_velocity", "apply_force",
        "update_physics", "check_collision"
    ])
    pkg.create_module("graphics3d.physics", "collision", [
        "Collider", "BoxCollider", "SphereCollider", "MeshCollider",
        "detect_collision", "resolve_collision"
    ])
    
    return pkg

# ============================================================================
# __INIT__.PY FILE EXAMPLES
# ============================================================================

def create_init_files(pkg: PackageStructure):
    """Create __init__.py files for the package structure"""
    print("\n=== Creating __init__.py Files ===\n")
    
    # Main package __init__.py
    main_exports = [
        "math", "geometry", "rendering", "scene", "animation", "physics"
    ]
    pkg.create_init_file("graphics3d", exports=main_exports)
    
    # Math package __init__.py
    math_imports = [
        "from .vectors import Vector3, vector_add, vector_subtract",
        "from .matrices import Matrix4, matrix_multiply, create_rotation_matrix",
        "from .quaternions import Quaternion, quaternion_multiply"
    ]
    math_exports = [
        "Vector3", "Matrix4", "Quaternion",
        "vector_add", "vector_subtract", "matrix_multiply", "quaternion_multiply"
    ]
    pkg.create_init_file("graphics3d.math", imports=math_imports, exports=math_exports)
    
    # Geometry package __init__.py
    geometry_imports = [
        "from .primitives import create_cube, create_sphere, create_cylinder",
        "from .mesh import Mesh, load_mesh, save_mesh"
    ]
    geometry_exports = [
        "create_cube", "create_sphere", "create_cylinder",
        "Mesh", "load_mesh", "save_mesh"
    ]
    pkg.create_init_file("graphics3d.geometry", imports=geometry_imports, exports=geometry_exports)
    
    # Rendering package __init__.py
    rendering_imports = [
        "from .renderer import Renderer, setup_renderer",
        "from .shaders import Shader, load_shader, compile_shader",
        "from .materials import Material, create_material"
    ]
    rendering_exports = [
        "Renderer", "Shader", "Material",
        "setup_renderer", "load_shader", "create_material"
    ]
    pkg.create_init_file("graphics3d.rendering", imports=rendering_imports, exports=rendering_exports)
    
    # Scene package __init__.py
    scene_imports = [
        "from .scene import Scene, add_object, render_scene",
        "from .camera import Camera, set_position, set_target",
        "from .lighting import Light, DirectionalLight, PointLight"
    ]
    scene_exports = [
        "Scene", "Camera", "Light", "DirectionalLight", "PointLight",
        "add_object", "render_scene", "set_position", "set_target"
    ]
    pkg.create_init_file("graphics3d.scene", imports=scene_imports, exports=scene_exports)
    
    # Animation package __init__.py
    animation_imports = [
        "from .animator import Animator, add_animation, play_animation",
        "from .keyframes import Keyframe, create_keyframe, interpolate_keyframes"
    ]
    animation_exports = [
        "Animator", "Keyframe",
        "add_animation", "play_animation", "create_keyframe"
    ]
    pkg.create_init_file("graphics3d.animation", imports=animation_imports, exports=animation_exports)
    
    # Physics package __init__.py
    physics_imports = [
        "from .rigidbody import RigidBody, set_mass, apply_force",
        "from .collision import Collider, BoxCollider, SphereCollider, detect_collision"
    ]
    physics_exports = [
        "RigidBody", "Collider", "BoxCollider", "SphereCollider",
        "set_mass", "apply_force", "detect_collision"
    ]
    pkg.create_init_file("graphics3d.physics", imports=physics_imports, exports=physics_exports)

# ============================================================================
# PACKAGE USAGE EXAMPLES
# ============================================================================

def demonstrate_package_imports():
    """Demonstrate different ways to import from packages"""
    print("\n=== Package Import Examples ===\n")
    
    # Simulate different import styles
    import_examples = [
        "# Import entire package",
        "import graphics3d",
        "",
        "# Import specific subpackage",
        "import graphics3d.math",
        "import graphics3d.rendering",
        "",
        "# Import specific modules",
        "from graphics3d.math import vectors",
        "from graphics3d.geometry import primitives",
        "",
        "# Import specific functions/classes",
        "from graphics3d.math.vectors import Vector3, vector_add",
        "from graphics3d.rendering.renderer import Renderer",
        "",
        "# Import with aliases",
        "import graphics3d.math as math3d",
        "from graphics3d.rendering import renderer as rdr",
        "",
        "# Import everything (not recommended)",
        "from graphics3d.math.vectors import *",
        "",
        "# Relative imports (within package)",
        "from .math import vectors",
        "from ..rendering import renderer",
        "from ...scene import camera"
    ]
    
    for example in import_examples:
        print(example)

def demonstrate_package_usage():
    """Demonstrate how to use the package structure"""
    print("\n=== Package Usage Examples ===\n")
    
    usage_examples = [
        "# Create a 3D scene",
        "from graphics3d.scene import Scene, Camera",
        "from graphics3d.geometry import create_cube, create_sphere",
        "from graphics3d.math import Vector3",
        "",
        "# Initialize scene",
        "scene = Scene()",
        "camera = Camera()",
        "camera.set_position(Vector3(0, 5, -10))",
        "camera.set_target(Vector3(0, 0, 0))",
        "",
        "# Add objects to scene",
        "cube = create_cube(size=2.0)",
        "sphere = create_sphere(radius=1.0)",
        "scene.add_object(cube)",
        "scene.add_object(sphere)",
        "",
        "# Setup rendering",
        "from graphics3d.rendering import setup_renderer, Renderer",
        "renderer = setup_renderer(width=800, height=600)",
        "",
        "# Render loop",
        "while True:",
        "    renderer.begin_frame()",
        "    scene.render_scene(renderer, camera)",
        "    renderer.end_frame()",
        "",
        "# Add physics",
        "from graphics3d.physics import RigidBody, apply_force",
        "rigidbody = RigidBody(mass=1.0)",
        "apply_force(rigidbody, Vector3(0, -9.81, 0))  # Gravity",
        "",
        "# Add animation",
        "from graphics3d.animation import Animator, create_keyframe",
        "animator = Animator()",
        "keyframe1 = create_keyframe(time=0, position=Vector3(0, 0, 0))",
        "keyframe2 = create_keyframe(time=1, position=Vector3(0, 5, 0))",
        "animator.add_animation('bounce', [keyframe1, keyframe2])",
        "animator.play_animation('bounce')"
    ]
    
    for example in usage_examples:
        print(example)

# ============================================================================
# PACKAGE STRUCTURE VISUALIZATION
# ============================================================================

def visualize_package_structure(pkg: PackageStructure):
    """Visualize the complete package structure"""
    print("\n=== Package Structure Visualization ===\n")
    
    def print_tree(package_name: str, indent: str = ""):
        """Print package structure as a tree"""
        if package_name not in pkg.packages:
            return
        
        package = pkg.packages[package_name]
        print(f"{indent}📦 {package['name']}/")
        
        # Print __init__.py
        if package_name in pkg.init_files:
            init = pkg.init_files[package_name]
            print(f"{indent}   📄 __init__.py")
            if init['exports']:
                print(f"{indent}      Exports: {', '.join(init['exports'][:3])}{'...' if len(init['exports']) > 3 else ''}")
        
        # Print modules
        for module_name in package['modules']:
            module_key = f"{package_name}.{module_name}"
            if module_key in pkg.modules:
                module = pkg.modules[module_key]
                print(f"{indent}   📄 {module_name}.py")
                if module['functions']:
                    print(f"{indent}      Functions: {', '.join(module['functions'][:3])}{'...' if len(module['functions']) > 3 else ''}")
        
        # Print subpackages
        for subpackage in package.get('subpackages', []):
            print_tree(subpackage, indent + "   ")
    
    # Print main package structure
    print_tree("graphics3d")

def show_package_statistics(pkg: PackageStructure):
    """Show statistics about the package structure"""
    print("\n=== Package Statistics ===\n")
    
    total_modules = len(pkg.modules)
    total_packages = len(pkg.packages)
    total_init_files = len(pkg.init_files)
    
    total_functions = 0
    for module in pkg.modules.values():
        total_functions += len(module['functions'])
    
    print(f"📊 Package Statistics:")
    print(f"   Total packages: {total_packages}")
    print(f"   Total modules: {total_modules}")
    print(f"   Total __init__.py files: {total_init_files}")
    print(f"   Total functions: {total_functions}")
    
    print(f"\n📦 Package Breakdown:")
    for package_name, package in pkg.packages.items():
        module_count = len(package['modules'])
        print(f"   {package_name}: {module_count} modules")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate package structure"""
    print("=== 3D Graphics Package Structure Demo ===\n")
    
    # Create the package structure
    pkg = create_3d_graphics_package()
    
    # Create __init__.py files
    create_init_files(pkg)
    
    # Demonstrate imports and usage
    demonstrate_package_imports()
    demonstrate_package_usage()
    
    # Visualize the structure
    visualize_package_structure(pkg)
    show_package_statistics(pkg)
    
    print("\n" + "="*60)
    print("Package structure demo completed successfully!")
    print("\nKey benefits of this package structure:")
    print("✓ Modular organization: Related functionality grouped together")
    print("✓ Clear separation of concerns: Math, geometry, rendering, etc.")
    print("✓ Easy imports: Users can import what they need")
    print("✓ Scalable: Easy to add new modules and packages")
    print("✓ Maintainable: Clear structure makes code easier to maintain")
    print("✓ Reusable: Individual components can be used independently")

if __name__ == "__main__":
    main()
