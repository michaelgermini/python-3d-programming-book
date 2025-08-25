#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Module Attributes Example

Demonstrates understanding of module attributes and metadata
with 3D graphics applications.
"""

import sys
import os
import math
from typing import Dict, List, Any, Optional

# ============================================================================
# MODULE METADATA
# ============================================================================

# Module-level attributes and metadata
__version__ = "1.0.0"
__author__ = "3D Graphics Library Team"
__description__ = "Module attributes demonstration for 3D graphics"
__license__ = "MIT"
__maintainer__ = "Graphics Team"
__email__ = "graphics@example.com"

# Module constants
PI = math.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# Module configuration
DEFAULT_VERTEX_COUNT = 1000
DEFAULT_TEXTURE_SIZE = 512
DEFAULT_SHADER_VERSION = "330"

# ============================================================================
# MODULE ATTRIBUTES EXPLANATION
# ============================================================================

def explain_module_attributes():
    """Explain the key module attributes and their purposes"""
    print("=== Module Attributes Explanation ===\n")
    
    print("1. __name__ - Module name:")
    print(f"   Current module name: {__name__}")
    print("   - '__main__' when run directly")
    print("   - Module name when imported")
    print()
    
    print("2. __file__ - Module file path:")
    print(f"   Current module file: {__file__}")
    print("   - Path to the module's source file")
    print("   - None for built-in modules")
    print()
    
    print("3. __doc__ - Module documentation:")
    print(f"   Module docstring: {__doc__[:50]}...")
    print("   - First string in the module")
    print("   - Used for help() and documentation")
    print()
    
    print("4. __version__ - Module version:")
    print(f"   Current version: {__version__}")
    print("   - Semantic versioning (major.minor.patch)")
    print("   - Used for compatibility checking")
    print()
    
    print("5. __author__ - Module author:")
    print(f"   Author: {__author__}")
    print("   - Credits and attribution")
    print("   - Contact information")
    print()

def demonstrate_name_main():
    """Demonstrate the __name__ == '__main__' pattern"""
    print("\n=== __name__ == '__main__' Pattern ===\n")
    
    print("1. When module is run directly:")
    print(f"   __name__ = {__name__}")
    if __name__ == "__main__":
        print("   ✓ This code runs when module is executed directly")
    else:
        print("   ✗ This code runs when module is imported")
    print()
    
    print("2. Common pattern:")
    print("   if __name__ == '__main__':")
    print("       main()  # Run main function")
    print("   else:")
    print("       print('Module imported')")
    print()
    
    print("3. Benefits:")
    print("   ✓ Code can be both imported and run directly")
    print("   ✓ Prevents unwanted execution when imported")
    print("   ✓ Allows testing and demonstration")
    print()

# ============================================================================
# MODULE METADATA FUNCTIONS
# ============================================================================

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about this module"""
    return {
        'name': __name__,
        'file': __file__,
        'doc': __doc__,
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'maintainer': __maintainer__,
        'email': __email__,
        'constants': {
            'PI': PI,
            'DEG_TO_RAD': DEG_TO_RAD,
            'RAD_TO_DEG': RAD_TO_DEG
        },
        'config': {
            'DEFAULT_VERTEX_COUNT': DEFAULT_VERTEX_COUNT,
            'DEFAULT_TEXTURE_SIZE': DEFAULT_TEXTURE_SIZE,
            'DEFAULT_SHADER_VERSION': DEFAULT_SHADER_VERSION
        }
    }

def print_module_info():
    """Print formatted module information"""
    print("\n=== Module Information ===\n")
    
    info = get_module_info()
    
    print("📦 Module Details:")
    print(f"   Name: {info['name']}")
    print(f"   File: {info['file']}")
    print(f"   Version: {info['version']}")
    print(f"   Author: {info['author']}")
    print(f"   Description: {info['description']}")
    print(f"   License: {info['license']}")
    print(f"   Maintainer: {info['maintainer']}")
    print(f"   Email: {info['email']}")
    
    print("\n🔢 Constants:")
    for name, value in info['constants'].items():
        print(f"   {name}: {value}")
    
    print("\n⚙️ Configuration:")
    for name, value in info['config'].items():
        print(f"   {name}: {value}")

# ============================================================================
# MODULE ATTRIBUTES IN PRACTICE
# ============================================================================

def demonstrate_module_usage():
    """Demonstrate how module attributes are used in practice"""
    print("\n=== Module Attributes in Practice ===\n")
    
    print("1. Version checking:")
    print("   import module_attributes")
    print("   if module_attributes.__version__ >= '1.0.0':")
    print("       print('Compatible version')")
    print()
    
    print("2. Configuration access:")
    print("   from module_attributes import DEFAULT_VERTEX_COUNT")
    print(f"   vertex_count = {DEFAULT_VERTEX_COUNT}")
    print()
    
    print("3. Constants usage:")
    print("   from module_attributes import PI, DEG_TO_RAD")
    print(f"   angle_rad = 90 * {DEG_TO_RAD}")
    print(f"   angle_rad = {90 * DEG_TO_RAD}")
    print()
    
    print("4. Module information:")
    print("   import module_attributes")
    print("   print(module_attributes.__doc__)")
    print("   print(module_attributes.__author__)")

# ============================================================================
# 3D GRAPHICS MODULE SIMULATION
# ============================================================================

class GraphicsModule:
    """Simulate a 3D graphics module with attributes"""
    
    def __init__(self, name: str):
        self.name = name
        self.__version__ = "1.0.0"
        self.__author__ = "Graphics Team"
        self.__description__ = f"3D Graphics module: {name}"
        self.functions = []
        self.constants = {}
    
    def add_function(self, name: str, description: str):
        """Add a function to the module"""
        self.functions.append({'name': name, 'description': description})
    
    def add_constant(self, name: str, value: Any):
        """Add a constant to the module"""
        self.constants[name] = value
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information"""
        return {
            'name': self.name,
            'version': self.__version__,
            'author': self.__author__,
            'description': self.__description__,
            'functions': self.functions,
            'constants': self.constants
        }

def create_graphics_modules():
    """Create simulated 3D graphics modules"""
    print("\n=== Creating 3D Graphics Modules ===\n")
    
    # Math module
    math_module = GraphicsModule("math3d")
    math_module.add_function("vector_add", "Add two 3D vectors")
    math_module.add_function("vector_scale", "Scale a 3D vector")
    math_module.add_function("matrix_multiply", "Multiply two 4x4 matrices")
    math_module.add_constant("PI", 3.14159)
    math_module.add_constant("DEG_TO_RAD", 0.0174533)
    math_module.add_constant("RAD_TO_DEG", 57.2958)
    
    # Geometry module
    geometry_module = GraphicsModule("geometry")
    geometry_module.add_function("create_cube", "Create a cube mesh")
    geometry_module.add_function("create_sphere", "Create a sphere mesh")
    geometry_module.add_function("create_cylinder", "Create a cylinder mesh")
    geometry_module.add_constant("DEFAULT_SEGMENTS", 16)
    geometry_module.add_constant("DEFAULT_RADIUS", 1.0)
    
    # Rendering module
    rendering_module = GraphicsModule("rendering")
    rendering_module.add_function("setup_renderer", "Initialize the renderer")
    rendering_module.add_function("render_scene", "Render a 3D scene")
    rendering_module.add_function("create_shader", "Create a shader program")
    rendering_module.add_constant("DEFAULT_WIDTH", 800)
    rendering_module.add_constant("DEFAULT_HEIGHT", 600)
    rendering_module.add_constant("DEFAULT_FOV", 45.0)
    
    return math_module, geometry_module, rendering_module

def demonstrate_module_discovery():
    """Demonstrate discovering module attributes"""
    print("\n=== Module Discovery ===\n")
    
    # Create modules
    math_module, geometry_module, rendering_module = create_graphics_modules()
    modules = [math_module, geometry_module, rendering_module]
    
    print("1. Discovering module information:")
    for module in modules:
        info = module.get_info()
        print(f"\n📦 {info['name']}:")
        print(f"   Version: {info['version']}")
        print(f"   Author: {info['author']}")
        print(f"   Description: {info['description']}")
        
        print(f"   Functions ({len(info['functions'])}):")
        for func in info['functions']:
            print(f"     - {func['name']}: {func['description']}")
        
        print(f"   Constants ({len(info['constants'])}):")
        for name, value in info['constants'].items():
            print(f"     - {name}: {value}")
    
    print("\n2. Version compatibility check:")
    for module in modules:
        version = module.__version__
        if version >= "1.0.0":
            print(f"   ✓ {module.name} version {version} is compatible")
        else:
            print(f"   ✗ {module.name} version {version} needs update")

# ============================================================================
# MODULE ATTRIBUTES FOR CONFIGURATION
# ============================================================================

def demonstrate_configuration():
    """Demonstrate using module attributes for configuration"""
    print("\n=== Module Configuration ===\n")
    
    print("1. Default configuration:")
    print(f"   Default vertex count: {DEFAULT_VERTEX_COUNT}")
    print(f"   Default texture size: {DEFAULT_TEXTURE_SIZE}")
    print(f"   Default shader version: {DEFAULT_SHADER_VERSION}")
    print()
    
    print("2. Configuration override:")
    print("   # In user code:")
    print("   import module_attributes")
    print("   module_attributes.DEFAULT_VERTEX_COUNT = 2000")
    print("   module_attributes.DEFAULT_TEXTURE_SIZE = 1024")
    print()
    
    print("3. Environment-based configuration:")
    import os
    vertex_count = os.getenv('GRAPHICS_VERTEX_COUNT', DEFAULT_VERTEX_COUNT)
    texture_size = os.getenv('GRAPHICS_TEXTURE_SIZE', DEFAULT_TEXTURE_SIZE)
    shader_version = os.getenv('GRAPHICS_SHADER_VERSION', DEFAULT_SHADER_VERSION)
    
    print(f"   Environment vertex count: {vertex_count}")
    print(f"   Environment texture size: {texture_size}")
    print(f"   Environment shader version: {shader_version}")

# ============================================================================
# MODULE DOCUMENTATION
# ============================================================================

def demonstrate_documentation():
    """Demonstrate module documentation attributes"""
    print("\n=== Module Documentation ===\n")
    
    print("1. Module docstring:")
    print("   help(module_attributes)")
    print("   print(module_attributes.__doc__)")
    print()
    
    print("2. Function documentation:")
    print("   help(module_attributes.get_module_info)")
    print("   print(module_attributes.get_module_info.__doc__)")
    print()
    
    print("3. Class documentation:")
    print("   help(module_attributes.GraphicsModule)")
    print("   print(module_attributes.GraphicsModule.__doc__)")
    print()
    
    print("4. Interactive help:")
    print("   >>> import module_attributes")
    print("   >>> help(module_attributes)")
    print("   >>> dir(module_attributes)")

# ============================================================================
# MODULE ATTRIBUTES FOR DEBUGGING
# ============================================================================

def demonstrate_debugging():
    """Demonstrate using module attributes for debugging"""
    print("\n=== Module Debugging ===\n")
    
    print("1. Module location debugging:")
    print(f"   Module file: {__file__}")
    print(f"   Module name: {__name__}")
    print(f"   Module path: {os.path.dirname(__file__)}")
    print()
    
    print("2. Module content inspection:")
    print("   dir(module_attributes)")
    print("   [attr for attr in dir(module_attributes) if not attr.startswith('_')]")
    print()
    
    print("3. Module version debugging:")
    print(f"   Module version: {__version__}")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    print()
    
    print("4. Module import debugging:")
    print("   import sys")
    print("   print(sys.modules.keys())")
    print("   print('module_attributes' in sys.modules)")

# ============================================================================
# PRACTICAL 3D GRAPHICS EXAMPLE
# ============================================================================

def demonstrate_practical_usage():
    """Demonstrate practical usage of module attributes in 3D graphics"""
    print("\n=== Practical 3D Graphics Example ===\n")
    
    # Simulate importing a 3D graphics module
    print("1. Module import and version check:")
    print("   import graphics3d")
    print("   if graphics3d.__version__ >= '1.0.0':")
    print("       print('Using compatible graphics library')")
    print()
    
    print("2. Configuration setup:")
    print("   # Use module constants for configuration")
    print(f"   vertex_count = {DEFAULT_VERTEX_COUNT}")
    print(f"   texture_size = {DEFAULT_TEXTURE_SIZE}")
    print(f"   shader_version = '{DEFAULT_SHADER_VERSION}'")
    print()
    
    print("3. Module information display:")
    print("   # Display module information to user")
    print(f"   print('Graphics Library: {__description__}')")
    print(f"   print('Version: {__version__}')")
    print(f"   print('Author: {__author__}')")
    print()
    
    print("4. Constants usage:")
    print("   # Use mathematical constants")
    print(f"   angle_degrees = 90")
    print(f"   angle_radians = angle_degrees * {DEG_TO_RAD}")
    print(f"   angle_radians = {90 * DEG_TO_RAD}")
    print()
    
    print("5. Module discovery:")
    print("   # Discover available functions and constants")
    print("   available_functions = [f for f in dir(graphics3d) if callable(getattr(graphics3d, f))]")
    print("   available_constants = [c for c in dir(graphics3d) if not c.startswith('_') and not callable(getattr(graphics3d, c))]")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate module attributes"""
    print("=== Python Module Attributes Demo ===\n")
    
    # Explain module attributes
    explain_module_attributes()
    demonstrate_name_main()
    
    # Show module information
    print_module_info()
    
    # Demonstrate practical usage
    demonstrate_module_usage()
    demonstrate_module_discovery()
    demonstrate_configuration()
    demonstrate_documentation()
    demonstrate_debugging()
    demonstrate_practical_usage()
    
    print("\n" + "="*60)
    print("Module attributes demo completed successfully!")
    print("\nKey takeaways:")
    print("✓ __name__ determines if module is run directly or imported")
    print("✓ __file__ provides the module's file path")
    print("✓ __doc__ contains module documentation")
    print("✓ __version__ enables version compatibility checking")
    print("✓ Module constants provide configuration defaults")
    print("✓ Module attributes enable discovery and debugging")
    print("✓ Use module attributes for professional library development")

if __name__ == "__main__":
    main()
