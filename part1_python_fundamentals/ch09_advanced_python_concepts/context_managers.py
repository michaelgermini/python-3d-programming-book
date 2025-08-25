#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Context Managers Example

Demonstrates context managers, the with statement, custom context managers,
resource cleanup, and exception handling for 3D graphics applications.
"""

import time
import os
import tempfile
import json
from typing import List, Dict, Any, Optional, Union, TextIO
from contextlib import contextmanager
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Context Managers"
__description__ = "Context managers for 3D graphics applications"

# ============================================================================
# BASIC CONTEXT MANAGERS
# ============================================================================

class TimerContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"⏱️  Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"✅ {self.name} completed in {duration:.4f} seconds")
        
        if exc_type is not None:
            print(f"❌ {self.name} failed with {exc_type.__name__}: {exc_val}")
            return False  # Re-raise the exception
        return True

class FileManager:
    """Context manager for file operations"""
    
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self) -> TextIO:
        self.file = open(self.filename, self.mode)
        print(f"📁 Opened file: {self.filename}")
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"📁 Closed file: {self.filename}")
        
        if exc_type is not None:
            print(f"❌ File operation failed: {exc_val}")
            return False
        return True

class TemporaryFileManager:
    """Context manager for temporary files"""
    
    def __init__(self, suffix: str = '.tmp', prefix: str = 'temp_'):
        self.suffix = suffix
        self.prefix = prefix
        self.temp_file = None
        self.filename = None
    
    def __enter__(self) -> TextIO:
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w+',
            suffix=self.suffix,
            prefix=self.prefix,
            delete=False
        )
        self.filename = self.temp_file.name
        print(f"📄 Created temporary file: {self.filename}")
        return self.temp_file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file:
            self.temp_file.close()
            try:
                os.unlink(self.filename)
                print(f"🗑️  Deleted temporary file: {self.filename}")
            except OSError as e:
                print(f"⚠️  Could not delete temporary file: {e}")
        
        if exc_type is not None:
            print(f"❌ Temporary file operation failed: {exc_val}")
            return False
        return True

# ============================================================================
# 3D GRAPHICS CONTEXT MANAGERS
# ============================================================================

class OpenGLContext:
    """Simulated OpenGL context manager"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.is_active = False
    
    def __enter__(self):
        print(f"🎮 Initializing OpenGL context ({self.width}x{self.height})...")
        # Simulate OpenGL initialization
        time.sleep(0.1)
        self.is_active = True
        print("✅ OpenGL context ready")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("🎮 Cleaning up OpenGL context...")
        # Simulate OpenGL cleanup
        time.sleep(0.05)
        self.is_active = False
        print("✅ OpenGL context cleaned up")
        
        if exc_type is not None:
            print(f"❌ OpenGL operation failed: {exc_val}")
            return False
        return True
    
    def render_frame(self):
        """Simulate rendering a frame"""
        if not self.is_active:
            raise RuntimeError("OpenGL context not active")
        print("🎨 Rendering frame...")
        time.sleep(0.01)

class SceneManager:
    """Context manager for 3D scene operations"""
    
    def __init__(self, scene_name: str):
        self.scene_name = scene_name
        self.objects = []
        self.is_loaded = False
    
    def __enter__(self):
        print(f"🎬 Loading scene: {self.scene_name}")
        # Simulate scene loading
        time.sleep(0.1)
        self.is_loaded = True
        print(f"✅ Scene '{self.scene_name}' loaded")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"🎬 Unloading scene: {self.scene_name}")
        # Simulate scene unloading
        time.sleep(0.05)
        self.objects.clear()
        self.is_loaded = False
        print(f"✅ Scene '{self.scene_name}' unloaded")
        
        if exc_type is not None:
            print(f"❌ Scene operation failed: {exc_val}")
            return False
        return True
    
    def add_object(self, obj_name: str):
        """Add object to scene"""
        if not self.is_loaded:
            raise RuntimeError("Scene not loaded")
        self.objects.append(obj_name)
        print(f"➕ Added object '{obj_name}' to scene")

class ResourceManager:
    """Context manager for resource management"""
    
    def __init__(self, resource_type: str):
        self.resource_type = resource_type
        self.resources = []
        self.is_allocated = False
    
    def __enter__(self):
        print(f"🔧 Allocating {self.resource_type} resources...")
        # Simulate resource allocation
        time.sleep(0.05)
        self.is_allocated = True
        print(f"✅ {self.resource_type} resources allocated")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"🔧 Deallocating {self.resource_type} resources...")
        # Simulate resource deallocation
        time.sleep(0.03)
        self.resources.clear()
        self.is_allocated = False
        print(f"✅ {self.resource_type} resources deallocated")
        
        if exc_type is not None:
            print(f"❌ Resource operation failed: {exc_val}")
            return False
        return True
    
    def acquire_resource(self, resource_name: str):
        """Acquire a resource"""
        if not self.is_allocated:
            raise RuntimeError("Resources not allocated")
        self.resources.append(resource_name)
        print(f"🔗 Acquired resource: {resource_name}")

# ============================================================================
# CONTEXT MANAGER UTILITIES
# ============================================================================

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    print(f"📊 Starting performance monitoring: {operation_name}")
    
    try:
        yield
    except Exception as e:
        print(f"❌ {operation_name} failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        print(f"📊 {operation_name} completed in {duration:.4f} seconds")

@contextmanager
def error_handler(operation_name: str, default_value: Any = None):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        print(f"⚠️  {operation_name} encountered error: {e}")
        if default_value is not None:
            print(f"🔄 Using default value: {default_value}")
            return default_value
        raise

@contextmanager
def state_saver(obj, attribute_name: str):
    """Context manager to save and restore object state"""
    original_value = getattr(obj, attribute_name)
    print(f"💾 Saving {attribute_name} = {original_value}")
    
    try:
        yield
    finally:
        setattr(obj, attribute_name, original_value)
        print(f"🔄 Restored {attribute_name} = {original_value}")

@contextmanager
def temporary_directory():
    """Context manager for temporary directory"""
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Created temporary directory: {temp_dir}")
    
    try:
        yield temp_dir
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🗑️  Deleted temporary directory: {temp_dir}")
        except OSError as e:
            print(f"⚠️  Could not delete temporary directory: {e}")

# ============================================================================
# NESTED CONTEXT MANAGERS
# ============================================================================

class NestedContextManager:
    """Demonstrate nested context managers"""
    
    def __init__(self, name: str):
        self.name = name
        self.level = 0
    
    def __enter__(self):
        self.level += 1
        indent = "  " * (self.level - 1)
        print(f"{indent}🚪 Entering {self.name} (level {self.level})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        indent = "  " * (self.level - 1)
        print(f"{indent}🚪 Exiting {self.name} (level {self.level})")
        self.level -= 1
        
        if exc_type is not None:
            print(f"{indent}❌ {self.name} failed: {exc_val}")
            return False
        return True

# ============================================================================
# EXAMPLE CLASSES
# ============================================================================

@dataclass
class Vector3D:
    """Simple 3D vector class"""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"

class GraphicsObject:
    """Example graphics object"""
    
    def __init__(self, name: str, position: Vector3D):
        self.name = name
        self.position = position
        self.visible = True
    
    def __str__(self):
        return f"GraphicsObject('{self.name}', {self.position})"

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_context_managers():
    """Demonstrate basic context managers"""
    print("=== Basic Context Managers Demo ===\n")
    
    # Timer context manager
    print("1. Timer context manager:")
    with TimerContext("Data Processing"):
        time.sleep(0.1)  # Simulate work
        print("   Processing data...")
    
    # File manager
    print("\n2. File manager:")
    with FileManager("test_data.json", 'w') as f:
        data = {"name": "test", "value": 42}
        json.dump(data, f)
        print("   Wrote data to file")
    
    # Temporary file manager
    print("\n3. Temporary file manager:")
    with TemporaryFileManager(suffix='.json', prefix='data_') as temp_file:
        temp_file.write('{"temp": "data"}')
        temp_file.flush()
        print(f"   Wrote to temporary file: {temp_file.name}")
    
    print()

def demonstrate_3d_graphics_context_managers():
    """Demonstrate 3D graphics context managers"""
    print("=== 3D Graphics Context Managers Demo ===\n")
    
    # OpenGL context
    print("1. OpenGL context manager:")
    with OpenGLContext(1024, 768) as gl_context:
        gl_context.render_frame()
        gl_context.render_frame()
        print("   Rendered multiple frames")
    
    # Scene manager
    print("\n2. Scene manager:")
    with SceneManager("Main Scene") as scene:
        scene.add_object("Player")
        scene.add_object("Enemy")
        scene.add_object("Terrain")
        print(f"   Scene contains {len(scene.objects)} objects")
    
    # Resource manager
    print("\n3. Resource manager:")
    with ResourceManager("Texture") as texture_manager:
        texture_manager.acquire_resource("player_texture.png")
        texture_manager.acquire_resource("enemy_texture.png")
        print(f"   Acquired {len(texture_manager.resources)} textures")
    
    print()

def demonstrate_context_manager_utilities():
    """Demonstrate context manager utilities"""
    print("=== Context Manager Utilities Demo ===\n")
    
    # Performance monitor
    print("1. Performance monitor:")
    with performance_monitor("Complex Calculation"):
        time.sleep(0.05)  # Simulate work
        result = sum(i**2 for i in range(1000))
        print(f"   Calculation result: {result}")
    
    # Error handler
    print("\n2. Error handler:")
    with error_handler("Risky Operation", default_value="fallback"):
        if random.random() < 0.5:  # 50% chance of error
            raise ValueError("Random error occurred")
        print("   Operation completed successfully")
    
    # State saver
    print("\n3. State saver:")
    obj = GraphicsObject("Test", Vector3D(1, 2, 3))
    print(f"   Original position: {obj.position}")
    
    with state_saver(obj, 'position'):
        obj.position = Vector3D(10, 20, 30)
        print(f"   Modified position: {obj.position}")
    
    print(f"   Final position: {obj.position}")
    
    # Temporary directory
    print("\n4. Temporary directory:")
    with temporary_directory() as temp_dir:
        print(f"   Working in: {temp_dir}")
        # Create some files in the temp directory
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test data")
        print(f"   Created file: {test_file}")
    
    print()

def demonstrate_nested_context_managers():
    """Demonstrate nested context managers"""
    print("=== Nested Context Managers Demo ===\n")
    
    print("1. Nested context managers:")
    with NestedContextManager("Outer") as outer:
        with NestedContextManager("Middle") as middle:
            with NestedContextManager("Inner") as inner:
                print("      Performing nested operations...")
                time.sleep(0.01)
    
    print("\n2. Complex nested scenario:")
    with TimerContext("Complete 3D Operation"):
        with OpenGLContext(800, 600) as gl:
            with SceneManager("Game Scene") as scene:
                with ResourceManager("Shader") as shader_manager:
                    scene.add_object("Player")
                    shader_manager.acquire_resource("vertex_shader.glsl")
                    shader_manager.acquire_resource("fragment_shader.glsl")
                    gl.render_frame()
                    print("   Completed complex 3D operation")
    
    print()

def demonstrate_exception_handling():
    """Demonstrate exception handling in context managers"""
    print("=== Exception Handling Demo ===\n")
    
    # Exception in context manager
    print("1. Exception in context manager:")
    try:
        with TimerContext("Failing Operation"):
            print("   Starting operation...")
            raise ValueError("Something went wrong!")
    except ValueError as e:
        print(f"   Caught exception: {e}")
    
    # Exception with error handler
    print("\n2. Exception with error handler:")
    with error_handler("Unreliable Operation", default_value="safe_value"):
        if random.random() < 0.7:  # 70% chance of error
            raise RuntimeError("Operation failed")
        print("   Operation succeeded")
    
    # Exception in nested context
    print("\n3. Exception in nested context:")
    try:
        with OpenGLContext() as gl:
            with SceneManager("Test Scene") as scene:
                scene.add_object("Test Object")
                raise IOError("File not found")
    except IOError as e:
        print(f"   Caught nested exception: {e}")
    
    print()

def demonstrate_file_operations():
    """Demonstrate file operations with context managers"""
    print("=== File Operations Demo ===\n")
    
    # JSON file operations
    print("1. JSON file operations:")
    data = {
        "scene": "Main Scene",
        "objects": [
            {"name": "Player", "position": [0, 0, 0]},
            {"name": "Enemy", "position": [10, 0, 0]}
        ],
        "settings": {"fov": 60, "resolution": [1920, 1080]}
    }
    
    with FileManager("scene_config.json", 'w') as f:
        json.dump(data, f, indent=2)
        print("   Wrote scene configuration")
    
    with FileManager("scene_config.json", 'r') as f:
        loaded_data = json.load(f)
        print(f"   Loaded scene: {loaded_data['scene']}")
        print(f"   Objects: {len(loaded_data['objects'])}")
    
    # Temporary file operations
    print("\n2. Temporary file operations:")
    with TemporaryFileManager(suffix='.log', prefix='debug_') as temp_file:
        temp_file.write("Debug information\n")
        temp_file.write("Performance metrics\n")
        temp_file.write("Error logs\n")
        temp_file.flush()
        print(f"   Wrote debug info to: {temp_file.name}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate context managers"""
    print("=== Context Managers Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_context_managers()
    demonstrate_3d_graphics_context_managers()
    demonstrate_context_manager_utilities()
    demonstrate_nested_context_managers()
    demonstrate_exception_handling()
    demonstrate_file_operations()
    
    print("="*60)
    print("Context Managers demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic context managers: Timer, file, temporary file management")
    print("✓ 3D graphics context managers: OpenGL, scene, resource management")
    print("✓ Context manager utilities: Performance monitoring, error handling")
    print("✓ Nested context managers: Complex resource management scenarios")
    print("✓ Exception handling: Proper cleanup and error propagation")
    print("✓ Resource management: Automatic cleanup and state restoration")

if __name__ == "__main__":
    main()
