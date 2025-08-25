"""
Chapter 11: Decorators and Context Managers - Context Managers
============================================================

This module demonstrates how to use context managers to manage resources
in 3D graphics applications, including OpenGL contexts, file operations,
performance monitoring, and error handling.

Key Concepts:
- Context manager protocol
- Resource management
- Performance profiling
- Error handling and recovery
- OpenGL context management
- File and memory management
"""

import time
import math
import random
import threading
import tempfile
import os
from typing import Any, Dict, List, Optional, Generator
from contextlib import contextmanager
from dataclasses import dataclass
import json


@dataclass
class Vector3D:
    """3D vector for context manager examples."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class Mesh:
    """3D mesh data structure."""
    vertices: List[Vector3D]
    faces: List[List[int]]
    name: str
    
    def __str__(self):
        return f"Mesh({self.name}, {len(self.vertices)} vertices, {len(self.faces)} faces)"


# Basic Context Managers
class PerformanceProfiler:
    """Context manager for profiling code performance."""
    
    def __init__(self, operation_name: str, threshold: float = 0.1):
        self.operation_name = operation_name
        self.threshold = threshold
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"  🚀 Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if duration > self.threshold:
            print(f"  ⚠️  {self.operation_name} took {duration:.6f}s (threshold: {self.threshold}s)")
        else:
            print(f"  ✅ {self.operation_name} completed in {duration:.6f}s")
        
        if exc_type is not None:
            print(f"  ❌ {self.operation_name} failed with {exc_type.__name__}: {exc_val}")
            return False  # Re-raise the exception
        return True


class ErrorHandler:
    """Context manager for handling and logging errors."""
    
    def __init__(self, operation_name: str, fallback_value: Any = None):
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.error_occurred = False
    
    def __enter__(self):
        print(f"  🔍 Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            print(f"  ❌ Error in {self.operation_name}: {exc_type.__name__}: {exc_val}")
            if self.fallback_value is not None:
                print(f"  🔄 Using fallback value: {self.fallback_value}")
            return True  # Suppress the exception
        else:
            print(f"  ✅ {self.operation_name} completed successfully")
        return True


class ResourceManager:
    """Context manager for managing 3D graphics resources."""
    
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        self.resource_id = None
        self.allocated = False
    
    def __enter__(self):
        # Simulate resource allocation
        self.resource_id = random.randint(1000, 9999)
        self.allocated = True
        print(f"  📦 Allocated {self.resource_name} (ID: {self.resource_id})")
        return self.resource_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.allocated:
            print(f"  🗑️  Deallocated {self.resource_name} (ID: {self.resource_id})")
            self.allocated = False
        
        if exc_type is not None:
            print(f"  ❌ Error during {self.resource_name} operation: {exc_type.__name__}")
            return False
        return True


# OpenGL Context Management
class OpenGLContext:
    """Context manager for OpenGL context management."""
    
    def __init__(self, window_title: str = "3D Application", width: int = 800, height: int = 600):
        self.window_title = window_title
        self.width = width
        self.height = height
        self.context_active = False
        self.render_count = 0
    
    def __enter__(self):
        # Simulate OpenGL context creation
        print(f"  🖥️  Creating OpenGL context: {self.window_title} ({self.width}x{self.height})")
        print(f"  🔧 Initializing OpenGL state...")
        self.context_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context_active:
            print(f"  🖥️  Destroying OpenGL context")
            print(f"  📊 Total renders: {self.render_count}")
            self.context_active = False
        
        if exc_type is not None:
            print(f"  ❌ OpenGL context error: {exc_type.__name__}")
            return False
        return True
    
    def render_frame(self):
        """Simulate rendering a frame."""
        if self.context_active:
            self.render_count += 1
            print(f"  🎨 Rendering frame #{self.render_count}")
        else:
            raise RuntimeError("OpenGL context not active")


# File and Data Management
class MeshFileManager:
    """Context manager for 3D mesh file operations."""
    
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.mesh_data = None
    
    def __enter__(self):
        print(f"  📁 Opening mesh file: {self.filename}")
        # Simulate file operations
        if self.mode == 'r':
            # Simulate reading mesh data
            self.mesh_data = Mesh(
                vertices=[Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 1, 0)],
                faces=[[0, 1, 2]],
                name="triangle"
            )
            print(f"  📖 Loaded mesh: {self.mesh_data}")
        return self.mesh_data
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"  📁 Closing mesh file: {self.filename}")
        if exc_type is not None:
            print(f"  ❌ File operation error: {exc_type.__name__}")
            return False
        return True


class TemporaryMeshStorage:
    """Context manager for temporary mesh storage."""
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.temp_file = None
    
    def __enter__(self):
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        print(f"  📝 Creating temporary storage: {self.temp_file.name}")
        
        # Save mesh to temporary file
        mesh_data = {
            'name': self.mesh.name,
            'vertices': [[v.x, v.y, v.z] for v in self.mesh.vertices],
            'faces': self.mesh.faces
        }
        json.dump(mesh_data, self.temp_file)
        self.temp_file.close()
        
        return self.temp_file.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
            print(f"  🗑️  Cleaned up temporary file: {self.temp_file.name}")
        
        if exc_type is not None:
            print(f"  ❌ Temporary storage error: {exc_type.__name__}")
            return False
        return True


# Threading and Concurrency
class ThreadSafeOperation:
    """Context manager for thread-safe operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.lock = threading.Lock()
    
    def __enter__(self):
        print(f"  🔒 Acquiring lock for {self.operation_name}")
        self.lock.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
        print(f"  🔓 Released lock for {self.operation_name}")
        
        if exc_type is not None:
            print(f"  ❌ Thread-safe operation error: {exc_type.__name__}")
            return False
        return True


class BatchProcessor:
    """Context manager for batch processing operations."""
    
    def __init__(self, batch_name: str, items: List[Any]):
        self.batch_name = batch_name
        self.items = items
        self.processed_count = 0
        self.failed_count = 0
    
    def __enter__(self):
        print(f"  📦 Starting batch processing: {self.batch_name} ({len(self.items)} items)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"  📊 Batch completed: {self.processed_count} processed, {self.failed_count} failed")
        
        if exc_type is not None:
            print(f"  ❌ Batch processing error: {exc_type.__name__}")
            return False
        return True
    
    def process_item(self, item: Any) -> bool:
        """Process a single item in the batch."""
        try:
            # Simulate processing
            time.sleep(0.01)
            self.processed_count += 1
            return True
        except Exception as e:
            self.failed_count += 1
            print(f"  ❌ Failed to process item: {e}")
            return False


# 3D Graphics Specific Context Managers
class SceneManager:
    """Context manager for 3D scene management."""
    
    def __init__(self, scene_name: str):
        self.scene_name = scene_name
        self.objects = []
        self.lights = []
        self.camera = None
    
    def __enter__(self):
        print(f"  🎬 Creating scene: {self.scene_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"  🎬 Destroying scene: {self.scene_name}")
        print(f"  📊 Scene stats: {len(self.objects)} objects, {len(self.lights)} lights")
        
        if exc_type is not None:
            print(f"  ❌ Scene error: {exc_type.__name__}")
            return False
        return True
    
    def add_object(self, obj: Any):
        """Add an object to the scene."""
        self.objects.append(obj)
        print(f"  ➕ Added object to scene: {obj}")
    
    def add_light(self, light: Any):
        """Add a light to the scene."""
        self.lights.append(light)
        print(f"  💡 Added light to scene: {light}")


class ShaderProgram:
    """Context manager for OpenGL shader program management."""
    
    def __init__(self, vertex_shader: str, fragment_shader: str):
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.program_id = None
        self.compiled = False
    
    def __enter__(self):
        # Simulate shader compilation
        self.program_id = random.randint(1000, 9999)
        print(f"  🔧 Compiling shader program (ID: {self.program_id})")
        print(f"  📝 Vertex shader: {len(self.vertex_shader)} characters")
        print(f"  📝 Fragment shader: {len(self.fragment_shader)} characters")
        self.compiled = True
        return self.program_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.compiled:
            print(f"  🗑️  Deleting shader program (ID: {self.program_id})")
        
        if exc_type is not None:
            print(f"  ❌ Shader compilation error: {exc_type.__name__}")
            return False
        return True


# Function-based Context Managers
@contextmanager
def performance_monitor(operation_name: str, threshold: float = 0.1):
    """Function-based context manager for performance monitoring."""
    start_time = time.time()
    print(f"  🚀 Starting {operation_name}")
    
    try:
        yield
    except Exception as e:
        print(f"  ❌ {operation_name} failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        if duration > threshold:
            print(f"  ⚠️  {operation_name} took {duration:.6f}s (threshold: {threshold}s)")
        else:
            print(f"  ✅ {operation_name} completed in {duration:.6f}s")


@contextmanager
def error_recovery(operation_name: str, fallback_func: callable = None):
    """Function-based context manager for error recovery."""
    print(f"  🔍 Starting {operation_name}")
    
    try:
        yield
        print(f"  ✅ {operation_name} completed successfully")
    except Exception as e:
        print(f"  ❌ {operation_name} failed: {e}")
        if fallback_func:
            print(f"  🔄 Executing fallback function")
            fallback_func()
        raise


@contextmanager
def resource_cleanup(resource_name: str, cleanup_func: callable = None):
    """Function-based context manager for resource cleanup."""
    print(f"  📦 Allocating {resource_name}")
    
    try:
        yield
    finally:
        if cleanup_func:
            cleanup_func()
        print(f"  🗑️  Cleaned up {resource_name}")


# Example Usage and Demonstration
def demonstrate_context_managers():
    """Demonstrates various context manager patterns for 3D graphics."""
    print("=== Context Managers for 3D Graphics ===\n")
    
    # Basic context managers
    print("=== Basic Context Managers ===")
    
    print("Performance profiler:")
    with PerformanceProfiler("3D calculation", threshold=0.05):
        # Simulate 3D calculation
        time.sleep(0.1)
        result = Vector3D(1, 2, 3)
        print(f"  Calculated: {result}")
    
    print("\nError handler:")
    with ErrorHandler("risky operation", fallback_value=Vector3D(0, 0, 0)):
        if random.random() < 0.5:
            raise ValueError("Random error occurred")
        print(f"  Operation succeeded")
    
    print("\nResource manager:")
    with ResourceManager("texture") as texture_id:
        print(f"  Using texture ID: {texture_id}")
        # Simulate texture operations
        time.sleep(0.01)
    
    # OpenGL context management
    print("\n=== OpenGL Context Management ===")
    
    with OpenGLContext("3D Scene Viewer", 1024, 768) as gl_context:
        gl_context.render_frame()
        gl_context.render_frame()
        gl_context.render_frame()
    
    # File and data management
    print("\n=== File and Data Management ===")
    
    print("Mesh file manager:")
    with MeshFileManager("cube.obj", 'r') as mesh:
        print(f"  Working with mesh: {mesh}")
    
    print("\nTemporary mesh storage:")
    test_mesh = Mesh(
        vertices=[Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(0, 1, 0)],
        faces=[[0, 1, 2]],
        name="test_triangle"
    )
    
    with TemporaryMeshStorage(test_mesh) as temp_path:
        print(f"  Mesh stored at: {temp_path}")
        # Simulate processing
        time.sleep(0.01)
    
    # Threading and concurrency
    print("\n=== Threading and Concurrency ===")
    
    with ThreadSafeOperation("mesh modification") as lock:
        print(f"  Performing thread-safe mesh modification")
        time.sleep(0.01)
    
    print("\nBatch processor:")
    items = [f"mesh_{i}" for i in range(5)]
    with BatchProcessor("mesh_processing", items) as batch:
        for item in items:
            batch.process_item(item)
    
    # 3D graphics specific
    print("\n=== 3D Graphics Specific ===")
    
    print("Scene manager:")
    with SceneManager("Main Scene") as scene:
        scene.add_object("Cube")
        scene.add_object("Sphere")
        scene.add_light("Directional Light")
        scene.add_light("Point Light")
    
    print("\nShader program:")
    vertex_shader = "void main() { gl_Position = vec4(0.0); }"
    fragment_shader = "void main() { gl_FragColor = vec4(1.0); }"
    
    with ShaderProgram(vertex_shader, fragment_shader) as program_id:
        print(f"  Using shader program: {program_id}")
        # Simulate shader usage
        time.sleep(0.01)
    
    # Function-based context managers
    print("\n=== Function-based Context Managers ===")
    
    print("Performance monitor:")
    with performance_monitor("vector calculation", threshold=0.05):
        # Simulate calculation
        time.sleep(0.1)
        result = Vector3D(1, 2, 3).magnitude()
        print(f"  Vector magnitude: {result}")
    
    print("\nError recovery:")
    def fallback():
        print(f"  Executing fallback: using default values")
    
    with error_recovery("data loading", fallback):
        if random.random() < 0.7:
            raise RuntimeError("Data loading failed")
        print(f"  Data loaded successfully")
    
    print("\nResource cleanup:")
    def cleanup():
        print(f"  Performing custom cleanup")
    
    with resource_cleanup("custom_resource", cleanup):
        print(f"  Using custom resource")
        time.sleep(0.01)


if __name__ == "__main__":
    demonstrate_context_managers()
