"""
Chapter 12: Working with External Libraries - Library Management
===============================================================

This module demonstrates advanced library management patterns for 3D graphics
applications, including dependency management, plugin systems, and cross-platform
compatibility.

Key Concepts:
- Dependency management and versioning
- Plugin architecture
- Cross-platform compatibility
- Library abstraction layers
- Error handling and fallbacks
- Performance optimization
"""

import importlib
import sys
import os
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import platform


class LibraryStatus(Enum):
    """Status of library availability."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass
class LibraryInfo:
    """Information about a library."""
    name: str
    version: str
    status: LibraryStatus
    description: str
    dependencies: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    performance_rating: float = 1.0  # 0.0 to 1.0
    
    def is_available(self) -> bool:
        """Check if library is available."""
        return self.status in [LibraryStatus.AVAILABLE, LibraryStatus.OPTIONAL, LibraryStatus.REQUIRED]
    
    def is_required(self) -> bool:
        """Check if library is required."""
        return self.status == LibraryStatus.REQUIRED


class LibraryManager:
    """Manages external libraries for 3D graphics applications."""
    
    def __init__(self):
        self.libraries: Dict[str, LibraryInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self.fallback_libraries: Dict[str, str] = {}
        self.performance_metrics: Dict[str, float] = {}
    
    def register_library(self, library: LibraryInfo):
        """Register a library with the manager."""
        self.libraries[library.name] = library
        print(f"  📚 Registered library: {library.name} v{library.version}")
    
    def check_availability(self, library_name: str) -> bool:
        """Check if a library is available."""
        if library_name not in self.libraries:
            return False
        
        library = self.libraries[library_name]
        
        # Check if already loaded
        if library_name in self.loaded_modules:
            return True
        
        # Try to import the library
        try:
            module = importlib.import_module(library_name)
            self.loaded_modules[library_name] = module
            library.status = LibraryStatus.AVAILABLE
            print(f"  ✅ Library {library_name} is available")
            return True
        except ImportError:
            library.status = LibraryStatus.UNAVAILABLE
            print(f"  ❌ Library {library_name} is not available")
            return False
    
    def load_library(self, library_name: str) -> Optional[Any]:
        """Load a library and return the module."""
        if library_name in self.loaded_modules:
            return self.loaded_modules[library_name]
        
        if not self.check_availability(library_name):
            # Try fallback library
            if library_name in self.fallback_libraries:
                fallback_name = self.fallback_libraries[library_name]
                print(f"  🔄 Trying fallback library: {fallback_name}")
                return self.load_library(fallback_name)
            return None
        
        return self.loaded_modules[library_name]
    
    def get_library_info(self, library_name: str) -> Optional[LibraryInfo]:
        """Get information about a library."""
        return self.libraries.get(library_name)
    
    def set_fallback(self, primary_library: str, fallback_library: str):
        """Set a fallback library."""
        self.fallback_libraries[primary_library] = fallback_library
        print(f"  🔄 Set fallback: {primary_library} -> {fallback_library}")
    
    def get_available_libraries(self) -> List[str]:
        """Get list of available libraries."""
        return [name for name, lib in self.libraries.items() if lib.is_available()]
    
    def get_required_libraries(self) -> List[str]:
        """Get list of required libraries."""
        return [name for name, lib in self.libraries.items() if lib.is_required()]


class PluginManager:
    """Manages plugins for 3D graphics applications."""
    
    def __init__(self, library_manager: LibraryManager):
        self.library_manager = library_manager
        self.plugins: Dict[str, Any] = {}
        self.plugin_factories: Dict[str, Callable] = {}
        self.active_plugins: List[str] = []
    
    def register_plugin(self, name: str, factory_func: Callable, 
                       dependencies: List[str] = None):
        """Register a plugin factory."""
        self.plugin_factories[name] = factory_func
        if dependencies:
            print(f"  🔌 Registered plugin: {name} (dependencies: {dependencies})")
        else:
            print(f"  🔌 Registered plugin: {name}")
    
    def load_plugin(self, name: str) -> bool:
        """Load a plugin."""
        if name not in self.plugin_factories:
            print(f"  ❌ Plugin {name} not found")
            return False
        
        if name in self.plugins:
            print(f"  ⚠️  Plugin {name} already loaded")
            return True
        
        try:
            factory = self.plugin_factories[name]
            plugin = factory(self.library_manager)
            self.plugins[name] = plugin
            self.active_plugins.append(name)
            print(f"  ✅ Loaded plugin: {name}")
            return True
        except Exception as e:
            print(f"  ❌ Failed to load plugin {name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[Any]:
        """Get a loaded plugin."""
        return self.plugins.get(name)
    
    def unload_plugin(self, name: str):
        """Unload a plugin."""
        if name in self.plugins:
            del self.plugins[name]
            if name in self.active_plugins:
                self.active_plugins.remove(name)
            print(f"  🔌 Unloaded plugin: {name}")
    
    def get_active_plugins(self) -> List[str]:
        """Get list of active plugins."""
        return self.active_plugins.copy()


class CrossPlatformAdapter:
    """Adapter for cross-platform library compatibility."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.adapters: Dict[str, Dict[str, Any]] = {}
        self.current_adapter = None
    
    def register_adapter(self, library_name: str, platform_name: str, adapter: Any):
        """Register a platform-specific adapter."""
        if library_name not in self.adapters:
            self.adapters[library_name] = {}
        
        self.adapters[library_name][platform_name] = adapter
        print(f"  🔧 Registered {platform_name} adapter for {library_name}")
    
    def get_adapter(self, library_name: str) -> Optional[Any]:
        """Get the appropriate adapter for the current platform."""
        if library_name not in self.adapters:
            return None
        
        platform_adapters = self.adapters[library_name]
        
        # Try exact platform match
        if self.platform in platform_adapters:
            self.current_adapter = platform_adapters[self.platform]
            print(f"  🔧 Using {self.platform} adapter for {library_name}")
            return self.current_adapter
        
        # Try generic fallback
        if "generic" in platform_adapters:
            self.current_adapter = platform_adapters["generic"]
            print(f"  🔧 Using generic adapter for {library_name}")
            return self.current_adapter
        
        return None
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get current platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }


class PerformanceMonitor:
    """Monitors performance of library operations."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.thresholds: Dict[str, float] = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(time.time())
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        start_time = self.metrics[operation].pop()
        duration = time.time() - start_time
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        
        # Check threshold
        if operation in self.thresholds and duration > self.thresholds[operation]:
            print(f"  ⚠️  Operation {operation} took {duration:.6f}s (threshold: {self.thresholds[operation]:.6f}s)")
        
        return duration
    
    def set_threshold(self, operation: str, threshold: float):
        """Set performance threshold for an operation."""
        self.thresholds[operation] = threshold
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        report = {}
        for operation, times in self.metrics.items():
            if times:
                report[operation] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        return report


class LibraryAbstractionLayer:
    """Abstraction layer for 3D graphics libraries."""
    
    def __init__(self, library_manager: LibraryManager, 
                 plugin_manager: PluginManager,
                 platform_adapter: CrossPlatformAdapter,
                 performance_monitor: PerformanceMonitor):
        self.library_manager = library_manager
        self.plugin_manager = plugin_manager
        self.platform_adapter = platform_adapter
        self.performance_monitor = performance_monitor
        self.renderers: Dict[str, Any] = {}
        self.current_renderer = None
    
    def register_renderer(self, name: str, renderer_factory: Callable):
        """Register a renderer factory."""
        self.renderers[name] = renderer_factory
        print(f"  🎨 Registered renderer: {name}")
    
    def create_renderer(self, name: str) -> Optional[Any]:
        """Create a renderer instance."""
        if name not in self.renderers:
            print(f"  ❌ Renderer {name} not found")
            return None
        
        try:
            self.performance_monitor.start_timing(f"create_renderer_{name}")
            factory = self.renderers[name]
            renderer = factory(self.library_manager, self.plugin_manager)
            self.performance_monitor.end_timing(f"create_renderer_{name}")
            
            self.current_renderer = renderer
            print(f"  ✅ Created renderer: {name}")
            return renderer
        except Exception as e:
            print(f"  ❌ Failed to create renderer {name}: {e}")
            return None
    
    def get_current_renderer(self) -> Optional[Any]:
        """Get the current renderer."""
        return self.current_renderer
    
    def switch_renderer(self, name: str) -> bool:
        """Switch to a different renderer."""
        renderer = self.create_renderer(name)
        if renderer:
            self.current_renderer = renderer
            return True
        return False


# Example Library Implementations
class NumPyRenderer:
    """NumPy-based renderer implementation."""
    
    def __init__(self, library_manager: LibraryManager, plugin_manager: PluginManager):
        self.library_manager = library_manager
        self.plugin_manager = plugin_manager
        self.numpy = library_manager.load_library("numpy")
        
        if not self.numpy:
            raise RuntimeError("NumPy is required for NumPyRenderer")
    
    def render_scene(self, scene_data: Dict[str, Any]):
        """Render a scene using NumPy."""
        print(f"  🎨 NumPyRenderer: Rendering scene with {len(scene_data.get('objects', []))} objects")
        
        # Simulate NumPy-based rendering
        vertices = self.numpy.array(scene_data.get('vertices', []))
        indices = self.numpy.array(scene_data.get('indices', []))
        
        print(f"  📊 Processed {len(vertices)} vertices and {len(indices)} indices")


class OpenGLRenderer:
    """OpenGL-based renderer implementation."""
    
    def __init__(self, library_manager: LibraryManager, plugin_manager: PluginManager):
        self.library_manager = library_manager
        self.plugin_manager = plugin_manager
        
        # Try to load OpenGL libraries
        self.gl = library_manager.load_library("OpenGL")
        self.glfw = library_manager.load_library("glfw")
        
        if not self.gl or not self.glfw:
            raise RuntimeError("OpenGL and GLFW are required for OpenGLRenderer")
    
    def render_scene(self, scene_data: Dict[str, Any]):
        """Render a scene using OpenGL."""
        print(f"  🎨 OpenGLRenderer: Rendering scene with {len(scene_data.get('objects', []))} objects")
        
        # Simulate OpenGL-based rendering
        print(f"  🔧 Setting up OpenGL context")
        print(f"  📦 Creating vertex buffers")
        print(f"  🎨 Drawing objects")


class VulkanRenderer:
    """Vulkan-based renderer implementation."""
    
    def __init__(self, library_manager: LibraryManager, plugin_manager: PluginManager):
        self.library_manager = library_manager
        self.plugin_manager = plugin_manager
        
        # Try to load Vulkan libraries
        self.vulkan = library_manager.load_library("vulkan")
        
        if not self.vulkan:
            raise RuntimeError("Vulkan is required for VulkanRenderer")
    
    def render_scene(self, scene_data: Dict[str, Any]):
        """Render a scene using Vulkan."""
        print(f"  🎨 VulkanRenderer: Rendering scene with {len(scene_data.get('objects', []))} objects")
        
        # Simulate Vulkan-based rendering
        print(f"  🔧 Setting up Vulkan instance")
        print(f"  📦 Creating command buffers")
        print(f"  🎨 Submitting draw commands")


# Example Usage and Demonstration
def demonstrate_library_management():
    """Demonstrates library management for 3D graphics."""
    print("=== Library Management for 3D Graphics ===\n")
    
    # Create library manager
    library_manager = LibraryManager()
    
    # Register libraries
    libraries = [
        LibraryInfo("numpy", "1.21.0", LibraryStatus.REQUIRED, "Numerical computing library"),
        LibraryInfo("OpenGL", "3.3", LibraryStatus.OPTIONAL, "OpenGL graphics library"),
        LibraryInfo("glfw", "3.3", LibraryStatus.OPTIONAL, "GLFW window management"),
        LibraryInfo("vulkan", "1.2", LibraryStatus.OPTIONAL, "Vulkan graphics API"),
        LibraryInfo("matplotlib", "3.5.0", LibraryStatus.OPTIONAL, "Plotting library"),
        LibraryInfo("pillow", "8.3.0", LibraryStatus.OPTIONAL, "Image processing library")
    ]
    
    for library in libraries:
        library_manager.register_library(library)
    
    # Set fallbacks
    library_manager.set_fallback("OpenGL", "numpy")
    library_manager.set_fallback("vulkan", "OpenGL")
    
    # Create plugin manager
    plugin_manager = PluginManager(library_manager)
    
    # Register renderer plugins
    plugin_manager.register_plugin("numpy_renderer", 
                                 lambda lm, pm: NumPyRenderer(lm, pm))
    plugin_manager.register_plugin("opengl_renderer", 
                                 lambda lm, pm: OpenGLRenderer(lm, pm))
    plugin_manager.register_plugin("vulkan_renderer", 
                                 lambda lm, pm: VulkanRenderer(lm, pm))
    
    # Create platform adapter
    platform_adapter = CrossPlatformAdapter()
    platform_info = platform_adapter.get_platform_info()
    print(f"  💻 Platform: {platform_info['system']} {platform_info['release']}")
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    performance_monitor.set_threshold("create_renderer", 0.1)
    performance_monitor.set_threshold("render_scene", 0.05)
    
    # Create abstraction layer
    abstraction_layer = LibraryAbstractionLayer(
        library_manager, plugin_manager, platform_adapter, performance_monitor
    )
    
    # Register renderers
    abstraction_layer.register_renderer("numpy", 
                                      lambda lm, pm: NumPyRenderer(lm, pm))
    abstraction_layer.register_renderer("opengl", 
                                      lambda lm, pm: OpenGLRenderer(lm, pm))
    abstraction_layer.register_renderer("vulkan", 
                                      lambda lm, pm: VulkanRenderer(lm, pm))
    
    # Check library availability
    print("\n=== Library Availability ===")
    available_libraries = library_manager.get_available_libraries()
    print(f"Available libraries: {available_libraries}")
    
    required_libraries = library_manager.get_required_libraries()
    print(f"Required libraries: {required_libraries}")
    
    # Try to create renderers
    print("\n=== Renderer Creation ===")
    
    renderers_to_try = ["numpy", "opengl", "vulkan"]
    
    for renderer_name in renderers_to_try:
        print(f"\nTrying to create {renderer_name} renderer:")
        try:
            renderer = abstraction_layer.create_renderer(renderer_name)
            if renderer:
                # Test rendering
                scene_data = {
                    "objects": ["cube", "sphere", "cylinder"],
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "indices": [0, 1, 2]
                }
                
                performance_monitor.start_timing("render_scene")
                renderer.render_scene(scene_data)
                performance_monitor.end_timing("render_scene")
                
                break  # Use the first successful renderer
        except Exception as e:
            print(f"  ❌ Failed to create {renderer_name} renderer: {e}")
    
    # Load plugins
    print("\n=== Plugin Loading ===")
    plugin_manager.load_plugin("numpy_renderer")
    
    # Performance report
    print("\n=== Performance Report ===")
    report = performance_monitor.get_performance_report()
    for operation, metrics in report.items():
        print(f"  {operation}: {metrics['count']} calls, "
              f"avg: {metrics['average']:.6f}s, "
              f"total: {metrics['total']:.6f}s")
    
    # Library information
    print("\n=== Library Information ===")
    for name, library in library_manager.libraries.items():
        status_icon = "✅" if library.is_available() else "❌"
        print(f"  {status_icon} {name} v{library.version}: {library.description}")


if __name__ == "__main__":
    demonstrate_library_management()
