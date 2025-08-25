#!/usr/bin/env python3
"""
Chapter 15: Advanced 3D Graphics Libraries and Tools
Cross-Platform Graphics Framework

Develops a framework that can seamlessly switch between different graphics APIs
based on platform capabilities and provides fallback mechanisms.
"""

import sys
import platform
import os
import time
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import weakref

class GraphicsAPI(Enum):
    """Supported graphics APIs"""
    OPENGL = "OpenGL"
    VULKAN = "Vulkan"
    DIRECTX = "DirectX"
    METAL = "Metal"
    SOFTWARE = "Software"

class Platform(Enum):
    """Supported platforms"""
    WINDOWS = "Windows"
    MACOS = "macOS"
    LINUX = "Linux"
    ANDROID = "Android"
    IOS = "iOS"

@dataclass
class GraphicsCapabilities:
    """Graphics capabilities of a platform"""
    platform: Platform
    supported_apis: List[GraphicsAPI]
    max_texture_size: int
    max_vertex_attributes: int
    max_uniform_blocks: int
    max_texture_units: int
    shader_version: str
    glsl_version: str
    extensions: List[str]
    vendor: str
    renderer: str
    version: str

@dataclass
class RenderTarget:
    """Abstract render target"""
    width: int
    height: int
    format: str
    samples: int = 1

@dataclass
class ShaderProgram:
    """Abstract shader program"""
    vertex_shader: str
    fragment_shader: str
    uniforms: Dict[str, Any] = None
    attributes: Dict[str, Any] = None

class GraphicsBackend:
    """Abstract graphics backend interface"""
    
    def __init__(self, api: GraphicsAPI):
        self.api = api
        self.initialized = False
        self.capabilities = None
    
    def initialize(self) -> bool:
        """Initialize the graphics backend"""
        raise NotImplementedError("Subclasses must implement initialize")
    
    def shutdown(self):
        """Shutdown the graphics backend"""
        raise NotImplementedError("Subclasses must implement shutdown")
    
    def create_window(self, title: str, width: int, height: int) -> bool:
        """Create a window"""
        raise NotImplementedError("Subclasses must implement create_window")
    
    def create_shader_program(self, vertex_src: str, fragment_src: str) -> ShaderProgram:
        """Create a shader program"""
        raise NotImplementedError("Subclasses must implement create_shader_program")
    
    def create_render_target(self, width: int, height: int, format: str = "RGBA8") -> RenderTarget:
        """Create a render target"""
        raise NotImplementedError("Subclasses must implement create_render_target")
    
    def begin_frame(self):
        """Begin rendering a frame"""
        raise NotImplementedError("Subclasses must implement begin_frame")
    
    def end_frame(self):
        """End rendering a frame"""
        raise NotImplementedError("Subclasses must implement end_frame")
    
    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Clear the screen"""
        raise NotImplementedError("Subclasses must implement clear")
    
    def get_capabilities(self) -> GraphicsCapabilities:
        """Get graphics capabilities"""
        return self.capabilities

class OpenGLBackend(GraphicsBackend):
    """OpenGL graphics backend"""
    
    def __init__(self):
        super().__init__(GraphicsAPI.OPENGL)
        self.context = None
        self.window = None
    
    def initialize(self) -> bool:
        """Initialize OpenGL backend"""
        try:
            import moderngl
            import glfw
            
            if not glfw.init():
                return False
            
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
            self.context = moderngl.create_context()
            self.initialized = True
            
            # Get capabilities
            self.capabilities = GraphicsCapabilities(
                platform=self._detect_platform(),
                supported_apis=[GraphicsAPI.OPENGL],
                max_texture_size=2048,  # Default value
                max_vertex_attributes=16,
                max_uniform_blocks=14,
                max_texture_units=16,
                shader_version="330",
                glsl_version="330",
                extensions=[],
                vendor="Unknown",
                renderer="Unknown",
                version="Unknown"
            )
            
            return True
            
        except ImportError:
            print("OpenGL backend not available: moderngl or glfw not installed")
            return False
        except Exception as e:
            print(f"Failed to initialize OpenGL backend: {e}")
            return False
    
    def shutdown(self):
        """Shutdown OpenGL backend"""
        if self.window:
            import glfw
            glfw.terminate()
        self.initialized = False
    
    def create_window(self, title: str, width: int, height: int) -> bool:
        """Create OpenGL window"""
        try:
            import glfw
            
            self.window = glfw.create_window(width, height, title, None, None)
            if not self.window:
                return False
            
            glfw.make_context_current(self.window)
            return True
            
        except Exception as e:
            print(f"Failed to create OpenGL window: {e}")
            return False
    
    def create_shader_program(self, vertex_src: str, fragment_src: str) -> ShaderProgram:
        """Create OpenGL shader program"""
        if not self.initialized:
            return None
        
        try:
            program = self.context.program(
                vertex_shader=vertex_src,
                fragment_shader=fragment_src
            )
            
            return ShaderProgram(
                vertex_shader=vertex_src,
                fragment_shader=fragment_src,
                uniforms={},
                attributes={}
            )
            
        except Exception as e:
            print(f"Failed to create OpenGL shader program: {e}")
            return None
    
    def create_render_target(self, width: int, height: int, format: str = "RGBA8") -> RenderTarget:
        """Create OpenGL render target"""
        if not self.initialized:
            return None
        
        try:
            # Create framebuffer
            framebuffer = self.context.framebuffer()
            
            # Create texture
            texture = self.context.texture((width, height), 4)
            framebuffer.color_attachments[0] = texture
            
            return RenderTarget(width, height, format)
            
        except Exception as e:
            print(f"Failed to create OpenGL render target: {e}")
            return None
    
    def begin_frame(self):
        """Begin OpenGL frame"""
        if self.window:
            import glfw
            glfw.poll_events()
    
    def end_frame(self):
        """End OpenGL frame"""
        if self.window:
            import glfw
            glfw.swap_buffers(self.window)
    
    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Clear OpenGL screen"""
        if self.initialized:
            self.context.clear(r, g, b, a)
    
    def _detect_platform(self) -> Platform:
        """Detect current platform"""
        system = platform.system().lower()
        if system == "windows":
            return Platform.WINDOWS
        elif system == "darwin":
            return Platform.MACOS
        elif system == "linux":
            return Platform.LINUX
        else:
            return Platform.LINUX

class SoftwareBackend(GraphicsBackend):
    """Software rendering backend (fallback)"""
    
    def __init__(self):
        super().__init__(GraphicsAPI.SOFTWARE)
        self.framebuffer = None
        self.width = 0
        self.height = 0
    
    def initialize(self) -> bool:
        """Initialize software backend"""
        try:
            import numpy as np
            
            self.capabilities = GraphicsCapabilities(
                platform=self._detect_platform(),
                supported_apis=[GraphicsAPI.SOFTWARE],
                max_texture_size=1024,
                max_vertex_attributes=8,
                max_uniform_blocks=1,
                max_texture_units=1,
                shader_version="100",
                glsl_version="100",
                extensions=[],
                vendor="Software Renderer",
                renderer="CPU",
                version="1.0"
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize software backend: {e}")
            return False
    
    def shutdown(self):
        """Shutdown software backend"""
        self.initialized = False
        self.framebuffer = None
    
    def create_window(self, title: str, width: int, height: int) -> bool:
        """Create software window"""
        try:
            import numpy as np
            
            self.width = width
            self.height = height
            self.framebuffer = np.zeros((height, width, 4), dtype=np.uint8)
            
            print(f"Software window created: {title} ({width}x{height})")
            return True
            
        except Exception as e:
            print(f"Failed to create software window: {e}")
            return False
    
    def create_shader_program(self, vertex_src: str, fragment_src: str) -> ShaderProgram:
        """Create software shader program (simplified)"""
        return ShaderProgram(
            vertex_shader=vertex_src,
            fragment_shader=fragment_src,
            uniforms={},
            attributes={}
        )
    
    def create_render_target(self, width: int, height: int, format: str = "RGBA8") -> RenderTarget:
        """Create software render target"""
        return RenderTarget(width, height, format)
    
    def begin_frame(self):
        """Begin software frame"""
        pass
    
    def end_frame(self):
        """End software frame"""
        # In a real implementation, this would display the framebuffer
        pass
    
    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Clear software screen"""
        if self.framebuffer is not None:
            import numpy as np
            color = np.array([r * 255, g * 255, b * 255, a * 255], dtype=np.uint8)
            self.framebuffer[:] = color
    
    def _detect_platform(self) -> Platform:
        """Detect current platform"""
        system = platform.system().lower()
        if system == "windows":
            return Platform.WINDOWS
        elif system == "darwin":
            return Platform.MACOS
        elif system == "linux":
            return Platform.LINUX
        else:
            return Platform.LINUX

class CrossPlatformGraphicsFramework:
    """Main cross-platform graphics framework"""
    
    def __init__(self):
        self.backend = None
        self.current_api = None
        self.platform = self._detect_platform()
        self.available_apis = []
        self.preferred_apis = [
            GraphicsAPI.VULKAN,
            GraphicsAPI.OPENGL,
            GraphicsAPI.DIRECTX,
            GraphicsAPI.METAL,
            GraphicsAPI.SOFTWARE
        ]
        
        # Platform-specific API preferences
        self.platform_preferences = {
            Platform.WINDOWS: [GraphicsAPI.DIRECTX, GraphicsAPI.VULKAN, GraphicsAPI.OPENGL],
            Platform.MACOS: [GraphicsAPI.METAL, GraphicsAPI.OPENGL],
            Platform.LINUX: [GraphicsAPI.VULKAN, GraphicsAPI.OPENGL],
            Platform.ANDROID: [GraphicsAPI.VULKAN, GraphicsAPI.OPENGL],
            Platform.IOS: [GraphicsAPI.METAL, GraphicsAPI.OPENGL]
        }
    
    def _detect_platform(self) -> Platform:
        """Detect current platform"""
        system = platform.system().lower()
        if system == "windows":
            return Platform.WINDOWS
        elif system == "darwin":
            return Platform.MACOS
        elif system == "linux":
            return Platform.LINUX
        else:
            return Platform.LINUX
    
    def detect_available_apis(self) -> List[GraphicsAPI]:
        """Detect available graphics APIs"""
        available = []
        
        # Check OpenGL
        try:
            import moderngl
            available.append(GraphicsAPI.OPENGL)
        except ImportError:
            pass
        
        # Check Vulkan
        try:
            import vulkan
            available.append(GraphicsAPI.VULKAN)
        except ImportError:
            pass
        
        # Check DirectX (Windows only)
        if self.platform == Platform.WINDOWS:
            try:
                import d3d12
                available.append(GraphicsAPI.DIRECTX)
            except ImportError:
                pass
        
        # Check Metal (macOS only)
        if self.platform == Platform.MACOS:
            try:
                import metal
                available.append(GraphicsAPI.METAL)
            except ImportError:
                pass
        
        # Software rendering is always available
        available.append(GraphicsAPI.SOFTWARE)
        
        self.available_apis = available
        return available
    
    def select_best_api(self) -> GraphicsAPI:
        """Select the best available graphics API"""
        if not self.available_apis:
            self.detect_available_apis()
        
        # Get platform preferences
        platform_prefs = self.platform_preferences.get(self.platform, self.preferred_apis)
        
        # Find the best available API
        for api in platform_prefs:
            if api in self.available_apis:
                return api
        
        # Fallback to software rendering
        return GraphicsAPI.SOFTWARE
    
    def initialize(self, preferred_api: GraphicsAPI = None) -> bool:
        """Initialize the graphics framework"""
        print(f"Initializing graphics framework on {self.platform.value}")
        
        # Detect available APIs
        available_apis = self.detect_available_apis()
        print(f"Available APIs: {[api.value for api in available_apis]}")
        
        # Select API
        if preferred_api and preferred_api in available_apis:
            selected_api = preferred_api
        else:
            selected_api = self.select_best_api()
        
        print(f"Selected API: {selected_api.value}")
        
        # Create backend
        self.backend = self._create_backend(selected_api)
        if not self.backend:
            return False
        
        # Initialize backend
        if not self.backend.initialize():
            print(f"Failed to initialize {selected_api.value} backend")
            return False
        
        self.current_api = selected_api
        print(f"Graphics framework initialized with {selected_api.value}")
        return True
    
    def _create_backend(self, api: GraphicsAPI) -> GraphicsBackend:
        """Create a graphics backend for the specified API"""
        if api == GraphicsAPI.OPENGL:
            return OpenGLBackend()
        elif api == GraphicsAPI.SOFTWARE:
            return SoftwareBackend()
        elif api == GraphicsAPI.VULKAN:
            # VulkanBackend would be implemented here
            print("Vulkan backend not implemented, falling back to software")
            return SoftwareBackend()
        elif api == GraphicsAPI.DIRECTX:
            # DirectXBackend would be implemented here
            print("DirectX backend not implemented, falling back to software")
            return SoftwareBackend()
        elif api == GraphicsAPI.METAL:
            # MetalBackend would be implemented here
            print("Metal backend not implemented, falling back to software")
            return SoftwareBackend()
        else:
            return SoftwareBackend()
    
    def create_window(self, title: str, width: int, height: int) -> bool:
        """Create a window"""
        if not self.backend:
            return False
        
        return self.backend.create_window(title, width, height)
    
    def create_shader_program(self, vertex_src: str, fragment_src: str) -> ShaderProgram:
        """Create a shader program"""
        if not self.backend:
            return None
        
        return self.backend.create_shader_program(vertex_src, fragment_src)
    
    def create_render_target(self, width: int, height: int, format: str = "RGBA8") -> RenderTarget:
        """Create a render target"""
        if not self.backend:
            return None
        
        return self.backend.create_render_target(width, height, format)
    
    def begin_frame(self):
        """Begin rendering a frame"""
        if self.backend:
            self.backend.begin_frame()
    
    def end_frame(self):
        """End rendering a frame"""
        if self.backend:
            self.backend.end_frame()
    
    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Clear the screen"""
        if self.backend:
            self.backend.clear(r, g, b, a)
    
    def get_capabilities(self) -> GraphicsCapabilities:
        """Get graphics capabilities"""
        if self.backend:
            return self.backend.get_capabilities()
        return None
    
    def shutdown(self):
        """Shutdown the graphics framework"""
        if self.backend:
            self.backend.shutdown()
        self.backend = None
        self.current_api = None
        print("Graphics framework shutdown")
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get information about the current API"""
        if not self.backend:
            return {}
        
        capabilities = self.backend.get_capabilities()
        return {
            "current_api": self.current_api.value if self.current_api else "None",
            "platform": self.platform.value,
            "available_apis": [api.value for api in self.available_apis],
            "capabilities": {
                "max_texture_size": capabilities.max_texture_size,
                "max_vertex_attributes": capabilities.max_vertex_attributes,
                "max_uniform_blocks": capabilities.max_uniform_blocks,
                "max_texture_units": capabilities.max_texture_units,
                "shader_version": capabilities.shader_version,
                "glsl_version": capabilities.glsl_version,
                "vendor": capabilities.vendor,
                "renderer": capabilities.renderer,
                "version": capabilities.version
            } if capabilities else {}
        }

class GraphicsApplication:
    """Example application using the cross-platform framework"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.framework = CrossPlatformGraphicsFramework()
        self.running = False
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def initialize(self) -> bool:
        """Initialize the application"""
        print("=== Cross-Platform Graphics Application ===\n")
        
        # Initialize framework
        if not self.framework.initialize():
            print("Failed to initialize graphics framework")
            return False
        
        # Create window
        if not self.framework.create_window("Cross-Platform Graphics", self.width, self.height):
            print("Failed to create window")
            return False
        
        # Get API information
        api_info = self.framework.get_api_info()
        print("API Information:")
        print(f"  Current API: {api_info.get('current_api', 'Unknown')}")
        print(f"  Platform: {api_info.get('platform', 'Unknown')}")
        print(f"  Available APIs: {', '.join(api_info.get('available_apis', []))}")
        
        capabilities = api_info.get('capabilities', {})
        if capabilities:
            print("Capabilities:")
            print(f"  Max Texture Size: {capabilities.get('max_texture_size', 'Unknown')}")
            print(f"  Shader Version: {capabilities.get('shader_version', 'Unknown')}")
            print(f"  Vendor: {capabilities.get('vendor', 'Unknown')}")
            print(f"  Renderer: {capabilities.get('renderer', 'Unknown')}")
        
        # Create shader program
        vertex_shader = """
        #version 330
        in vec2 in_position;
        in vec3 in_color;
        out vec3 color;
        void main() {
            color = in_color;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        """
        
        self.shader_program = self.framework.create_shader_program(vertex_shader, fragment_shader)
        if not self.shader_program:
            print("Failed to create shader program")
            return False
        
        print("\nApplication initialized successfully!")
        return True
    
    def run(self):
        """Run the application"""
        if not self.initialize():
            return
        
        self.running = True
        print("\nStarting render loop...")
        print("Press Ctrl+C to exit")
        
        try:
            while self.running:
                # Begin frame
                self.framework.begin_frame()
                
                # Clear screen
                self.framework.clear(0.1, 0.1, 0.1, 1.0)
                
                # Render frame
                self.render_frame()
                
                # End frame
                self.framework.end_frame()
                
                # Update statistics
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 2.0:
                    fps = self.frame_count / (current_time - self.last_fps_time)
                    print(f"FPS: {fps:.1f}")
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Simple frame rate limiting
                time.sleep(1/60)
                
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def render_frame(self):
        """Render a single frame"""
        # This would contain the actual rendering code
        # For now, we just clear the screen
        pass
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.framework.shutdown()
        print("Application shutdown complete")

def main():
    """Main function"""
    print("=== Cross-Platform Graphics Framework ===\n")
    print("This framework provides:")
    print("  • Automatic API detection and selection")
    print("  • Platform-specific optimizations")
    print("  • Fallback mechanisms")
    print("  • Unified interface across platforms")
    print("  • Graceful degradation")
    
    # Create and run application
    app = GraphicsApplication(800, 600)
    app.run()

if __name__ == "__main__":
    main()
