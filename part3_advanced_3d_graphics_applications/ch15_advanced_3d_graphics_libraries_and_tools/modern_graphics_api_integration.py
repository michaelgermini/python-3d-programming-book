#!/usr/bin/env python3
"""
Chapter 15: Advanced 3D Graphics Libraries and Tools
Modern Graphics API Integration

Integrates modern graphics APIs like Vulkan, Metal, and DirectX for
high-performance rendering across different platforms.
"""

import sys
import platform
import time
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import weakref

class GraphicsAPI(Enum):
    """Modern graphics APIs"""
    VULKAN = "Vulkan"
    METAL = "Metal"
    DIRECTX = "DirectX"
    OPENGL = "OpenGL"

class Platform(Enum):
    """Supported platforms"""
    WINDOWS = "Windows"
    MACOS = "macOS"
    LINUX = "Linux"
    ANDROID = "Android"
    IOS = "iOS"

@dataclass
class DeviceInfo:
    """Graphics device information"""
    name: str
    vendor: str
    device_type: str
    memory_size: int
    api_version: str
    driver_version: str
    features: List[str]

@dataclass
class SwapChainInfo:
    """Swap chain configuration"""
    width: int
    height: int
    format: str
    present_mode: str
    image_count: int
    vsync: bool

class ModernGraphicsAPI:
    """Base class for modern graphics APIs"""
    
    def __init__(self, api: GraphicsAPI):
        self.api = api
        self.initialized = False
        self.device_info = None
        self.swap_chain = None
    
    def initialize(self) -> bool:
        """Initialize the graphics API"""
        raise NotImplementedError("Subclasses must implement initialize")
    
    def shutdown(self):
        """Shutdown the graphics API"""
        raise NotImplementedError("Subclasses must implement shutdown")
    
    def create_device(self) -> bool:
        """Create graphics device"""
        raise NotImplementedError("Subclasses must implement create_device")
    
    def create_swap_chain(self, width: int, height: int) -> bool:
        """Create swap chain"""
        raise NotImplementedError("Subclasses must implement create_swap_chain")
    
    def create_shader_module(self, shader_source: str, stage: str) -> Any:
        """Create shader module"""
        raise NotImplementedError("Subclasses must implement create_shader_module")
    
    def create_pipeline(self, vertex_shader: Any, fragment_shader: Any) -> Any:
        """Create graphics pipeline"""
        raise NotImplementedError("Subclasses must implement create_pipeline")
    
    def begin_frame(self) -> int:
        """Begin frame and get image index"""
        raise NotImplementedError("Subclasses must implement begin_frame")
    
    def end_frame(self, image_index: int):
        """End frame and present"""
        raise NotImplementedError("Subclasses must implement end_frame")
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information"""
        return self.device_info

class VulkanAPI(ModernGraphicsAPI):
    """Vulkan graphics API implementation"""
    
    def __init__(self):
        super().__init__(GraphicsAPI.VULKAN)
        self.instance = None
        self.device = None
        self.physical_device = None
        self.surface = None
        self.command_pool = None
        self.command_buffers = []
        self.semaphores = []
        self.fences = []
    
    def initialize(self) -> bool:
        """Initialize Vulkan"""
        try:
            import vulkan as vk
            
            # Create Vulkan instance
            app_info = vk.VkApplicationInfo(
                application_name="Modern Graphics API Demo",
                application_version=vk.VK_MAKE_VERSION(1, 0, 0),
                engine_name="No Engine",
                engine_version=vk.VK_MAKE_VERSION(1, 0, 0),
                api_version=vk.VK_API_VERSION_1_0
            )
            
            create_info = vk.VkInstanceCreateInfo(
                p_application_info=app_info,
                enabled_layer_count=0,
                enabled_extension_count=0
            )
            
            self.instance = vk.vkCreateInstance(create_info, None)
            
            # Get physical device
            devices = vk.vkEnumeratePhysicalDevices(self.instance)
            if not devices:
                print("No Vulkan devices found")
                return False
            
            self.physical_device = devices[0]
            
            # Get device properties
            properties = vk.vkGetPhysicalDeviceProperties(self.physical_device)
            self.device_info = DeviceInfo(
                name=properties.device_name.decode(),
                vendor="Unknown",
                device_type="Discrete GPU",
                memory_size=0,
                api_version=f"{properties.api_version}",
                driver_version=f"{properties.driver_version}",
                features=[]
            )
            
            print(f"Vulkan initialized with device: {self.device_info.name}")
            return True
            
        except ImportError:
            print("Vulkan not available: vulkan package not installed")
            return False
        except Exception as e:
            print(f"Failed to initialize Vulkan: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Vulkan"""
        if self.instance:
            import vulkan as vk
            vk.vkDestroyInstance(self.instance, None)
        self.initialized = False
    
    def create_device(self) -> bool:
        """Create Vulkan device"""
        try:
            import vulkan as vk
            
            # Get queue family properties
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
            
            # Find graphics queue family
            graphics_queue_family = None
            for i, queue_family in enumerate(queue_families):
                if queue_family.queue_flags & vk.VK_QUEUE_GRAPHICS_BIT:
                    graphics_queue_family = i
                    break
            
            if graphics_queue_family is None:
                print("No graphics queue family found")
                return False
            
            # Create logical device
            queue_create_info = vk.VkDeviceQueueCreateInfo(
                queue_family_index=graphics_queue_family,
                queue_count=1,
                p_queue_priorities=[1.0]
            )
            
            device_create_info = vk.VkDeviceCreateInfo(
                queue_create_info_count=1,
                p_queue_create_infos=[queue_create_info],
                enabled_extension_count=0,
                enabled_layer_count=0
            )
            
            self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)
            
            return True
            
        except Exception as e:
            print(f"Failed to create Vulkan device: {e}")
            return False
    
    def create_swap_chain(self, width: int, height: int) -> bool:
        """Create Vulkan swap chain"""
        # This would create the swap chain
        # Implementation depends on window system integration
        return True
    
    def create_shader_module(self, shader_source: str, stage: str) -> Any:
        """Create Vulkan shader module"""
        try:
            import vulkan as vk
            
            # Compile shader (in real implementation, you'd use a shader compiler)
            shader_code = shader_source.encode()
            
            create_info = vk.VkShaderModuleCreateInfo(
                code_size=len(shader_code),
                p_code=shader_code
            )
            
            return vk.vkCreateShaderModule(self.device, create_info, None)
            
        except Exception as e:
            print(f"Failed to create Vulkan shader module: {e}")
            return None
    
    def create_pipeline(self, vertex_shader: Any, fragment_shader: Any) -> Any:
        """Create Vulkan graphics pipeline"""
        # This would create the graphics pipeline
        # Implementation is complex and depends on specific requirements
        return None
    
    def begin_frame(self) -> int:
        """Begin Vulkan frame"""
        # This would acquire the next image from swap chain
        return 0
    
    def end_frame(self, image_index: int):
        """End Vulkan frame"""
        # This would present the image
        pass

class MetalAPI(ModernGraphicsAPI):
    """Metal graphics API implementation (macOS/iOS)"""
    
    def __init__(self):
        super().__init__(GraphicsAPI.METAL)
        self.device = None
        self.command_queue = None
        self.library = None
    
    def initialize(self) -> bool:
        """Initialize Metal"""
        try:
            import metal
            
            # Create Metal device
            self.device = metal.MTLCreateSystemDefaultDevice()
            if not self.device:
                print("No Metal device found")
                return False
            
            # Create command queue
            self.command_queue = self.device.newCommandQueue()
            
            # Get device info
            self.device_info = DeviceInfo(
                name=self.device.name,
                vendor="Apple",
                device_type="Integrated GPU",
                memory_size=0,
                api_version="Metal 2.0",
                driver_version="Unknown",
                features=[]
            )
            
            print(f"Metal initialized with device: {self.device_info.name}")
            return True
            
        except ImportError:
            print("Metal not available: metal package not installed")
            return False
        except Exception as e:
            print(f"Failed to initialize Metal: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Metal"""
        self.device = None
        self.command_queue = None
        self.initialized = False
    
    def create_device(self) -> bool:
        """Metal device is created during initialization"""
        return self.device is not None
    
    def create_swap_chain(self, width: int, height: int) -> bool:
        """Create Metal swap chain"""
        # Metal uses CAMetalLayer for swap chain
        return True
    
    def create_shader_module(self, shader_source: str, stage: str) -> Any:
        """Create Metal shader module"""
        try:
            # Compile Metal shader
            library = self.device.newLibraryWithSource_options_error(
                shader_source, None, None
            )
            return library
            
        except Exception as e:
            print(f"Failed to create Metal shader module: {e}")
            return None
    
    def create_pipeline(self, vertex_shader: Any, fragment_shader: Any) -> Any:
        """Create Metal graphics pipeline"""
        try:
            # Create render pipeline descriptor
            pipeline_descriptor = metal.MTLRenderPipelineDescriptor()
            pipeline_descriptor.vertexFunction = vertex_shader
            pipeline_descriptor.fragmentFunction = fragment_shader
            
            # Create pipeline state
            pipeline_state = self.device.newRenderPipelineStateWithDescriptor_error(
                pipeline_descriptor, None
            )
            
            return pipeline_state
            
        except Exception as e:
            print(f"Failed to create Metal pipeline: {e}")
            return None
    
    def begin_frame(self) -> int:
        """Begin Metal frame"""
        # Get command buffer
        command_buffer = self.command_queue.commandBuffer()
        return 0
    
    def end_frame(self, image_index: int):
        """End Metal frame"""
        # Commit command buffer
        pass

class DirectXAPI(ModernGraphicsAPI):
    """DirectX 12 graphics API implementation (Windows)"""
    
    def __init__(self):
        super().__init__(GraphicsAPI.DIRECTX)
        self.device = None
        self.command_queue = None
        self.swap_chain = None
    
    def initialize(self) -> bool:
        """Initialize DirectX 12"""
        try:
            import d3d12
            
            # Create DirectX 12 device
            self.device = d3d12.CreateDevice()
            if not self.device:
                print("No DirectX 12 device found")
                return False
            
            # Create command queue
            self.command_queue = self.device.CreateCommandQueue()
            
            # Get device info
            self.device_info = DeviceInfo(
                name="DirectX 12 Device",
                vendor="Microsoft",
                device_type="Hardware",
                memory_size=0,
                api_version="DirectX 12",
                driver_version="Unknown",
                features=[]
            )
            
            print("DirectX 12 initialized")
            return True
            
        except ImportError:
            print("DirectX 12 not available: d3d12 package not installed")
            return False
        except Exception as e:
            print(f"Failed to initialize DirectX 12: {e}")
            return False
    
    def shutdown(self):
        """Shutdown DirectX 12"""
        self.device = None
        self.command_queue = None
        self.initialized = False
    
    def create_device(self) -> bool:
        """DirectX device is created during initialization"""
        return self.device is not None
    
    def create_swap_chain(self, width: int, height: int) -> bool:
        """Create DirectX swap chain"""
        # Create swap chain with DXGI
        return True
    
    def create_shader_module(self, shader_source: str, stage: str) -> Any:
        """Create DirectX shader module"""
        try:
            # Compile HLSL shader
            shader_blob = d3d12.CompileShader(shader_source, stage)
            return shader_blob
            
        except Exception as e:
            print(f"Failed to create DirectX shader module: {e}")
            return None
    
    def create_pipeline(self, vertex_shader: Any, fragment_shader: Any) -> Any:
        """Create DirectX graphics pipeline"""
        try:
            # Create pipeline state object
            pso = self.device.CreateGraphicsPipelineState(vertex_shader, fragment_shader)
            return pso
            
        except Exception as e:
            print(f"Failed to create DirectX pipeline: {e}")
            return None
    
    def begin_frame(self) -> int:
        """Begin DirectX frame"""
        # Get command allocator and command list
        return 0
    
    def end_frame(self, image_index: int):
        """End DirectX frame"""
        # Present frame
        pass

class ModernGraphicsAPIManager:
    """Manager for modern graphics APIs"""
    
    def __init__(self):
        self.current_api = None
        self.api_instance = None
        self.platform = self._detect_platform()
        self.available_apis = []
        
        # Platform-specific API preferences
        self.platform_preferences = {
            Platform.WINDOWS: [GraphicsAPI.DIRECTX, GraphicsAPI.VULKAN],
            Platform.MACOS: [GraphicsAPI.METAL, GraphicsAPI.VULKAN],
            Platform.LINUX: [GraphicsAPI.VULKAN],
            Platform.ANDROID: [GraphicsAPI.VULKAN],
            Platform.IOS: [GraphicsAPI.METAL]
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
        
        # Check Vulkan
        try:
            import vulkan
            available.append(GraphicsAPI.VULKAN)
        except ImportError:
            pass
        
        # Check Metal (macOS/iOS only)
        if self.platform in [Platform.MACOS, Platform.IOS]:
            try:
                import metal
                available.append(GraphicsAPI.METAL)
            except ImportError:
                pass
        
        # Check DirectX (Windows only)
        if self.platform == Platform.WINDOWS:
            try:
                import d3d12
                available.append(GraphicsAPI.DIRECTX)
            except ImportError:
                pass
        
        self.available_apis = available
        return available
    
    def select_best_api(self) -> GraphicsAPI:
        """Select the best available graphics API"""
        if not self.available_apis:
            self.detect_available_apis()
        
        # Get platform preferences
        platform_prefs = self.platform_preferences.get(self.platform, [])
        
        # Find the best available API
        for api in platform_prefs:
            if api in self.available_apis:
                return api
        
        # Fallback to first available
        if self.available_apis:
            return self.available_apis[0]
        
        return None
    
    def initialize(self, preferred_api: GraphicsAPI = None) -> bool:
        """Initialize the graphics API manager"""
        print(f"Initializing modern graphics API manager on {self.platform.value}")
        
        # Detect available APIs
        available_apis = self.detect_available_apis()
        print(f"Available APIs: {[api.value for api in available_apis]}")
        
        # Select API
        if preferred_api and preferred_api in available_apis:
            selected_api = preferred_api
        else:
            selected_api = self.select_best_api()
        
        if not selected_api:
            print("No suitable graphics API found")
            return False
        
        print(f"Selected API: {selected_api.value}")
        
        # Create API instance
        self.api_instance = self._create_api_instance(selected_api)
        if not self.api_instance:
            return False
        
        # Initialize API
        if not self.api_instance.initialize():
            print(f"Failed to initialize {selected_api.value}")
            return False
        
        self.current_api = selected_api
        print(f"Modern graphics API initialized with {selected_api.value}")
        return True
    
    def _create_api_instance(self, api: GraphicsAPI) -> ModernGraphicsAPI:
        """Create an API instance"""
        if api == GraphicsAPI.VULKAN:
            return VulkanAPI()
        elif api == GraphicsAPI.METAL:
            return MetalAPI()
        elif api == GraphicsAPI.DIRECTX:
            return DirectXAPI()
        else:
            return None
    
    def create_device(self) -> bool:
        """Create graphics device"""
        if not self.api_instance:
            return False
        
        return self.api_instance.create_device()
    
    def create_swap_chain(self, width: int, height: int) -> bool:
        """Create swap chain"""
        if not self.api_instance:
            return False
        
        return self.api_instance.create_swap_chain(width, height)
    
    def create_shader_module(self, shader_source: str, stage: str) -> Any:
        """Create shader module"""
        if not self.api_instance:
            return None
        
        return self.api_instance.create_shader_module(shader_source, stage)
    
    def create_pipeline(self, vertex_shader: Any, fragment_shader: Any) -> Any:
        """Create graphics pipeline"""
        if not self.api_instance:
            return None
        
        return self.api_instance.create_pipeline(vertex_shader, fragment_shader)
    
    def begin_frame(self) -> int:
        """Begin frame"""
        if not self.api_instance:
            return -1
        
        return self.api_instance.begin_frame()
    
    def end_frame(self, image_index: int):
        """End frame"""
        if self.api_instance:
            self.api_instance.end_frame(image_index)
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information"""
        if self.api_instance:
            return self.api_instance.get_device_info()
        return None
    
    def shutdown(self):
        """Shutdown the graphics API manager"""
        if self.api_instance:
            self.api_instance.shutdown()
        self.api_instance = None
        self.current_api = None
        print("Modern graphics API manager shutdown")

class ModernGraphicsApplication:
    """Example application using modern graphics APIs"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.api_manager = ModernGraphicsAPIManager()
        self.running = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Shader programs
        self.vertex_shader = None
        self.fragment_shader = None
        self.pipeline = None
    
    def initialize(self) -> bool:
        """Initialize the application"""
        print("=== Modern Graphics API Application ===\n")
        
        # Initialize API manager
        if not self.api_manager.initialize():
            print("Failed to initialize graphics API manager")
            return False
        
        # Create device
        if not self.api_manager.create_device():
            print("Failed to create graphics device")
            return False
        
        # Create swap chain
        if not self.api_manager.create_swap_chain(self.width, self.height):
            print("Failed to create swap chain")
            return False
        
        # Get device information
        device_info = self.api_manager.get_device_info()
        if device_info:
            print("Device Information:")
            print(f"  Name: {device_info.name}")
            print(f"  Vendor: {device_info.vendor}")
            print(f"  API Version: {device_info.api_version}")
            print(f"  Driver Version: {device_info.driver_version}")
        
        # Create shaders
        self._create_shaders()
        
        # Create pipeline
        if self.vertex_shader and self.fragment_shader:
            self.pipeline = self.api_manager.create_pipeline(self.vertex_shader, self.fragment_shader)
        
        print("\nApplication initialized successfully!")
        return True
    
    def _create_shaders(self):
        """Create shader programs based on current API"""
        api = self.api_manager.current_api
        
        if api == GraphicsAPI.VULKAN:
            # Vulkan GLSL shaders
            vertex_source = """
            #version 450
            layout(location = 0) in vec2 inPosition;
            layout(location = 1) in vec3 inColor;
            layout(location = 0) out vec3 fragColor;
            
            void main() {
                gl_Position = vec4(inPosition, 0.0, 1.0);
                fragColor = inColor;
            }
            """
            
            fragment_source = """
            #version 450
            layout(location = 0) in vec3 fragColor;
            layout(location = 0) out vec4 outColor;
            
            void main() {
                outColor = vec4(fragColor, 1.0);
            }
            """
            
        elif api == GraphicsAPI.METAL:
            # Metal shaders
            vertex_source = """
            #include <metal_stdlib>
            using namespace metal;
            
            struct VertexIn {
                float2 position [[attribute(0)]];
                float3 color [[attribute(1)]];
            };
            
            struct VertexOut {
                float4 position [[position]];
                float3 color;
            };
            
            vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
                VertexOut out;
                out.position = float4(in.position, 0.0, 1.0);
                out.color = in.color;
                return out;
            }
            """
            
            fragment_source = """
            #include <metal_stdlib>
            using namespace metal;
            
            fragment float4 fragment_main(float3 color [[stage_in]]) {
                return float4(color, 1.0);
            }
            """
            
        elif api == GraphicsAPI.DIRECTX:
            # DirectX HLSL shaders
            vertex_source = """
            struct VSInput {
                float2 position : POSITION;
                float3 color : COLOR;
            };
            
            struct VSOutput {
                float4 position : SV_POSITION;
                float3 color : COLOR;
            };
            
            VSOutput VSMain(VSInput input) {
                VSOutput output;
                output.position = float4(input.position, 0.0, 1.0);
                output.color = input.color;
                return output;
            }
            """
            
            fragment_source = """
            struct PSInput {
                float4 position : SV_POSITION;
                float3 color : COLOR;
            };
            
            float4 PSMain(PSInput input) : SV_TARGET {
                return float4(input.color, 1.0);
            }
            """
        
        else:
            return
        
        # Create shader modules
        self.vertex_shader = self.api_manager.create_shader_module(vertex_source, "vertex")
        self.fragment_shader = self.api_manager.create_shader_module(fragment_source, "fragment")
    
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
                image_index = self.api_manager.begin_frame()
                
                # Render frame
                self.render_frame(image_index)
                
                # End frame
                self.api_manager.end_frame(image_index)
                
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
    
    def render_frame(self, image_index: int):
        """Render a single frame"""
        # This would contain the actual rendering code using the modern API
        # Implementation depends on the specific API being used
        pass
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.api_manager.shutdown()
        print("Application shutdown complete")

def main():
    """Main function"""
    print("=== Modern Graphics API Integration ===\n")
    print("This demonstrates integration of modern graphics APIs:")
    print("  • Vulkan - Cross-platform, low-level API")
    print("  • Metal - Apple's graphics API")
    print("  • DirectX 12 - Microsoft's graphics API")
    print("  • Automatic API selection")
    print("  • Platform-specific optimizations")
    
    # Create and run application
    app = ModernGraphicsApplication(800, 600)
    app.run()

if __name__ == "__main__":
    main()
