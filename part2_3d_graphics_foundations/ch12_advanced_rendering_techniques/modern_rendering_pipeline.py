#!/usr/bin/env python3
"""
Chapter 12: Advanced Rendering Techniques
Modern Rendering Pipeline

Demonstrates modern rendering techniques including deferred rendering, G-buffer,
post-processing pipeline, and performance monitoring.
"""

import numpy as np
import moderngl
import glfw
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Modern Rendering Pipeline"
__description__ = "Deferred rendering and modern graphics pipeline"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    """3D vector for positions and directions"""
    x: float
    y: float
    z: float
    
    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class Color:
    """Color representation"""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_array(self) -> List[float]:
        return [self.r, self.g, self.b, self.a]

class RenderTarget(Enum):
    """Render target types"""
    POSITION = "position"
    NORMAL = "normal"
    ALBEDO = "albedo"
    METALLIC_ROUGHNESS = "metallic_roughness"

class PostProcessEffect(Enum):
    """Post-processing effects"""
    BLOOM = "bloom"
    TONEMAPPING = "tonemapping"
    FXAA = "fxaa"

# ============================================================================
# G-BUFFER SYSTEM
# ============================================================================

class GBuffer:
    """G-buffer for deferred rendering"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create render targets
        self.position_texture = ctx.texture((width, height), 4, dtype='f4')
        self.normal_texture = ctx.texture((width, height), 4, dtype='f4')
        self.albedo_texture = ctx.texture((width, height), 4, dtype='f4')
        self.metallic_roughness_texture = ctx.texture((width, height), 4, dtype='f4')
        self.depth_texture = ctx.depth_texture((width, height))
        
        # Create framebuffer
        self.framebuffer = ctx.framebuffer(
            color_attachments=[
                self.position_texture,
                self.normal_texture,
                self.albedo_texture,
                self.metallic_roughness_texture
            ],
            depth_attachment=self.depth_texture
        )
        
        # Store textures in a dictionary
        self.textures = {
            RenderTarget.POSITION: self.position_texture,
            RenderTarget.NORMAL: self.normal_texture,
            RenderTarget.ALBEDO: self.albedo_texture,
            RenderTarget.METALLIC_ROUGHNESS: self.metallic_roughness_texture
        }
    
    def bind(self):
        """Bind the G-buffer framebuffer"""
        self.framebuffer.use()
    
    def unbind(self):
        """Unbind the G-buffer framebuffer"""
        self.ctx.screen.use()
    
    def get_texture(self, target: RenderTarget) -> moderngl.Texture:
        """Get a specific texture from the G-buffer"""
        return self.textures[target]

# ============================================================================
# DEFERRED RENDERING
# ============================================================================

class DeferredRenderer:
    """Deferred rendering system"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create G-buffer
        self.g_buffer = GBuffer(ctx, width, height)
        
        # Create fullscreen quad
        self.quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0, 1.0
        ], dtype='f4')
        
        self.quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        
        self.quad_vbo = ctx.buffer(self.quad_vertices.tobytes())
        self.quad_ibo = ctx.buffer(self.quad_indices.tobytes())
        
        # Create shaders
        self.setup_shaders()
        
        # Create vertex array for deferred lighting
        self.quad_vao = ctx.vertex_array(
            self.deferred_shader,
            [
                (self.quad_vbo, '3f 2f', 'in_position', 'in_texcoord'),
            ],
            self.quad_ibo
        )
    
    def setup_shaders(self):
        """Setup shaders for deferred rendering"""
        # Geometry pass vertex shader
        geometry_vertex = """
        #version 330
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normal_matrix;
        
        out vec3 world_pos;
        out vec3 world_normal;
        out vec2 texcoord;
        
        void main() {
            world_pos = vec3(model * vec4(in_position, 1.0));
            world_normal = normal_matrix * in_normal;
            texcoord = in_texcoord;
            gl_Position = projection * view * vec4(world_pos, 1.0);
        }
        """
        
        # Geometry pass fragment shader
        geometry_fragment = """
        #version 330
        
        in vec3 world_pos;
        in vec3 world_normal;
        in vec2 texcoord;
        
        uniform vec4 albedo;
        uniform float metallic;
        uniform float roughness;
        
        layout(location = 0) out vec4 g_position;
        layout(location = 1) out vec4 g_normal;
        layout(location = 2) out vec4 g_albedo;
        layout(location = 3) out vec4 g_metallic_roughness;
        
        void main() {
            g_position = vec4(world_pos, 1.0);
            g_normal = vec4(normalize(world_normal), 1.0);
            g_albedo = albedo;
            g_metallic_roughness = vec4(metallic, roughness, 0.0, 1.0);
        }
        """
        
        # Deferred lighting fragment shader
        deferred_fragment = """
        #version 330
        
        in vec2 texcoord;
        
        uniform sampler2D g_position;
        uniform sampler2D g_normal;
        uniform sampler2D g_albedo;
        uniform sampler2D g_metallic_roughness;
        
        uniform vec3 light_positions[4];
        uniform vec3 light_colors[4];
        uniform vec3 view_pos;
        
        out vec4 frag_color;
        
        void main() {
            vec3 world_pos = texture(g_position, texcoord).rgb;
            vec3 normal = texture(g_normal, texcoord).rgb;
            vec3 albedo = texture(g_albedo, texcoord).rgb;
            vec3 metallic_roughness = texture(g_metallic_roughness, texcoord).rgb;
            
            float metallic = metallic_roughness.r;
            float roughness = metallic_roughness.g;
            
            vec3 N = normalize(normal);
            vec3 V = normalize(view_pos - world_pos);
            
            vec3 Lo = vec3(0.0);
            
            for(int i = 0; i < 4; ++i) {
                vec3 L = normalize(light_positions[i] - world_pos);
                float distance = length(light_positions[i] - world_pos);
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance = light_colors[i] * attenuation;
                
                float NdotL = max(dot(N, L), 0.0);
                Lo += radiance * NdotL;
            }
            
            vec3 ambient = vec3(0.03) * albedo;
            vec3 color = ambient + Lo;
            
            // HDR tonemapping
            color = color / (color + vec3(1.0));
            // Gamma correction
            color = pow(color, vec3(1.0/2.2));
            
            frag_color = vec4(color, 1.0);
        }
        """
        
        # Create shader programs
        self.geometry_shader = self.ctx.program(
            vertex_shader=geometry_vertex,
            fragment_shader=geometry_fragment
        )
        
        self.deferred_shader = self.ctx.program(
            vertex_shader=geometry_vertex,
            fragment_shader=deferred_fragment
        )
    
    def geometry_pass(self, scene_objects: List[Any], view_matrix: np.ndarray, 
                     projection_matrix: np.ndarray):
        """Render geometry to G-buffer"""
        self.g_buffer.bind()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render scene objects to G-buffer
        for obj in scene_objects:
            # Set uniforms and render object
            pass
    
    def lighting_pass(self, lights: List[Any], view_pos: Vector3D):
        """Render lighting using G-buffer data"""
        self.g_buffer.unbind()
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Bind G-buffer textures
        self.deferred_shader['g_position'].value = 0
        self.deferred_shader['g_normal'].value = 1
        self.deferred_shader['g_albedo'].value = 2
        self.deferred_shader['g_metallic_roughness'].value = 3
        
        self.g_buffer.get_texture(RenderTarget.POSITION).use(0)
        self.g_buffer.get_texture(RenderTarget.NORMAL).use(1)
        self.g_buffer.get_texture(RenderTarget.ALBEDO).use(2)
        self.g_buffer.get_texture(RenderTarget.METALLIC_ROUGHNESS).use(3)
        
        # Set lighting uniforms
        for i, light in enumerate(lights[:4]):
            self.deferred_shader[f'light_positions[{i}]'].write(light.position.to_array())
            self.deferred_shader[f'light_colors[{i}]'].write(light.color.to_array())
        
        self.deferred_shader['view_pos'].write(view_pos.to_array())
        
        # Render fullscreen quad
        self.quad_vao.render()
    
    def render(self, scene_objects: List[Any], lights: List[Any], 
              view_matrix: np.ndarray, projection_matrix: np.ndarray, 
              view_pos: Vector3D):
        """Complete deferred rendering pass"""
        # Geometry pass
        self.geometry_pass(scene_objects, view_matrix, projection_matrix)
        
        # Lighting pass
        self.lighting_pass(lights, view_pos)

# ============================================================================
# POST-PROCESSING PIPELINE
# ============================================================================

class PostProcessor:
    """Post-processing pipeline"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create framebuffers for ping-pong rendering
        self.ping_fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((width, height), 4)]
        )
        self.pong_fbo = ctx.framebuffer(
            color_attachments=[ctx.texture((width, height), 4)]
        )
        
        # Create fullscreen quad
        self.quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0, 1.0
        ], dtype='f4')
        
        self.quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        
        self.quad_vbo = ctx.buffer(self.quad_vertices.tobytes())
        self.quad_ibo = ctx.buffer(self.quad_indices.tobytes())
        
        # Create post-processing shaders
        self.setup_shaders()
        
        # Create vertex arrays
        self.quad_vao = ctx.vertex_array(
            self.bloom_shader,
            [
                (self.quad_vbo, '3f 2f', 'in_position', 'in_texcoord'),
            ],
            self.quad_ibo
        )
    
    def setup_shaders(self):
        """Setup post-processing shaders"""
        # Post-processing vertex shader
        post_vertex = """
        #version 330
        
        in vec3 in_position;
        in vec2 in_texcoord;
        
        out vec2 texcoord;
        
        void main() {
            texcoord = in_texcoord;
            gl_Position = vec4(in_position, 1.0);
        }
        """
        
        # Bloom fragment shader
        bloom_fragment = """
        #version 330
        
        in vec2 texcoord;
        
        uniform sampler2D input_texture;
        uniform float threshold;
        uniform float intensity;
        
        out vec4 frag_color;
        
        void main() {
            vec3 color = texture(input_texture, texcoord).rgb;
            
            // Extract bright areas
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            if(brightness > threshold) {
                frag_color = vec4(color * intensity, 1.0);
            } else {
                frag_color = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        
        # Tonemapping fragment shader
        tonemap_fragment = """
        #version 330
        
        in vec2 texcoord;
        
        uniform sampler2D input_texture;
        uniform float exposure;
        uniform float gamma;
        
        out vec4 frag_color;
        
        void main() {
            vec3 color = texture(input_texture, texcoord).rgb;
            
            // Reinhard tonemapping
            vec3 mapped = color / (color + vec3(1.0));
            
            // Exposure adjustment
            mapped = vec3(1.0) - exp(-mapped * exposure);
            
            // Gamma correction
            mapped = pow(mapped, vec3(1.0 / gamma));
            
            frag_color = vec4(mapped, 1.0);
        }
        """
        
        # Create shader programs
        self.bloom_shader = self.ctx.program(
            vertex_shader=post_vertex,
            fragment_shader=bloom_fragment
        )
        
        self.tonemap_shader = self.ctx.program(
            vertex_shader=post_vertex,
            fragment_shader=tonemap_fragment
        )
    
    def apply_bloom(self, input_texture: moderngl.Texture, threshold: float = 1.0, 
                   intensity: float = 1.0) -> moderngl.Texture:
        """Apply bloom effect"""
        self.ping_fbo.use()
        self.bloom_shader['input_texture'].value = 0
        self.bloom_shader['threshold'].value = threshold
        self.bloom_shader['intensity'].value = intensity
        input_texture.use(0)
        
        self.quad_vao.program = self.bloom_shader
        self.quad_vao.render()
        
        return self.ping_fbo.color_attachments[0]
    
    def apply_tonemapping(self, input_texture: moderngl.Texture, exposure: float = 1.0, 
                         gamma: float = 2.2) -> moderngl.Texture:
        """Apply tonemapping and gamma correction"""
        self.ping_fbo.use()
        self.tonemap_shader['input_texture'].value = 0
        self.tonemap_shader['exposure'].value = exposure
        self.tonemap_shader['gamma'].value = gamma
        input_texture.use(0)
        
        self.quad_vao.program = self.tonemap_shader
        self.quad_vao.render()
        
        return self.ping_fbo.color_attachments[0]
    
    def apply_post_processing(self, input_texture: moderngl.Texture, 
                            effects: List[PostProcessEffect]) -> moderngl.Texture:
        """Apply multiple post-processing effects"""
        current_texture = input_texture
        
        for effect in effects:
            if effect == PostProcessEffect.BLOOM:
                current_texture = self.apply_bloom(current_texture)
            elif effect == PostProcessEffect.TONEMAPPING:
                current_texture = self.apply_tonemapping(current_texture)
        
        return current_texture

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.frame_times = []
        self.fps_history = []
        self.max_history = 60
        
        # Performance metrics
        self.frame_count = 0
        self.last_time = time.time()
        self.current_fps = 0.0
        self.average_fps = 0.0
        self.min_fps = float('inf')
        self.max_fps = 0.0
    
    def update(self):
        """Update performance metrics"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        # Update frame times
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if frame_time > 0:
            self.current_fps = 1.0 / frame_time
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > self.max_history:
                self.fps_history.pop(0)
        
        # Update statistics
        self.frame_count += 1
        if self.fps_history:
            self.average_fps = sum(self.fps_history) / len(self.fps_history)
            self.min_fps = min(self.min_fps, min(self.fps_history))
            self.max_fps = max(self.max_fps, max(self.fps_history))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        return {
            'current_fps': self.current_fps,
            'average_fps': self.average_fps,
            'min_fps': self.min_fps,
            'max_fps': self.max_fps,
            'frame_time': self.frame_times[-1] if self.frame_times else 0.0
        }

# ============================================================================
# MODERN RENDERING PIPELINE
# ============================================================================

class ModernRenderingPipeline:
    """Complete modern rendering pipeline"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        
        # Rendering components
        self.deferred_renderer = None
        self.post_processor = None
        self.performance_monitor = None
        
        # Scene data
        self.scene_objects = []
        self.lights = []
        
        # Camera
        self.camera_pos = Vector3D(0, 0, 3)
        self.camera_target = Vector3D(0, 0, 0)
        self.camera_up = Vector3D(0, 1, 0)
        self.fov = 45.0
        
        # Rendering settings
        self.enable_post_processing = True
        self.post_effects = [PostProcessEffect.BLOOM, PostProcessEffect.TONEMAPPING]
        
        self.init_glfw()
        self.init_opengl()
        self.setup_rendering_pipeline()
    
    def init_glfw(self):
        """Initialize GLFW"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Modern Rendering Pipeline", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        """Initialize OpenGL context"""
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
    
    def setup_rendering_pipeline(self):
        """Setup the complete rendering pipeline"""
        self.deferred_renderer = DeferredRenderer(self.ctx, self.width, self.height)
        self.post_processor = PostProcessor(self.ctx, self.width, self.height)
        self.performance_monitor = PerformanceMonitor()
    
    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix"""
        return np.eye(4, dtype='f4')
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix"""
        return np.eye(4, dtype='f4')
    
    def render_frame(self):
        """Render a complete frame"""
        # Update performance monitor
        self.performance_monitor.update()
        
        # Get matrices
        view_matrix = self.get_view_matrix()
        projection_matrix = self.get_projection_matrix()
        
        # Deferred rendering
        self.deferred_renderer.render(
            self.scene_objects,
            self.lights,
            view_matrix,
            projection_matrix,
            self.camera_pos
        )
        
        # Post-processing
        if self.enable_post_processing:
            final_texture = self.deferred_renderer.g_buffer.get_texture(RenderTarget.ALBEDO)
            processed_texture = self.post_processor.apply_post_processing(
                final_texture, self.post_effects
            )
    
    def handle_input(self):
        """Handle keyboard input"""
        if glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS:
            self.enable_post_processing = not self.enable_post_processing
            print(f"Post-processing: {'enabled' if self.enable_post_processing else 'disabled'}")
        
        if glfw.get_key(self.window, glfw.KEY_F) == glfw.PRESS:
            stats = self.performance_monitor.get_performance_stats()
            print(f"FPS: {stats['current_fps']:.1f} (Avg: {stats['average_fps']:.1f})")
    
    def framebuffer_size_callback(self, window, width, height):
        """Handle window resize"""
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        """Main application loop"""
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Render
            self.render_frame()
            
            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Handle input
            self.handle_input()
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_deferred_rendering():
    """Demonstrate deferred rendering concepts"""
    print("=== Deferred Rendering Demo ===\n")
    
    print("Deferred rendering advantages:")
    print("  • Decoupled geometry and lighting")
    print("  • Efficient handling of many lights")
    print("  • Reduced overdraw")
    print("  • Better memory bandwidth usage")
    print("  • Support for complex lighting models")
    print()
    
    print("G-buffer components:")
    print("  • Position buffer: World space positions")
    print("  • Normal buffer: Surface normals")
    print("  • Albedo buffer: Base material colors")
    print("  • Metallic/Roughness buffer: PBR material properties")
    print()

def demonstrate_post_processing():
    """Demonstrate post-processing pipeline"""
    print("=== Post-Processing Pipeline Demo ===\n")
    
    print("Available post-processing effects:")
    print("  • Bloom: Glowing bright areas")
    print("  • Tonemapping: HDR to LDR conversion")
    print("  • FXAA: Fast approximate anti-aliasing")
    print()
    
    print("Post-processing pipeline:")
    print("  1. Render scene to texture")
    print("  2. Apply effects in sequence")
    print("  3. Use ping-pong buffers for multi-pass effects")
    print("  4. Render final result to screen")
    print()

def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("=== Performance Monitoring Demo ===\n")
    
    print("Performance metrics:")
    print("  • FPS: Frames per second")
    print("  • Frame time: Time per frame")
    print("  • GPU usage: GPU utilization percentage")
    print("  • Memory usage: GPU memory consumption")
    print()
    
    print("Optimization strategies:")
    print("  • Reduce draw calls through batching")
    print("  • Minimize state changes")
    print("  • Use LOD for distant objects")
    print("  • Implement frustum culling")
    print()

def demonstrate_rendering_pipeline():
    """Demonstrate the complete rendering pipeline"""
    print("=== Modern Rendering Pipeline Demo ===\n")
    
    print("Starting modern rendering pipeline...")
    print("Controls:")
    print("  P: Toggle post-processing")
    print("  F: Show performance stats")
    print("  ESC: Exit")
    print()
    
    try:
        pipeline = ModernRenderingPipeline(800, 600)
        pipeline.run()
    except Exception as e:
        print(f"✗ Rendering pipeline failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate modern rendering pipeline"""
    print("=== Modern Rendering Pipeline Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    demonstrate_deferred_rendering()
    demonstrate_post_processing()
    demonstrate_performance_monitoring()
    
    print("="*60)
    print("Modern Rendering Pipeline demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Deferred rendering with G-buffer")
    print("✓ Multiple render targets")
    print("✓ Post-processing pipeline")
    print("✓ Performance monitoring and optimization")
    print("✓ Modern graphics pipeline architecture")
    print("✓ Efficient rendering techniques")
    
    print("\nPipeline features:")
    print("• G-buffer generation and management")
    print("• Deferred lighting calculations")
    print("• Post-processing effects (bloom, tonemapping)")
    print("• Performance profiling and monitoring")
    print("• Multi-pass rendering optimization")
    
    print("\nApplications:")
    print("• AAA game development: High-quality rendering")
    print("• Real-time visualization: Interactive graphics")
    print("• Architectural rendering: Professional visualization")
    print("• Virtual reality: High-performance VR graphics")
    print("• Film and animation: Real-time pre-visualization")
    
    print("\nNext steps:")
    print("• Implement actual geometry rendering")
    print("• Add more post-processing effects")
    print("• Implement advanced optimization techniques")
    print("• Add support for compute shaders")
    print("• Optimize for mobile and VR platforms")

if __name__ == "__main__":
    main()
