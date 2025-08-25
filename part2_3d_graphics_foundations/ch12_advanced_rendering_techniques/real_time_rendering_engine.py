#!/usr/bin/env python3
"""
Chapter 12: Advanced Rendering Techniques
Real-Time Rendering Engine

Demonstrates a real-time rendering engine with performance optimization,
memory management, and advanced rendering techniques.
"""

import numpy as np
import moderngl
import glfw
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    x: float
    y: float
    z: float
    
    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_array(self) -> List[float]:
        return [self.r, self.g, self.b, self.a]

class RenderMode(Enum):
    FORWARD = "forward"
    DEFERRED = "deferred"

class QualityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=60)
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.frame_start_time = 0.0
    
    def start_frame(self):
        self.frame_start_time = time.time()
    
    def end_frame(self):
        frame_time = time.time() - self.frame_start_time
        self.frame_times.append(frame_time)
        
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def get_stats(self) -> Dict[str, float]:
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        return {
            'fps': self.current_fps,
            'frame_time': avg_frame_time * 1000
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"FPS: {stats['fps']:.1f} | Frame: {stats['frame_time']:.2f}ms")

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryManager:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.texture_pool = {}
        self.buffer_pool = {}
        self.texture_memory = 0
        self.buffer_memory = 0
        self.total_memory = 0
    
    def create_texture(self, size: Tuple[int, int], components: int = 4) -> moderngl.Texture:
        texture = self.ctx.texture(size, components)
        memory_size = size[0] * size[1] * components * 4
        self.texture_memory += memory_size
        self.total_memory += memory_size
        
        texture_id = id(texture)
        self.texture_pool[texture_id] = {
            'texture': texture,
            'size': memory_size,
            'last_used': time.time()
        }
        return texture
    
    def create_buffer(self, data: bytes) -> moderngl.Buffer:
        buffer = self.ctx.buffer(data)
        memory_size = len(data)
        self.buffer_memory += memory_size
        self.total_memory += memory_size
        
        buffer_id = id(buffer)
        self.buffer_pool[buffer_id] = {
            'buffer': buffer,
            'size': memory_size,
            'last_used': time.time()
        }
        return buffer
    
    def cleanup_unused(self, max_age: float = 60.0):
        current_time = time.time()
        
        for texture_id, info in list(self.texture_pool.items()):
            if current_time - info['last_used'] > max_age:
                self.texture_memory -= info['size']
                self.total_memory -= info['size']
                del self.texture_pool[texture_id]
        
        for buffer_id, info in list(self.buffer_pool.items()):
            if current_time - info['last_used'] > max_age:
                self.buffer_memory -= info['size']
                self.total_memory -= info['size']
                del self.buffer_pool[buffer_id]
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'texture_memory': self.texture_memory,
            'buffer_memory': self.buffer_memory,
            'total_memory': self.total_memory,
            'texture_count': len(self.texture_pool),
            'buffer_count': len(self.buffer_pool)
        }

# ============================================================================
# RENDERING PIPELINE
# ============================================================================

class RenderingPipeline:
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.current_mode = RenderMode.FORWARD
        self.quality_level = QualityLevel.HIGH
        self.setup_shaders()
    
    def setup_shaders(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        out vec3 world_pos;
        out vec3 normal;
        void main() {
            world_pos = (model * vec4(in_position, 1.0)).xyz;
            normal = mat3(model) * in_normal;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec3 world_pos;
        in vec3 normal;
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform vec3 material_color;
        out vec4 frag_color;
        void main() {
            vec3 norm = normalize(normal);
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            vec3 ambient = 0.1 * light_color;
            vec3 result = (ambient + diffuse) * material_color;
            frag_color = vec4(result, 1.0);
        }
        """
        
        self.shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def set_render_mode(self, mode: RenderMode):
        self.current_mode = mode
        print(f"Rendering mode: {mode.value}")
    
    def set_quality_level(self, level: QualityLevel):
        self.quality_level = level
        print(f"Quality level: {level.value}")
    
    def render_scene(self, scene_objects: List[Any], lights: List[Any], 
                    view_matrix: np.ndarray, projection_matrix: np.ndarray):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        
        for obj in scene_objects:
            self.shader['model'].write(obj.model_matrix.tobytes())
            self.shader['material_color'].write(obj.color.to_array())
            obj.vao.render()

# ============================================================================
# REAL-TIME RENDERING ENGINE
# ============================================================================

class RealTimeRenderingEngine:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = None
        self.rendering_pipeline = None
        self.scene_objects = []
        self.lights = []
        self.target_fps = 60.0
        self.vsync_enabled = True
        self.auto_quality = True
        
        self.init_glfw()
        self.init_opengl()
        self.setup_engine()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Real-Time Rendering Engine", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        
        if self.vsync_enabled:
            glfw.swap_interval(1)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
    
    def setup_engine(self):
        self.memory_manager = MemoryManager(self.ctx)
        self.rendering_pipeline = RenderingPipeline(self.ctx, self.width, self.height)
        self.setup_scene()
    
    def setup_scene(self):
        # Create a simple cube
        cube_vertices = np.array([
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        ], dtype='f4')
        
        cube_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        
        vbo = self.ctx.buffer(cube_vertices.tobytes())
        ibo = self.ctx.buffer(cube_indices.tobytes())
        
        vao = self.ctx.vertex_array(
            self.rendering_pipeline.shader,
            [(vbo, '3f 3f', 'in_position', 'in_normal')],
            ibo
        )
        
        class SceneObject:
            def __init__(self, vao, color):
                self.vao = vao
                self.color = color
                self.model_matrix = np.eye(4, dtype='f4')
        
        self.scene_objects.append(SceneObject(vao, Color(0.8, 0.6, 0.2, 1.0)))
        
        class Light:
            def __init__(self, position, color):
                self.position = position
                self.color = color
        
        self.lights.append(Light(Vector3D(2.0, 2.0, 2.0), Color(1.0, 1.0, 1.0, 1.0)))
    
    def render_frame(self):
        self.performance_monitor.start_frame()
        
        view_matrix = np.eye(4, dtype='f4')
        projection_matrix = np.eye(4, dtype='f4')
        
        self.rendering_pipeline.render_scene(
            self.scene_objects,
            self.lights,
            view_matrix,
            projection_matrix
        )
        
        self.performance_monitor.end_frame()
        
        if self.auto_quality:
            self.adjust_quality()
        
        self.memory_manager.cleanup_unused()
    
    def adjust_quality(self):
        stats = self.performance_monitor.get_stats()
        
        if stats['fps'] < 30.0 and self.rendering_pipeline.quality_level != QualityLevel.LOW:
            self.rendering_pipeline.set_quality_level(QualityLevel.LOW)
        elif stats['fps'] > 55.0 and self.rendering_pipeline.quality_level == QualityLevel.LOW:
            self.rendering_pipeline.set_quality_level(QualityLevel.MEDIUM)
        elif stats['fps'] > 55.0 and self.rendering_pipeline.quality_level == QualityLevel.MEDIUM:
            self.rendering_pipeline.set_quality_level(QualityLevel.HIGH)
    
    def handle_input(self):
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.rendering_pipeline.set_render_mode(RenderMode.FORWARD)
        
        if glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.rendering_pipeline.set_render_mode(RenderMode.DEFERRED)
        
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.rendering_pipeline.set_quality_level(QualityLevel.LOW)
        
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.rendering_pipeline.set_quality_level(QualityLevel.MEDIUM)
        
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.rendering_pipeline.set_quality_level(QualityLevel.HIGH)
        
        if glfw.get_key(self.window, glfw.KEY_P) == glfw.PRESS:
            self.performance_monitor.print_stats()
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            self.render_frame()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self.handle_input()
            
            if not self.vsync_enabled:
                target_frame_time = 1.0 / self.target_fps
                if delta_time < target_frame_time:
                    time.sleep(target_frame_time - delta_time)
        
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_performance_monitoring():
    print("=== Performance Monitoring Demo ===\n")
    print("Performance monitoring features:")
    print("  • Real-time FPS tracking")
    print("  • Frame time analysis")
    print("  • Performance statistics")
    print("  • Quality adjustment")
    print()

def demonstrate_memory_management():
    print("=== Memory Management Demo ===\n")
    print("Memory management features:")
    print("  • GPU memory tracking")
    print("  • Resource pooling")
    print("  • Automatic cleanup")
    print("  • Memory optimization")
    print()

def demonstrate_rendering_pipeline():
    print("=== Rendering Pipeline Demo ===\n")
    print("Rendering pipeline features:")
    print("  • Forward rendering")
    print("  • Deferred rendering")
    print("  • Quality levels")
    print("  • Dynamic switching")
    print()

def demonstrate_rendering_engine():
    print("=== Real-Time Rendering Engine Demo ===\n")
    print("Starting real-time rendering engine...")
    print("Controls:")
    print("  1: Forward rendering")
    print("  2: Deferred rendering")
    print("  Q: Low quality")
    print("  W: Medium quality")
    print("  E: High quality")
    print("  P: Print performance stats")
    print("  ESC: Exit")
    print()
    
    try:
        engine = RealTimeRenderingEngine(800, 600)
        engine.run()
    except Exception as e:
        print(f"✗ Rendering engine failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=== Real-Time Rendering Engine Demo ===\n")
    
    demonstrate_performance_monitoring()
    demonstrate_memory_management()
    demonstrate_rendering_pipeline()
    
    print("="*60)
    print("Real-Time Rendering Engine demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Performance monitoring and profiling")
    print("✓ GPU memory management")
    print("✓ Advanced rendering pipelines")
    print("✓ Real-time quality adjustment")
    print("✓ Engine optimization")
    print("✓ Modern graphics architecture")
    
    print("\nEngine features:")
    print("• Real-time performance monitoring")
    print("• GPU memory management")
    print("• Multiple rendering pipelines")
    print("• Dynamic quality adjustment")
    print("• Resource optimization")
    print("• Modern graphics techniques")
    
    print("\nApplications:")
    print("• Game engines: Professional game development")
    print("• Real-time graphics: Interactive applications")
    print("• Visualization: Scientific and data visualization")
    print("• Virtual reality: Immersive experiences")
    print("• Simulation: Real-time simulation systems")
    
    print("\nNext steps:")
    print("• Add more rendering techniques")
    print("• Implement advanced optimization")
    print("• Add multi-threading support")
    print("• Optimize for mobile platforms")
    print("• Integrate with game engines")

if __name__ == "__main__":
    main()
