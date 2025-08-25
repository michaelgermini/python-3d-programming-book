#!/usr/bin/env python3
"""
Chapter 14: Computer Graphics Research and Development
Volume Rendering Research System

Demonstrates advanced volume rendering techniques including ray marching,
transfer functions, and medical imaging applications.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class TransferFunction:
    """Transfer function for volume rendering"""
    opacity_scale: float = 1.0
    color_scale: float = 1.0
    density_threshold: float = 0.1
    gradient_threshold: float = 0.05
    
    def get_opacity(self, density: float, gradient_magnitude: float) -> float:
        """Calculate opacity based on density and gradient"""
        if density < self.density_threshold:
            return 0.0
        
        opacity = density * self.opacity_scale
        if gradient_magnitude > self.gradient_threshold:
            opacity *= 1.5  # Enhance edges
        
        return min(opacity, 1.0)
    
    def get_color(self, density: float, gradient_magnitude: float) -> Tuple[float, float, float]:
        """Calculate color based on density and gradient"""
        if density < self.density_threshold:
            return (0.0, 0.0, 0.0)
        
        # Color mapping based on density
        if density < 0.3:
            color = (0.2, 0.8, 0.2)  # Green for low density
        elif density < 0.6:
            color = (0.8, 0.8, 0.2)  # Yellow for medium density
        else:
            color = (0.8, 0.2, 0.2)  # Red for high density
        
        # Enhance with gradient
        if gradient_magnitude > self.gradient_threshold:
            color = tuple(c * 1.3 for c in color)
        
        return tuple(min(c * self.color_scale, 1.0) for c in color)

class VolumeData:
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth
        self.data = np.zeros((width, height, depth), dtype=np.float32)
        self.gradient_data = np.zeros((width, height, depth, 3), dtype=np.float32)
        
    def generate_test_data(self):
        """Generate test volume data"""
        # Create a sphere
        center_x, center_y, center_z = self.width // 2, self.height // 2, self.depth // 2
        radius = min(self.width, self.height, self.depth) // 3
        
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                    if distance < radius:
                        # Create a sphere with varying density
                        density = 1.0 - (distance / radius)
                        self.data[x, y, z] = density
        
        # Add some noise
        noise = np.random.normal(0, 0.1, self.data.shape)
        self.data = np.clip(self.data + noise, 0, 1)
        
        # Calculate gradients
        self.calculate_gradients()
    
    def generate_medical_data(self):
        """Generate medical imaging-like data"""
        # Create multiple structures
        center_x, center_y, center_z = self.width // 2, self.height // 2, self.depth // 2
        
        # Main structure (like a brain)
        radius = min(self.width, self.height, self.depth) // 4
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                    if distance < radius:
                        density = 0.8 * (1.0 - (distance / radius))
                        self.data[x, y, z] = max(self.data[x, y, z], density)
        
        # Add smaller structures (like blood vessels)
        for _ in range(5):
            vessel_x = random.randint(0, self.width)
            vessel_y = random.randint(0, self.height)
            vessel_z = random.randint(0, self.depth)
            vessel_radius = random.randint(2, 8)
            
            for x in range(max(0, vessel_x - vessel_radius), min(self.width, vessel_x + vessel_radius)):
                for y in range(max(0, vessel_y - vessel_radius), min(self.height, vessel_y + vessel_radius)):
                    for z in range(max(0, vessel_z - vessel_radius), min(self.depth, vessel_z + vessel_radius)):
                        distance = math.sqrt((x - vessel_x)**2 + (y - vessel_y)**2 + (z - vessel_z)**2)
                        if distance < vessel_radius:
                            density = 0.6 * (1.0 - (distance / vessel_radius))
                            self.data[x, y, z] = max(self.data[x, y, z], density)
        
        # Calculate gradients
        self.calculate_gradients()
    
    def calculate_gradients(self):
        """Calculate gradient vectors for each voxel"""
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                for z in range(1, self.depth - 1):
                    # Central differences
                    grad_x = (self.data[x + 1, y, z] - self.data[x - 1, y, z]) / 2.0
                    grad_y = (self.data[x, y + 1, z] - self.data[x, y - 1, z]) / 2.0
                    grad_z = (self.data[x, y, z + 1] - self.data[x, y, z - 1]) / 2.0
                    
                    self.gradient_data[x, y, z] = [grad_x, grad_y, grad_z]
    
    def get_density(self, x: float, y: float, z: float) -> float:
        """Get density at continuous coordinates using trilinear interpolation"""
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        z = max(0, min(self.depth - 1, z))
        
        x0, y0, z0 = int(x), int(y), int(z)
        x1, y1, z1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1), min(z0 + 1, self.depth - 1)
        
        # Interpolation weights
        wx = x - x0
        wy = y - y0
        wz = z - z0
        
        # Trilinear interpolation
        c000 = self.data[x0, y0, z0]
        c001 = self.data[x0, y0, z1]
        c010 = self.data[x0, y1, z0]
        c011 = self.data[x0, y1, z1]
        c100 = self.data[x1, y0, z0]
        c101 = self.data[x1, y0, z1]
        c110 = self.data[x1, y1, z0]
        c111 = self.data[x1, y1, z1]
        
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        
        return c0 * (1 - wz) + c1 * wz
    
    def get_gradient(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Get gradient at continuous coordinates"""
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        z = max(0, min(self.depth - 1, z))
        
        x0, y0, z0 = int(x), int(y), int(z)
        x1, y1, z1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1), min(z0 + 1, self.depth - 1)
        
        # Interpolation weights
        wx = x - x0
        wy = y - y0
        wz = z - z0
        
        # Trilinear interpolation for gradient
        g000 = self.gradient_data[x0, y0, z0]
        g001 = self.gradient_data[x0, y0, z1]
        g010 = self.gradient_data[x0, y1, z0]
        g011 = self.gradient_data[x0, y1, z1]
        g100 = self.gradient_data[x1, y0, z0]
        g101 = self.gradient_data[x1, y0, z1]
        g110 = self.gradient_data[x1, y1, z0]
        g111 = self.gradient_data[x1, y1, z1]
        
        g00 = g000 * (1 - wx) + g100 * wx
        g01 = g001 * (1 - wx) + g101 * wx
        g10 = g010 * (1 - wx) + g110 * wx
        g11 = g011 * (1 - wx) + g111 * wx
        
        g0 = g00 * (1 - wy) + g10 * wy
        g1 = g01 * (1 - wy) + g11 * wy
        
        gradient = g0 * (1 - wz) + g1 * wz
        return tuple(gradient)

class RayMarcher:
    def __init__(self, volume_data: VolumeData, transfer_function: TransferFunction):
        self.volume_data = volume_data
        self.transfer_function = transfer_function
        self.step_size = 0.1
        self.max_steps = 1000
    
    def ray_march(self, ray_origin: Tuple[float, float, float], 
                  ray_direction: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """Perform ray marching through volume data"""
        accumulated_color = [0.0, 0.0, 0.0]
        accumulated_opacity = 0.0
        
        # Ray-box intersection
        t_min, t_max = self.ray_box_intersection(ray_origin, ray_direction)
        if t_min >= t_max:
            return (0.0, 0.0, 0.0, 0.0)
        
        # Start marching
        current_pos = list(ray_origin)
        for i in range(3):
            current_pos[i] += ray_direction[i] * t_min
        
        t = t_min
        while t < t_max and accumulated_opacity < 0.95:
            # Sample volume
            density = self.volume_data.get_density(current_pos[0], current_pos[1], current_pos[2])
            gradient = self.volume_data.get_gradient(current_pos[0], current_pos[1], current_pos[2])
            gradient_magnitude = math.sqrt(sum(g * g for g in gradient))
            
            # Apply transfer function
            opacity = self.transfer_function.get_opacity(density, gradient_magnitude)
            color = self.transfer_function.get_color(density, gradient_magnitude)
            
            # Front-to-back compositing
            if opacity > 0:
                alpha = opacity * self.step_size
                for i in range(3):
                    accumulated_color[i] += color[i] * alpha * (1.0 - accumulated_opacity)
                accumulated_opacity += alpha * (1.0 - accumulated_opacity)
            
            # Advance ray
            for i in range(3):
                current_pos[i] += ray_direction[i] * self.step_size
            t += self.step_size
        
        return tuple(accumulated_color) + (accumulated_opacity,)
    
    def ray_box_intersection(self, ray_origin: Tuple[float, float, float], 
                           ray_direction: Tuple[float, float, float]) -> Tuple[float, float]:
        """Calculate ray-box intersection"""
        box_min = [0, 0, 0]
        box_max = [self.volume_data.width - 1, self.volume_data.height - 1, self.volume_data.depth - 1]
        
        t_min = 0.0
        t_max = float('inf')
        
        for i in range(3):
            if abs(ray_direction[i]) < 1e-6:
                if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                    return (1.0, 0.0)
            else:
                t1 = (box_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (box_max[i] - ray_origin[i]) / ray_direction[i]
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        
        if t_min > t_max:
            return (1.0, 0.0)
        
        return (t_min, t_max)

class VolumeRenderer:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Volume data
        self.volume_data = VolumeData(64, 64, 64)
        self.transfer_function = TransferFunction()
        self.ray_marcher = RayMarcher(self.volume_data, self.transfer_function)
        
        # Camera
        self.camera_distance = 100.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        
        # Rendering
        self.render_target = None
        self.framebuffer = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.render_times = []
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.setup_volume_data()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Volume Rendering Research", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.setup_shaders()
        self.create_geometry()
        self.setup_framebuffer()
    
    def setup_shaders(self):
        vertex_shader = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 texcoord;
        void main() {
            texcoord = in_texcoord;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec2 texcoord;
        out vec4 out_color;
        uniform sampler2D volume_texture;
        uniform vec3 camera_position;
        uniform vec3 camera_direction;
        uniform vec3 camera_up;
        uniform vec3 camera_right;
        uniform float camera_fov;
        uniform float aspect_ratio;
        
        void main() {
            // Calculate ray direction
            vec2 screen_pos = texcoord * 2.0 - 1.0;
            vec3 ray_direction = normalize(camera_direction + 
                                         screen_pos.x * camera_right * camera_fov * aspect_ratio +
                                         screen_pos.y * camera_up * camera_fov);
            
            // Simple volume rendering (placeholder)
            float density = texture(volume_texture, texcoord).r;
            vec3 color = vec3(density, density, density);
            out_color = vec4(color, 1.0);
        }
        """
        
        self.shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def create_geometry(self):
        # Full-screen quad
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
        ], dtype='f4')
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.shader,
            [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')],
            self.ibo
        )
    
    def setup_framebuffer(self):
        self.render_target = self.ctx.texture((self.width, self.height), 4)
        self.framebuffer = self.ctx.framebuffer(self.render_target)
    
    def setup_volume_data(self):
        # Generate test data
        self.volume_data.generate_medical_data()
        
        # Create 3D texture from volume data
        volume_texture_data = self.volume_data.data.astype(np.float32)
        self.volume_texture = self.ctx.texture3d(
            (self.volume_data.width, self.volume_data.height, self.volume_data.depth),
            1,
            volume_texture_data.tobytes()
        )
    
    def render_volume(self):
        """Render volume using ray marching"""
        start_time = time.time()
        
        # Clear framebuffer
        self.framebuffer.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Calculate camera parameters
        camera_pos = (
            self.camera_distance * math.cos(self.camera_rotation_y) * math.cos(self.camera_rotation_x),
            self.camera_distance * math.sin(self.camera_rotation_x),
            self.camera_distance * math.sin(self.camera_rotation_y) * math.cos(self.camera_rotation_x)
        )
        
        # Render each pixel using ray marching
        image_data = np.zeros((self.height, self.width, 4), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                # Calculate ray direction
                screen_x = (x / self.width) * 2.0 - 1.0
                screen_y = (y / self.height) * 2.0 - 1.0
                
                # Simple ray direction calculation
                ray_direction = (
                    screen_x * 0.5,
                    screen_y * 0.5,
                    1.0
                )
                
                # Normalize ray direction
                length = math.sqrt(sum(d * d for d in ray_direction))
                ray_direction = tuple(d / length for d in ray_direction)
                
                # Ray march
                color = self.ray_marcher.ray_march(camera_pos, ray_direction)
                image_data[y, x] = color
        
        # Update texture
        self.render_target.write(image_data.tobytes())
        
        render_time = time.time() - start_time
        self.render_times.append(render_time)
    
    def render_scene(self):
        # Render volume
        self.render_volume()
        
        # Render to screen
        self.ctx.screen.use()
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        self.volume_texture.use(0)
        self.shader['volume_texture'] = 0
        
        self.vao.render()
    
    def handle_input(self):
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            # Generate test data
            self.volume_data.generate_test_data()
            self.setup_volume_data()
        
        if glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            # Generate medical data
            self.volume_data.generate_medical_data()
            self.setup_volume_data()
        
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.transfer_function.opacity_scale *= 1.1
        
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.transfer_function.opacity_scale *= 0.9
        
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.transfer_function.density_threshold *= 0.9
        
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.transfer_function.density_threshold *= 1.1
    
    def mouse_callback(self, window, xpos, ypos):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.camera_rotation_y += xpos * 0.01
            self.camera_rotation_x += ypos * 0.01
    
    def scroll_callback(self, window, xoffset, yoffset):
        self.camera_distance += yoffset * 5.0
        self.camera_distance = max(10.0, min(200.0, self.camera_distance))
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self.setup_framebuffer()
    
    def run(self):
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            self.handle_input()
            self.render_scene()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Print statistics
            self.frame_count += 1
            if current_time - self.last_fps_time >= 2.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                avg_render_time = np.mean(self.render_times[-30:]) if self.render_times else 0
                print(f"FPS: {fps:.1f}, Avg Render Time: {avg_render_time*1000:.2f}ms, "
                      f"Volume Size: {self.volume_data.width}x{self.volume_data.height}x{self.volume_data.depth}")
                self.frame_count = 0
                self.last_fps_time = current_time
        
        glfw.terminate()

def main():
    print("=== Volume Rendering Research System ===\n")
    print("Volume rendering features:")
    print("  • Ray marching algorithm")
    print("  • Transfer functions")
    print("  • Medical imaging simulation")
    print("  • Real-time volume visualization")
    print("  • Performance optimization research")
    
    print("\nControls:")
    print("• Mouse: Rotate camera")
    print("• Scroll: Zoom in/out")
    print("• 1: Generate test sphere data")
    print("• 2: Generate medical-like data")
    print("• Arrow keys: Adjust transfer function")
    
    print("\nResearch applications:")
    print("• Medical imaging and diagnosis")
    print("• Scientific visualization")
    print("• Volume data analysis")
    print("• Rendering algorithm research")
    print("• Performance benchmarking")
    
    try:
        volume_renderer = VolumeRenderer(800, 600)
        volume_renderer.run()
    except Exception as e:
        print(f"✗ Volume rendering system failed to start: {e}")

if __name__ == "__main__":
    main()
