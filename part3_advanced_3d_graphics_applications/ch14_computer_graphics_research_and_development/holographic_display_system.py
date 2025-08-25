#!/usr/bin/env python3
"""
Chapter 14: Computer Graphics Research and Development
Holographic Display System

Demonstrates holographic display techniques including light field rendering,
interference patterns, and 3D visualization without glasses.
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
class LightField:
    """Represents a light field for holographic display"""
    width: int
    height: int
    num_views: int
    data: np.ndarray  # Shape: (height, width, num_views, 3) for RGB
    
    def __init__(self, width: int, height: int, num_views: int):
        self.width = width
        self.height = height
        self.num_views = num_views
        self.data = np.zeros((height, width, num_views, 3), dtype=np.float32)
    
    def set_pixel(self, x: int, y: int, view: int, color: Tuple[float, float, float]):
        """Set a pixel in the light field"""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= view < self.num_views:
            self.data[y, x, view] = color
    
    def get_pixel(self, x: int, y: int, view: int) -> Tuple[float, float, float]:
        """Get a pixel from the light field"""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= view < self.num_views:
            return tuple(self.data[y, x, view])
        return (0.0, 0.0, 0.0)

class HolographicPattern:
    """Generates holographic interference patterns"""
    
    def __init__(self, width: int, height: int, wavelength: float = 633e-9):
        self.width = width
        self.height = height
        self.wavelength = wavelength  # Wavelength in meters (red laser)
        self.pixel_pitch = 8e-6  # Pixel pitch in meters (8 microns)
    
    def generate_point_hologram(self, point_3d: Tuple[float, float, float], 
                               reference_distance: float = 0.1) -> np.ndarray:
        """Generate holographic pattern for a single 3D point"""
        x, y, z = point_3d
        
        # Convert to meters
        x_m = x * 0.001  # mm to m
        y_m = y * 0.001
        z_m = z * 0.001
        
        pattern = np.zeros((self.height, self.width), dtype=np.float32)
        
        for i in range(self.height):
            for j in range(self.width):
                # Calculate distance from hologram pixel to 3D point
                pixel_x = (j - self.width // 2) * self.pixel_pitch
                pixel_y = (i - self.height // 2) * self.pixel_pitch
                
                distance = math.sqrt((pixel_x - x_m)**2 + (pixel_y - y_m)**2 + z_m**2)
                
                # Calculate phase difference
                phase = 2 * math.pi * distance / self.wavelength
                
                # Generate interference pattern
                pattern[i, j] = math.cos(phase)
        
        return pattern
    
    def generate_object_hologram(self, points_3d: List[Tuple[float, float, float]], 
                                intensities: List[float]) -> np.ndarray:
        """Generate holographic pattern for multiple 3D points"""
        pattern = np.zeros((self.height, self.width), dtype=np.float32)
        
        for point, intensity in zip(points_3d, intensities):
            point_pattern = self.generate_point_hologram(point)
            pattern += point_pattern * intensity
        
        # Normalize
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        return pattern

class LenticularDisplay:
    """Simulates lenticular lens-based 3D display"""
    
    def __init__(self, width: int, height: int, num_views: int, lens_pitch: float = 0.5):
        self.width = width
        self.height = height
        self.num_views = num_views
        self.lens_pitch = lens_pitch  # Lens pitch in pixels
        self.light_field = LightField(width, height, num_views)
    
    def render_3d_object(self, object_3d: List[Tuple[float, float, float]], 
                        colors: List[Tuple[float, float, float]]):
        """Render a 3D object into the light field"""
        for view in range(self.num_views):
            # Calculate view angle
            view_angle = (view - self.num_views // 2) * 5.0  # 5 degrees between views
            view_rad = math.radians(view_angle)
            
            # Render object from this viewpoint
            for i, (point_3d, color) in enumerate(zip(object_3d, colors)):
                x, y, z = point_3d
                
                # Apply perspective projection
                if z > 0:
                    screen_x = int((x / z) * 100 + self.width // 2)
                    screen_y = int((y / z) * 100 + self.height // 2)
                    
                    # Apply view angle offset
                    screen_x += int(math.tan(view_rad) * z * 10)
                    
                    if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                        self.light_field.set_pixel(screen_x, screen_y, view, color)
    
    def generate_lenticular_image(self) -> np.ndarray:
        """Generate the final lenticular image"""
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                # Calculate which lens this pixel belongs to
                lens_index = int(x / self.lens_pitch)
                view_index = lens_index % self.num_views
                
                # Get color from light field
                color = self.light_field.get_pixel(x, y, view_index)
                image[y, x] = color
        
        return image

class HolographicDisplay:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Holographic components
        self.holographic_pattern = HolographicPattern(256, 256)
        self.lenticular_display = LenticularDisplay(width, height, 8)
        
        # 3D objects
        self.objects_3d = []
        self.object_colors = []
        
        # Display mode
        self.display_mode = "lenticular"  # "lenticular" or "holographic"
        
        # Camera
        self.camera_distance = 100.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.render_times = []
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.create_test_objects()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Holographic Display System", None, None)
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
        uniform sampler2D display_texture;
        uniform int display_mode;
        uniform float time;
        
        void main() {
            vec4 color = texture(display_texture, texcoord);
            
            if (display_mode == 1) {
                // Holographic mode - add interference patterns
                float interference = sin(texcoord.x * 100.0 + time) * 0.1;
                color.rgb += interference;
            }
            
            out_color = color;
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
    
    def create_test_objects(self):
        """Create test 3D objects"""
        # Create a cube
        cube_points = []
        cube_colors = []
        
        for x in [-20, 20]:
            for y in [-20, 20]:
                for z in [-20, 20]:
                    cube_points.append((x, y, z))
                    cube_colors.append((0.8, 0.2, 0.2))
        
        # Create a sphere
        sphere_points = []
        sphere_colors = []
        
        for i in range(50):
            phi = 2 * math.pi * i / 50
            for j in range(25):
                theta = math.pi * j / 25
                x = 40 * math.sin(theta) * math.cos(phi)
                y = 40 * math.sin(theta) * math.sin(phi)
                z = 40 * math.cos(theta)
                sphere_points.append((x, y, z))
                sphere_colors.append((0.2, 0.8, 0.2))
        
        self.objects_3d = cube_points + sphere_points
        self.object_colors = cube_colors + sphere_colors
    
    def generate_holographic_pattern(self) -> np.ndarray:
        """Generate holographic interference pattern"""
        # Create points for holographic pattern
        points = []
        intensities = []
        
        # Add some test points
        for i in range(10):
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            z = random.uniform(20, 100)
            points.append((x, y, z))
            intensities.append(random.uniform(0.5, 1.0))
        
        # Generate holographic pattern
        pattern = self.holographic_pattern.generate_object_hologram(points, intensities)
        
        # Convert to RGB texture
        pattern_rgb = np.zeros((pattern.shape[0], pattern.shape[1], 3), dtype=np.float32)
        pattern_rgb[:, :, 0] = pattern  # Red channel
        pattern_rgb[:, :, 1] = pattern  # Green channel
        pattern_rgb[:, :, 2] = pattern  # Blue channel
        
        return pattern_rgb
    
    def generate_lenticular_image(self) -> np.ndarray:
        """Generate lenticular 3D image"""
        # Clear light field
        self.lenticular_display.light_field.data.fill(0)
        
        # Render 3D objects
        self.lenticular_display.render_3d_object(self.objects_3d, self.object_colors)
        
        # Generate final image
        return self.lenticular_display.generate_lenticular_image()
    
    def render_display(self):
        """Render the appropriate display mode"""
        start_time = time.time()
        
        if self.display_mode == "holographic":
            # Generate holographic pattern
            pattern = self.generate_holographic_pattern()
            
            # Create texture
            if hasattr(self, 'holographic_texture'):
                self.holographic_texture.release()
            
            self.holographic_texture = self.ctx.texture(
                (pattern.shape[1], pattern.shape[0]), 3, pattern.tobytes()
            )
            self.display_texture = self.holographic_texture
        
        else:  # lenticular
            # Generate lenticular image
            image = self.generate_lenticular_image()
            
            # Create texture
            if hasattr(self, 'lenticular_texture'):
                self.lenticular_texture.release()
            
            self.lenticular_texture = self.ctx.texture(
                (image.shape[1], image.shape[0]), 3, image.tobytes()
            )
            self.display_texture = self.lenticular_texture
        
        render_time = time.time() - start_time
        self.render_times.append(render_time)
    
    def render_scene(self):
        # Render display
        self.render_display()
        
        # Render to screen
        self.ctx.screen.use()
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        self.display_texture.use(0)
        self.shader['display_texture'] = 0
        self.shader['display_mode'] = 1 if self.display_mode == "holographic" else 0
        self.shader['time'] = time.time()
        
        self.vao.render()
    
    def handle_input(self):
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            # Switch to lenticular mode
            self.display_mode = "lenticular"
            print("Switched to Lenticular Display Mode")
        
        if glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            # Switch to holographic mode
            self.display_mode = "holographic"
            print("Switched to Holographic Display Mode")
        
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            # Add new 3D object
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            z = random.uniform(20, 100)
            color = (random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0))
            
            self.objects_3d.append((x, y, z))
            self.object_colors.append(color)
            print(f"Added 3D object at ({x:.1f}, {y:.1f}, {z:.1f})")
    
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
                print(f"FPS: {fps:.1f}, Mode: {self.display_mode}, "
                      f"Objects: {len(self.objects_3d)}, "
                      f"Avg Render Time: {avg_render_time*1000:.2f}ms")
                self.frame_count = 0
                self.last_fps_time = current_time
        
        glfw.terminate()

def main():
    print("=== Holographic Display System ===\n")
    print("Holographic display features:")
    print("  • Lenticular lens simulation")
    print("  • Holographic interference patterns")
    print("  • Light field rendering")
    print("  • 3D visualization without glasses")
    print("  • Real-time display switching")
    
    print("\nControls:")
    print("• 1: Switch to Lenticular Display Mode")
    print("• 2: Switch to Holographic Display Mode")
    print("• SPACE: Add new 3D object")
    print("• Mouse: Rotate camera view")
    print("• Scroll: Zoom in/out")
    
    print("\nResearch applications:")
    print("• 3D display technology research")
    print("• Holographic projection systems")
    print("• Light field camera simulation")
    print("• Medical imaging displays")
    print("• Entertainment and gaming")
    print("• Scientific visualization")
    
    try:
        holographic_display = HolographicDisplay(800, 600)
        holographic_display.run()
    except Exception as e:
        print(f"✗ Holographic display system failed to start: {e}")

if __name__ == "__main__":
    main()
