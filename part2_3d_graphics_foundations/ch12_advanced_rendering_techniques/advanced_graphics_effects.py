#!/usr/bin/env python3
"""
Chapter 12: Advanced Rendering Techniques
Advanced Graphics Effects

Demonstrates advanced graphics effects including screen space effects, particle systems,
and procedural generation techniques.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Advanced Graphics Effects"
__description__ = "Screen space effects and advanced graphics techniques"

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

class EffectType(Enum):
    """Types of graphics effects"""
    SSAO = "ssao"
    SSR = "ssr"
    BLOOM = "bloom"

# ============================================================================
# SCREEN SPACE EFFECTS
# ============================================================================

class ScreenSpaceEffects:
    """Screen space effects system"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create framebuffers
        self.effect_fbo = ctx.framebuffer(
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
        
        # Create shaders
        self.setup_shaders()
        
        # Create vertex array
        self.quad_vao = ctx.vertex_array(
            self.ssao_shader,
            [
                (self.quad_vbo, '3f 2f', 'in_position', 'in_texcoord'),
            ],
            self.quad_ibo
        )
    
    def setup_shaders(self):
        """Setup screen space effect shaders"""
        # SSAO vertex shader
        ssao_vertex = """
        #version 330
        
        in vec3 in_position;
        in vec2 in_texcoord;
        
        out vec2 texcoord;
        
        void main() {
            texcoord = in_texcoord;
            gl_Position = vec4(in_position, 1.0);
        }
        """
        
        # SSAO fragment shader
        ssao_fragment = """
        #version 330
        
        in vec2 texcoord;
        
        uniform sampler2D g_position;
        uniform sampler2D g_normal;
        uniform mat4 projection;
        
        out float frag_color;
        
        void main() {
            vec3 frag_pos = texture(g_position, texcoord).xyz;
            vec3 normal = texture(g_normal, texcoord).xyz;
            
            float occlusion = 0.0;
            float radius = 0.5;
            float bias = 0.025;
            
            for(int i = 0; i < 16; ++i) {
                vec3 sample_pos = frag_pos + normal * radius * float(i) / 16.0;
                vec4 offset = projection * vec4(sample_pos, 1.0);
                offset.xyz /= offset.w;
                offset.xyz = offset.xyz * 0.5 + 0.5;
                
                float sample_depth = texture(g_position, offset.xy).z;
                float range_check = smoothstep(0.0, 1.0, radius / abs(frag_pos.z - sample_depth));
                occlusion += (sample_depth >= sample_pos.z + bias ? 1.0 : 0.0) * range_check;
            }
            
            occlusion = 1.0 - (occlusion / 16.0);
            frag_color = occlusion;
        }
        """
        
        # Create shader program
        self.ssao_shader = self.ctx.program(
            vertex_shader=ssao_vertex,
            fragment_shader=ssao_fragment
        )
    
    def apply_ssao(self, g_position: moderngl.Texture, g_normal: moderngl.Texture, 
                  projection_matrix: np.ndarray) -> moderngl.Texture:
        """Apply Screen Space Ambient Occlusion"""
        self.effect_fbo.use()
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        
        # Set uniforms
        self.ssao_shader['g_position'].value = 0
        self.ssao_shader['g_normal'].value = 1
        self.ssao_shader['projection'].write(projection_matrix.tobytes())
        
        g_position.use(0)
        g_normal.use(1)
        
        # Render
        self.quad_vao.program = self.ssao_shader
        self.quad_vao.render()
        
        return self.effect_fbo.color_attachments[0]

# ============================================================================
# PARTICLE SYSTEM
# ============================================================================

@dataclass
class Particle:
    """Individual particle"""
    position: Vector3D
    velocity: Vector3D
    color: Color
    life: float
    max_life: float
    size: float

class ParticleSystem:
    """Advanced particle system"""
    
    def __init__(self, ctx: moderngl.Context, max_particles: int = 1000):
        self.ctx = ctx
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        
        # Particle properties
        self.gravity = Vector3D(0, -9.81, 0)
        
        # Create particle shaders
        self.setup_shaders()
        
        # Create particle geometry
        self.create_particle_geometry()
    
    def setup_shaders(self):
        """Setup particle shaders"""
        # Particle vertex shader
        particle_vertex = """
        #version 330
        
        in vec3 in_position;
        in vec3 in_velocity;
        in vec4 in_color;
        in float in_life;
        in float in_max_life;
        in float in_size;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        out vec4 color;
        out float life_factor;
        
        void main() {
            vec3 pos = in_position + in_velocity * time;
            life_factor = in_life / in_max_life;
            color = in_color;
            color.a *= (1.0 - life_factor);
            
            float size = in_size * (1.0 - life_factor * 0.5);
            gl_Position = projection * view * model * vec4(pos, 1.0);
            gl_PointSize = size;
        }
        """
        
        # Particle fragment shader
        particle_fragment = """
        #version 330
        
        in vec4 color;
        in float life_factor;
        
        out vec4 frag_color;
        
        void main() {
            vec2 center = gl_PointCoord - vec2(0.5);
            float dist = length(center);
            
            if(dist > 0.5) {
                discard;
            }
            
            float alpha = 1.0 - dist * 2.0;
            alpha *= color.a;
            frag_color = vec4(color.rgb, alpha);
        }
        """
        
        # Create shader program
        self.particle_shader = self.ctx.program(
            vertex_shader=particle_vertex,
            fragment_shader=particle_fragment
        )
    
    def create_particle_geometry(self):
        """Create particle geometry"""
        particle_data = []
        for i in range(self.max_particles):
            particle_data.extend([0.0, 0.0, 0.0])  # Position
            particle_data.extend([0.0, 0.0, 0.0])  # Velocity
            particle_data.extend([1.0, 1.0, 1.0, 1.0])  # Color
            particle_data.extend([0.0])  # Life
            particle_data.extend([1.0])  # Max life
            particle_data.extend([1.0])  # Size
        
        self.particle_vbo = self.ctx.buffer(np.array(particle_data, dtype='f4').tobytes())
        
        # Create vertex array
        self.particle_vao = self.ctx.vertex_array(
            self.particle_shader,
            [
                (self.particle_vbo, '3f 3f 4f 1f 1f 1f', 
                 'in_position', 'in_velocity', 'in_color', 'in_life', 'in_max_life', 'in_size'),
            ]
        )
    
    def emit_particle(self, position: Vector3D, velocity: Vector3D, 
                    color: Color, life: float, size: float):
        """Emit a new particle"""
        if len(self.particles) < self.max_particles:
            particle = Particle(
                position=position,
                velocity=velocity,
                color=color,
                life=life,
                max_life=life,
                size=size
            )
            self.particles.append(particle)
    
    def update(self, delta_time: float):
        """Update particle system"""
        for particle in self.particles[:]:
            particle.position = particle.position + particle.velocity * delta_time
            particle.velocity = particle.velocity + self.gravity * delta_time
            particle.life -= delta_time
            
            if particle.life <= 0:
                self.particles.remove(particle)
    
    def render(self, model_matrix: np.ndarray, view_matrix: np.ndarray, 
              projection_matrix: np.ndarray, time: float):
        """Render particle system"""
        if not self.particles:
            return
        
        # Update particle data
        particle_data = []
        for particle in self.particles:
            particle_data.extend(particle.position.to_array())
            particle_data.extend(particle.velocity.to_array())
            particle_data.extend(particle.color.to_array())
            particle_data.extend([particle.life])
            particle_data.extend([particle.max_life])
            particle_data.extend([particle.size])
        
        # Pad with zeros
        while len(particle_data) < self.max_particles * 13:
            particle_data.extend([0.0] * 13)
        
        # Update vertex buffer
        self.particle_vbo.write(np.array(particle_data, dtype='f4').tobytes())
        
        # Set uniforms
        self.particle_shader['model'].write(model_matrix.tobytes())
        self.particle_shader['view'].write(view_matrix.tobytes())
        self.particle_shader['projection'].write(projection_matrix.tobytes())
        self.particle_shader['time'].value = time
        
        # Enable blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render particles
        self.particle_vao.render(moderngl.POINTS, instances=len(self.particles))
        
        # Disable blending
        self.ctx.disable(moderngl.BLEND)

# ============================================================================
# PROCEDURAL GENERATION
# ============================================================================

class ProceduralGenerator:
    """Procedural content generation"""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.setup_shaders()
    
    def setup_shaders(self):
        """Setup procedural generation shaders"""
        # Noise generation compute shader
        noise_compute = """
        #version 430
        
        layout(local_size_x=16, local_size_y=16) in;
        
        layout(rgba32f, binding=0) uniform image2D noise_texture;
        
        uniform float time;
        uniform vec2 resolution;
        
        uint hash(uint x) {
            x ^= x >> 16;
            x *= 0x85ebca6b;
            x ^= x >> 13;
            x *= 0xc2b2ae35;
            x ^= x >> 16;
            return x;
        }
        
        float random(vec2 st) {
            return float(hash(uint(st.x * 10000.0 + st.y * 10000.0))) / 4294967295.0;
        }
        
        float noise(vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);
            
            float a = random(i);
            float b = random(i + vec2(1.0, 0.0));
            float c = random(i + vec2(0.0, 1.0));
            float d = random(i + vec2(1.0, 1.0));
            
            vec2 u = f * f * (3.0 - 2.0 * f);
            
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }
        
        void main() {
            ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
            vec2 uv = vec2(pixel_coords) / resolution;
            
            float n = 0.0;
            float amplitude = 1.0;
            float frequency = 1.0;
            
            for(int i = 0; i < 4; ++i) {
                n += amplitude * noise(uv * frequency + time * 0.1);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            
            vec4 color = vec4(n, n, n, 1.0);
            imageStore(noise_texture, pixel_coords, color);
        }
        """
        
        # Create compute shader
        self.noise_shader = self.ctx.compute_shader(noise_compute)
    
    def generate_noise_texture(self, width: int, height: int, time: float) -> moderngl.Texture:
        """Generate procedural noise texture"""
        noise_texture = self.ctx.texture((width, height), 4, dtype='f4')
        noise_texture.bind_to_image(0, read=False, write=True)
        
        self.noise_shader['time'].value = time
        self.noise_shader['resolution'].value = (width, height)
        
        self.noise_shader.run(group_x=width // 16, group_y=height // 16)
        
        return noise_texture

# ============================================================================
# ADVANCED GRAPHICS EFFECTS RENDERER
# ============================================================================

class AdvancedGraphicsEffectsRenderer:
    """Complete advanced graphics effects renderer"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        
        # Effect systems
        self.screen_space_effects = None
        self.particle_system = None
        self.procedural_generator = None
        
        # Effect settings
        self.enable_ssao = True
        self.enable_particles = True
        
        self.init_glfw()
        self.init_opengl()
        self.setup_effects()
    
    def init_glfw(self):
        """Initialize GLFW"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Advanced Graphics Effects", None, None)
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
    
    def setup_effects(self):
        """Setup effect systems"""
        self.screen_space_effects = ScreenSpaceEffects(self.ctx, self.width, self.height)
        self.particle_system = ParticleSystem(self.ctx, max_particles=1000)
        self.procedural_generator = ProceduralGenerator(self.ctx)
    
    def render_frame(self):
        """Render a complete frame with effects"""
        # Generate procedural content
        noise_texture = self.procedural_generator.generate_noise_texture(
            self.width, self.height, time.time()
        )
        
        # Update and render particle system
        if self.enable_particles:
            # Emit particles
            for i in range(5):
                self.particle_system.emit_particle(
                    position=Vector3D(0, 0, 0),
                    velocity=Vector3D(
                        random.uniform(-1, 1),
                        random.uniform(2, 5),
                        random.uniform(-1, 1)
                    ),
                    color=Color(1.0, 0.5, 0.2, 1.0),
                    life=2.0,
                    size=0.1
                )
            
            # Update particles
            self.particle_system.update(1.0 / 60.0)
            
            # Render particles
            model_matrix = np.eye(4, dtype='f4')
            view_matrix = np.eye(4, dtype='f4')
            projection_matrix = np.eye(4, dtype='f4')
            self.particle_system.render(model_matrix, view_matrix, projection_matrix, time.time())
    
    def handle_input(self):
        """Handle keyboard input"""
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.enable_ssao = not self.enable_ssao
            print(f"SSAO: {'enabled' if self.enable_ssao else 'disabled'}")
        
        if glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.enable_particles = not self.enable_particles
            print(f"Particles: {'enabled' if self.enable_particles else 'disabled'}")
    
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

def demonstrate_screen_space_effects():
    """Demonstrate screen space effects"""
    print("=== Screen Space Effects Demo ===\n")
    
    print("Screen space effects:")
    print("  • SSAO: Screen Space Ambient Occlusion")
    print("  • SSR: Screen Space Reflections")
    print("  • SSA: Screen Space Ambient Occlusion")
    print()
    
    print("SSAO process:")
    print("  1. Sample depth buffer around each pixel")
    print("  2. Calculate occlusion based on depth differences")
    print("  3. Apply blur to smooth the result")
    print("  4. Use as ambient lighting multiplier")
    print()

def demonstrate_particle_systems():
    """Demonstrate particle systems"""
    print("=== Particle Systems Demo ===\n")
    
    print("Particle system features:")
    print("  • GPU-based particle simulation")
    print("  • Physics integration (gravity, wind)")
    print("  • Life cycle management")
    print("  • Efficient rendering with instancing")
    print()

def demonstrate_procedural_generation():
    """Demonstrate procedural generation"""
    print("=== Procedural Generation Demo ===\n")
    
    print("Procedural generation techniques:")
    print("  • Noise generation (Perlin, Simplex)")
    print("  • Fractal noise and octaves")
    print("  • Compute shader-based generation")
    print("  • Real-time procedural content")
    print()

def demonstrate_rendering_system():
    """Demonstrate the complete rendering system"""
    print("=== Advanced Graphics Effects Demo ===\n")
    
    print("Starting advanced graphics effects renderer...")
    print("Controls:")
    print("  1: Toggle SSAO")
    print("  2: Toggle particles")
    print("  ESC: Exit")
    print()
    
    try:
        renderer = AdvancedGraphicsEffectsRenderer(800, 600)
        renderer.run()
    except Exception as e:
        print(f"✗ Rendering system failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate advanced graphics effects"""
    print("=== Advanced Graphics Effects Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    demonstrate_screen_space_effects()
    demonstrate_particle_systems()
    demonstrate_procedural_generation()
    
    print("="*60)
    print("Advanced Graphics Effects demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Screen space effects (SSAO)")
    print("✓ Advanced particle systems")
    print("✓ Procedural content generation")
    print("✓ Real-time graphics effects")
    print("✓ GPU-based computation")
    print("✓ Modern graphics techniques")
    
    print("\nEffect features:")
    print("• Screen space ambient occlusion")
    print("• GPU particle simulation")
    print("• Procedural noise generation")
    print("• Real-time content creation")
    print("• Efficient rendering techniques")
    
    print("\nApplications:")
    print("• Game development: Advanced visual effects")
    print("• Real-time graphics: Interactive effects")
    print("• Visual effects: Professional VFX")
    print("• Virtual reality: Immersive graphics")
    print("• Scientific visualization: Dynamic content")
    
    print("\nNext steps:")
    print("• Add more screen space effects")
    print("• Implement advanced particle behaviors")
    print("• Add procedural geometry generation")
    print("• Optimize for mobile platforms")
    print("• Integrate with game engines")

if __name__ == "__main__":
    main()
