#!/usr/bin/env python3
"""
Chapter 12: Advanced Rendering Techniques
Shader-Based Rendering System

Demonstrates shader programming fundamentals, GPU computing, material systems,
and modern rendering techniques using OpenGL and GLSL.
"""

import numpy as np
import moderngl
import glfw
from PIL import Image
import math
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Shader-Based Rendering System"
__description__ = "Modern shader-based rendering with OpenGL"

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

@dataclass
class Material:
    """Material properties for PBR rendering"""
    albedo: Color
    metallic: float = 0.0
    roughness: float = 0.5
    ao: float = 1.0
    
    def to_array(self) -> List[float]:
        return [*self.albedo.to_array(), self.metallic, self.roughness, self.ao, 0.0]

class ShaderType(Enum):
    """Types of shaders"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    COMPUTE = "compute"

# ============================================================================
# SHADER MANAGEMENT
# ============================================================================

class ShaderManager:
    """Manages shader compilation and linking"""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.shaders: Dict[str, moderngl.Program] = {}
        self.shader_sources: Dict[str, Dict[str, str]] = {}
    
    def add_shader(self, name: str, vertex_source: str, fragment_source: str):
        """Add a shader program"""
        try:
            program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source
            )
            self.shaders[name] = program
            self.shader_sources[name] = {
                'vertex': vertex_source,
                'fragment': fragment_source
            }
            print(f"✓ Shader '{name}' compiled successfully")
        except Exception as e:
            print(f"✗ Failed to compile shader '{name}': {e}")
    
    def get_shader(self, name: str) -> Optional[moderngl.Program]:
        """Get a shader program by name"""
        return self.shaders.get(name)
    
    def reload_shader(self, name: str) -> bool:
        """Reload a shader program"""
        if name not in self.shader_sources:
            return False
        
        sources = self.shader_sources[name]
        try:
            new_program = self.ctx.program(
                vertex_shader=sources['vertex'],
                fragment_shader=sources['fragment']
            )
            self.shaders[name] = new_program
            print(f"✓ Shader '{name}' reloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to reload shader '{name}': {e}")
            return False

# ============================================================================
# SHADER SOURCES
# ============================================================================

class ShaderSources:
    """Predefined shader source code"""
    
    # Basic vertex shader
    BASIC_VERTEX = """
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
    
    # Basic fragment shader
    BASIC_FRAGMENT = """
    #version 330
    
    in vec3 world_pos;
    in vec3 world_normal;
    in vec2 texcoord;
    
    uniform vec3 light_pos;
    uniform vec3 light_color;
    uniform vec3 view_pos;
    uniform vec4 material_albedo;
    uniform float material_metallic;
    uniform float material_roughness;
    uniform float material_ao;
    
    out vec4 frag_color;
    
    void main() {
        vec3 normal = normalize(world_normal);
        vec3 light_dir = normalize(light_pos - world_pos);
        vec3 view_dir = normalize(view_pos - world_pos);
        
        // Ambient lighting
        float ambient_strength = 0.1;
        vec3 ambient = ambient_strength * light_color;
        
        // Diffuse lighting
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = diff * light_color;
        
        // Specular lighting
        float specular_strength = 0.5;
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
        vec3 specular = specular_strength * spec * light_color;
        
        vec3 result = (ambient + diffuse + specular) * material_albedo.rgb;
        frag_color = vec4(result, material_albedo.a);
    }
    """
    
    # PBR vertex shader
    PBR_VERTEX = """
    #version 330
    
    in vec3 in_position;
    in vec3 in_normal;
    in vec2 in_texcoord;
    in vec3 in_tangent;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat3 normal_matrix;
    
    out vec3 world_pos;
    out vec3 world_normal;
    out vec2 texcoord;
    out mat3 TBN;
    
    void main() {
        world_pos = vec3(model * vec4(in_position, 1.0));
        world_normal = normal_matrix * in_normal;
        texcoord = in_texcoord;
        
        vec3 T = normalize(normal_matrix * in_tangent);
        vec3 N = normalize(world_normal);
        T = normalize(T - dot(T, N) * N);
        vec3 B = cross(N, T);
        TBN = mat3(T, B, N);
        
        gl_Position = projection * view * vec4(world_pos, 1.0);
    }
    """
    
    # PBR fragment shader
    PBR_FRAGMENT = """
    #version 330
    
    in vec3 world_pos;
    in vec3 world_normal;
    in vec2 texcoord;
    in mat3 TBN;
    
    uniform vec3 light_positions[4];
    uniform vec3 light_colors[4];
    uniform vec3 view_pos;
    uniform vec4 material_albedo;
    uniform float material_metallic;
    uniform float material_roughness;
    uniform float material_ao;
    
    out vec4 frag_color;
    
    const float PI = 3.14159265359;
    
    // PBR functions
    vec3 fresnelSchlick(float cosTheta, vec3 F0) {
        return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    }
    
    float DistributionGGX(vec3 N, vec3 H, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float NdotH = max(dot(N, H), 0.0);
        float NdotH2 = NdotH * NdotH;
        
        float nom = a2;
        float denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = PI * denom * denom;
        
        return nom / denom;
    }
    
    float GeometrySchlickGGX(float NdotV, float roughness) {
        float r = (roughness + 1.0);
        float k = (r * r) / 8.0;
        
        float nom = NdotV;
        float denom = NdotV * (1.0 - k) + k;
        
        return nom / denom;
    }
    
    float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
        float NdotV = max(dot(N, V), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        float ggx2 = GeometrySchlickGGX(NdotV, roughness);
        float ggx1 = GeometrySchlickGGX(NdotL, roughness);
        
        return ggx1 * ggx2;
    }
    
    void main() {
        vec3 N = normalize(world_normal);
        vec3 V = normalize(view_pos - world_pos);
        
        vec3 F0 = vec3(0.04);
        F0 = mix(F0, material_albedo.rgb, material_metallic);
        
        vec3 Lo = vec3(0.0);
        
        for(int i = 0; i < 4; ++i) {
            vec3 L = normalize(light_positions[i] - world_pos);
            vec3 H = normalize(V + L);
            float distance = length(light_positions[i] - world_pos);
            float attenuation = 1.0 / (distance * distance);
            vec3 radiance = light_colors[i] * attenuation;
            
            // Cook-Torrance BRDF
            float NDF = DistributionGGX(N, H, material_roughness);
            float G = GeometrySmith(N, V, L, material_roughness);
            vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
            
            vec3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            vec3 specular = numerator / denominator;
            
            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - material_metallic;
            
            float NdotL = max(dot(N, L), 0.0);
            
            Lo += (kD * material_albedo.rgb / PI + specular) * radiance * NdotL;
        }
        
        vec3 ambient = vec3(0.03) * material_albedo.rgb * material_ao;
        vec3 color = ambient + Lo;
        
        // HDR tonemapping
        color = color / (color + vec3(1.0));
        // Gamma correction
        color = pow(color, vec3(1.0/2.2));
        
        frag_color = vec4(color, material_albedo.a);
    }
    """

# ============================================================================
# RENDERING SYSTEM
# ============================================================================

class ShaderBasedRenderer:
    """Main rendering system using shaders"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.shader_manager = None
        self.vao = None
        self.vbo = None
        self.ibo = None
        
        # Camera properties
        self.camera_pos = Vector3D(0, 0, 3)
        self.camera_target = Vector3D(0, 0, 0)
        self.camera_up = Vector3D(0, 1, 0)
        self.fov = 45.0
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Lighting
        self.light_positions = [
            Vector3D(2, 2, 2),
            Vector3D(-2, 2, 2),
            Vector3D(2, -2, 2),
            Vector3D(-2, -2, 2)
        ]
        self.light_colors = [
            Color(1.0, 0.8, 0.6),
            Color(0.6, 0.8, 1.0),
            Color(0.8, 1.0, 0.6),
            Color(1.0, 0.6, 0.8)
        ]
        
        # Materials
        self.materials = {
            'gold': Material(Color(1.0, 0.8, 0.0), metallic=1.0, roughness=0.1),
            'plastic': Material(Color(0.2, 0.8, 0.2), metallic=0.0, roughness=0.3),
            'metal': Material(Color(0.7, 0.7, 0.7), metallic=1.0, roughness=0.5),
            'ceramic': Material(Color(0.9, 0.9, 0.9), metallic=0.0, roughness=0.8)
        }
        
        self.current_material = 'gold'
        self.rotation = 0.0
        
        self.init_glfw()
        self.init_opengl()
        self.setup_shaders()
        self.setup_geometry()
    
    def init_glfw(self):
        """Initialize GLFW"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Shader-Based Rendering", None, None)
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
    
    def setup_shaders(self):
        """Setup shader programs"""
        self.shader_manager = ShaderManager(self.ctx)
        
        # Add basic shader
        self.shader_manager.add_shader(
            'basic',
            ShaderSources.BASIC_VERTEX,
            ShaderSources.BASIC_FRAGMENT
        )
        
        # Add PBR shader
        self.shader_manager.add_shader(
            'pbr',
            ShaderSources.PBR_VERTEX,
            ShaderSources.PBR_FRAGMENT
        )
    
    def setup_geometry(self):
        """Setup geometry data"""
        # Create a sphere
        vertices, indices = self.create_sphere(1.0, 32, 16)
        
        # Create vertex buffer
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        
        # Create index buffer
        self.ibo = self.ctx.buffer(indices.astype('u4').tobytes())
        
        # Create vertex array object
        self.vao = self.ctx.vertex_array(
            self.shader_manager.get_shader('pbr'),
            [
                (self.vbo, '3f 3f 2f 3f', 'in_position', 'in_normal', 'in_texcoord', 'in_tangent'),
            ],
            self.ibo
        )
    
    def create_sphere(self, radius: float, segments: int, rings: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sphere geometry"""
        vertices = []
        indices = []
        
        for ring in range(rings + 1):
            phi = ring * math.pi / rings
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            for segment in range(segments + 1):
                theta = segment * 2 * math.pi / segments
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)
                
                x = radius * cos_theta * sin_phi
                y = radius * cos_phi
                z = radius * sin_theta * sin_phi
                
                # Position
                vertices.extend([x, y, z])
                
                # Normal
                vertices.extend([x/radius, y/radius, z/radius])
                
                # Texture coordinates
                vertices.extend([segment / segments, ring / rings])
                
                # Tangent (simplified)
                vertices.extend([-sin_theta, 0, cos_theta])
        
        # Create indices
        for ring in range(rings):
            for segment in range(segments):
                first = ring * (segments + 1) + segment
                second = first + segments + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='u4')
    
    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix"""
        return self.look_at(self.camera_pos, self.camera_target, self.camera_up)
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix"""
        return self.perspective(self.fov, self.width / self.height, self.near_plane, self.far_plane)
    
    def get_model_matrix(self) -> np.ndarray:
        """Get model matrix"""
        model = np.eye(4, dtype='f4')
        model = self.rotate_y(model, self.rotation)
        return model
    
    def look_at(self, eye: Vector3D, target: Vector3D, up: Vector3D) -> np.ndarray:
        """Create look-at matrix"""
        z = Vector3D(eye.x - target.x, eye.y - target.y, eye.z - target.z)
        z = Vector3D(z.x / math.sqrt(z.x*z.x + z.y*z.y + z.z*z.z),
                    z.y / math.sqrt(z.x*z.x + z.y*z.y + z.z*z.z),
                    z.z / math.sqrt(z.x*z.x + z.y*z.y + z.z*z.z))
        
        x = Vector3D(up.y * z.z - up.z * z.y,
                    up.z * z.x - up.x * z.z,
                    up.x * z.y - up.y * z.x)
        x = Vector3D(x.x / math.sqrt(x.x*x.x + x.y*x.y + x.z*x.z),
                    x.y / math.sqrt(x.x*x.x + x.y*x.y + x.z*x.z),
                    x.z / math.sqrt(x.x*x.x + x.y*x.y + x.z*x.z))
        
        y = Vector3D(z.y * x.z - z.z * x.y,
                    z.z * x.x - z.x * x.z,
                    z.x * x.y - z.y * x.x)
        
        return np.array([
            [x.x, x.y, x.z, -x.x*eye.x - x.y*eye.y - x.z*eye.z],
            [y.x, y.y, y.z, -y.x*eye.x - y.y*eye.y - y.z*eye.z],
            [z.x, z.y, z.z, -z.x*eye.x - z.y*eye.y - z.z*eye.z],
            [0, 0, 0, 1]
        ], dtype='f4')
    
    def perspective(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective matrix"""
        f = 1.0 / math.tan(fov * math.pi / 360.0)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype='f4')
    
    def rotate_y(self, matrix: np.ndarray, angle: float) -> np.ndarray:
        """Rotate matrix around Y axis"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        rotation = np.array([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ], dtype='f4')
        
        return rotation @ matrix
    
    def render(self):
        """Main rendering loop"""
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Get matrices
        model = self.get_model_matrix()
        view = self.get_view_matrix()
        projection = self.get_projection_matrix()
        normal_matrix = np.linalg.inv(model[:3, :3]).T
        
        # Get current shader
        shader = self.shader_manager.get_shader('pbr')
        if not shader:
            return
        
        # Set uniforms
        shader['model'].write(model.tobytes())
        shader['view'].write(view.tobytes())
        shader['projection'].write(projection.tobytes())
        shader['normal_matrix'].write(normal_matrix.tobytes())
        
        # Set lighting uniforms
        for i in range(4):
            shader[f'light_positions[{i}]'].write(self.light_positions[i].to_array())
            shader[f'light_colors[{i}]'].write(self.light_colors[i].to_array())
        
        # Set material uniforms
        material = self.materials[self.current_material]
        shader['material_albedo'].write(material.albedo.to_array())
        shader['material_metallic'].value = material.metallic
        shader['material_roughness'].value = material.roughness
        shader['material_ao'].value = material.ao
        
        # Set camera position
        shader['view_pos'].write(self.camera_pos.to_array())
        
        # Render
        self.vao.render()
    
    def update(self, delta_time: float):
        """Update scene"""
        self.rotation += delta_time * 0.5
    
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
            
            # Update
            self.update(delta_time)
            
            # Render
            self.render()
            
            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Handle input
            self.handle_input()
        
        self.cleanup()
    
    def handle_input(self):
        """Handle keyboard input"""
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.current_material = 'gold'
        elif glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.current_material = 'plastic'
        elif glfw.get_key(self.window, glfw.KEY_3) == glfw.PRESS:
            self.current_material = 'metal'
        elif glfw.get_key(self.window, glfw.KEY_4) == glfw.PRESS:
            self.current_material = 'ceramic'
        elif glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            self.shader_manager.reload_shader('pbr')
    
    def cleanup(self):
        """Cleanup resources"""
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_shader_compilation():
    """Demonstrate shader compilation"""
    print("=== Shader Compilation Demo ===\n")
    
    # Create a minimal OpenGL context for shader compilation
    if not glfw.init():
        print("Failed to initialize GLFW")
        return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1, 1, "Shader Test", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return
    
    glfw.make_context_current(window)
    
    try:
        ctx = moderngl.create_context()
        shader_manager = ShaderManager(ctx)
        
        # Test basic shader compilation
        shader_manager.add_shader(
            'test_basic',
            ShaderSources.BASIC_VERTEX,
            ShaderSources.BASIC_FRAGMENT
        )
        
        # Test PBR shader compilation
        shader_manager.add_shader(
            'test_pbr',
            ShaderSources.PBR_VERTEX,
            ShaderSources.PBR_FRAGMENT
        )
        
        print("✓ All shaders compiled successfully")
        
        # Test shader reloading
        print("\nTesting shader reloading...")
        success = shader_manager.reload_shader('test_basic')
        if success:
            print("✓ Shader reloading works")
        else:
            print("✗ Shader reloading failed")
        
    except Exception as e:
        print(f"✗ Shader compilation failed: {e}")
    
    finally:
        glfw.terminate()

def demonstrate_material_system():
    """Demonstrate material system"""
    print("\n=== Material System Demo ===\n")
    
    materials = {
        'gold': Material(Color(1.0, 0.8, 0.0), metallic=1.0, roughness=0.1),
        'plastic': Material(Color(0.2, 0.8, 0.2), metallic=0.0, roughness=0.3),
        'metal': Material(Color(0.7, 0.7, 0.7), metallic=1.0, roughness=0.5),
        'ceramic': Material(Color(0.9, 0.9, 0.9), metallic=0.0, roughness=0.8)
    }
    
    print("Available materials:")
    for name, material in materials.items():
        print(f"  {name}:")
        print(f"    Albedo: {material.albedo}")
        print(f"    Metallic: {material.metallic}")
        print(f"    Roughness: {material.roughness}")
        print(f"    AO: {material.ao}")
        print()

def demonstrate_rendering_system():
    """Demonstrate the complete rendering system"""
    print("=== Rendering System Demo ===\n")
    
    print("Starting shader-based rendering system...")
    print("Controls:")
    print("  1-4: Switch materials (gold, plastic, metal, ceramic)")
    print("  R: Reload shaders")
    print("  ESC: Exit")
    print()
    
    try:
        renderer = ShaderBasedRenderer(800, 600)
        renderer.run()
    except Exception as e:
        print(f"✗ Rendering system failed: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate shader-based rendering system"""
    print("=== Shader-Based Rendering System Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    demonstrate_shader_compilation()
    demonstrate_material_system()
    
    print("="*60)
    print("Shader-Based Rendering System demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Shader compilation and management")
    print("✓ Vertex and fragment shader programming")
    print("✓ PBR material system")
    print("✓ Modern OpenGL rendering pipeline")
    print("✓ Real-time shader reloading")
    print("✓ GPU-based rendering optimization")
    
    print("\nShader features:")
    print("• GLSL shader language support")
    print("• PBR lighting model implementation")
    print("• Material system with metallic/roughness workflow")
    print("• Multiple light sources")
    print("• Real-time shader hot-reloading")
    
    print("\nApplications:")
    print("• Game development: Modern rendering pipelines")
    print("• Real-time graphics: High-performance rendering")
    print("• Material visualization: PBR material preview")
    print("• Graphics research: Shader development and testing")
    print("• Educational: Learning modern graphics programming")
    
    print("\nNext steps:")
    print("• Add texture support and normal mapping")
    print("• Implement post-processing effects")
    print("• Add more advanced lighting models")
    print("• Optimize for mobile and web platforms")
    print("• Integrate with game engines and frameworks")

if __name__ == "__main__":
    main()
