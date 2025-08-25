#!/usr/bin/env python3
"""
Chapter 15: Advanced 3D Graphics Libraries and Tools
Advanced Rendering Pipeline

Demonstrates how to build a sophisticated rendering pipeline using multiple
libraries and techniques for maximum performance and visual quality.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import weakref

@dataclass
class RenderStats:
    """Statistics for rendering performance"""
    frame_count: int = 0
    fps: float = 0.0
    frame_time: float = 0.0
    draw_calls: int = 0
    triangles: int = 0
    vertices: int = 0
    memory_usage: float = 0.0
    gpu_time: float = 0.0
    cpu_time: float = 0.0

class RenderStage(Enum):
    """Different stages of the rendering pipeline"""
    SHADOW_MAP = "shadow_map"
    GEOMETRY = "geometry"
    LIGHTING = "lighting"
    POST_PROCESS = "post_process"
    UI = "ui"

class RenderTarget:
    """Represents a render target (texture or framebuffer)"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int, 
                 color_attachments: int = 1, depth_attachment: bool = True):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.color_attachments = color_attachments
        self.depth_attachment = depth_attachment
        
        # Create framebuffer
        self.framebuffer = self.create_framebuffer()
        self.textures = self.create_textures()
    
    def create_framebuffer(self) -> moderngl.Framebuffer:
        """Create the framebuffer object"""
        return self.ctx.framebuffer()
    
    def create_textures(self) -> List[moderngl.Texture]:
        """Create texture attachments"""
        textures = []
        
        # Create color attachments
        for i in range(self.color_attachments):
            texture = self.ctx.texture((self.width, self.height), 4)
            self.framebuffer.color_attachments[i] = texture
            textures.append(texture)
        
        # Create depth attachment if needed
        if self.depth_attachment:
            depth_texture = self.ctx.depth_texture((self.width, self.height))
            self.framebuffer.depth_attachment = depth_texture
            textures.append(depth_texture)
        
        return textures
    
    def use(self):
        """Bind this render target"""
        self.framebuffer.use()
    
    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0):
        """Clear the render target"""
        self.framebuffer.clear(r, g, b, a)
    
    def get_texture(self, index: int = 0) -> moderngl.Texture:
        """Get a texture attachment"""
        if 0 <= index < len(self.textures):
            return self.textures[index]
        return None

class ShaderProgram:
    """Wrapper for shader programs with uniform management"""
    
    def __init__(self, ctx: moderngl.Context, vertex_src: str, fragment_src: str):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        self.uniforms = {}
    
    def set_uniform(self, name: str, value):
        """Set a uniform value"""
        if name in self.program:
            self.program[name].value = value
            self.uniforms[name] = value
    
    def set_uniform_matrix(self, name: str, matrix: np.ndarray):
        """Set a matrix uniform"""
        if name in self.program:
            self.program[name].write(matrix.astype('f4').tobytes())
            self.uniforms[name] = matrix
    
    def use(self):
        """Use this shader program"""
        return self.program

class GeometryBuffer:
    """Geometry buffer for deferred rendering"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Create G-Buffer with multiple attachments
        self.framebuffer = ctx.framebuffer()
        
        # Position texture (RGB for position, A for depth)
        self.position_texture = ctx.texture((width, height), 4, dtype='f4')
        self.framebuffer.color_attachments[0] = self.position_texture
        
        # Normal texture (RGB for normal, A for unused)
        self.normal_texture = ctx.texture((width, height), 4, dtype='f4')
        self.framebuffer.color_attachments[1] = self.normal_texture
        
        # Albedo texture (RGB for color, A for metallic)
        self.albedo_texture = ctx.texture((width, height), 4, dtype='f4')
        self.framebuffer.color_attachments[2] = self.albedo_texture
        
        # Material texture (R for roughness, G for AO, B for unused, A for unused)
        self.material_texture = ctx.texture((width, height), 4, dtype='f4')
        self.framebuffer.color_attachments[3] = self.material_texture
        
        # Depth texture
        self.depth_texture = ctx.depth_texture((width, height))
        self.framebuffer.depth_attachment = self.depth_texture
    
    def use(self):
        """Bind the G-Buffer"""
        self.framebuffer.use()
    
    def clear(self):
        """Clear the G-Buffer"""
        self.framebuffer.clear(0.0, 0.0, 0.0, 1.0)
    
    def get_textures(self) -> List[moderngl.Texture]:
        """Get all G-Buffer textures"""
        return [self.position_texture, self.normal_texture, 
                self.albedo_texture, self.material_texture]

class RenderPipeline:
    """Advanced rendering pipeline with multiple stages"""
    
    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # Pipeline stages
        self.stages: Dict[RenderStage, Callable] = {}
        self.stage_order = [
            RenderStage.SHADOW_MAP,
            RenderStage.GEOMETRY,
            RenderStage.LIGHTING,
            RenderStage.POST_PROCESS,
            RenderStage.UI
        ]
        
        # Render targets
        self.shadow_map = RenderTarget(ctx, 2048, 2048, 1, True)
        self.geometry_buffer = GeometryBuffer(ctx, width, height)
        self.lighting_buffer = RenderTarget(ctx, width, height, 1, False)
        self.post_process_buffer = RenderTarget(ctx, width, height, 1, False)
        
        # Shader programs
        self.shaders = self.create_shaders()
        
        # Statistics
        self.stats = RenderStats()
        self.frame_times = []
        
        # Performance monitoring
        self.enable_profiling = True
        self.stage_times = {}
    
    def create_shaders(self) -> Dict[str, ShaderProgram]:
        """Create all shader programs"""
        shaders = {}
        
        # Shadow mapping shader
        shadow_vertex = """
        #version 330
        in vec3 in_position;
        uniform mat4 lightSpaceMatrix;
        uniform mat4 model;
        
        void main() {
            gl_Position = lightSpaceMatrix * model * vec4(in_position, 1.0);
        }
        """
        
        shadow_fragment = """
        #version 330
        void main() {
            // Depth is automatically written
        }
        """
        
        shaders['shadow'] = ShaderProgram(self.ctx, shadow_vertex, shadow_fragment)
        
        # Geometry pass shader
        geometry_vertex = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec2 in_texcoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        void main() {
            FragPos = vec3(model * vec4(in_position, 1.0));
            Normal = mat3(transpose(inverse(model))) * in_normal;
            TexCoord = in_texcoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        geometry_fragment = """
        #version 330
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        layout (location = 0) out vec4 gPosition;
        layout (location = 1) out vec4 gNormal;
        layout (location = 2) out vec4 gAlbedo;
        layout (location = 3) out vec4 gMaterial;
        
        uniform vec3 color;
        uniform float metallic;
        uniform float roughness;
        
        void main() {
            gPosition = vec4(FragPos, 1.0);
            gNormal = vec4(normalize(Normal), 1.0);
            gAlbedo = vec4(color, metallic);
            gMaterial = vec4(roughness, 0.0, 0.0, 1.0);
        }
        """
        
        shaders['geometry'] = ShaderProgram(self.ctx, geometry_vertex, geometry_fragment)
        
        # Lighting pass shader
        lighting_vertex = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = in_texcoord;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """
        
        lighting_fragment = """
        #version 330
        in vec2 TexCoord;
        
        out vec4 FragColor;
        
        uniform sampler2D gPosition;
        uniform sampler2D gNormal;
        uniform sampler2D gAlbedo;
        uniform sampler2D gMaterial;
        uniform sampler2D shadowMap;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 viewPos;
        uniform mat4 lightSpaceMatrix;
        
        float ShadowCalculation(vec4 fragPosLightSpace) {
            vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
            projCoords = projCoords * 0.5 + 0.5;
            float closestDepth = texture(shadowMap, projCoords.xy).r;
            float currentDepth = projCoords.z;
            float bias = 0.005;
            return currentDepth - bias > closestDepth ? 1.0 : 0.0;
        }
        
        void main() {
            vec3 FragPos = texture(gPosition, TexCoord).rgb;
            vec3 Normal = texture(gNormal, TexCoord).rgb;
            vec3 Albedo = texture(gAlbedo, TexCoord).rgb;
            float Metallic = texture(gAlbedo, TexCoord).a;
            float Roughness = texture(gMaterial, TexCoord).r;
            
            vec3 lighting = Albedo * 0.1; // ambient
            
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            vec4 fragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
            float shadow = ShadowCalculation(fragPosLightSpace);
            
            lighting += (1.0 - shadow) * diffuse;
            
            FragColor = vec4(lighting, 1.0);
        }
        """
        
        shaders['lighting'] = ShaderProgram(self.ctx, lighting_vertex, lighting_fragment)
        
        # Post-processing shader
        post_vertex = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = in_texcoord;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """
        
        post_fragment = """
        #version 330
        in vec2 TexCoord;
        
        out vec4 FragColor;
        
        uniform sampler2D screenTexture;
        uniform float time;
        
        void main() {
            vec3 color = texture(screenTexture, TexCoord).rgb;
            
            // Simple bloom effect
            vec3 bloom = vec3(0.0);
            for(int i = 0; i < 9; i++) {
                for(int j = 0; j < 9; j++) {
                    vec2 offset = vec2(i - 4, j - 4) * 0.001;
                    bloom += texture(screenTexture, TexCoord + offset).rgb;
                }
            }
            bloom /= 81.0;
            
            // Add some color grading
            color = pow(color, vec3(1.0 / 2.2)); // gamma correction
            color += bloom * 0.1;
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        shaders['post_process'] = ShaderProgram(self.ctx, post_vertex, post_fragment)
        
        return shaders
    
    def add_stage(self, stage: RenderStage, callback: Callable):
        """Add a rendering stage"""
        self.stages[stage] = callback
    
    def render_frame(self, scene_data: Dict[str, Any]):
        """Render a complete frame through all stages"""
        start_time = time.time()
        
        # Clear statistics
        self.stats.draw_calls = 0
        self.stats.triangles = 0
        self.stats.vertices = 0
        
        # Execute each stage in order
        for stage in self.stage_order:
            if stage in self.stages:
                stage_start = time.time()
                self.stages[stage](scene_data)
                stage_time = time.time() - stage_start
                
                if self.enable_profiling:
                    self.stage_times[stage.value] = stage_time
        
        # Update statistics
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        
        # Keep only last 60 frames for FPS calculation
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        self.stats.frame_count += 1
        self.stats.frame_time = frame_time
        self.stats.fps = 1.0 / frame_time if frame_time > 0 else 0.0
    
    def get_stats(self) -> RenderStats:
        """Get current rendering statistics"""
        return self.stats
    
    def get_stage_times(self) -> Dict[str, float]:
        """Get timing information for each stage"""
        return self.stage_times.copy()

class AdvancedRenderer:
    """Main renderer class that uses the advanced pipeline"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Pipeline
        self.pipeline = None
        
        # Scene data
        self.scene_objects = []
        self.lights = []
        self.camera = {
            'position': [0, 0, 5],
            'target': [0, 0, 0],
            'up': [0, 1, 0],
            'fov': 45.0,
            'near': 0.1,
            'far': 100.0
        }
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.setup_pipeline()
        self.create_scene()
    
    def init_glfw(self):
        """Initialize GLFW"""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Advanced Rendering Pipeline", None, None)
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
    
    def setup_pipeline(self):
        """Setup the rendering pipeline"""
        self.pipeline = RenderPipeline(self.ctx, self.width, self.height)
        
        # Add pipeline stages
        self.pipeline.add_stage(RenderStage.SHADOW_MAP, self.render_shadow_map)
        self.pipeline.add_stage(RenderStage.GEOMETRY, self.render_geometry)
        self.pipeline.add_stage(RenderStage.LIGHTING, self.render_lighting)
        self.pipeline.add_stage(RenderStage.POST_PROCESS, self.render_post_process)
        self.pipeline.add_stage(RenderStage.UI, self.render_ui)
    
    def create_scene(self):
        """Create test scene objects"""
        # Create a simple cube
        cube_vertices = np.array([
            # Front face
            -1, -1,  1,  0,  0,  1,  0, 0,
             1, -1,  1,  0,  0,  1,  1, 0,
             1,  1,  1,  0,  0,  1,  1, 1,
            -1,  1,  1,  0,  0,  1,  0, 1,
            # Back face
            -1, -1, -1,  0,  0, -1,  1, 0,
             1, -1, -1,  0,  0, -1,  0, 0,
             1,  1, -1,  0,  0, -1,  0, 1,
            -1,  1, -1,  0,  0, -1,  1, 1,
        ], dtype='f4')
        
        cube_indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            1, 5, 6, 6, 2, 1,  # Right
            5, 4, 7, 7, 6, 5,  # Back
            4, 0, 3, 3, 7, 4,  # Left
            3, 2, 6, 6, 7, 3,  # Top
            4, 5, 1, 1, 0, 4   # Bottom
        ], dtype='u4')
        
        # Create vertex buffer and vertex array
        self.cube_vbo = self.ctx.buffer(cube_vertices.tobytes())
        self.cube_ibo = self.ctx.buffer(cube_indices.tobytes())
        
        self.cube_vao = self.ctx.vertex_array(
            self.pipeline.shaders['geometry'].use(),
            [(self.cube_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord')],
            self.cube_ibo
        )
        
        # Add scene objects
        self.scene_objects.append({
            'type': 'cube',
            'position': [0, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            'color': [0.8, 0.2, 0.2],
            'metallic': 0.0,
            'roughness': 0.5
        })
        
        # Add lights
        self.lights.append({
            'position': [5, 5, 5],
            'color': [1.0, 1.0, 1.0],
            'intensity': 1.0
        })
    
    def render_shadow_map(self, scene_data: Dict[str, Any]):
        """Render shadow map stage"""
        self.pipeline.shadow_map.use()
        self.pipeline.shadow_map.clear()
        
        # Set up light space matrix
        light_pos = self.lights[0]['position']
        light_space_matrix = self.calculate_light_space_matrix(light_pos)
        
        shader = self.pipeline.shaders['shadow'].use()
        shader.set_uniform_matrix('lightSpaceMatrix', light_space_matrix)
        
        # Render scene objects to shadow map
        for obj in self.scene_objects:
            model_matrix = self.calculate_model_matrix(obj)
            shader.set_uniform_matrix('model', model_matrix)
            
            self.cube_vao.render()
            self.pipeline.stats.draw_calls += 1
            self.pipeline.stats.triangles += 12  # 2 triangles per face * 6 faces
    
    def render_geometry(self, scene_data: Dict[str, Any]):
        """Render geometry stage (G-Buffer)"""
        self.pipeline.geometry_buffer.use()
        self.pipeline.geometry_buffer.clear()
        
        # Set up view and projection matrices
        view_matrix = self.calculate_view_matrix()
        projection_matrix = self.calculate_projection_matrix()
        
        shader = self.pipeline.shaders['geometry'].use()
        shader.set_uniform_matrix('view', view_matrix)
        shader.set_uniform_matrix('projection', projection_matrix)
        
        # Render scene objects
        for obj in self.scene_objects:
            model_matrix = self.calculate_model_matrix(obj)
            shader.set_uniform_matrix('model', model_matrix)
            shader.set_uniform('color', obj['color'])
            shader.set_uniform('metallic', obj['metallic'])
            shader.set_uniform('roughness', obj['roughness'])
            
            self.cube_vao.render()
            self.pipeline.stats.draw_calls += 1
            self.pipeline.stats.triangles += 12
            self.pipeline.stats.vertices += 24
    
    def render_lighting(self, scene_data: Dict[str, Any]):
        """Render lighting stage"""
        self.pipeline.lighting_buffer.use()
        self.pipeline.lighting_buffer.clear()
        
        # Bind G-Buffer textures
        g_textures = self.pipeline.geometry_buffer.get_textures()
        for i, texture in enumerate(g_textures):
            texture.use(i)
        
        # Bind shadow map
        self.pipeline.shadow_map.get_texture(0).use(len(g_textures))
        
        shader = self.pipeline.shaders['lighting'].use()
        shader.set_uniform('gPosition', 0)
        shader.set_uniform('gNormal', 1)
        shader.set_uniform('gAlbedo', 2)
        shader.set_uniform('gMaterial', 3)
        shader.set_uniform('shadowMap', len(g_textures))
        
        # Set light properties
        light = self.lights[0]
        shader.set_uniform('lightPos', light['position'])
        shader.set_uniform('lightColor', light['color'])
        shader.set_uniform('viewPos', self.camera['position'])
        
        # Calculate light space matrix for shadow mapping
        light_space_matrix = self.calculate_light_space_matrix(light['position'])
        shader.set_uniform_matrix('lightSpaceMatrix', light_space_matrix)
        
        # Render full-screen quad
        self.render_fullscreen_quad()
    
    def render_post_process(self, scene_data: Dict[str, Any]):
        """Render post-processing stage"""
        self.pipeline.post_process_buffer.use()
        self.pipeline.post_process_buffer.clear()
        
        # Bind lighting buffer texture
        self.pipeline.lighting_buffer.get_texture(0).use(0)
        
        shader = self.pipeline.shaders['post_process'].use()
        shader.set_uniform('screenTexture', 0)
        shader.set_uniform('time', time.time())
        
        # Render full-screen quad
        self.render_fullscreen_quad()
    
    def render_ui(self, scene_data: Dict[str, Any]):
        """Render UI stage (final output)"""
        self.ctx.screen.use()
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Bind post-process buffer texture
        self.pipeline.post_process_buffer.get_texture(0).use(0)
        
        # Use simple shader for final output
        vertex_src = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 TexCoord;
        void main() {
            TexCoord = in_texcoord;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """
        
        fragment_src = """
        #version 330
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
        """
        
        shader = self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)
        shader['screenTexture'] = 0
        
        # Render full-screen quad
        self.render_fullscreen_quad()
    
    def render_fullscreen_quad(self):
        """Render a full-screen quad"""
        vertices = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
             1,  1,  1, 1,
            -1,  1,  0, 1,
        ], dtype='f4')
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        vao = self.ctx.vertex_array(
            self.ctx.program(vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec2 in_texcoord;
                out vec2 TexCoord;
                void main() {
                    TexCoord = in_texcoord;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """, fragment_shader="""
                #version 330
                in vec2 TexCoord;
                out vec4 FragColor;
                uniform sampler2D screenTexture;
                void main() {
                    FragColor = texture(screenTexture, TexCoord);
                }
            """),
            [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
            ibo
        )
        
        vao.render()
    
    def calculate_model_matrix(self, obj: Dict[str, Any]) -> np.ndarray:
        """Calculate model matrix for an object"""
        # Simple identity matrix for now
        return np.eye(4, dtype='f4')
    
    def calculate_view_matrix(self) -> np.ndarray:
        """Calculate view matrix"""
        # Simple look-at matrix
        return np.eye(4, dtype='f4')
    
    def calculate_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix"""
        fov = math.radians(self.camera['fov'])
        aspect = self.width / self.height
        near = self.camera['near']
        far = self.camera['far']
        
        f = 1.0 / math.tan(fov / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype='f4')
    
    def calculate_light_space_matrix(self, light_pos: List[float]) -> np.ndarray:
        """Calculate light space matrix for shadow mapping"""
        # Simple orthographic projection for directional light
        return np.eye(4, dtype='f4')
    
    def framebuffer_size_callback(self, window, width, height):
        """Handle framebuffer size changes"""
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        
        # Recreate pipeline with new dimensions
        self.pipeline = RenderPipeline(self.ctx, width, height)
        self.setup_pipeline()
    
    def run(self):
        """Main render loop"""
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Render frame
            scene_data = {
                'delta_time': delta_time,
                'time': current_time,
                'objects': self.scene_objects,
                'lights': self.lights,
                'camera': self.camera
            }
            
            self.pipeline.render_frame(scene_data)
            
            # Update statistics
            self.frame_count += 1
            if current_time - self.last_fps_time >= 2.0:
                stats = self.pipeline.get_stats()
                stage_times = self.pipeline.get_stage_times()
                
                print(f"FPS: {stats.fps:.1f}, Draw Calls: {stats.draw_calls}, "
                      f"Triangles: {stats.triangles}, Frame Time: {stats.frame_time*1000:.2f}ms")
                
                if stage_times:
                    print("Stage Times:")
                    for stage, time_taken in stage_times.items():
                        print(f"  {stage}: {time_taken*1000:.2f}ms")
                
                self.frame_count = 0
                self.last_fps_time = current_time
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        glfw.terminate()

def main():
    print("=== Advanced Rendering Pipeline ===\n")
    print("This demonstrates a sophisticated rendering pipeline with:")
    print("  • Deferred rendering with G-Buffer")
    print("  • Shadow mapping")
    print("  • Post-processing effects")
    print("  • Performance monitoring")
    print("  • Multi-stage rendering pipeline")
    
    try:
        renderer = AdvancedRenderer(800, 600)
        renderer.run()
    except Exception as e:
        print(f"✗ Advanced rendering pipeline failed to start: {e}")

if __name__ == "__main__":
    main()
