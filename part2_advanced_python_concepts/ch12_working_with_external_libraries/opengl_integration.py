"""
Chapter 12: Working with External Libraries - OpenGL Integration
===============================================================

This module demonstrates how to integrate OpenGL with Python for 3D graphics
applications, including context management, shader programs, and rendering
pipelines.

Key Concepts:
- OpenGL context management
- Shader program creation and management
- Vertex buffer objects (VBOs)
- Vertex array objects (VAOs)
- Texture management
- Rendering pipeline setup
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class Vector3D:
    """3D vector for OpenGL operations."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def to_array(self) -> np.ndarray:
        """Convert to NumPy array for OpenGL."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Color:
    """Color for OpenGL operations."""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def __str__(self):
        return f"Color({self.r:.3f}, {self.g:.3f}, {self.b:.3f}, {self.a:.3f})"
    
    def to_array(self) -> np.ndarray:
        """Convert to NumPy array for OpenGL."""
        return np.array([self.r, self.g, self.b, self.a], dtype=np.float32)


class OpenGLContext:
    """Simulated OpenGL context for demonstration."""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.context_active = False
        self.shaders = {}
        self.vbos = {}
        self.vaos = {}
        self.textures = {}
        self.current_program = None
        
    def __enter__(self):
        """Initialize OpenGL context."""
        print(f"  🖥️  Initializing OpenGL context ({self.width}x{self.height})")
        self.context_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up OpenGL context."""
        if self.context_active:
            print(f"  🖥️  Cleaning up OpenGL context")
            self.cleanup()
            self.context_active = False
    
    def cleanup(self):
        """Clean up OpenGL resources."""
        print(f"  🗑️  Deleting {len(self.shaders)} shaders")
        print(f"  🗑️  Deleting {len(self.vbos)} VBOs")
        print(f"  🗑️  Deleting {len(self.vaos)} VAOs")
        print(f"  🗑️  Deleting {len(self.textures)} textures")
        
        self.shaders.clear()
        self.vbos.clear()
        self.vaos.clear()
        self.textures.clear()


class ShaderProgram:
    """OpenGL shader program wrapper."""
    
    def __init__(self, context: OpenGLContext, name: str):
        self.context = context
        self.name = name
        self.program_id = None
        self.vertex_shader = None
        self.fragment_shader = None
        self.compiled = False
    
    def create_vertex_shader(self, source: str) -> 'ShaderProgram':
        """Create vertex shader."""
        print(f"  🔧 Creating vertex shader for {self.name}")
        self.vertex_shader = source
        return self
    
    def create_fragment_shader(self, source: str) -> 'ShaderProgram':
        """Create fragment shader."""
        print(f"  🔧 Creating fragment shader for {self.name}")
        self.fragment_shader = source
        return self
    
    def compile(self) -> 'ShaderProgram':
        """Compile shader program."""
        if not self.vertex_shader or not self.fragment_shader:
            raise ValueError("Both vertex and fragment shaders must be set")
        
        print(f"  🔧 Compiling shader program: {self.name}")
        
        # Simulate compilation process
        self.program_id = len(self.context.shaders) + 1
        self.compiled = True
        
        # Store in context
        self.context.shaders[self.name] = self
        
        print(f"  ✅ Shader program compiled successfully (ID: {self.program_id})")
        return self
    
    def use(self):
        """Use this shader program."""
        if not self.compiled:
            raise RuntimeError("Shader program not compiled")
        
        self.context.current_program = self
        print(f"  🎨 Using shader program: {self.name}")
    
    def set_uniform(self, name: str, value: Any):
        """Set uniform variable in shader."""
        if not self.compiled:
            raise RuntimeError("Shader program not compiled")
        
        print(f"  📝 Setting uniform '{name}' = {value} in {self.name}")


class VertexBufferObject:
    """OpenGL Vertex Buffer Object wrapper."""
    
    def __init__(self, context: OpenGLContext, name: str):
        self.context = context
        self.name = name
        self.vbo_id = None
        self.data = None
        self.size = 0
    
    def bind(self):
        """Bind this VBO."""
        print(f"  📦 Binding VBO: {self.name}")
    
    def unbind(self):
        """Unbind this VBO."""
        print(f"  📦 Unbinding VBO: {self.name}")
    
    def set_data(self, data: np.ndarray, usage: str = "STATIC_DRAW"):
        """Set vertex data."""
        self.data = data
        self.size = len(data)
        self.vbo_id = len(self.context.vbos) + 1
        
        # Store in context
        self.context.vbos[self.name] = self
        
        print(f"  📦 Set VBO data: {self.name} ({self.size} vertices, {usage})")
    
    def get_size(self) -> int:
        """Get buffer size in bytes."""
        return self.size * self.data.itemsize if self.data is not None else 0


class VertexArrayObject:
    """OpenGL Vertex Array Object wrapper."""
    
    def __init__(self, context: OpenGLContext, name: str):
        self.context = context
        self.name = name
        self.vao_id = None
        self.attributes = {}
    
    def bind(self):
        """Bind this VAO."""
        print(f"  📋 Binding VAO: {self.name}")
    
    def unbind(self):
        """Unbind this VAO."""
        print(f"  📋 Unbinding VAO: {self.name}")
    
    def set_attribute(self, location: int, vbo: VertexBufferObject, 
                     size: int, data_type: str = "FLOAT", 
                     normalized: bool = False, stride: int = 0, offset: int = 0):
        """Set vertex attribute."""
        self.attributes[location] = {
            'vbo': vbo,
            'size': size,
            'type': data_type,
            'normalized': normalized,
            'stride': stride,
            'offset': offset
        }
        
        self.vao_id = len(self.context.vaos) + 1
        self.context.vaos[self.name] = self
        
        print(f"  📋 Set attribute {location} in VAO {self.name}: "
              f"size={size}, type={data_type}")


class Texture:
    """OpenGL Texture wrapper."""
    
    def __init__(self, context: OpenGLContext, name: str):
        self.context = context
        self.name = name
        self.texture_id = None
        self.width = 0
        self.height = 0
        self.format = None
    
    def bind(self, unit: int = 0):
        """Bind texture to texture unit."""
        print(f"  🖼️  Binding texture {self.name} to unit {unit}")
    
    def set_data(self, width: int, height: int, data: np.ndarray, 
                 format: str = "RGBA", data_type: str = "UNSIGNED_BYTE"):
        """Set texture data."""
        self.width = width
        self.height = height
        self.format = format
        self.texture_id = len(self.context.textures) + 1
        
        # Store in context
        self.context.textures[self.name] = self
        
        print(f"  🖼️  Set texture data: {self.name} ({width}x{height}, {format})")
    
    def set_parameters(self, min_filter: str = "LINEAR", 
                      mag_filter: str = "LINEAR",
                      wrap_s: str = "CLAMP_TO_EDGE",
                      wrap_t: str = "CLAMP_TO_EDGE"):
        """Set texture parameters."""
        print(f"  🖼️  Set texture parameters for {self.name}: "
              f"min={min_filter}, mag={mag_filter}, wrap=({wrap_s}, {wrap_t})")


class OpenGLRenderer:
    """OpenGL renderer for 3D graphics."""
    
    def __init__(self, context: OpenGLContext):
        self.context = context
        self.clear_color = Color(0.2, 0.3, 0.3, 1.0)
        self.viewport_set = False
    
    def set_clear_color(self, color: Color):
        """Set clear color."""
        self.clear_color = color
        print(f"  🎨 Set clear color: {color}")
    
    def set_viewport(self, x: int, y: int, width: int, height: int):
        """Set viewport."""
        self.viewport_set = True
        print(f"  📐 Set viewport: ({x}, {y}, {width}, {height})")
    
    def clear(self, buffers: List[str] = None):
        """Clear specified buffers."""
        if buffers is None:
            buffers = ["COLOR_BUFFER_BIT", "DEPTH_BUFFER_BIT"]
        
        print(f"  🧹 Clearing buffers: {', '.join(buffers)}")
    
    def draw_arrays(self, mode: str, first: int, count: int):
        """Draw arrays."""
        print(f"  🎨 Drawing {count} vertices using {mode} (starting at {first})")
    
    def draw_elements(self, mode: str, count: int, data_type: str = "UNSIGNED_INT", offset: int = 0):
        """Draw elements using index buffer."""
        print(f"  🎨 Drawing {count} elements using {mode} ({data_type}, offset={offset})")


class MeshRenderer:
    """Mesh renderer using OpenGL."""
    
    def __init__(self, context: OpenGLContext):
        self.context = context
        self.renderer = OpenGLRenderer(context)
        self.meshes = {}
    
    def create_mesh(self, name: str, vertices: List[Vector3D], 
                   indices: List[int] = None, colors: List[Color] = None) -> Dict[str, Any]:
        """Create a mesh with OpenGL buffers."""
        print(f"  📦 Creating mesh: {name}")
        
        # Create VBO for vertices
        vertex_vbo = VertexBufferObject(self.context, f"{name}_vertices")
        vertex_data = np.array([v.to_array() for v in vertices], dtype=np.float32)
        vertex_vbo.set_data(vertex_data)
        
        mesh_data = {
            'name': name,
            'vertex_vbo': vertex_vbo,
            'vertex_count': len(vertices)
        }
        
        # Create VBO for colors if provided
        if colors:
            color_vbo = VertexBufferObject(self.context, f"{name}_colors")
            color_data = np.array([c.to_array() for c in colors], dtype=np.float32)
            color_vbo.set_data(color_data)
            mesh_data['color_vbo'] = color_vbo
        
        # Create VBO for indices if provided
        if indices:
            index_vbo = VertexBufferObject(self.context, f"{name}_indices")
            index_data = np.array(indices, dtype=np.uint32)
            index_vbo.set_data(index_data)
            mesh_data['index_vbo'] = index_vbo
            mesh_data['index_count'] = len(indices)
        
        # Create VAO
        vao = VertexArrayObject(self.context, f"{name}_vao")
        vao.bind()
        
        # Set vertex attributes
        vertex_vbo.bind()
        vao.set_attribute(0, vertex_vbo, 3, "FLOAT")  # Position
        
        if colors:
            color_vbo.bind()
            vao.set_attribute(1, color_vbo, 4, "FLOAT")  # Color
        
        vao.unbind()
        mesh_data['vao'] = vao
        
        self.meshes[name] = mesh_data
        return mesh_data
    
    def render_mesh(self, name: str, shader: ShaderProgram, 
                   transform_matrix: np.ndarray = None):
        """Render a mesh."""
        if name not in self.meshes:
            raise ValueError(f"Mesh '{name}' not found")
        
        mesh = self.meshes[name]
        print(f"  🎨 Rendering mesh: {name}")
        
        # Use shader
        shader.use()
        
        # Set transform matrix if provided
        if transform_matrix is not None:
            shader.set_uniform("modelMatrix", transform_matrix)
        
        # Bind VAO and render
        mesh['vao'].bind()
        
        if 'index_vbo' in mesh:
            # Use indexed rendering
            self.renderer.draw_elements("TRIANGLES", mesh['index_count'])
        else:
            # Use array rendering
            self.renderer.draw_arrays("TRIANGLES", 0, mesh['vertex_count'])
        
        mesh['vao'].unbind()


class ShaderLibrary:
    """Library of common shader programs."""
    
    @staticmethod
    def create_basic_shader(context: OpenGLContext) -> ShaderProgram:
        """Create basic vertex/fragment shader program."""
        vertex_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec4 aColor;
        
        out vec4 vertexColor;
        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 projectionMatrix;
        
        void main()
        {
            gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(aPos, 1.0);
            vertexColor = aColor;
        }
        """
        
        fragment_source = """
        #version 330 core
        in vec4 vertexColor;
        out vec4 FragColor;
        
        void main()
        {
            FragColor = vertexColor;
        }
        """
        
        shader = ShaderProgram(context, "basic")
        shader.create_vertex_shader(vertex_source)
        shader.create_fragment_shader(fragment_source)
        shader.compile()
        
        return shader
    
    @staticmethod
    def create_phong_shader(context: OpenGLContext) -> ShaderProgram:
        """Create Phong lighting shader program."""
        vertex_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        
        out vec3 FragPos;
        out vec3 Normal;
        
        uniform mat4 modelMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 projectionMatrix;
        uniform mat3 normalMatrix;
        
        void main()
        {
            FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
            Normal = normalMatrix * aNormal;
            gl_Position = projectionMatrix * viewMatrix * vec4(FragPos, 1.0);
        }
        """
        
        fragment_source = """
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        
        out vec4 FragColor;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        
        void main()
        {
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            vec3 result = (diffuse + 0.1) * objectColor;
            FragColor = vec4(result, 1.0);
        }
        """
        
        shader = ShaderProgram(context, "phong")
        shader.create_vertex_shader(vertex_source)
        shader.create_fragment_shader(fragment_source)
        shader.compile()
        
        return shader


# Example Usage and Demonstration
def demonstrate_opengl_integration():
    """Demonstrates OpenGL integration for 3D graphics."""
    print("=== OpenGL Integration for 3D Graphics ===\n")
    
    # Create OpenGL context
    with OpenGLContext(1024, 768) as context:
        # Create renderer
        renderer = MeshRenderer(context)
        
        # Create shaders
        print("=== Shader Creation ===")
        basic_shader = ShaderLibrary.create_basic_shader(context)
        phong_shader = ShaderLibrary.create_phong_shader(context)
        
        # Create a simple triangle mesh
        print("\n=== Mesh Creation ===")
        vertices = [
            Vector3D(-0.5, -0.5, 0.0),
            Vector3D(0.5, -0.5, 0.0),
            Vector3D(0.0, 0.5, 0.0)
        ]
        
        colors = [
            Color(1.0, 0.0, 0.0, 1.0),  # Red
            Color(0.0, 1.0, 0.0, 1.0),  # Green
            Color(0.0, 0.0, 1.0, 1.0)   # Blue
        ]
        
        triangle_mesh = renderer.create_mesh("triangle", vertices, colors=colors)
        
        # Create a cube mesh
        cube_vertices = [
            # Front face
            Vector3D(-0.5, -0.5, 0.5), Vector3D(0.5, -0.5, 0.5), Vector3D(0.5, 0.5, 0.5), Vector3D(-0.5, 0.5, 0.5),
            # Back face
            Vector3D(-0.5, -0.5, -0.5), Vector3D(0.5, -0.5, -0.5), Vector3D(0.5, 0.5, -0.5), Vector3D(-0.5, 0.5, -0.5)
        ]
        
        cube_indices = [
            # Front
            0, 1, 2, 2, 3, 0,
            # Back
            4, 6, 5, 6, 4, 7,
            # Left
            4, 0, 3, 3, 7, 4,
            # Right
            1, 5, 6, 6, 2, 1,
            # Top
            3, 2, 6, 6, 7, 3,
            # Bottom
            4, 5, 1, 1, 0, 4
        ]
        
        cube_colors = [Color(0.8, 0.8, 0.8, 1.0)] * 8  # Gray for all vertices
        cube_mesh = renderer.create_mesh("cube", cube_vertices, cube_indices, cube_colors)
        
        # Create texture
        print("\n=== Texture Creation ===")
        texture = Texture(context, "checkerboard")
        
        # Create checkerboard pattern
        width, height = 64, 64
        texture_data = np.zeros((height, width, 4), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                if (x // 8 + y // 8) % 2 == 0:
                    texture_data[y, x] = [255, 255, 255, 255]  # White
                else:
                    texture_data[y, x] = [0, 0, 0, 255]  # Black
        
        texture.set_data(width, height, texture_data)
        texture.set_parameters()
        
        # Rendering loop simulation
        print("\n=== Rendering Loop ===")
        
        # Set up renderer
        renderer.renderer.set_clear_color(Color(0.2, 0.3, 0.3, 1.0))
        renderer.renderer.set_viewport(0, 0, 1024, 768)
        
        # Simulate rendering frames
        for frame in range(3):
            print(f"\n--- Frame {frame + 1} ---")
            
            # Clear buffers
            renderer.renderer.clear()
            
            # Render triangle
            renderer.render_mesh("triangle", basic_shader)
            
            # Render cube with transformation
            transform = np.eye(4, dtype=np.float32)
            transform[0, 0] = 0.5  # Scale down
            transform[1, 1] = 0.5
            transform[2, 2] = 0.5
            transform[0, 3] = 1.0  # Translate right
            
            renderer.render_mesh("cube", phong_shader, transform)
            
            print(f"  📊 Frame {frame + 1} rendered successfully")
        
        print(f"\n=== OpenGL Statistics ===")
        print(f"  📦 VBOs created: {len(context.vbos)}")
        print(f"  📋 VAOs created: {len(context.vaos)}")
        print(f"  🔧 Shaders created: {len(context.shaders)}")
        print(f"  🖼️  Textures created: {len(context.textures)}")


if __name__ == "__main__":
    demonstrate_opengl_integration()
