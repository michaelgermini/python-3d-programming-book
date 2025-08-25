#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Graphics Pipeline Example

Demonstrates a modular graphics rendering pipeline with separate
modules for rendering, materials, and shaders for 3D graphics applications.
"""

import math
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Pipeline Library"
__description__ = "Modular graphics rendering pipeline for 3D applications"

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ShaderType(Enum):
    """Shader types"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"

class RenderMode(Enum):
    """Rendering modes"""
    POINTS = "points"
    LINES = "lines"
    TRIANGLES = "triangles"
    TRIANGLE_STRIP = "triangle_strip"

class BlendMode(Enum):
    """Blending modes"""
    NONE = "none"
    ALPHA = "alpha"
    ADDITIVE = "additive"
    MULTIPLY = "multiply"

# ============================================================================
# VECTOR3 CLASS (SIMPLIFIED)
# ============================================================================

@dataclass
class Vector3:
    """3D Vector class"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def normalize(self) -> 'Vector3':
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

# ============================================================================
# SHADER MODULE
# ============================================================================

class Shader:
    """Shader class for GPU programs"""
    
    def __init__(self, name: str, shader_type: ShaderType):
        self.name = name
        self.shader_type = shader_type
        self.source_code = ""
        self.compiled = False
        self.uniforms: Dict[str, Any] = {}
        self.attributes: Dict[str, Any] = {}
    
    def set_source(self, source: str):
        """Set shader source code"""
        self.source_code = source
        self.compiled = False
    
    def compile(self) -> bool:
        """Compile the shader"""
        if not self.source_code:
            print(f"Error: No source code for shader {self.name}")
            return False
        
        # Simulate compilation
        print(f"Compiling {self.shader_type.value} shader: {self.name}")
        self.compiled = True
        return True
    
    def set_uniform(self, name: str, value: Any):
        """Set uniform variable"""
        self.uniforms[name] = value
    
    def get_uniform(self, name: str) -> Any:
        """Get uniform variable"""
        return self.uniforms.get(name)
    
    def __str__(self) -> str:
        return f"Shader({self.name}, {self.shader_type.value})"

class ShaderProgram:
    """Complete shader program with vertex and fragment shaders"""
    
    def __init__(self, name: str):
        self.name = name
        self.vertex_shader: Optional[Shader] = None
        self.fragment_shader: Optional[Shader] = None
        self.geometry_shader: Optional[Shader] = None
        self.linked = False
        self.uniforms: Dict[str, Any] = {}
    
    def attach_shader(self, shader: Shader):
        """Attach a shader to the program"""
        if shader.shader_type == ShaderType.VERTEX:
            self.vertex_shader = shader
        elif shader.shader_type == ShaderType.FRAGMENT:
            self.fragment_shader = shader
        elif shader.shader_type == ShaderType.GEOMETRY:
            self.geometry_shader = shader
    
    def link(self) -> bool:
        """Link the shader program"""
        if not self.vertex_shader or not self.fragment_shader:
            print(f"Error: Program {self.name} needs vertex and fragment shaders")
            return False
        
        if not self.vertex_shader.compiled or not self.fragment_shader.compiled:
            print(f"Error: Shaders in program {self.name} must be compiled first")
            return False
        
        print(f"Linking shader program: {self.name}")
        self.linked = True
        return True
    
    def use(self):
        """Use this shader program"""
        if not self.linked:
            print(f"Error: Cannot use unlinked program {self.name}")
            return
        
        print(f"Using shader program: {self.name}")
    
    def set_uniform(self, name: str, value: Any):
        """Set uniform variable"""
        self.uniforms[name] = value
    
    def __str__(self) -> str:
        return f"ShaderProgram({self.name})"

# ============================================================================
# MATERIAL MODULE
# ============================================================================

@dataclass
class Texture:
    """Texture class for material textures"""
    name: str
    width: int
    height: int
    channels: int
    data: Optional[bytes] = None
    
    def __str__(self) -> str:
        return f"Texture({self.name}, {self.width}x{self.height})"

@dataclass
class Material:
    """Material class for surface properties"""
    name: str
    diffuse_color: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    specular_color: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    ambient_color: Vector3 = field(default_factory=lambda: Vector3(0.1, 0.1, 0.1))
    shininess: float = 32.0
    opacity: float = 1.0
    diffuse_texture: Optional[Texture] = None
    normal_texture: Optional[Texture] = None
    specular_texture: Optional[Texture] = None
    blend_mode: BlendMode = BlendMode.ALPHA
    
    def apply_to_shader(self, shader: ShaderProgram):
        """Apply material properties to shader"""
        shader.set_uniform("material.diffuse", self.diffuse_color)
        shader.set_uniform("material.specular", self.specular_color)
        shader.set_uniform("material.ambient", self.ambient_color)
        shader.set_uniform("material.shininess", self.shininess)
        shader.set_uniform("material.opacity", self.opacity)
        shader.set_uniform("material.blend_mode", self.blend_mode.value)
        
        if self.diffuse_texture:
            shader.set_uniform("material.has_diffuse_texture", True)
            shader.set_uniform("material.diffuse_texture", self.diffuse_texture)
        else:
            shader.set_uniform("material.has_diffuse_texture", False)
    
    def __str__(self) -> str:
        return f"Material({self.name})"

class MaterialLibrary:
    """Library for managing materials"""
    
    def __init__(self):
        self.materials: Dict[str, Material] = {}
    
    def add_material(self, material: Material):
        """Add material to library"""
        self.materials[material.name] = material
        print(f"Added material: {material.name}")
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get material by name"""
        return self.materials.get(name)
    
    def list_materials(self) -> List[str]:
        """List all material names"""
        return list(self.materials.keys())
    
    def create_default_material(self, name: str) -> Material:
        """Create a default material"""
        material = Material(name)
        self.add_material(material)
        return material

# ============================================================================
# MESH MODULE
# ============================================================================

@dataclass
class Vertex:
    """Vertex with position, normal, and texture coordinates"""
    position: Vector3
    normal: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    tex_coords: Tuple[float, float] = (0.0, 0.0)
    color: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

class Mesh:
    """3D mesh with vertices and indices"""
    
    def __init__(self, name: str):
        self.name = name
        self.vertices: List[Vertex] = []
        self.indices: List[int] = []
        self.material: Optional[Material] = None
        self.vao_id: Optional[int] = None
        self.vbo_id: Optional[int] = None
        self.ebo_id: Optional[int] = None
    
    def add_vertex(self, vertex: Vertex):
        """Add vertex to mesh"""
        self.vertices.append(vertex)
    
    def add_face(self, v1: int, v2: int, v3: int):
        """Add triangular face"""
        self.indices.extend([v1, v2, v3])
    
    def set_material(self, material: Material):
        """Set material for mesh"""
        self.material = material
    
    def create_buffers(self):
        """Create GPU buffers for mesh"""
        print(f"Creating buffers for mesh: {self.name}")
        print(f"  Vertices: {len(self.vertices)}")
        print(f"  Indices: {len(self.indices)}")
        
        # Simulate buffer creation
        self.vao_id = 1
        self.vbo_id = 2
        self.ebo_id = 3
    
    def render(self, shader: ShaderProgram):
        """Render the mesh"""
        if not self.vao_id:
            print(f"Error: Mesh {self.name} buffers not created")
            return
        
        # Apply material
        if self.material:
            self.material.apply_to_shader(shader)
        
        # Simulate rendering
        print(f"Rendering mesh: {self.name}")
        print(f"  Using shader: {shader.name}")
        print(f"  Material: {self.material.name if self.material else 'None'}")
        print(f"  Triangles: {len(self.indices) // 3}")
    
    def __str__(self) -> str:
        return f"Mesh({self.name}, {len(self.vertices)} vertices, {len(self.indices)} indices)"

# ============================================================================
# RENDERER MODULE
# ============================================================================

@dataclass
class Camera:
    """Camera for 3D scene viewing"""
    position: Vector3
    target: Vector3
    up: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    fov: float = 45.0
    aspect_ratio: float = 16.0 / 9.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    def get_view_matrix(self) -> List[List[float]]:
        """Get view matrix (simplified)"""
        # Simplified view matrix calculation
        return [
            [1.0, 0.0, 0.0, -self.position.x],
            [0.0, 1.0, 0.0, -self.position.y],
            [0.0, 0.0, 1.0, -self.position.z],
            [0.0, 0.0, 0.0, 1.0]
        ]
    
    def get_projection_matrix(self) -> List[List[float]]:
        """Get projection matrix (simplified)"""
        # Simplified perspective projection
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        return [
            [f / self.aspect_ratio, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane), -1.0],
            [0.0, 0.0, (2.0 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane), 0.0]
        ]

class Renderer:
    """Main renderer class"""
    
    def __init__(self):
        self.camera: Optional[Camera] = None
        self.meshes: List[Mesh] = []
        self.current_shader: Optional[ShaderProgram] = None
        self.clear_color: Vector3 = Vector3(0.2, 0.3, 0.3)
        self.enabled_features: Dict[str, bool] = {
            "depth_test": True,
            "blending": True,
            "culling": True,
            "wireframe": False
        }
    
    def set_camera(self, camera: Camera):
        """Set the camera"""
        self.camera = camera
        print(f"Camera set: position={camera.position}, target={camera.target}")
    
    def add_mesh(self, mesh: Mesh):
        """Add mesh to renderer"""
        self.meshes.append(mesh)
        print(f"Added mesh to renderer: {mesh.name}")
    
    def use_shader(self, shader: ShaderProgram):
        """Use shader program"""
        self.current_shader = shader
        shader.use()
    
    def clear(self):
        """Clear the screen"""
        print(f"Clearing screen with color: {self.clear_color}")
    
    def render_scene(self):
        """Render all meshes in the scene"""
        if not self.camera:
            print("Error: No camera set")
            return
        
        if not self.current_shader:
            print("Error: No shader program active")
            return
        
        print("\n=== Rendering Scene ===")
        print(f"Camera: {self.camera.position} -> {self.camera.target}")
        print(f"Shader: {self.current_shader.name}")
        print(f"Meshes: {len(self.meshes)}")
        
        # Update shader uniforms
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix()
        
        self.current_shader.set_uniform("viewMatrix", view_matrix)
        self.current_shader.set_uniform("projectionMatrix", proj_matrix)
        self.current_shader.set_uniform("cameraPosition", self.camera.position)
        
        # Render each mesh
        for mesh in self.meshes:
            mesh.render(self.current_shader)
        
        print("Scene rendering complete!")
    
    def enable_feature(self, feature: str, enabled: bool = True):
        """Enable or disable rendering feature"""
        if feature in self.enabled_features:
            self.enabled_features[feature] = enabled
            print(f"{'Enabled' if enabled else 'Disabled'} {feature}")
        else:
            print(f"Unknown feature: {feature}")
    
    def set_clear_color(self, color: Vector3):
        """Set clear color"""
        self.clear_color = color
        print(f"Clear color set to: {color}")

# ============================================================================
# SCENE MODULE
# ============================================================================

class Scene:
    """3D scene containing objects and lights"""
    
    def __init__(self, name: str):
        self.name = name
        self.meshes: List[Mesh] = []
        self.lights: List[Dict[str, Any]] = []
        self.camera: Optional[Camera] = None
        self.ambient_light: Vector3 = Vector3(0.1, 0.1, 0.1)
    
    def add_mesh(self, mesh: Mesh):
        """Add mesh to scene"""
        self.meshes.append(mesh)
        print(f"Added mesh to scene: {mesh.name}")
    
    def add_light(self, position: Vector3, color: Vector3, intensity: float = 1.0):
        """Add light to scene"""
        light = {
            "position": position,
            "color": color,
            "intensity": intensity
        }
        self.lights.append(light)
        print(f"Added light: {position} with color {color}")
    
    def set_camera(self, camera: Camera):
        """Set scene camera"""
        self.camera = camera
        print(f"Set scene camera: {camera.position}")
    
    def render(self, renderer: Renderer):
        """Render the scene"""
        print(f"\n=== Rendering Scene: {self.name} ===")
        
        # Set up renderer
        if self.camera:
            renderer.set_camera(self.camera)
        
        # Add meshes to renderer
        for mesh in self.meshes:
            renderer.add_mesh(mesh)
        
        # Render scene
        renderer.render_scene()
    
    def __str__(self) -> str:
        return f"Scene({self.name}, {len(self.meshes)} meshes, {len(self.lights)} lights)"

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_shaders():
    """Demonstrate shader creation and usage"""
    print("=== Shader System Demo ===\n")
    
    # Create vertex shader
    vertex_shader = Shader("basic_vertex", ShaderType.VERTEX)
    vertex_source = """
    #version 330 core
    layout (location = 0) in vec3 position;
    layout (location = 1) in vec3 normal;
    layout (location = 2) in vec2 texCoord;
    
    uniform mat4 modelMatrix;
    uniform mat4 viewMatrix;
    uniform mat4 projectionMatrix;
    
    out vec3 FragPos;
    out vec3 Normal;
    out vec2 TexCoord;
    
    void main() {
        FragPos = vec3(modelMatrix * vec4(position, 1.0));
        Normal = mat3(transpose(inverse(modelMatrix))) * normal;
        TexCoord = texCoord;
        
        gl_Position = projectionMatrix * viewMatrix * vec4(FragPos, 1.0);
    }
    """
    vertex_shader.set_source(vertex_source)
    vertex_shader.compile()
    
    # Create fragment shader
    fragment_shader = Shader("basic_fragment", ShaderType.FRAGMENT)
    fragment_source = """
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec2 TexCoord;
    
    uniform vec3 lightPos;
    uniform vec3 lightColor;
    uniform vec3 viewPos;
    uniform Material material;
    
    out vec4 FragColor;
    
    void main() {
        vec3 ambient = lightColor * material.ambient;
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = lightColor * diff * material.diffuse;
        
        FragColor = vec4(ambient + diffuse, 1.0);
    }
    """
    fragment_shader.set_source(fragment_source)
    fragment_shader.compile()
    
    # Create shader program
    program = ShaderProgram("basic_program")
    program.attach_shader(vertex_shader)
    program.attach_shader(fragment_shader)
    program.link()
    
    print(f"Created shader program: {program}")
    print()

def demonstrate_materials():
    """Demonstrate material system"""
    print("=== Material System Demo ===\n")
    
    # Create material library
    library = MaterialLibrary()
    
    # Create default material
    default_mat = library.create_default_material("default")
    
    # Create custom materials
    red_material = Material("red_material")
    red_material.diffuse_color = Vector3(1.0, 0.0, 0.0)
    red_material.specular_color = Vector3(0.5, 0.0, 0.0)
    red_material.shininess = 64.0
    library.add_material(red_material)
    
    blue_material = Material("blue_material")
    blue_material.diffuse_color = Vector3(0.0, 0.0, 1.0)
    blue_material.specular_color = Vector3(0.0, 0.0, 0.5)
    blue_material.shininess = 32.0
    library.add_material(blue_material)
    
    # Create texture
    texture = Texture("checkerboard", 256, 256, 3)
    
    # Create textured material
    textured_material = Material("textured_material")
    textured_material.diffuse_texture = texture
    textured_material.diffuse_color = Vector3(1.0, 1.0, 1.0)
    library.add_material(textured_material)
    
    print("Available materials:")
    for name in library.list_materials():
        material = library.get_material(name)
        print(f"  - {name}: {material}")
    print()

def demonstrate_meshes():
    """Demonstrate mesh creation"""
    print("=== Mesh System Demo ===\n")
    
    # Create a simple cube mesh
    cube = Mesh("cube")
    
    # Add vertices (simplified cube)
    vertices = [
        Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1),
        Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(1, 1, 1), Vector3(-1, 1, 1)
    ]
    
    for pos in vertices:
        vertex = Vertex(position=pos, normal=pos.normalize())
        cube.add_vertex(vertex)
    
    # Add faces (triangles)
    faces = [
        (0, 1, 2), (2, 3, 0),  # Front
        (1, 5, 6), (6, 2, 1),  # Right
        (5, 4, 7), (7, 6, 5),  # Back
        (4, 0, 3), (3, 7, 4),  # Left
        (3, 2, 6), (6, 7, 3),  # Top
        (4, 5, 1), (1, 0, 4)   # Bottom
    ]
    
    for face in faces:
        cube.add_face(*face)
    
    cube.create_buffers()
    
    # Create a sphere mesh (simplified)
    sphere = Mesh("sphere")
    
    # Add sphere vertices (simplified)
    for i in range(8):
        angle = i * math.pi / 4
        x = math.cos(angle)
        z = math.sin(angle)
        pos = Vector3(x, 0, z)
        vertex = Vertex(position=pos, normal=pos.normalize())
        sphere.add_vertex(vertex)
    
    sphere.create_buffers()
    
    print(f"Created meshes:")
    print(f"  - {cube}")
    print(f"  - {sphere}")
    print()

def demonstrate_renderer():
    """Demonstrate renderer functionality"""
    print("=== Renderer Demo ===\n")
    
    # Create renderer
    renderer = Renderer()
    
    # Create camera
    camera = Camera(
        position=Vector3(0, 0, -5),
        target=Vector3(0, 0, 0),
        up=Vector3(0, 1, 0),
        fov=45.0,
        aspect_ratio=16.0/9.0
    )
    
    # Create shader program
    program = ShaderProgram("demo_program")
    program.linked = True  # Simulate linking
    
    # Create meshes
    cube = Mesh("demo_cube")
    cube.vertices = [Vertex(Vector3(0, 0, 0))] * 8
    cube.indices = list(range(36))
    cube.create_buffers()
    
    sphere = Mesh("demo_sphere")
    sphere.vertices = [Vertex(Vector3(0, 0, 0))] * 8
    sphere.indices = list(range(24))
    sphere.create_buffers()
    
    # Set up renderer
    renderer.set_camera(camera)
    renderer.use_shader(program)
    renderer.add_mesh(cube)
    renderer.add_mesh(sphere)
    
    # Configure renderer
    renderer.set_clear_color(Vector3(0.1, 0.1, 0.2))
    renderer.enable_feature("depth_test", True)
    renderer.enable_feature("blending", True)
    renderer.enable_feature("culling", True)
    
    # Render
    renderer.render_scene()
    print()

def demonstrate_scene():
    """Demonstrate scene management"""
    print("=== Scene Management Demo ===\n")
    
    # Create scene
    scene = Scene("demo_scene")
    
    # Create camera
    camera = Camera(
        position=Vector3(0, 2, -5),
        target=Vector3(0, 0, 0),
        up=Vector3(0, 1, 0)
    )
    scene.set_camera(camera)
    
    # Create meshes
    cube = Mesh("scene_cube")
    cube.vertices = [Vertex(Vector3(0, 0, 0))] * 8
    cube.indices = list(range(36))
    cube.create_buffers()
    
    sphere = Mesh("scene_sphere")
    sphere.vertices = [Vertex(Vector3(0, 0, 0))] * 8
    sphere.indices = list(range(24))
    sphere.create_buffers()
    
    # Add meshes to scene
    scene.add_mesh(cube)
    scene.add_mesh(sphere)
    
    # Add lights
    scene.add_light(Vector3(2, 2, 2), Vector3(1, 1, 1), 1.0)
    scene.add_light(Vector3(-2, 1, 1), Vector3(0.5, 0.5, 1.0), 0.5)
    
    # Create renderer and render scene
    renderer = Renderer()
    program = ShaderProgram("scene_program")
    program.linked = True
    renderer.use_shader(program)
    
    scene.render(renderer)
    print()

def demonstrate_complete_pipeline():
    """Demonstrate complete graphics pipeline"""
    print("=== Complete Graphics Pipeline Demo ===\n")
    
    # 1. Create shaders
    print("1. Creating shaders...")
    vertex_shader = Shader("pipeline_vertex", ShaderType.VERTEX)
    vertex_shader.set_source("#version 330 core\nvoid main() { gl_Position = vec4(0,0,0,1); }")
    vertex_shader.compile()
    
    fragment_shader = Shader("pipeline_fragment", ShaderType.FRAGMENT)
    fragment_shader.set_source("#version 330 core\nout vec4 FragColor;\nvoid main() { FragColor = vec4(1,1,1,1); }")
    fragment_shader.compile()
    
    program = ShaderProgram("pipeline_program")
    program.attach_shader(vertex_shader)
    program.attach_shader(fragment_shader)
    program.link()
    
    # 2. Create materials
    print("\n2. Creating materials...")
    material_lib = MaterialLibrary()
    red_mat = Material("red")
    red_mat.diffuse_color = Vector3(1, 0, 0)
    material_lib.add_material(red_mat)
    
    # 3. Create meshes
    print("\n3. Creating meshes...")
    mesh = Mesh("pipeline_mesh")
    mesh.vertices = [Vertex(Vector3(0, 0, 0))] * 4
    mesh.indices = [0, 1, 2, 2, 3, 0]
    mesh.material = red_mat
    mesh.create_buffers()
    
    # 4. Create scene
    print("\n4. Creating scene...")
    scene = Scene("pipeline_scene")
    scene.add_mesh(mesh)
    scene.set_camera(Camera(Vector3(0, 0, -3), Vector3(0, 0, 0)))
    scene.add_light(Vector3(1, 1, 1), Vector3(1, 1, 1))
    
    # 5. Create renderer and render
    print("\n5. Rendering...")
    renderer = Renderer()
    renderer.use_shader(program)
    scene.render(renderer)
    
    print("\nPipeline demonstration complete!")
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate graphics pipeline"""
    print("=== Graphics Pipeline Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_shaders()
    demonstrate_materials()
    demonstrate_meshes()
    demonstrate_renderer()
    demonstrate_scene()
    demonstrate_complete_pipeline()
    
    print("="*60)
    print("Graphics Pipeline demo completed successfully!")
    print("\nKey features:")
    print("✓ Shader system: Vertex, fragment, and geometry shaders")
    print("✓ Material system: Surface properties and textures")
    print("✓ Mesh system: 3D geometry with vertices and indices")
    print("✓ Renderer: Complete rendering pipeline")
    print("✓ Scene management: Objects, lights, and cameras")
    print("✓ Modular design: Separate modules for different components")

if __name__ == "__main__":
    main()
