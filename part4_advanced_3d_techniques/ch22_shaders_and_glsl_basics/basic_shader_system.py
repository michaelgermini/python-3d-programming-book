"""
Chapter 22: Shaders and GLSL Basics - Basic Shader System
========================================================

This module demonstrates basic shader system implementation for 3D graphics.

Key Concepts:
- Vertex and fragment shader management
- Uniform variables and attribute handling
- Shader program creation and binding
- Basic lighting calculations in GLSL
"""

import os
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import OpenGL.GL as gl


class ShaderType(Enum):
    """Shader type enumeration."""
    VERTEX = gl.GL_VERTEX_SHADER
    FRAGMENT = gl.GL_FRAGMENT_SHADER
    GEOMETRY = gl.GL_GEOMETRY_SHADER
    COMPUTE = gl.GL_COMPUTE_SHADER


@dataclass
class ShaderInfo:
    """Information about a shader."""
    shader_id: int
    shader_type: ShaderType
    source_code: str
    compiled: bool = False
    error_log: str = ""


class Shader:
    """Represents a single shader (vertex, fragment, etc.)."""
    
    def __init__(self, shader_type: ShaderType, source_code: str = ""):
        self.shader_type = shader_type
        self.source_code = source_code
        self.shader_id = 0
        self.compiled = False
        self.error_log = ""
        
    def compile(self) -> bool:
        """Compile the shader."""
        if not self.source_code:
            self.error_log = "No source code provided"
            return False
            
        # Create shader
        self.shader_id = gl.glCreateShader(self.shader_type.value)
        
        # Set source code
        gl.glShaderSource(self.shader_id, self.source_code)
        
        # Compile shader
        gl.glCompileShader(self.shader_id)
        
        # Check compilation status
        success = gl.glGetShaderiv(self.shader_id, gl.GL_COMPILE_STATUS)
        if not success:
            self.error_log = gl.glGetShaderInfoLog(self.shader_id).decode('utf-8')
            gl.glDeleteShader(self.shader_id)
            self.shader_id = 0
            return False
            
        self.compiled = True
        return True
        
    def cleanup(self):
        """Clean up shader resources."""
        if self.shader_id:
            gl.glDeleteShader(self.shader_id)
            self.shader_id = 0
            self.compiled = False


class ShaderProgram:
    """Represents a complete shader program with vertex and fragment shaders."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.program_id = 0
        self.shaders: List[Shader] = []
        self.linked = False
        self.error_log = ""
        self.uniform_locations: Dict[str, int] = {}
        self.attribute_locations: Dict[str, int] = {}
        
    def add_shader(self, shader: Shader) -> bool:
        """Add a shader to the program."""
        if not shader.compiled:
            if not shader.compile():
                self.error_log = f"Failed to compile shader: {shader.error_log}"
                return False
                
        self.shaders.append(shader)
        return True
        
    def link(self) -> bool:
        """Link the shader program."""
        if not self.shaders:
            self.error_log = "No shaders added to program"
            return False
            
        # Create program
        self.program_id = gl.glCreateProgram()
        
        # Attach shaders
        for shader in self.shaders:
            gl.glAttachShader(self.program_id, shader.shader_id)
            
        # Link program
        gl.glLinkProgram(self.program_id)
        
        # Check linking status
        success = gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS)
        if not success:
            self.error_log = gl.glGetProgramInfoLog(self.program_id).decode('utf-8')
            gl.glDeleteProgram(self.program_id)
            self.program_id = 0
            return False
            
        self.linked = True
        
        # Cache uniform and attribute locations
        self._cache_locations()
        
        return True
        
    def _cache_locations(self):
        """Cache uniform and attribute locations for performance."""
        # Get uniform locations
        uniform_count = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_UNIFORMS)
        for i in range(uniform_count):
            name = gl.glGetActiveUniform(self.program_id, i)[0].decode('utf-8')
            location = gl.glGetUniformLocation(self.program_id, name)
            self.uniform_locations[name] = location
            
        # Get attribute locations
        attribute_count = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_ATTRIBUTES)
        for i in range(attribute_count):
            name = gl.glGetActiveAttrib(self.program_id, i)[0].decode('utf-8')
            location = gl.glGetAttribLocation(self.program_id, name)
            self.attribute_locations[name] = location
            
    def use(self):
        """Use this shader program."""
        if self.linked:
            gl.glUseProgram(self.program_id)
            
    def stop(self):
        """Stop using this shader program."""
        gl.glUseProgram(0)
        
    def get_uniform_location(self, name: str) -> int:
        """Get uniform location by name."""
        return self.uniform_locations.get(name, -1)
        
    def get_attribute_location(self, name: str) -> int:
        """Get attribute location by name."""
        return self.attribute_locations.get(name, -1)
        
    def set_uniform_1f(self, name: str, value: float):
        """Set uniform float value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniform1f(location, value)
            
    def set_uniform_2f(self, name: str, x: float, y: float):
        """Set uniform vec2 value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniform2f(location, x, y)
            
    def set_uniform_3f(self, name: str, x: float, y: float, z: float):
        """Set uniform vec3 value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniform3f(location, x, y, z)
            
    def set_uniform_4f(self, name: str, x: float, y: float, z: float, w: float):
        """Set uniform vec4 value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniform4f(location, x, y, z, w)
            
    def set_uniform_matrix4fv(self, name: str, matrix: np.ndarray, transpose: bool = False):
        """Set uniform mat4 value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniformMatrix4fv(location, 1, transpose, matrix.astype(np.float32))
            
    def set_uniform_1i(self, name: str, value: int):
        """Set uniform integer value."""
        location = self.get_uniform_location(name)
        if location != -1:
            gl.glUniform1i(location, value)
            
    def cleanup(self):
        """Clean up shader program resources."""
        if self.program_id:
            gl.glDeleteProgram(self.program_id)
            self.program_id = 0
            self.linked = False
            
        for shader in self.shaders:
            shader.cleanup()
        self.shaders.clear()


class ShaderPreset:
    """Predefined shader presets for common use cases."""
    
    @staticmethod
    def create_basic_vertex_shader() -> str:
        """Create basic vertex shader."""
        return """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        void main() {
            FragPos = vec3(model * vec4(position, 1.0));
            Normal = mat3(transpose(inverse(model))) * normal;
            TexCoord = texCoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
    @staticmethod
    def create_basic_fragment_shader() -> str:
        """Create basic fragment shader."""
        return """
        #version 330 core
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        uniform sampler2D diffuseTexture;
        
        out vec4 FragColor;
        
        void main() {
            // Ambient lighting
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Combine lighting
            vec3 result = (ambient + diffuse) * objectColor;
            
            // Apply texture
            vec4 texColor = texture(diffuseTexture, TexCoord);
            result *= texColor.rgb;
            
            FragColor = vec4(result, 1.0);
        }
        """
        
    @staticmethod
    def create_phong_vertex_shader() -> str:
        """Create Phong lighting vertex shader."""
        return """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        out vec3 ViewPos;
        
        void main() {
            FragPos = vec3(model * vec4(position, 1.0));
            Normal = mat3(transpose(inverse(model))) * normal;
            TexCoord = texCoord;
            ViewPos = vec3(inverse(view)[3]);
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
    @staticmethod
    def create_phong_fragment_shader() -> str:
        """Create Phong lighting fragment shader."""
        return """
        #version 330 core
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        in vec3 ViewPos;
        
        uniform vec3 lightPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        uniform float shininess;
        uniform sampler2D diffuseTexture;
        
        out vec4 FragColor;
        
        void main() {
            // Ambient lighting
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular lighting
            vec3 viewDir = normalize(ViewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
            vec3 specular = spec * lightColor;
            
            // Combine lighting
            vec3 result = (ambient + diffuse + specular) * objectColor;
            
            // Apply texture
            vec4 texColor = texture(diffuseTexture, TexCoord);
            result *= texColor.rgb;
            
            FragColor = vec4(result, 1.0);
        }
        """
        
    @staticmethod
    def create_unlit_vertex_shader() -> str:
        """Create unlit vertex shader."""
        return """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 2) in vec2 texCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
    @staticmethod
    def create_unlit_fragment_shader() -> str:
        """Create unlit fragment shader."""
        return """
        #version 330 core
        
        in vec2 TexCoord;
        
        uniform vec3 objectColor;
        uniform sampler2D diffuseTexture;
        
        out vec4 FragColor;
        
        void main() {
            vec4 texColor = texture(diffuseTexture, TexCoord);
            FragColor = vec4(objectColor * texColor.rgb, texColor.a);
        }
        """


class ShaderManager:
    """Manages shader programs and their resources."""
    
    def __init__(self):
        self.shader_programs: Dict[str, ShaderProgram] = {}
        self.active_program: Optional[ShaderProgram] = None
        
    def create_shader_program(self, name: str, vertex_source: str, fragment_source: str) -> Optional[ShaderProgram]:
        """Create a shader program from source code."""
        # Create vertex shader
        vertex_shader = Shader(ShaderType.VERTEX, vertex_source)
        
        # Create fragment shader
        fragment_shader = Shader(ShaderType.FRAGMENT, fragment_source)
        
        # Create program
        program = ShaderProgram(name)
        
        # Add shaders to program
        if not program.add_shader(vertex_shader):
            return None
        if not program.add_shader(fragment_shader):
            return None
            
        # Link program
        if not program.link():
            return None
            
        # Store program
        self.shader_programs[name] = program
        return program
        
    def create_preset_program(self, preset_name: str) -> Optional[ShaderProgram]:
        """Create shader program from preset."""
        presets = {
            "basic": (ShaderPreset.create_basic_vertex_shader(), 
                     ShaderPreset.create_basic_fragment_shader()),
            "phong": (ShaderPreset.create_phong_vertex_shader(), 
                     ShaderPreset.create_phong_fragment_shader()),
            "unlit": (ShaderPreset.create_unlit_vertex_shader(), 
                     ShaderPreset.create_unlit_fragment_shader())
        }
        
        if preset_name in presets:
            vertex_source, fragment_source = presets[preset_name]
            return self.create_shader_program(preset_name, vertex_source, fragment_source)
        else:
            print(f"Unknown preset: {preset_name}")
            return None
            
    def get_shader_program(self, name: str) -> Optional[ShaderProgram]:
        """Get shader program by name."""
        return self.shader_programs.get(name)
        
    def use_shader_program(self, name: str) -> bool:
        """Use shader program by name."""
        program = self.get_shader_program(name)
        if program:
            program.use()
            self.active_program = program
            return True
        return False
        
    def get_active_program(self) -> Optional[ShaderProgram]:
        """Get currently active shader program."""
        return self.active_program
        
    def cleanup(self):
        """Clean up all shader programs."""
        for program in self.shader_programs.values():
            program.cleanup()
        self.shader_programs.clear()
        self.active_program = None


def demonstrate_basic_shader_system():
    """Demonstrate basic shader system functionality."""
    print("=== Basic Shader System Demonstration ===\n")
    
    # Create shader manager
    manager = ShaderManager()
    
    # Create preset shader programs
    print("1. Creating preset shader programs:")
    
    basic_program = manager.create_preset_program("basic")
    if basic_program:
        print(f"   - Created basic shader program: {basic_program.name}")
        print(f"     Uniforms: {list(basic_program.uniform_locations.keys())}")
        print(f"     Attributes: {list(basic_program.attribute_locations.keys())}")
        
    phong_program = manager.create_preset_program("phong")
    if phong_program:
        print(f"   - Created phong shader program: {phong_program.name}")
        print(f"     Uniforms: {list(phong_program.uniform_locations.keys())}")
        
    unlit_program = manager.create_preset_program("unlit")
    if unlit_program:
        print(f"   - Created unlit shader program: {unlit_program.name}")
        print(f"     Uniforms: {list(unlit_program.uniform_locations.keys())}")
        
    # Test shader program operations
    print("\n2. Testing shader program operations:")
    
    # Use basic program
    if manager.use_shader_program("basic"):
        print("   - Successfully activated basic shader program")
        
        # Set uniform values
        basic_program.set_uniform_3f("lightPos", 2.0, 2.0, 2.0)
        basic_program.set_uniform_3f("lightColor", 1.0, 1.0, 1.0)
        basic_program.set_uniform_3f("objectColor", 0.8, 0.2, 0.2)
        print("   - Set uniform values for lighting")
        
    # Test uniform location caching
    print("\n3. Testing uniform location caching:")
    
    for program_name in ["basic", "phong", "unlit"]:
        program = manager.get_shader_program(program_name)
        if program:
            print(f"   - {program_name} program uniforms:")
            for uniform_name, location in program.uniform_locations.items():
                print(f"     {uniform_name}: location {location}")
                
    # Test shader compilation error handling
    print("\n4. Testing shader compilation error handling:")
    
    # Create invalid shader
    invalid_vertex = Shader(ShaderType.VERTEX, "invalid glsl code")
    if not invalid_vertex.compile():
        print(f"   - Invalid vertex shader compilation failed as expected")
        print(f"     Error: {invalid_vertex.error_log[:50]}...")
        
    # Test program linking error handling
    print("\n5. Testing program linking:")
    
    # Create program with mismatched shaders
    test_program = ShaderProgram("test")
    vertex_shader = Shader(ShaderType.VERTEX, ShaderPreset.create_basic_vertex_shader())
    fragment_shader = Shader(ShaderType.FRAGMENT, "invalid fragment shader")
    
    test_program.add_shader(vertex_shader)
    test_program.add_shader(fragment_shader)
    
    if not test_program.link():
        print(f"   - Program linking failed as expected")
        print(f"     Error: {test_program.error_log[:50]}...")
        
    # Cleanup
    print("\n6. Cleaning up shader resources...")
    manager.cleanup()
    test_program.cleanup()
    print("   - All shader resources cleaned up")


if __name__ == "__main__":
    demonstrate_basic_shader_system()
