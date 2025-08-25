"""
Chapter 22: Shaders and GLSL Basics - Shader Compilation
=======================================================

This module demonstrates shader compilation, linking, and error handling.

Key Concepts:
- Shader compilation and linking process
- Error handling and validation
- Shader debugging techniques
- Performance optimization and validation
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import OpenGL.GL as gl


class ShaderErrorType(Enum):
    """Shader error type enumeration."""
    COMPILATION = "compilation"
    LINKING = "linking"
    VALIDATION = "validation"
    RUNTIME = "runtime"


@dataclass
class ShaderError:
    """Represents a shader error with details."""
    error_type: ShaderErrorType
    shader_id: int
    line_number: int
    message: str
    source_line: str = ""
    severity: str = "error"


class ShaderValidator:
    """Validates shader code and provides error checking."""
    
    def __init__(self):
        self.errors: List[ShaderError] = []
        self.warnings: List[ShaderError] = []
        
    def validate_vertex_shader(self, source_code: str) -> bool:
        """Validate vertex shader source code."""
        self.errors.clear()
        self.warnings.clear()
        
        # Check for required elements
        if not self._check_version_declaration(source_code):
            self.errors.append(ShaderError(
                ShaderErrorType.COMPILATION, 0, 1,
                "Missing #version declaration"
            ))
            
        if not self._check_main_function(source_code):
            self.errors.append(ShaderError(
                ShaderErrorType.COMPILATION, 0, 1,
                "Missing main() function"
            ))
            
        if not self._check_gl_position_assignment(source_code):
            self.errors.append(ShaderError(
                ShaderErrorType.COMPILATION, 0, 1,
                "Missing gl_Position assignment in main()"
            ))
            
        # Check for common issues
        self._check_uniform_declarations(source_code)
        self._check_attribute_declarations(source_code)
        self._check_output_variables(source_code)
        
        return len(self.errors) == 0
        
    def validate_fragment_shader(self, source_code: str) -> bool:
        """Validate fragment shader source code."""
        self.errors.clear()
        self.warnings.clear()
        
        # Check for required elements
        if not self._check_version_declaration(source_code):
            self.errors.append(ShaderError(
                ShaderErrorType.COMPILATION, 0, 1,
                "Missing #version declaration"
            ))
            
        if not self._check_main_function(source_code):
            self.errors.append(ShaderError(
                ShaderErrorType.COMPILATION, 0, 1,
                "Missing main() function"
            ))
            
        # Check for common issues
        self._check_uniform_declarations(source_code)
        self._check_input_variables(source_code)
        self._check_output_variables(source_code)
        
        return len(self.errors) == 0
        
    def _check_version_declaration(self, source_code: str) -> bool:
        """Check if version declaration is present."""
        return "#version" in source_code
        
    def _check_main_function(self, source_code: str) -> bool:
        """Check if main function is present."""
        return "void main()" in source_code or "void main (" in source_code
        
    def _check_gl_position_assignment(self, source_code: str) -> bool:
        """Check if gl_Position is assigned in vertex shader."""
        return "gl_Position" in source_code
        
    def _check_uniform_declarations(self, source_code: str):
        """Check uniform declarations for issues."""
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'uniform' in line:
                # Check for missing semicolon
                if not line.strip().endswith(';'):
                    self.errors.append(ShaderError(
                        ShaderErrorType.COMPILATION, 0, i + 1,
                        "Missing semicolon in uniform declaration",
                        line.strip()
                    ))
                    
                # Check for invalid type names
                invalid_types = ['void', 'main', 'return', 'if', 'else', 'for', 'while']
                for invalid_type in invalid_types:
                    if f"uniform {invalid_type}" in line:
                        self.errors.append(ShaderError(
                            ShaderErrorType.COMPILATION, 0, i + 1,
                            f"Invalid uniform type: {invalid_type}",
                            line.strip()
                        ))
                        
    def _check_attribute_declarations(self, source_code: str):
        """Check attribute declarations for issues."""
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'attribute' in line or 'in' in line:
                # Check for missing semicolon
                if not line.strip().endswith(';'):
                    self.errors.append(ShaderError(
                        ShaderErrorType.COMPILATION, 0, i + 1,
                        "Missing semicolon in attribute declaration",
                        line.strip()
                    ))
                    
    def _check_input_variables(self, source_code: str):
        """Check input variable declarations."""
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'in ' in line and not line.strip().startswith('//'):
                # Check for missing semicolon
                if not line.strip().endswith(';'):
                    self.errors.append(ShaderError(
                        ShaderErrorType.COMPILATION, 0, i + 1,
                        "Missing semicolon in input variable declaration",
                        line.strip()
                    ))
                    
    def _check_output_variables(self, source_code: str):
        """Check output variable declarations."""
        lines = source_code.split('\n')
        for i, line in enumerate(lines):
            if 'out ' in line and not line.strip().startswith('//'):
                # Check for missing semicolon
                if not line.strip().endswith(';'):
                    self.errors.append(ShaderError(
                        ShaderErrorType.COMPILATION, 0, i + 1,
                        "Missing semicolon in output variable declaration",
                        line.strip()
                    ))
                    
    def get_errors(self) -> List[ShaderError]:
        """Get all validation errors."""
        return self.errors
        
    def get_warnings(self) -> List[ShaderError]:
        """Get all validation warnings."""
        return self.warnings


class ShaderCompiler:
    """Handles shader compilation with error checking."""
    
    def __init__(self):
        self.validator = ShaderValidator()
        self.compilation_log = []
        
    def compile_shader(self, shader_type: int, source_code: str) -> Tuple[bool, int, str]:
        """Compile a shader and return success status, shader ID, and error log."""
        # Validate source code first
        if shader_type == gl.GL_VERTEX_SHADER:
            if not self.validator.validate_vertex_shader(source_code):
                errors = self.validator.get_errors()
                error_log = "\n".join([f"Line {e.line_number}: {e.message}" for e in errors])
                return False, 0, error_log
        elif shader_type == gl.GL_FRAGMENT_SHADER:
            if not self.validator.validate_fragment_shader(source_code):
                errors = self.validator.get_errors()
                error_log = "\n".join([f"Line {e.line_number}: {e.message}" for e in errors])
                return False, 0, error_log
                
        # Create shader
        shader_id = gl.glCreateShader(shader_type)
        
        # Set source code
        gl.glShaderSource(shader_id, source_code)
        
        # Compile shader
        gl.glCompileShader(shader_id)
        
        # Check compilation status
        success = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
        if not success:
            error_log = gl.glGetShaderInfoLog(shader_id).decode('utf-8')
            gl.glDeleteShader(shader_id)
            return False, 0, error_log
            
        # Log successful compilation
        self.compilation_log.append(f"Successfully compiled {self._get_shader_type_name(shader_type)}")
        
        return True, shader_id, ""
        
    def link_program(self, shader_ids: List[int]) -> Tuple[bool, int, str]:
        """Link shader program and return success status, program ID, and error log."""
        if not shader_ids:
            return False, 0, "No shaders provided for linking"
            
        # Create program
        program_id = gl.glCreateProgram()
        
        # Attach shaders
        for shader_id in shader_ids:
            gl.glAttachShader(program_id, shader_id)
            
        # Link program
        gl.glLinkProgram(program_id)
        
        # Check linking status
        success = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        if not success:
            error_log = gl.glGetProgramInfoLog(program_id).decode('utf-8')
            gl.glDeleteProgram(program_id)
            return False, 0, error_log
            
        # Log successful linking
        self.compilation_log.append(f"Successfully linked program with {len(shader_ids)} shaders")
        
        return True, program_id, ""
        
    def validate_program(self, program_id: int) -> Tuple[bool, str]:
        """Validate a linked shader program."""
        gl.glValidateProgram(program_id)
        
        # Check validation status
        success = gl.glGetProgramiv(program_id, gl.GL_VALIDATE_STATUS)
        if not success:
            error_log = gl.glGetProgramInfoLog(program_id).decode('utf-8')
            return False, error_log
            
        return True, ""
        
    def _get_shader_type_name(self, shader_type: int) -> str:
        """Get shader type name."""
        type_names = {
            gl.GL_VERTEX_SHADER: "vertex shader",
            gl.GL_FRAGMENT_SHADER: "fragment shader",
            gl.GL_GEOMETRY_SHADER: "geometry shader",
            gl.GL_COMPUTE_SHADER: "compute shader"
        }
        return type_names.get(shader_type, "unknown shader")


class ShaderDebugger:
    """Provides debugging tools for shaders."""
    
    def __init__(self):
        self.debug_info: Dict[str, Any] = {}
        
    def enable_debug_output(self):
        """Enable OpenGL debug output."""
        gl.glEnable(gl.GL_DEBUG_OUTPUT)
        gl.glEnable(gl.GL_DEBUG_OUTPUT_SYNCHRONOUS)
        
    def get_shader_info(self, shader_id: int) -> Dict[str, Any]:
        """Get information about a compiled shader."""
        info = {}
        
        # Get shader type
        shader_type = gl.glGetShaderiv(shader_id, gl.GL_SHADER_TYPE)
        info['type'] = self._get_shader_type_name(shader_type)
        
        # Get source code length
        source_length = gl.glGetShaderiv(shader_id, gl.GL_SHADER_SOURCE_LENGTH)
        info['source_length'] = source_length
        
        # Get source code
        source_code = gl.glGetShaderSource(shader_id)
        info['source_code'] = source_code
        
        # Get compilation status
        compile_status = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
        info['compiled'] = bool(compile_status)
        
        # Get info log
        info_log = gl.glGetShaderInfoLog(shader_id)
        info['info_log'] = info_log
        
        return info
        
    def get_program_info(self, program_id: int) -> Dict[str, Any]:
        """Get information about a linked program."""
        info = {}
        
        # Get link status
        link_status = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        info['linked'] = bool(link_status)
        
        # Get validation status
        validate_status = gl.glGetProgramiv(program_id, gl.GL_VALIDATE_STATUS)
        info['validated'] = bool(validate_status)
        
        # Get info log
        info_log = gl.glGetProgramInfoLog(program_id)
        info['info_log'] = info_log
        
        # Get active uniforms
        uniform_count = gl.glGetProgramiv(program_id, gl.GL_ACTIVE_UNIFORMS)
        info['uniform_count'] = uniform_count
        
        uniforms = {}
        for i in range(uniform_count):
            name = gl.glGetActiveUniform(program_id, i)[0].decode('utf-8')
            location = gl.glGetUniformLocation(program_id, name)
            uniforms[name] = location
        info['uniforms'] = uniforms
        
        # Get active attributes
        attribute_count = gl.glGetProgramiv(program_id, gl.GL_ACTIVE_ATTRIBUTES)
        info['attribute_count'] = attribute_count
        
        attributes = {}
        for i in range(attribute_count):
            name = gl.glGetActiveAttrib(program_id, i)[0].decode('utf-8')
            location = gl.glGetAttribLocation(program_id, name)
            attributes[name] = location
        info['attributes'] = attributes
        
        return info
        
    def _get_shader_type_name(self, shader_type: int) -> str:
        """Get shader type name."""
        type_names = {
            gl.GL_VERTEX_SHADER: "vertex shader",
            gl.GL_FRAGMENT_SHADER: "fragment shader",
            gl.GL_GEOMETRY_SHADER: "geometry shader",
            gl.GL_COMPUTE_SHADER: "compute shader"
        }
        return type_names.get(shader_type, "unknown shader")


class ShaderPerformanceAnalyzer:
    """Analyzes shader performance and provides optimization suggestions."""
    
    def __init__(self):
        self.analysis_results: Dict[str, Any] = {}
        
    def analyze_shader_performance(self, source_code: str) -> Dict[str, Any]:
        """Analyze shader performance and return suggestions."""
        analysis = {
            'instruction_count': 0,
            'texture_samples': 0,
            'branches': 0,
            'loops': 0,
            'suggestions': []
        }
        
        lines = source_code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Count texture samples
            if 'texture(' in line:
                analysis['texture_samples'] += 1
                
            # Count branches
            if any(keyword in line for keyword in ['if ', 'else', 'switch']):
                analysis['branches'] += 1
                
            # Count loops
            if any(keyword in line for keyword in ['for ', 'while ', 'do ']):
                analysis['loops'] += 1
                
            # Count mathematical operations
            if any(op in line for op in ['*', '/', '+', '-', 'pow', 'sin', 'cos', 'tan']):
                analysis['instruction_count'] += 1
                
        # Generate suggestions
        if analysis['texture_samples'] > 4:
            analysis['suggestions'].append("Consider reducing texture samples for better performance")
            
        if analysis['branches'] > 2:
            analysis['suggestions'].append("Minimize branching in fragment shaders for better GPU utilization")
            
        if analysis['loops'] > 0:
            analysis['suggestions'].append("Consider unrolling loops or using texture lookups instead")
            
        if analysis['instruction_count'] > 100:
            analysis['suggestions'].append("Consider simplifying mathematical operations")
            
        return analysis


class ShaderErrorReporter:
    """Provides detailed error reporting for shader compilation."""
    
    def __init__(self):
        self.error_patterns = {
            r'(\d+):(\d+):\s*error:\s*(.+)': (1, 2, 3),  # Line:Column: error: message
            r'ERROR:\s*(\d+):(\d+):\s*(.+)': (1, 2, 3),  # ERROR: Line:Column: message
            r'(\d+):\s*error:\s*(.+)': (1, 2),  # Line: error: message
        }
        
    def parse_error_log(self, error_log: str) -> List[ShaderError]:
        """Parse error log and extract structured error information."""
        errors = []
        lines = error_log.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern, groups in self.error_patterns.items():
                match = re.match(pattern, line)
                if match:
                    if len(groups) == 3:
                        line_num = int(match.group(groups[0]))
                        col_num = int(match.group(groups[1]))
                        message = match.group(groups[2])
                    else:
                        line_num = int(match.group(groups[0]))
                        col_num = 0
                        message = match.group(groups[1])
                        
                    errors.append(ShaderError(
                        ShaderErrorType.COMPILATION,
                        0,  # shader_id not available in parsing
                        line_num,
                        message,
                        severity="error"
                    ))
                    break
                    
        return errors
        
    def format_error_report(self, errors: List[ShaderError], source_code: str = "") -> str:
        """Format error report with source code context."""
        if not errors:
            return "No errors found."
            
        report = "Shader Compilation Errors:\n"
        report += "=" * 50 + "\n\n"
        
        source_lines = source_code.split('\n') if source_code else []
        
        for error in errors:
            report += f"Error at line {error.line_number}:\n"
            report += f"  {error.message}\n"
            
            # Add source code context
            if 0 <= error.line_number - 1 < len(source_lines):
                report += f"  Source: {source_lines[error.line_number - 1].strip()}\n"
                
            report += "\n"
            
        return report


def demonstrate_shader_compilation():
    """Demonstrate shader compilation and error handling."""
    print("=== Shader Compilation Demonstration ===\n")
    
    # Create tools
    compiler = ShaderCompiler()
    debugger = ShaderDebugger()
    analyzer = ShaderPerformanceAnalyzer()
    reporter = ShaderErrorReporter()
    
    # Test valid shader compilation
    print("1. Testing valid shader compilation:")
    
    valid_vertex_source = """
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
    
    valid_fragment_source = """
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
    vec3 ambient = 0.1 * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    vec3 result = (ambient + diffuse) * objectColor;
    vec4 texColor = texture(diffuseTexture, TexCoord);
    FragColor = vec4(result * texColor.rgb, 1.0);
}
"""
    
    # Compile vertex shader
    success, vertex_id, error_log = compiler.compile_shader(gl.GL_VERTEX_SHADER, valid_vertex_source)
    if success:
        print("   - Vertex shader compiled successfully")
        
        # Analyze performance
        analysis = analyzer.analyze_shader_performance(valid_vertex_source)
        print(f"     Instructions: {analysis['instruction_count']}")
        print(f"     Suggestions: {len(analysis['suggestions'])}")
    else:
        print(f"   - Vertex shader compilation failed: {error_log}")
        
    # Compile fragment shader
    success, fragment_id, error_log = compiler.compile_shader(gl.GL_FRAGMENT_SHADER, valid_fragment_source)
    if success:
        print("   - Fragment shader compiled successfully")
        
        # Analyze performance
        analysis = analyzer.analyze_shader_performance(valid_fragment_source)
        print(f"     Instructions: {analysis['instruction_count']}")
        print(f"     Texture samples: {analysis['texture_samples']}")
        print(f"     Suggestions: {len(analysis['suggestions'])}")
    else:
        print(f"   - Fragment shader compilation failed: {error_log}")
        
    # Link program
    if vertex_id and fragment_id:
        success, program_id, error_log = compiler.link_program([vertex_id, fragment_id])
        if success:
            print("   - Shader program linked successfully")
            
            # Validate program
            valid, validation_log = compiler.validate_program(program_id)
            if valid:
                print("   - Shader program validated successfully")
            else:
                print(f"   - Shader program validation failed: {validation_log}")
                
            # Get program info
            program_info = debugger.get_program_info(program_id)
            print(f"     Active uniforms: {program_info['uniform_count']}")
            print(f"     Active attributes: {program_info['attribute_count']}")
        else:
            print(f"   - Shader program linking failed: {error_log}")
            
    # Test invalid shader compilation
    print("\n2. Testing invalid shader compilation:")
    
    invalid_source = """
#version 330 core
void main() {
    // Missing gl_Position assignment
    // This will cause a compilation error
}
"""
    
    success, shader_id, error_log = compiler.compile_shader(gl.GL_VERTEX_SHADER, invalid_source)
    if not success:
        print("   - Invalid shader compilation failed as expected")
        
        # Parse and format error report
        errors = reporter.parse_error_log(error_log)
        report = reporter.format_error_report(errors, invalid_source)
        print("   - Error report:")
        for line in report.split('\n')[:10]:  # Show first 10 lines
            print(f"     {line}")
            
    # Test validation
    print("\n3. Testing shader validation:")
    
    validator = ShaderValidator()
    
    # Test vertex shader validation
    valid = validator.validate_vertex_shader(valid_vertex_source)
    print(f"   - Valid vertex shader validation: {'PASS' if valid else 'FAIL'}")
    
    # Test invalid vertex shader
    invalid_vertex = """
#version 330 core
void main() {
    // Missing gl_Position
}
"""
    valid = validator.validate_vertex_shader(invalid_vertex)
    print(f"   - Invalid vertex shader validation: {'PASS' if valid else 'FAIL'}")
    
    if not valid:
        errors = validator.get_errors()
        print(f"     Found {len(errors)} validation errors:")
        for error in errors:
            print(f"       - {error.message}")
            
    # Show compilation log
    print("\n4. Compilation log:")
    for log_entry in compiler.compilation_log:
        print(f"   - {log_entry}")
        
    # Cleanup
    print("\n5. Cleaning up shader resources...")
    if vertex_id:
        gl.glDeleteShader(vertex_id)
    if fragment_id:
        gl.glDeleteShader(fragment_id)
    if 'program_id' in locals() and program_id:
        gl.glDeleteProgram(program_id)
    print("   - All shader resources cleaned up")


if __name__ == "__main__":
    demonstrate_shader_compilation()
