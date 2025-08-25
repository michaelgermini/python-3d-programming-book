"""
Chapter 22: Shaders and GLSL Basics - GLSL Language Features
===========================================================

This module demonstrates GLSL language features and advanced concepts.

Key Concepts:
- GLSL variables and data types
- Built-in functions and mathematical operations
- Custom functions and function overloading
- Advanced GLSL features and optimizations
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class GLSLDataType(Enum):
    """GLSL data type enumeration."""
    FLOAT = "float"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT2 = "mat2"
    MAT3 = "mat3"
    MAT4 = "mat4"
    INT = "int"
    IVEC2 = "ivec2"
    IVEC3 = "ivec3"
    IVEC4 = "ivec4"
    BOOL = "bool"
    BVEC2 = "bvec2"
    BVEC3 = "bvec3"
    BVEC4 = "bvec4"
    SAMPLER2D = "sampler2D"
    SAMPLER3D = "sampler3D"
    SAMPLERCUBE = "samplerCube"


class GLSLVariable:
    """Represents a GLSL variable with type and value."""
    
    def __init__(self, name: str, data_type: GLSLDataType, value: Any = None):
        self.name = name
        self.data_type = data_type
        self.value = value
        self.qualifier = ""  # uniform, attribute, varying, etc.
        
    def get_declaration(self) -> str:
        """Get GLSL variable declaration."""
        qualifier = f"{self.qualifier} " if self.qualifier else ""
        return f"{qualifier}{self.data_type.value} {self.name};"
        
    def get_initialization(self) -> str:
        """Get GLSL variable initialization."""
        if self.value is None:
            return ""
            
        if self.data_type == GLSLDataType.FLOAT:
            return f"{self.name} = {self.value}f;"
        elif self.data_type == GLSLDataType.VEC2:
            return f"{self.name} = vec2({self.value[0]}f, {self.value[1]}f);"
        elif self.data_type == GLSLDataType.VEC3:
            return f"{self.name} = vec3({self.value[0]}f, {self.value[1]}f, {self.value[2]}f);"
        elif self.data_type == GLSLDataType.VEC4:
            return f"{self.name} = vec4({self.value[0]}f, {self.value[1]}f, {self.value[2]}f, {self.value[3]}f);"
        elif self.data_type == GLSLDataType.INT:
            return f"{self.name} = {self.value};"
        elif self.data_type == GLSLDataType.BOOL:
            return f"{self.name} = {str(self.value).lower()};"
        else:
            return f"{self.name} = {self.value};"


class GLSLFunction:
    """Represents a GLSL function."""
    
    def __init__(self, name: str, return_type: GLSLDataType, parameters: List[Tuple[str, GLSLDataType]] = None):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters or []
        self.body = ""
        
    def add_parameter(self, name: str, data_type: GLSLDataType):
        """Add parameter to function."""
        self.parameters.append((name, data_type))
        
    def set_body(self, body: str):
        """Set function body."""
        self.body = body
        
    def get_declaration(self) -> str:
        """Get function declaration."""
        param_str = ", ".join([f"{param_type.value} {param_name}" for param_name, param_type in self.parameters])
        return f"{self.return_type.value} {self.name}({param_str});"
        
    def get_definition(self) -> str:
        """Get function definition."""
        param_str = ", ".join([f"{param_type.value} {param_name}" for param_name, param_type in self.parameters])
        return f"{self.return_type.value} {self.name}({param_str}) {{\n{self.body}\n}}"


class GLSLBuiltInFunctions:
    """Collection of GLSL built-in functions and their usage."""
    
    @staticmethod
    def get_mathematical_functions() -> Dict[str, str]:
        """Get mathematical built-in functions."""
        return {
            "abs": "Absolute value: abs(float x) -> float",
            "floor": "Floor function: floor(float x) -> float",
            "ceil": "Ceiling function: ceil(float x) -> float",
            "round": "Round to nearest: round(float x) -> float",
            "fract": "Fractional part: fract(float x) -> float",
            "mod": "Modulo: mod(float x, float y) -> float",
            "min": "Minimum: min(float x, float y) -> float",
            "max": "Maximum: max(float x, float y) -> float",
            "clamp": "Clamp value: clamp(float x, float min, float max) -> float",
            "mix": "Linear interpolation: mix(float x, float y, float a) -> float",
            "step": "Step function: step(float edge, float x) -> float",
            "smoothstep": "Smooth step: smoothstep(float edge0, float edge1, float x) -> float"
        }
        
    @staticmethod
    def get_trigonometric_functions() -> Dict[str, str]:
        """Get trigonometric built-in functions."""
        return {
            "sin": "Sine: sin(float x) -> float",
            "cos": "Cosine: cos(float x) -> float",
            "tan": "Tangent: tan(float x) -> float",
            "asin": "Arc sine: asin(float x) -> float",
            "acos": "Arc cosine: acos(float x) -> float",
            "atan": "Arc tangent: atan(float x) -> float",
            "atan2": "Arc tangent 2: atan2(float y, float x) -> float",
            "sinh": "Hyperbolic sine: sinh(float x) -> float",
            "cosh": "Hyperbolic cosine: cosh(float x) -> float",
            "tanh": "Hyperbolic tangent: tanh(float x) -> float"
        }
        
    @staticmethod
    def get_exponential_functions() -> Dict[str, str]:
        """Get exponential built-in functions."""
        return {
            "pow": "Power: pow(float x, float y) -> float",
            "exp": "Exponential: exp(float x) -> float",
            "log": "Natural logarithm: log(float x) -> float",
            "exp2": "Base-2 exponential: exp2(float x) -> float",
            "log2": "Base-2 logarithm: log2(float x) -> float",
            "sqrt": "Square root: sqrt(float x) -> float",
            "inversesqrt": "Inverse square root: inversesqrt(float x) -> float"
        }
        
    @staticmethod
    def get_geometric_functions() -> Dict[str, str]:
        """Get geometric built-in functions."""
        return {
            "length": "Vector length: length(vec3 v) -> float",
            "distance": "Distance between points: distance(vec3 p0, vec3 p1) -> float",
            "dot": "Dot product: dot(vec3 a, vec3 b) -> float",
            "cross": "Cross product: cross(vec3 a, vec3 b) -> vec3",
            "normalize": "Normalize vector: normalize(vec3 v) -> vec3",
            "reflect": "Reflection: reflect(vec3 I, vec3 N) -> vec3",
            "refract": "Refraction: refract(vec3 I, vec3 N, float eta) -> vec3",
            "faceforward": "Face forward: faceforward(vec3 N, vec3 I, vec3 Nref) -> vec3"
        }
        
    @staticmethod
    def get_texture_functions() -> Dict[str, str]:
        """Get texture built-in functions."""
        return {
            "texture": "Sample texture: texture(sampler2D sampler, vec2 coord) -> vec4",
            "textureLod": "Sample texture with LOD: textureLod(sampler2D sampler, vec2 coord, float lod) -> vec4",
            "textureProj": "Projective texture: textureProj(sampler2D sampler, vec3 coord) -> vec4",
            "textureOffset": "Offset texture: textureOffset(sampler2D sampler, vec2 coord, ivec2 offset) -> vec4",
            "texelFetch": "Fetch texel: texelFetch(sampler2D sampler, ivec2 coord, int lod) -> vec4"
        }
        
    @staticmethod
    def get_noise_functions() -> Dict[str, str]:
        """Get noise built-in functions."""
        return {
            "noise1": "1D noise: noise1(float x) -> float",
            "noise2": "2D noise: noise2(vec2 x) -> vec2",
            "noise3": "3D noise: noise3(vec3 x) -> vec3",
            "noise4": "4D noise: noise4(vec4 x) -> vec4"
        }


class GLSLCodeGenerator:
    """Generates GLSL code with various features."""
    
    def __init__(self):
        self.variables: List[GLSLVariable] = []
        self.functions: List[GLSLFunction] = []
        self.main_body = ""
        
    def add_variable(self, variable: GLSLVariable):
        """Add variable to code generator."""
        self.variables.append(variable)
        
    def add_function(self, function: GLSLFunction):
        """Add function to code generator."""
        self.functions.append(function)
        
    def set_main_body(self, body: str):
        """Set main function body."""
        self.main_body = body
        
    def generate_vertex_shader(self) -> str:
        """Generate complete vertex shader code."""
        code = "#version 330 core\n\n"
        
        # Add input attributes
        code += "// Input attributes\n"
        code += "layout(location = 0) in vec3 position;\n"
        code += "layout(location = 1) in vec3 normal;\n"
        code += "layout(location = 2) in vec2 texCoord;\n\n"
        
        # Add uniforms
        code += "// Uniforms\n"
        code += "uniform mat4 model;\n"
        code += "uniform mat4 view;\n"
        code += "uniform mat4 projection;\n\n"
        
        # Add output variables
        code += "// Output variables\n"
        code += "out vec3 FragPos;\n"
        code += "out vec3 Normal;\n"
        code += "out vec2 TexCoord;\n\n"
        
        # Add custom variables
        if self.variables:
            code += "// Custom variables\n"
            for var in self.variables:
                if var.qualifier == "uniform":
                    code += var.get_declaration() + "\n"
            code += "\n"
            
        # Add custom functions
        if self.functions:
            code += "// Custom functions\n"
            for func in self.functions:
                code += func.get_definition() + "\n\n"
                
        # Add main function
        code += "void main() {\n"
        code += "    FragPos = vec3(model * vec4(position, 1.0));\n"
        code += "    Normal = mat3(transpose(inverse(model))) * normal;\n"
        code += "    TexCoord = texCoord;\n"
        code += "    gl_Position = projection * view * vec4(FragPos, 1.0);\n"
        code += "}\n"
        
        return code
        
    def generate_fragment_shader(self) -> str:
        """Generate complete fragment shader code."""
        code = "#version 330 core\n\n"
        
        # Add input variables
        code += "// Input variables\n"
        code += "in vec3 FragPos;\n"
        code += "in vec3 Normal;\n"
        code += "in vec2 TexCoord;\n\n"
        
        # Add output variable
        code += "// Output variable\n"
        code += "out vec4 FragColor;\n\n"
        
        # Add uniforms
        code += "// Uniforms\n"
        code += "uniform vec3 lightPos;\n"
        code += "uniform vec3 lightColor;\n"
        code += "uniform vec3 objectColor;\n"
        code += "uniform sampler2D diffuseTexture;\n\n"
        
        # Add custom variables
        if self.variables:
            code += "// Custom variables\n"
            for var in self.variables:
                if var.qualifier == "uniform":
                    code += var.get_declaration() + "\n"
            code += "\n"
            
        # Add custom functions
        if self.functions:
            code += "// Custom functions\n"
            for func in self.functions:
                code += func.get_definition() + "\n\n"
                
        # Add main function
        code += "void main() {\n"
        code += self.main_body
        code += "}\n"
        
        return code


class GLSLUtilityFunctions:
    """Utility functions for common GLSL operations."""
    
    @staticmethod
    def create_phong_lighting_function() -> GLSLFunction:
        """Create Phong lighting calculation function."""
        func = GLSLFunction("calculatePhongLighting", GLSLDataType.VEC3)
        func.add_parameter("fragPos", GLSLDataType.VEC3)
        func.add_parameter("normal", GLSLDataType.VEC3)
        func.add_parameter("lightPos", GLSLDataType.VEC3)
        func.add_parameter("lightColor", GLSLDataType.VEC3)
        func.add_parameter("viewPos", GLSLDataType.VEC3)
        func.add_parameter("shininess", GLSLDataType.FLOAT)
        
        body = """
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse lighting
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular lighting
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * lightColor;
    
    return ambient + diffuse + specular;
"""
        func.set_body(body)
        return func
        
    @staticmethod
    def create_normal_mapping_function() -> GLSLFunction:
        """Create normal mapping function."""
        func = GLSLFunction("calculateNormalMapping", GLSLDataType.VEC3)
        func.add_parameter("normalMap", GLSLDataType.VEC3)
        func.add_parameter("normal", GLSLDataType.VEC3)
        func.add_parameter("fragPos", GLSLDataType.VEC3)
        func.add_parameter("texCoords", GLSLDataType.VEC2)
        
        body = """
    // Transform normal from tangent space to world space
    vec3 tangent = normalize(vec3(1.0, 0.0, 0.0));
    vec3 bitangent = normalize(cross(normal, tangent));
    mat3 TBN = mat3(tangent, bitangent, normal);
    
    // Transform normal from [0,1] to [-1,1]
    vec3 normalTS = normalize(normalMap * 2.0 - 1.0);
    
    return normalize(TBN * normalTS);
"""
        func.set_body(body)
        return func
        
    @staticmethod
    def create_fresnel_function() -> GLSLFunction:
        """Create Fresnel effect function."""
        func = GLSLFunction("calculateFresnel", GLSLDataType.FLOAT)
        func.add_parameter("viewDir", GLSLDataType.VEC3)
        func.add_parameter("normal", GLSLDataType.VEC3)
        func.add_parameter("fresnelPower", GLSLDataType.FLOAT)
        
        body = """
    float fresnel = pow(1.0 - max(dot(viewDir, normal), 0.0), fresnelPower);
    return fresnel;
"""
        func.set_body(body)
        return func
        
    @staticmethod
    def create_parallax_mapping_function() -> GLSLFunction:
        """Create parallax mapping function."""
        func = GLSLFunction("calculateParallaxMapping", GLSLDataType.VEC2)
        func.add_parameter("texCoords", GLSLDataType.VEC2)
        func.add_parameter("viewDir", GLSLDataType.VEC3)
        func.add_parameter("heightMap", GLSLDataType.SAMPLER2D)
        func.add_parameter("heightScale", GLSLDataType.FLOAT)
        
        body = """
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 P = viewDir.xy * heightScale;
    vec2 deltaTexCoords = P / numLayers;
    
    vec2 currentTexCoords = texCoords;
    float currentDepthMapValue = 1.0 - texture(heightMap, currentTexCoords).r;
    
    while(currentLayerDepth < currentDepthMapValue) {
        currentTexCoords -= deltaTexCoords;
        currentDepthMapValue = 1.0 - texture(heightMap, currentTexCoords).r;
        currentLayerDepth += layerDepth;
    }
    
    vec2 prevTexCoords = currentTexCoords + deltaTexCoords;
    float afterDepth = currentDepthMapValue - currentLayerDepth;
    float beforeDepth = 1.0 - texture(heightMap, prevTexCoords).r - currentLayerDepth + layerDepth;
    float weight = afterDepth / (afterDepth - beforeDepth);
    
    return mix(currentTexCoords, prevTexCoords, weight);
"""
        func.set_body(body)
        return func


def demonstrate_glsl_language_features():
    """Demonstrate GLSL language features."""
    print("=== GLSL Language Features Demonstration ===\n")
    
    # Show built-in functions
    print("1. GLSL Built-in Functions:")
    
    math_funcs = GLSLBuiltInFunctions.get_mathematical_functions()
    print("   Mathematical functions:")
    for func_name, description in list(math_funcs.items())[:5]:
        print(f"     {func_name}: {description}")
        
    trig_funcs = GLSLBuiltInFunctions.get_trigonometric_functions()
    print("   Trigonometric functions:")
    for func_name, description in list(trig_funcs.items())[:3]:
        print(f"     {func_name}: {description}")
        
    geom_funcs = GLSLBuiltInFunctions.get_geometric_functions()
    print("   Geometric functions:")
    for func_name, description in list(geom_funcs.items())[:4]:
        print(f"     {func_name}: {description}")
        
    # Create GLSL variables
    print("\n2. GLSL Variables:")
    
    variables = [
        GLSLVariable("ambientStrength", GLSLDataType.FLOAT, 0.1),
        GLSLVariable("lightPosition", GLSLDataType.VEC3, (2.0, 2.0, 2.0)),
        GLSLVariable("objectColor", GLSLDataType.VEC3, (0.8, 0.2, 0.2)),
        GLSLVariable("shininess", GLSLDataType.FLOAT, 32.0)
    ]
    
    for var in variables:
        print(f"   {var.get_declaration()}")
        print(f"   {var.get_initialization()}")
        
    # Create custom functions
    print("\n3. Custom GLSL Functions:")
    
    # Phong lighting function
    phong_func = GLSLUtilityFunctions.create_phong_lighting_function()
    print(f"   {phong_func.get_declaration()}")
    
    # Normal mapping function
    normal_func = GLSLUtilityFunctions.create_normal_mapping_function()
    print(f"   {normal_func.get_declaration()}")
    
    # Fresnel function
    fresnel_func = GLSLUtilityFunctions.create_fresnel_function()
    print(f"   {fresnel_func.get_declaration()}")
    
    # Generate complete shader code
    print("\n4. Generated Fragment Shader:")
    
    generator = GLSLCodeGenerator()
    
    # Add utility functions
    generator.add_function(phong_func)
    generator.add_function(fresnel_func)
    
    # Set main body
    main_body = """
    // Calculate lighting
    vec3 lighting = calculatePhongLighting(FragPos, Normal, lightPos, lightColor, vec3(0.0), 32.0);
    
    // Calculate Fresnel effect
    float fresnel = calculateFresnel(normalize(vec3(0.0) - FragPos), Normal, 5.0);
    
    // Sample texture
    vec4 texColor = texture(diffuseTexture, TexCoord);
    
    // Combine everything
    vec3 result = lighting * objectColor * texColor.rgb + fresnel * vec3(1.0);
    FragColor = vec4(result, 1.0);
"""
    generator.set_main_body(main_body)
    
    fragment_shader = generator.generate_fragment_shader()
    print("   Generated fragment shader code:")
    for line in fragment_shader.split('\n')[:20]:  # Show first 20 lines
        print(f"     {line}")
    print("     ...")
    
    # Show advanced features
    print("\n5. Advanced GLSL Features:")
    
    print("   - Function overloading: GLSL supports function overloading")
    print("   - Built-in variables: gl_Position, gl_FragColor, etc.")
    print("   - Swizzling: vec4 color = texture.rgba;")
    print("   - Component-wise operations: vec3 result = a * b;")
    print("   - Conditional compilation: #ifdef, #ifndef, #endif")
    print("   - Preprocessor directives: #version, #extension")
    
    print("\n6. GLSL Optimization Tips:")
    print("   - Use built-in functions when possible")
    print("   - Minimize branching in fragment shaders")
    print("   - Use appropriate precision qualifiers")
    print("   - Cache expensive calculations")
    print("   - Use texture lookups instead of complex calculations")


if __name__ == "__main__":
    demonstrate_glsl_language_features()
