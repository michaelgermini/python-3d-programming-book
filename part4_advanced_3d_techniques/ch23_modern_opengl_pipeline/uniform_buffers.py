"""
Chapter 23: Modern OpenGL Pipeline - Uniform Buffer Objects
=========================================================

This module demonstrates uniform buffer objects and uniform management in modern OpenGL.

Key Concepts:
- Uniform Buffer Objects (UBOs) for efficient uniform data transfer
- Uniform block layout and binding
- Matrix and transformation uniform management
- Performance optimization through uniform batching
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import struct


class UniformType(Enum):
    """Uniform data type enumeration."""
    FLOAT = "float"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT3 = "mat3"
    MAT4 = "mat4"
    INT = "int"
    BOOL = "bool"


@dataclass
class UniformInfo:
    """Information about a uniform variable."""
    name: str
    uniform_type: UniformType
    location: int
    size: int
    offset: int
    array_stride: int = 0
    matrix_stride: int = 0


@dataclass
class UniformBlockInfo:
    """Information about a uniform block."""
    name: str
    binding_point: int
    block_size: int
    uniforms: Dict[str, UniformInfo]


class UniformBufferObject:
    """Represents a Uniform Buffer Object (UBO)."""

    def __init__(self, binding_point: int, size: int):
        self.buffer_id = gl.glGenBuffers(1)
        self.binding_point = binding_point
        self.size = size
        self.data = bytearray(size)
        self.dirty = True

    def bind(self):
        """Bind the UBO to its binding point."""
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.buffer_id)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, self.binding_point, self.buffer_id)

    def unbind(self):
        """Unbind the UBO."""
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)

    def update_data(self, data: bytes, offset: int = 0):
        """Update data in the UBO."""
        if offset + len(data) > self.size:
            raise ValueError("Data exceeds buffer size")
        
        self.data[offset:offset + len(data)] = data
        self.dirty = True

    def set_float(self, offset: int, value: float):
        """Set a float value at the specified offset."""
        data = struct.pack('f', value)
        self.update_data(data, offset)

    def set_vec3(self, offset: int, x: float, y: float, z: float):
        """Set a vec3 value at the specified offset."""
        data = struct.pack('3f', x, y, z)
        self.update_data(data, offset)

    def set_mat4(self, offset: int, matrix: np.ndarray):
        """Set a mat4 value at the specified offset."""
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        
        # OpenGL expects column-major order
        matrix_gl = matrix.T.astype(np.float32)
        data = matrix_gl.tobytes()
        self.update_data(data, offset)

    def upload_to_gpu(self):
        """Upload data to GPU if dirty."""
        if self.dirty:
            gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.buffer_id)
            gl.glBufferData(gl.GL_UNIFORM_BUFFER, self.size, self.data, gl.GL_DYNAMIC_DRAW)
            self.dirty = False

    def cleanup(self):
        """Clean up UBO resources."""
        if self.buffer_id:
            gl.glDeleteBuffers(1, [self.buffer_id])
            self.buffer_id = 0


class UniformManager:
    """Manages uniform variables and uniform blocks."""

    def __init__(self):
        self.uniform_blocks: Dict[str, UniformBlockInfo] = {}
        self.ubos: Dict[str, UniformBufferObject] = {}
        self.next_binding_point = 0

    def create_uniform_block(self, name: str, uniforms: Dict[str, UniformType]) -> UniformBlockInfo:
        """Create a uniform block with the specified uniforms."""
        # Calculate block layout
        block_size = 0
        uniform_infos = {}
        
        for uniform_name, uniform_type in uniforms.items():
            offset = block_size
            
            # Calculate size and alignment for each type
            if uniform_type == UniformType.FLOAT:
                size = 4
                alignment = 4
            elif uniform_type == UniformType.VEC3:
                size = 12
                alignment = 16  # vec3 is aligned to vec4 boundary
            elif uniform_type == UniformType.VEC4:
                size = 16
                alignment = 16
            elif uniform_type == UniformType.MAT4:
                size = 64
                alignment = 16
            else:
                size = 4
                alignment = 4
            
            # Align offset
            offset = (offset + alignment - 1) & ~(alignment - 1)
            
            uniform_infos[uniform_name] = UniformInfo(
                name=uniform_name,
                uniform_type=uniform_type,
                location=0,  # Will be set by shader
                size=size,
                offset=offset
            )
            
            block_size = offset + size
        
        # Align final block size
        block_size = (block_size + 15) & ~15  # Align to 16 bytes
        
        block_info = UniformBlockInfo(
            name=name,
            binding_point=self.next_binding_point,
            block_size=block_size,
            uniforms=uniform_infos
        )
        
        self.uniform_blocks[name] = block_info
        self.next_binding_point += 1
        
        return block_info

    def create_ubo(self, block_name: str) -> Optional[UniformBufferObject]:
        """Create a UBO for a uniform block."""
        if block_name not in self.uniform_blocks:
            print(f"Uniform block '{block_name}' not found")
            return None
        
        block_info = self.uniform_blocks[block_name]
        ubo = UniformBufferObject(block_info.binding_point, block_info.block_size)
        self.ubos[block_name] = ubo
        return ubo

    def set_uniform_value(self, block_name: str, uniform_name: str, value: Any):
        """Set a uniform value in a UBO."""
        if block_name not in self.uniform_blocks:
            print(f"Uniform block '{block_name}' not found")
            return
        
        if uniform_name not in self.uniform_blocks[block_name].uniforms:
            print(f"Uniform '{uniform_name}' not found in block '{block_name}'")
            return
        
        if block_name not in self.ubos:
            print(f"UBO for block '{block_name}' not created")
            return
        
        block_info = self.uniform_blocks[block_name]
        uniform_info = block_info.uniforms[uniform_name]
        ubo = self.ubos[block_name]
        
        if uniform_info.uniform_type == UniformType.FLOAT:
            ubo.set_float(uniform_info.offset, value)
        elif uniform_info.uniform_type == UniformType.VEC3:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                ubo.set_vec3(uniform_info.offset, value[0], value[1], value[2])
            else:
                print(f"Invalid value for vec3 uniform '{uniform_name}'")
        elif uniform_info.uniform_type == UniformType.MAT4:
            if isinstance(value, np.ndarray):
                ubo.set_mat4(uniform_info.offset, value)
            else:
                print(f"Invalid value for mat4 uniform '{uniform_name}'")

    def upload_all_ubos(self):
        """Upload all UBOs to GPU."""
        for ubo in self.ubos.values():
            ubo.upload_to_gpu()

    def bind_ubo(self, block_name: str):
        """Bind a UBO."""
        if block_name in self.ubos:
            self.ubos[block_name].bind()

    def cleanup(self):
        """Clean up all UBO resources."""
        for ubo in self.ubos.values():
            ubo.cleanup()
        self.ubos.clear()


class TransformationUniforms:
    """Manages transformation matrices as uniforms."""

    def __init__(self, uniform_manager: UniformManager):
        self.uniform_manager = uniform_manager
        
        # Create transformation uniform block
        self.transform_block = uniform_manager.create_uniform_block("TransformBlock", {
            "model": UniformType.MAT4,
            "view": UniformType.MAT4,
            "projection": UniformType.MAT4,
            "normal_matrix": UniformType.MAT4
        })
        
        # Create UBO
        self.transform_ubo = uniform_manager.create_ubo("TransformBlock")
        
        # Initialize with identity matrices
        self.model_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.eye(4, dtype=np.float32)
        self.normal_matrix = np.eye(4, dtype=np.float32)

    def set_model_matrix(self, matrix: np.ndarray):
        """Set the model matrix."""
        self.model_matrix = matrix.astype(np.float32)
        self.uniform_manager.set_uniform_value("TransformBlock", "model", self.model_matrix)
        
        # Update normal matrix (inverse transpose of model matrix)
        self.normal_matrix = np.linalg.inv(self.model_matrix).T.astype(np.float32)
        self.uniform_manager.set_uniform_value("TransformBlock", "normal_matrix", self.normal_matrix)

    def set_view_matrix(self, matrix: np.ndarray):
        """Set the view matrix."""
        self.view_matrix = matrix.astype(np.float32)
        self.uniform_manager.set_uniform_value("TransformBlock", "view", self.view_matrix)

    def set_projection_matrix(self, matrix: np.ndarray):
        """Set the projection matrix."""
        self.projection_matrix = matrix.astype(np.float32)
        self.uniform_manager.set_uniform_value("TransformBlock", "projection", self.projection_matrix)

    def upload_matrices(self):
        """Upload all matrices to GPU."""
        self.uniform_manager.upload_all_ubos()

    def bind(self):
        """Bind the transformation UBO."""
        if self.transform_ubo:
            self.transform_ubo.bind()


class LightingUniforms:
    """Manages lighting uniforms."""

    def __init__(self, uniform_manager: UniformManager):
        self.uniform_manager = uniform_manager
        
        # Create lighting uniform block
        self.lighting_block = uniform_manager.create_uniform_block("LightingBlock", {
            "light_position": UniformType.VEC3,
            "light_color": UniformType.VEC3,
            "light_intensity": UniformType.FLOAT,
            "ambient_light": UniformType.VEC3,
            "view_position": UniformType.VEC3
        })
        
        # Create UBO
        self.lighting_ubo = uniform_manager.create_ubo("LightingBlock")
        
        # Initialize with default values
        self.light_position = [0.0, 5.0, 0.0]
        self.light_color = [1.0, 1.0, 1.0]
        self.light_intensity = 1.0
        self.ambient_light = [0.1, 0.1, 0.1]
        self.view_position = [0.0, 0.0, 5.0]

    def set_light_position(self, x: float, y: float, z: float):
        """Set light position."""
        self.light_position = [x, y, z]
        self.uniform_manager.set_uniform_value("LightingBlock", "light_position", self.light_position)

    def set_light_color(self, r: float, g: float, b: float):
        """Set light color."""
        self.light_color = [r, g, b]
        self.uniform_manager.set_uniform_value("LightingBlock", "light_color", self.light_color)

    def set_light_intensity(self, intensity: float):
        """Set light intensity."""
        self.light_intensity = intensity
        self.uniform_manager.set_uniform_value("LightingBlock", "light_intensity", self.light_intensity)

    def set_ambient_light(self, r: float, g: float, b: float):
        """Set ambient light."""
        self.ambient_light = [r, g, b]
        self.uniform_manager.set_uniform_value("LightingBlock", "ambient_light", self.ambient_light)

    def set_view_position(self, x: float, y: float, z: float):
        """Set view position."""
        self.view_position = [x, y, z]
        self.uniform_manager.set_uniform_value("LightingBlock", "view_position", self.view_position)

    def upload_lighting(self):
        """Upload all lighting data to GPU."""
        self.uniform_manager.upload_all_ubos()

    def bind(self):
        """Bind the lighting UBO."""
        if self.lighting_ubo:
            self.lighting_ubo.bind()


def demonstrate_uniform_buffers():
    """Demonstrate uniform buffer objects and uniform management."""
    print("=== Modern OpenGL Pipeline - Uniform Buffer Objects ===\n")

    # Create uniform manager
    uniform_manager = UniformManager()

    # Create transformation uniforms
    print("1. Creating transformation uniforms...")
    transform_uniforms = TransformationUniforms(uniform_manager)
    
    # Set some example matrices
    model_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    transform_uniforms.set_model_matrix(model_matrix)
    transform_uniforms.set_view_matrix(model_matrix)
    transform_uniforms.set_projection_matrix(model_matrix)
    
    print("Transformation uniforms created successfully")

    # Create lighting uniforms
    print("\n2. Creating lighting uniforms...")
    lighting_uniforms = LightingUniforms(uniform_manager)
    
    lighting_uniforms.set_light_position(0.0, 5.0, 0.0)
    lighting_uniforms.set_light_color(1.0, 1.0, 1.0)
    lighting_uniforms.set_light_intensity(1.0)
    lighting_uniforms.set_ambient_light(0.1, 0.1, 0.1)
    lighting_uniforms.set_view_position(0.0, 0.0, 5.0)
    
    print("Lighting uniforms created successfully")

    # Display uniform block information
    print(f"\n3. Uniform Block Information:")
    for block_name, block_info in uniform_manager.uniform_blocks.items():
        print(f"Block: {block_name}")
        print(f"  Binding Point: {block_info.binding_point}")
        print(f"  Block Size: {block_info.block_size} bytes")
        print(f"  Uniforms:")
        for uniform_name, uniform_info in block_info.uniforms.items():
            print(f"    {uniform_name}: {uniform_info.uniform_type.value} at offset {uniform_info.offset}")

    # Demonstrate UBO usage
    print(f"\n4. UBO Usage:")
    print("To upload to GPU: uniform_manager.upload_all_ubos()")
    print("To bind transformation UBO: transform_uniforms.bind()")
    print("To bind lighting UBO: lighting_uniforms.bind()")

    # Cleanup
    uniform_manager.cleanup()
    print("\n5. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_uniform_buffers()
