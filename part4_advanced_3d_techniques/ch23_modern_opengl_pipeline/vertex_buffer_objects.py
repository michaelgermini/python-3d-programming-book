"""
Chapter 23: Modern OpenGL Pipeline - Vertex Buffer Objects
========================================================

This module demonstrates modern OpenGL vertex buffer objects and vertex array objects.

Key Concepts:
- Vertex Buffer Objects (VBOs) for efficient vertex data storage
- Vertex Array Objects (VAOs) for vertex attribute configuration
- Modern OpenGL rendering pipeline
- Efficient data transfer between CPU and GPU
"""

import numpy as np
import OpenGL.GL as gl
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class BufferType(Enum):
    """OpenGL buffer type enumeration."""
    ARRAY_BUFFER = gl.GL_ARRAY_BUFFER
    ELEMENT_ARRAY_BUFFER = gl.GL_ELEMENT_ARRAY_BUFFER
    UNIFORM_BUFFER = gl.GL_UNIFORM_BUFFER


class BufferUsage(Enum):
    """OpenGL buffer usage enumeration."""
    STATIC_DRAW = gl.GL_STATIC_DRAW
    DYNAMIC_DRAW = gl.GL_DYNAMIC_DRAW
    STREAM_DRAW = gl.GL_STREAM_DRAW


@dataclass
class VertexAttribute:
    """Represents a vertex attribute configuration."""
    location: int
    size: int
    data_type: int
    normalized: bool
    stride: int
    offset: int


class VertexBufferObject:
    """Represents a Vertex Buffer Object (VBO)."""

    def __init__(self, buffer_type: BufferType = BufferType.ARRAY_BUFFER):
        self.buffer_id = gl.glGenBuffers(1)
        self.buffer_type = buffer_type
        self.bound = False
        self.data_size = 0

    def bind(self):
        """Bind the VBO."""
        gl.glBindBuffer(self.buffer_type.value, self.buffer_id)
        self.bound = True

    def unbind(self):
        """Unbind the VBO."""
        gl.glBindBuffer(self.buffer_type.value, 0)
        self.bound = False

    def upload_data(self, data: np.ndarray, usage: BufferUsage = BufferUsage.STATIC_DRAW):
        """Upload data to the VBO."""
        if not self.bound:
            self.bind()
        
        gl.glBufferData(self.buffer_type.value, data.nbytes, data, usage.value)
        self.data_size = data.nbytes

    def update_data(self, data: np.ndarray, offset: int = 0):
        """Update existing data in the VBO."""
        if not self.bound:
            self.bind()
        
        gl.glBufferSubData(self.buffer_type.value, offset, data.nbytes, data)

    def get_data_size(self) -> int:
        """Get the size of data in the VBO."""
        return self.data_size

    def cleanup(self):
        """Clean up VBO resources."""
        if self.buffer_id:
            gl.glDeleteBuffers(1, [self.buffer_id])
            self.buffer_id = 0


class VertexArrayObject:
    """Represents a Vertex Array Object (VAO)."""

    def __init__(self):
        self.vao_id = gl.glGenVertexArrays(1)
        self.attributes: Dict[int, VertexAttribute] = {}
        self.bound = False

    def bind(self):
        """Bind the VAO."""
        gl.glBindVertexArray(self.vao_id)
        self.bound = True

    def unbind(self):
        """Unbind the VAO."""
        gl.glBindVertexArray(0)
        self.bound = False

    def enable_attribute(self, attribute: VertexAttribute):
        """Enable and configure a vertex attribute."""
        if not self.bound:
            self.bind()
        
        gl.glEnableVertexAttribArray(attribute.location)
        gl.glVertexAttribPointer(
            attribute.location,
            attribute.size,
            attribute.data_type,
            attribute.normalized,
            attribute.stride,
            attribute.offset
        )
        self.attributes[attribute.location] = attribute

    def disable_attribute(self, location: int):
        """Disable a vertex attribute."""
        if not self.bound:
            self.bind()
        
        gl.glDisableVertexAttribArray(location)
        if location in self.attributes:
            del self.attributes[location]

    def get_attribute(self, location: int) -> Optional[VertexAttribute]:
        """Get vertex attribute configuration."""
        return self.attributes.get(location)

    def cleanup(self):
        """Clean up VAO resources."""
        if self.vao_id:
            gl.glDeleteVertexArrays(1, [self.vao_id])
            self.vao_id = 0


class MeshData:
    """Represents mesh data with vertices and indices."""

    def __init__(self, vertices: np.ndarray, indices: np.ndarray = None):
        self.vertices = vertices
        self.indices = indices
        self.vertex_count = len(vertices)
        self.index_count = len(indices) if indices is not None else 0

    def get_vertex_size(self) -> int:
        """Get the size of a single vertex in bytes."""
        return self.vertices.dtype.itemsize * self.vertices.shape[1]

    def get_index_size(self) -> int:
        """Get the size of a single index in bytes."""
        return self.indices.dtype.itemsize if self.indices is not None else 0


class ModernRenderer:
    """Modern OpenGL renderer using VBOs and VAOs."""

    def __init__(self):
        self.vaos: Dict[str, VertexArrayObject] = {}
        self.vbos: Dict[str, VertexBufferObject] = {}
        self.active_vao: Optional[VertexArrayObject] = None

    def create_mesh(self, name: str, mesh_data: MeshData) -> bool:
        """Create a mesh with VBOs and VAO."""
        try:
            # Create VAO
            vao = VertexArrayObject()
            vao.bind()

            # Create vertex VBO
            vertex_vbo = VertexBufferObject(BufferType.ARRAY_BUFFER)
            vertex_vbo.bind()
            vertex_vbo.upload_data(mesh_data.vertices, BufferUsage.STATIC_DRAW)

            # Configure vertex attributes (assuming position, normal, texcoord)
            stride = mesh_data.get_vertex_size()
            
            # Position attribute (location 0)
            position_attr = VertexAttribute(0, 3, gl.GL_FLOAT, False, stride, 0)
            vao.enable_attribute(position_attr)

            # Normal attribute (location 1) - if available
            if mesh_data.vertices.shape[1] >= 6:  # position + normal
                normal_attr = VertexAttribute(1, 3, gl.GL_FLOAT, False, stride, 12)
                vao.enable_attribute(normal_attr)

            # Texture coordinate attribute (location 2) - if available
            if mesh_data.vertices.shape[1] >= 8:  # position + normal + texcoord
                texcoord_attr = VertexAttribute(2, 2, gl.GL_FLOAT, False, stride, 24)
                vao.enable_attribute(texcoord_attr)

            # Create index VBO if indices are provided
            if mesh_data.indices is not None:
                index_vbo = VertexBufferObject(BufferType.ELEMENT_ARRAY_BUFFER)
                index_vbo.bind()
                index_vbo.upload_data(mesh_data.indices, BufferUsage.STATIC_DRAW)
                self.vbos[f"{name}_indices"] = index_vbo

            vao.unbind()
            vertex_vbo.unbind()

            # Store references
            self.vaos[name] = vao
            self.vbos[f"{name}_vertices"] = vertex_vbo

            return True

        except Exception as e:
            print(f"Error creating mesh '{name}': {e}")
            return False

    def render_mesh(self, name: str, render_mode: int = gl.GL_TRIANGLES):
        """Render a mesh using its VAO."""
        if name not in self.vaos:
            print(f"Mesh '{name}' not found")
            return

        vao = self.vaos[name]
        vao.bind()

        # Check if we have indices
        index_vbo_name = f"{name}_indices"
        if index_vbo_name in self.vbos:
            # Render with indices
            index_count = self.vbos[index_vbo_name].get_data_size() // 4  # Assuming 32-bit indices
            gl.glDrawElements(render_mode, index_count, gl.GL_UNSIGNED_INT, None)
        else:
            # Render without indices
            vertex_count = self.vbos[f"{name}_vertices"].get_data_size() // self.vbos[f"{name}_vertices"].get_data_size()
            gl.glDrawArrays(render_mode, 0, vertex_count)

        vao.unbind()

    def update_mesh_vertices(self, name: str, vertices: np.ndarray):
        """Update vertex data for a mesh."""
        vbo_name = f"{name}_vertices"
        if vbo_name in self.vbos:
            self.vbos[vbo_name].bind()
            self.vbos[vbo_name].update_data(vertices)
            self.vbos[vbo_name].unbind()

    def cleanup_mesh(self, name: str):
        """Clean up mesh resources."""
        if name in self.vaos:
            self.vaos[name].cleanup()
            del self.vaos[name]

        vbo_names = [f"{name}_vertices", f"{name}_indices"]
        for vbo_name in vbo_names:
            if vbo_name in self.vbos:
                self.vbos[vbo_name].cleanup()
                del self.vbos[vbo_name]

    def cleanup(self):
        """Clean up all renderer resources."""
        for vao in self.vaos.values():
            vao.cleanup()
        for vbo in self.vbos.values():
            vbo.cleanup()
        self.vaos.clear()
        self.vbos.clear()


def create_triangle_mesh() -> MeshData:
    """Create a simple triangle mesh."""
    vertices = np.array([
        # Position (x, y, z), Normal (nx, ny, nz), TexCoord (u, v)
        [-0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  0.0, 0.0],
        [ 0.5, -0.5, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0],
        [ 0.0,  0.5, 0.0,  0.0, 0.0, 1.0,  0.5, 1.0]
    ], dtype=np.float32)
    
    return MeshData(vertices)


def create_cube_mesh() -> MeshData:
    """Create a cube mesh with vertices and indices."""
    vertices = np.array([
        # Position (x, y, z), Normal (nx, ny, nz), TexCoord (u, v)
        # Front face
        [-0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 0.0],
        [ 0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0],
        [ 0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0],
        [-0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 1.0],
        # Back face
        [-0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  1.0, 0.0],
        [ 0.5, -0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 0.0],
        [ 0.5,  0.5, -0.5,  0.0, 0.0, -1.0,  0.0, 1.0],
        [-0.5,  0.5, -0.5,  0.0, 0.0, -1.0,  1.0, 1.0],
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,  # Front
        4, 5, 6, 6, 7, 4,  # Back
        0, 4, 7, 7, 3, 0,  # Left
        1, 5, 6, 6, 2, 1,  # Right
        3, 2, 6, 6, 7, 3,  # Top
        0, 1, 5, 5, 4, 0   # Bottom
    ], dtype=np.uint32)

    return MeshData(vertices, indices)


def demonstrate_vertex_buffer_objects():
    """Demonstrate vertex buffer objects and modern OpenGL rendering."""
    print("=== Modern OpenGL Pipeline - Vertex Buffer Objects ===\n")

    # Create renderer
    renderer = ModernRenderer()

    # Create triangle mesh
    print("1. Creating triangle mesh...")
    triangle_mesh = create_triangle_mesh()
    success = renderer.create_mesh("triangle", triangle_mesh)
    print(f"Triangle mesh created: {success}")

    # Create cube mesh
    print("\n2. Creating cube mesh...")
    cube_mesh = create_cube_mesh()
    success = renderer.create_mesh("cube", cube_mesh)
    print(f"Cube mesh created: {success}")

    # Display mesh information
    print(f"\n3. Mesh Information:")
    print(f"Triangle vertices: {triangle_mesh.vertex_count}")
    print(f"Triangle vertex size: {triangle_mesh.get_vertex_size()} bytes")
    print(f"Cube vertices: {cube_mesh.vertex_count}")
    print(f"Cube indices: {cube_mesh.index_count}")
    print(f"Cube vertex size: {cube_mesh.get_vertex_size()} bytes")

    # Demonstrate rendering (would require OpenGL context)
    print(f"\n4. Rendering demonstration:")
    print("To render triangle: renderer.render_mesh('triangle')")
    print("To render cube: renderer.render_mesh('cube')")

    # Cleanup
    renderer.cleanup()
    print("\n5. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_vertex_buffer_objects()
