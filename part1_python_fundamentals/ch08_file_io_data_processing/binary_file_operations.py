#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
Binary File Operations Example

Demonstrates binary file operations for efficient data storage and retrieval
in 3D graphics applications.
"""

import struct
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from datetime import datetime
import math
import zlib
import pickle

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Binary Handler"
__description__ = "Binary file operations for 3D graphics applications"

# ============================================================================
# BINARY FILE HANDLER CLASS
# ============================================================================

class BinaryFileHandler:
    """Class for handling binary file operations"""
    
    def __init__(self, base_directory: str = "binary_data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
    
    def write_binary_file(self, filename: str, data: bytes) -> bool:
        """Write binary data to file"""
        try:
            file_path = self.base_directory / filename
            
            with open(file_path, 'wb') as file:
                file.write(data)
            
            print(f"Successfully wrote binary file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing binary file {filename}: {e}")
            return False
    
    def read_binary_file(self, filename: str) -> Optional[bytes]:
        """Read binary data from file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"Binary file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as file:
                data = file.read()
            
            print(f"Successfully read binary file: {file_path}")
            return data
            
        except Exception as e:
            print(f"Error reading binary file {filename}: {e}")
            return None
    
    def append_binary_file(self, filename: str, data: bytes) -> bool:
        """Append binary data to existing file"""
        try:
            file_path = self.base_directory / filename
            
            with open(file_path, 'ab') as file:
                file.write(data)
            
            print(f"Successfully appended to binary file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error appending to binary file {filename}: {e}")
            return False
    
    def get_binary_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about binary file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"Binary file not found: {file_path}")
                return None
            
            stat = file_path.stat()
            info = {
                "filename": filename,
                "file_size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_file": file_path.is_file(),
                "is_readable": os.access(file_path, os.R_OK),
                "is_writable": os.access(file_path, os.W_OK)
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting binary file info for {filename}: {e}")
            return None

# ============================================================================
# BINARY DATA SERIALIZER
# ============================================================================

class BinaryDataSerializer:
    """Class for serializing data to binary format"""
    
    @staticmethod
    def serialize_float_list(data: List[float]) -> bytes:
        """Serialize a list of floats to binary"""
        return struct.pack(f'{len(data)}f', *data)
    
    @staticmethod
    def serialize_int_list(data: List[int]) -> bytes:
        """Serialize a list of integers to binary"""
        return struct.pack(f'{len(data)}i', *data)
    
    @staticmethod
    def serialize_vertex_data(vertices: List[List[float]], 
                            normals: Optional[List[List[float]]] = None,
                            tex_coords: Optional[List[List[float]]] = None) -> bytes:
        """Serialize vertex data to binary format"""
        try:
            # Header: vertex count, has normals, has tex coords
            vertex_count = len(vertices)
            has_normals = normals is not None and len(normals) == vertex_count
            has_tex_coords = tex_coords is not None and len(tex_coords) == vertex_count
            
            header = struct.pack('III', vertex_count, int(has_normals), int(has_tex_coords))
            
            # Serialize vertices
            vertex_data = b''
            for vertex in vertices:
                vertex_data += struct.pack('3f', *vertex)
            
            # Serialize normals if present
            normal_data = b''
            if has_normals:
                for normal in normals:
                    normal_data += struct.pack('3f', *normal)
            
            # Serialize texture coordinates if present
            tex_data = b''
            if has_tex_coords:
                for tex_coord in tex_coords:
                    tex_data += struct.pack('2f', *tex_coord)
            
            return header + vertex_data + normal_data + tex_data
            
        except Exception as e:
            print(f"Error serializing vertex data: {e}")
            return b''
    
    @staticmethod
    def serialize_mesh_data(mesh_name: str, vertices: List[List[float]], 
                          indices: List[int], material_id: int = 0) -> bytes:
        """Serialize mesh data to binary format"""
        try:
            # Header: name length, vertex count, index count, material id
            name_bytes = mesh_name.encode('utf-8')
            name_length = len(name_bytes)
            vertex_count = len(vertices)
            index_count = len(indices)
            
            header = struct.pack('IIIII', name_length, vertex_count, index_count, material_id, 0)  # Padding
            
            # Serialize name
            name_data = struct.pack(f'{name_length}s', name_bytes)
            
            # Serialize vertices
            vertex_data = b''
            for vertex in vertices:
                vertex_data += struct.pack('3f', *vertex)
            
            # Serialize indices
            index_data = struct.pack(f'{index_count}I', *indices)
            
            return header + name_data + vertex_data + index_data
            
        except Exception as e:
            print(f"Error serializing mesh data: {e}")
            return b''
    
    @staticmethod
    def serialize_transform_data(position: List[float], rotation: List[float], 
                               scale: List[float]) -> bytes:
        """Serialize transform data to binary format"""
        try:
            # Pack position, rotation, scale as 3 floats each
            data = struct.pack('9f', 
                             position[0], position[1], position[2],
                             rotation[0], rotation[1], rotation[2],
                             scale[0], scale[1], scale[2])
            return data
            
        except Exception as e:
            print(f"Error serializing transform data: {e}")
            return b''
    
    @staticmethod
    def serialize_material_data(material_name: str, diffuse_color: List[float],
                              specular_color: List[float], shininess: float) -> bytes:
        """Serialize material data to binary format"""
        try:
            # Header: name length, shininess
            name_bytes = material_name.encode('utf-8')
            name_length = len(name_bytes)
            
            header = struct.pack('If', name_length, shininess)
            
            # Serialize name
            name_data = struct.pack(f'{name_length}s', name_bytes)
            
            # Serialize colors
            color_data = struct.pack('6f', 
                                   diffuse_color[0], diffuse_color[1], diffuse_color[2],
                                   specular_color[0], specular_color[1], specular_color[2])
            
            return header + name_data + color_data
            
        except Exception as e:
            print(f"Error serializing material data: {e}")
            return b''

# ============================================================================
# BINARY DATA DESERIALIZER
# ============================================================================

class BinaryDataDeserializer:
    """Class for deserializing data from binary format"""
    
    @staticmethod
    def deserialize_float_list(data: bytes) -> List[float]:
        """Deserialize a list of floats from binary"""
        try:
            count = len(data) // 4  # 4 bytes per float
            return list(struct.unpack(f'{count}f', data))
        except Exception as e:
            print(f"Error deserializing float list: {e}")
            return []
    
    @staticmethod
    def deserialize_int_list(data: bytes) -> List[int]:
        """Deserialize a list of integers from binary"""
        try:
            count = len(data) // 4  # 4 bytes per int
            return list(struct.unpack(f'{count}i', data))
        except Exception as e:
            print(f"Error deserializing int list: {e}")
            return []
    
    @staticmethod
    def deserialize_vertex_data(data: bytes) -> Dict[str, Any]:
        """Deserialize vertex data from binary format"""
        try:
            # Read header
            header_size = struct.calcsize('III')
            vertex_count, has_normals, has_tex_coords = struct.unpack('III', data[:header_size])
            
            offset = header_size
            
            # Read vertices
            vertex_size = vertex_count * 3 * 4  # 3 floats per vertex, 4 bytes per float
            vertex_data = data[offset:offset + vertex_size]
            vertices = []
            for i in range(vertex_count):
                vertex = struct.unpack('3f', vertex_data[i*12:(i+1)*12])
                vertices.append(list(vertex))
            
            offset += vertex_size
            
            # Read normals if present
            normals = None
            if has_normals:
                normal_size = vertex_count * 3 * 4
                normal_data = data[offset:offset + normal_size]
                normals = []
                for i in range(vertex_count):
                    normal = struct.unpack('3f', normal_data[i*12:(i+1)*12])
                    normals.append(list(normal))
                offset += normal_size
            
            # Read texture coordinates if present
            tex_coords = None
            if has_tex_coords:
                tex_size = vertex_count * 2 * 4  # 2 floats per tex coord
                tex_data = data[offset:offset + tex_size]
                tex_coords = []
                for i in range(vertex_count):
                    tex_coord = struct.unpack('2f', tex_data[i*8:(i+1)*8])
                    tex_coords.append(list(tex_coord))
            
            return {
                "vertices": vertices,
                "normals": normals,
                "tex_coords": tex_coords
            }
            
        except Exception as e:
            print(f"Error deserializing vertex data: {e}")
            return {}
    
    @staticmethod
    def deserialize_mesh_data(data: bytes) -> Dict[str, Any]:
        """Deserialize mesh data from binary format"""
        try:
            # Read header
            header_size = struct.calcsize('IIIII')
            name_length, vertex_count, index_count, material_id, padding = struct.unpack('IIIII', data[:header_size])
            
            offset = header_size
            
            # Read name
            name_data = data[offset:offset + name_length]
            mesh_name = name_data.decode('utf-8')
            offset += name_length
            
            # Read vertices
            vertex_size = vertex_count * 3 * 4
            vertex_data = data[offset:offset + vertex_size]
            vertices = []
            for i in range(vertex_count):
                vertex = struct.unpack('3f', vertex_data[i*12:(i+1)*12])
                vertices.append(list(vertex))
            offset += vertex_size
            
            # Read indices
            index_size = index_count * 4
            index_data = data[offset:offset + index_size]
            indices = list(struct.unpack(f'{index_count}I', index_data))
            
            return {
                "name": mesh_name,
                "vertices": vertices,
                "indices": indices,
                "material_id": material_id
            }
            
        except Exception as e:
            print(f"Error deserializing mesh data: {e}")
            return {}
    
    @staticmethod
    def deserialize_transform_data(data: bytes) -> Dict[str, List[float]]:
        """Deserialize transform data from binary format"""
        try:
            # Unpack position, rotation, scale
            values = struct.unpack('9f', data)
            
            return {
                "position": list(values[0:3]),
                "rotation": list(values[3:6]),
                "scale": list(values[6:9])
            }
            
        except Exception as e:
            print(f"Error deserializing transform data: {e}")
            return {}
    
    @staticmethod
    def deserialize_material_data(data: bytes) -> Dict[str, Any]:
        """Deserialize material data from binary format"""
        try:
            # Read header
            header_size = struct.calcsize('If')
            name_length, shininess = struct.unpack('If', data[:header_size])
            
            offset = header_size
            
            # Read name
            name_data = data[offset:offset + name_length]
            material_name = name_data.decode('utf-8')
            offset += name_length
            
            # Read colors
            color_data = data[offset:offset + 24]  # 6 floats * 4 bytes
            colors = struct.unpack('6f', color_data)
            
            return {
                "name": material_name,
                "diffuse_color": list(colors[0:3]),
                "specular_color": list(colors[3:6]),
                "shininess": shininess
            }
            
        except Exception as e:
            print(f"Error deserializing material data: {e}")
            return {}

# ============================================================================
# 3D GRAPHICS BINARY PROCESSOR
# ============================================================================

class GraphicsBinaryProcessor:
    """Specialized binary processor for 3D graphics applications"""
    
    def __init__(self, base_directory: str = "graphics_binary"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.binary_handler = BinaryFileHandler(str(self.base_directory))
        self.serializer = BinaryDataSerializer()
        self.deserializer = BinaryDataDeserializer()
    
    def save_vertex_data(self, mesh_name: str, vertices: List[List[float]], 
                        normals: Optional[List[List[float]]] = None,
                        tex_coords: Optional[List[List[float]]] = None) -> bool:
        """Save vertex data to binary file"""
        try:
            # Serialize vertex data
            binary_data = self.serializer.serialize_vertex_data(vertices, normals, tex_coords)
            
            # Compress data
            compressed_data = zlib.compress(binary_data)
            
            # Save to file
            filename = f"{mesh_name}_vertices.bin"
            return self.binary_handler.write_binary_file(filename, compressed_data)
            
        except Exception as e:
            print(f"Error saving vertex data: {e}")
            return False
    
    def load_vertex_data(self, mesh_name: str) -> Optional[Dict[str, Any]]:
        """Load vertex data from binary file"""
        try:
            filename = f"{mesh_name}_vertices.bin"
            compressed_data = self.binary_handler.read_binary_file(filename)
            
            if compressed_data is None:
                return None
            
            # Decompress data
            binary_data = zlib.decompress(compressed_data)
            
            # Deserialize data
            return self.deserializer.deserialize_vertex_data(binary_data)
            
        except Exception as e:
            print(f"Error loading vertex data: {e}")
            return None
    
    def save_mesh_data(self, mesh_name: str, vertices: List[List[float]], 
                      indices: List[int], material_id: int = 0) -> bool:
        """Save mesh data to binary file"""
        try:
            # Serialize mesh data
            binary_data = self.serializer.serialize_mesh_data(mesh_name, vertices, indices, material_id)
            
            # Compress data
            compressed_data = zlib.compress(binary_data)
            
            # Save to file
            filename = f"{mesh_name}_mesh.bin"
            return self.binary_handler.write_binary_file(filename, compressed_data)
            
        except Exception as e:
            print(f"Error saving mesh data: {e}")
            return False
    
    def load_mesh_data(self, mesh_name: str) -> Optional[Dict[str, Any]]:
        """Load mesh data from binary file"""
        try:
            filename = f"{mesh_name}_mesh.bin"
            compressed_data = self.binary_handler.read_binary_file(filename)
            
            if compressed_data is None:
                return None
            
            # Decompress data
            binary_data = zlib.decompress(compressed_data)
            
            # Deserialize data
            return self.deserializer.deserialize_mesh_data(binary_data)
            
        except Exception as e:
            print(f"Error loading mesh data: {e}")
            return None
    
    def save_scene_data(self, scene_name: str, objects: List[Dict[str, Any]]) -> bool:
        """Save scene data to binary file"""
        try:
            # Create scene header
            object_count = len(objects)
            header = struct.pack('I', object_count)
            
            scene_data = header
            
            # Serialize each object
            for obj in objects:
                # Object type (1 byte)
                obj_type = obj.get('type', 'mesh').encode('utf-8')[:15].ljust(16, b'\0')
                scene_data += obj_type
                
                # Transform data
                position = obj.get('position', [0.0, 0.0, 0.0])
                rotation = obj.get('rotation', [0.0, 0.0, 0.0])
                scale = obj.get('scale', [1.0, 1.0, 1.0])
                transform_data = self.serializer.serialize_transform_data(position, rotation, scale)
                scene_data += transform_data
                
                # Object ID
                obj_id = obj.get('id', 0)
                scene_data += struct.pack('I', obj_id)
                
                # Material ID
                material_id = obj.get('material_id', 0)
                scene_data += struct.pack('I', material_id)
            
            # Compress data
            compressed_data = zlib.compress(scene_data)
            
            # Save to file
            filename = f"{scene_name}_scene.bin"
            return self.binary_handler.write_binary_file(filename, compressed_data)
            
        except Exception as e:
            print(f"Error saving scene data: {e}")
            return False
    
    def load_scene_data(self, scene_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load scene data from binary file"""
        try:
            filename = f"{scene_name}_scene.bin"
            compressed_data = self.binary_handler.read_binary_file(filename)
            
            if compressed_data is None:
                return None
            
            # Decompress data
            scene_data = zlib.decompress(compressed_data)
            
            # Read header
            object_count = struct.unpack('I', scene_data[:4])[0]
            
            objects = []
            offset = 4
            
            # Read each object
            for i in range(object_count):
                # Object type
                obj_type_data = scene_data[offset:offset + 16]
                obj_type = obj_type_data.decode('utf-8').rstrip('\0')
                offset += 16
                
                # Transform data
                transform_data = scene_data[offset:offset + 36]  # 9 floats * 4 bytes
                transform = self.deserializer.deserialize_transform_data(transform_data)
                offset += 36
                
                # Object ID
                obj_id = struct.unpack('I', scene_data[offset:offset + 4])[0]
                offset += 4
                
                # Material ID
                material_id = struct.unpack('I', scene_data[offset:offset + 4])[0]
                offset += 4
                
                objects.append({
                    'type': obj_type,
                    'position': transform['position'],
                    'rotation': transform['rotation'],
                    'scale': transform['scale'],
                    'id': obj_id,
                    'material_id': material_id
                })
            
            return objects
            
        except Exception as e:
            print(f"Error loading scene data: {e}")
            return None
    
    def save_material_library(self, library_name: str, materials: Dict[str, Dict[str, Any]]) -> bool:
        """Save material library to binary file"""
        try:
            # Create library header
            material_count = len(materials)
            header = struct.pack('I', material_count)
            
            library_data = header
            
            # Serialize each material
            for material_name, material_data in materials.items():
                diffuse_color = material_data.get('diffuse_color', [1.0, 1.0, 1.0])
                specular_color = material_data.get('specular_color', [0.0, 0.0, 0.0])
                shininess = material_data.get('shininess', 32.0)
                
                material_binary = self.serializer.serialize_material_data(
                    material_name, diffuse_color, specular_color, shininess
                )
                library_data += material_binary
            
            # Compress data
            compressed_data = zlib.compress(library_data)
            
            # Save to file
            filename = f"{library_name}_materials.bin"
            return self.binary_handler.write_binary_file(filename, compressed_data)
            
        except Exception as e:
            print(f"Error saving material library: {e}")
            return False

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_binary_operations():
    """Demonstrate basic binary file operations"""
    print("=== Basic Binary File Operations Demo ===\n")
    
    # Create binary file handler
    binary_handler = BinaryFileHandler("binary_demo")
    
    # Create sample binary data
    sample_data = b'Hello, this is binary data!\x00\x01\x02\x03\x04\x05'
    
    # Write binary file
    print("1. Writing binary file...")
    binary_handler.write_binary_file("sample.bin", sample_data)
    
    # Read binary file
    print("\n2. Reading binary file...")
    loaded_data = binary_handler.read_binary_file("sample.bin")
    if loaded_data:
        print(f"Loaded data: {loaded_data}")
        print(f"Data length: {len(loaded_data)} bytes")
    
    # Get file info
    print("\n3. Getting binary file info...")
    info = binary_handler.get_binary_file_info("sample.bin")
    if info:
        print("File info:")
        print(f"  Size: {info['file_size']} bytes")
        print(f"  Created: {info['created']}")
        print(f"  Modified: {info['modified']}")
        print(f"  Readable: {info['is_readable']}")
        print(f"  Writable: {info['is_writable']}")
    
    # Append to binary file
    print("\n4. Appending to binary file...")
    additional_data = b'\x06\x07\x08\x09\x0A'
    binary_handler.append_binary_file("sample.bin", additional_data)
    
    # Read updated file
    print("\n5. Reading updated binary file...")
    updated_data = binary_handler.read_binary_file("sample.bin")
    if updated_data:
        print(f"Updated data length: {len(updated_data)} bytes")
    
    print()

def demonstrate_binary_serialization():
    """Demonstrate binary data serialization"""
    print("=== Binary Data Serialization Demo ===\n")
    
    # Create serializer and deserializer
    serializer = BinaryDataSerializer()
    deserializer = BinaryDataDeserializer()
    
    # Serialize float list
    print("1. Serializing float list...")
    float_list = [1.0, 2.5, 3.14, -1.0, 0.0]
    float_binary = serializer.serialize_float_list(float_list)
    print(f"Original: {float_list}")
    print(f"Binary size: {len(float_binary)} bytes")
    
    # Deserialize float list
    deserialized_floats = deserializer.deserialize_float_list(float_binary)
    print(f"Deserialized: {deserialized_floats}")
    
    # Serialize vertex data
    print("\n2. Serializing vertex data...")
    vertices = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ]
    
    normals = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ]
    
    tex_coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    
    vertex_binary = serializer.serialize_vertex_data(vertices, normals, tex_coords)
    print(f"Vertex count: {len(vertices)}")
    print(f"Binary size: {len(vertex_binary)} bytes")
    
    # Deserialize vertex data
    deserialized_vertex_data = deserializer.deserialize_vertex_data(vertex_binary)
    print(f"Deserialized vertices: {len(deserialized_vertex_data.get('vertices', []))}")
    print(f"Has normals: {deserialized_vertex_data.get('normals') is not None}")
    print(f"Has tex coords: {deserialized_vertex_data.get('tex_coords') is not None}")
    
    # Serialize transform data
    print("\n3. Serializing transform data...")
    position = [1.0, 2.0, 3.0]
    rotation = [45.0, 0.0, 0.0]
    scale = [1.5, 1.5, 1.5]
    
    transform_binary = serializer.serialize_transform_data(position, rotation, scale)
    print(f"Transform binary size: {len(transform_binary)} bytes")
    
    # Deserialize transform data
    deserialized_transform = deserializer.deserialize_transform_data(transform_binary)
    print(f"Deserialized transform: {deserialized_transform}")
    
    print()

def demonstrate_graphics_binary_processing():
    """Demonstrate 3D graphics binary processing"""
    print("=== 3D Graphics Binary Processing Demo ===\n")
    
    # Create graphics binary processor
    graphics_processor = GraphicsBinaryProcessor("graphics_binary_demo")
    
    # Save vertex data
    print("1. Saving vertex data...")
    vertices = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ]
    
    normals = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ]
    
    tex_coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    
    graphics_processor.save_vertex_data("quad", vertices, normals, tex_coords)
    
    # Load vertex data
    print("\n2. Loading vertex data...")
    loaded_vertex_data = graphics_processor.load_vertex_data("quad")
    if loaded_vertex_data:
        print(f"Loaded vertices: {len(loaded_vertex_data['vertices'])}")
        print(f"Has normals: {loaded_vertex_data['normals'] is not None}")
        print(f"Has tex coords: {loaded_vertex_data['tex_coords'] is not None}")
    
    # Save mesh data
    print("\n3. Saving mesh data...")
    mesh_vertices = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0]
    ]
    
    mesh_indices = [
        0, 1, 2, 2, 3, 0,  # Front face
        1, 5, 6, 6, 2, 1,  # Right face
        5, 4, 7, 7, 6, 5,  # Back face
        4, 0, 3, 3, 7, 4,  # Left face
        3, 2, 6, 6, 7, 3,  # Top face
        4, 5, 1, 1, 0, 4   # Bottom face
    ]
    
    graphics_processor.save_mesh_data("cube", mesh_vertices, mesh_indices, material_id=1)
    
    # Load mesh data
    print("\n4. Loading mesh data...")
    loaded_mesh_data = graphics_processor.load_mesh_data("cube")
    if loaded_mesh_data:
        print(f"Mesh name: {loaded_mesh_data['name']}")
        print(f"Vertex count: {len(loaded_mesh_data['vertices'])}")
        print(f"Index count: {len(loaded_mesh_data['indices'])}")
        print(f"Material ID: {loaded_mesh_data['material_id']}")
    
    # Save scene data
    print("\n5. Saving scene data...")
    scene_objects = [
        {
            'type': 'mesh',
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0],
            'id': 1,
            'material_id': 1
        },
        {
            'type': 'light',
            'position': [5.0, 5.0, 5.0],
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0],
            'id': 2,
            'material_id': 0
        },
        {
            'type': 'camera',
            'position': [0.0, 3.0, -5.0],
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0],
            'id': 3,
            'material_id': 0
        }
    ]
    
    graphics_processor.save_scene_data("demo_scene", scene_objects)
    
    # Load scene data
    print("\n6. Loading scene data...")
    loaded_scene_data = graphics_processor.load_scene_data("demo_scene")
    if loaded_scene_data:
        print(f"Scene objects: {len(loaded_scene_data)}")
        for i, obj in enumerate(loaded_scene_data):
            print(f"  Object {i+1}: {obj['type']} at {obj['position']}")
    
    print()

def demonstrate_binary_utilities():
    """Demonstrate binary utility functions"""
    print("=== Binary Utilities Demo ===\n")
    
    # Create binary file handler
    binary_handler = BinaryFileHandler("utilities_demo")
    
    # Create complex binary data for utilities demonstration
    print("1. Creating complex binary data...")
    
    # Create a custom binary format for 3D object data
    object_count = 3
    
    # Header: object count, version, timestamp
    header = struct.pack('III', object_count, 1, int(datetime.now().timestamp()))
    
    # Object 1: Cube
    cube_data = struct.pack('16s9fII', 
                           b'cube'.ljust(16, b'\0'),  # Name (16 bytes)
                           0.0, 0.0, 0.0,            # Position
                           0.0, 45.0, 0.0,           # Rotation
                           1.0, 1.0, 1.0,            # Scale
                           1, 1)                      # Object ID, Material ID
    
    # Object 2: Sphere
    sphere_data = struct.pack('16s9fII',
                             b'sphere'.ljust(16, b'\0'),  # Name (16 bytes)
                             2.0, 0.0, 0.0,              # Position
                             0.0, 0.0, 0.0,              # Rotation
                             0.5, 0.5, 0.5,              # Scale
                             2, 2)                        # Object ID, Material ID
    
    # Object 3: Light
    light_data = struct.pack('16s9fII',
                            b'light'.ljust(16, b'\0'),    # Name (16 bytes)
                            5.0, 5.0, 5.0,               # Position
                            0.0, 0.0, 0.0,               # Rotation
                            1.0, 1.0, 1.0,               # Scale
                            3, 0)                         # Object ID, Material ID
    
    # Combine all data
    complex_data = header + cube_data + sphere_data + light_data
    
    # Compress the data
    compressed_data = zlib.compress(complex_data)
    
    # Save complex data
    print("2. Saving complex binary data...")
    binary_handler.write_binary_file("complex_objects.bin", compressed_data)
    
    # Read and analyze the data
    print("\n3. Reading and analyzing complex data...")
    loaded_compressed = binary_handler.read_binary_file("complex_objects.bin")
    if loaded_compressed:
        # Decompress data
        loaded_data = zlib.decompress(loaded_compressed)
        
        # Parse header
        header_size = struct.calcsize('III')
        object_count, version, timestamp = struct.unpack('III', loaded_data[:header_size])
        
        print(f"Object count: {object_count}")
        print(f"Version: {version}")
        print(f"Timestamp: {datetime.fromtimestamp(timestamp)}")
        
        # Parse objects
        offset = header_size
        object_size = struct.calcsize('16s9fII')  # 16 bytes name + 9 floats + 2 ints
        
        for i in range(object_count):
            obj_data = loaded_data[offset:offset + object_size]
            name, px, py, pz, rx, ry, rz, sx, sy, sz, obj_id, mat_id = struct.unpack('16s9fII', obj_data)
            
            # Decode name
            obj_name = name.decode('utf-8').rstrip('\0')
            
            print(f"\nObject {i+1}:")
            print(f"  Name: {obj_name}")
            print(f"  Position: [{px}, {py}, {pz}]")
            print(f"  Rotation: [{rx}, {ry}, {rz}]")
            print(f"  Scale: [{sx}, {sy}, {sz}]")
            print(f"  Object ID: {obj_id}")
            print(f"  Material ID: {mat_id}")
            
            offset += object_size
    
    # Get file information
    print("\n4. Getting file information...")
    info = binary_handler.get_binary_file_info("complex_objects.bin")
    if info:
        print(f"File size: {info['file_size']} bytes")
        print(f"Compression ratio: {len(complex_data) / info['file_size']:.2f}x")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate binary file operations"""
    print("=== Binary File Operations Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_binary_operations()
    demonstrate_binary_serialization()
    demonstrate_graphics_binary_processing()
    demonstrate_binary_utilities()
    
    print("="*60)
    print("Binary File Operations demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic binary operations: Read, write, append, info")
    print("✓ Data serialization: Float lists, vertex data, transforms")
    print("✓ Data deserialization: Binary to Python objects")
    print("✓ 3D graphics data: Vertex data, mesh data, scene data")
    print("✓ Compression: zlib compression for efficient storage")
    print("✓ Custom formats: Structured binary data formats")

if __name__ == "__main__":
    main()
