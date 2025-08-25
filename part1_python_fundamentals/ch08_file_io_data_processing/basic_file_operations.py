#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
Basic File Operations Example

Demonstrates fundamental file reading and writing operations,
including text files, binary files, and error handling for 3D graphics applications.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics File I/O Library"
__description__ = "Basic file operations for 3D graphics applications"

# ============================================================================
# FILE OPERATIONS CLASS
# ============================================================================

class FileOperations:
    """Class for handling basic file operations"""
    
    def __init__(self, base_directory: str = "data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.encoding = "utf-8"
    
    def write_text_file(self, filename: str, content: str, encoding: Optional[str] = None) -> bool:
        """Write text content to a file"""
        try:
            file_path = self.base_directory / filename
            encoding = encoding or self.encoding
            
            with open(file_path, 'w', encoding=encoding) as file:
                file.write(content)
            
            print(f"Successfully wrote text file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing text file {filename}: {e}")
            return False
    
    def read_text_file(self, filename: str, encoding: Optional[str] = None) -> Optional[str]:
        """Read text content from a file"""
        try:
            file_path = self.base_directory / filename
            encoding = encoding or self.encoding
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            print(f"Successfully read text file: {file_path}")
            return content
            
        except Exception as e:
            print(f"Error reading text file {filename}: {e}")
            return None
    
    def write_binary_file(self, filename: str, data: bytes) -> bool:
        """Write binary data to a file"""
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
        """Read binary data from a file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as file:
                data = file.read()
            
            print(f"Successfully read binary file: {file_path}")
            return data
            
        except Exception as e:
            print(f"Error reading binary file {filename}: {e}")
            return None
    
    def append_text_file(self, filename: str, content: str, encoding: Optional[str] = None) -> bool:
        """Append text content to a file"""
        try:
            file_path = self.base_directory / filename
            encoding = encoding or self.encoding
            
            with open(file_path, 'a', encoding=encoding) as file:
                file.write(content)
            
            print(f"Successfully appended to file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error appending to file {filename}: {e}")
            return False
    
    def file_exists(self, filename: str) -> bool:
        """Check if a file exists"""
        file_path = self.base_directory / filename
        return file_path.exists()
    
    def get_file_size(self, filename: str) -> Optional[int]:
        """Get file size in bytes"""
        try:
            file_path = self.base_directory / filename
            if file_path.exists():
                return file_path.stat().st_size
            return None
        except Exception as e:
            print(f"Error getting file size for {filename}: {e}")
            return None
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file"""
        try:
            file_path = self.base_directory / filename
            if file_path.exists():
                file_path.unlink()
                print(f"Successfully deleted file: {file_path}")
                return True
            else:
                print(f"File not found: {file_path}")
                return False
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
            return False

# ============================================================================
# 3D GRAPHICS FILE HANDLERS
# ============================================================================

class GraphicsFileHandler:
    """Specialized file handler for 3D graphics applications"""
    
    def __init__(self, base_directory: str = "graphics_data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.file_ops = FileOperations(str(self.base_directory))
    
    def save_scene_config(self, scene_name: str, config_data: Dict[str, Any]) -> bool:
        """Save 3D scene configuration to JSON-like format"""
        try:
            filename = f"{scene_name}_config.txt"
            content = self._dict_to_config_format(config_data)
            return self.file_ops.write_text_file(filename, content)
        except Exception as e:
            print(f"Error saving scene config: {e}")
            return False
    
    def load_scene_config(self, scene_name: str) -> Optional[Dict[str, Any]]:
        """Load 3D scene configuration from file"""
        try:
            filename = f"{scene_name}_config.txt"
            content = self.file_ops.read_text_file(filename)
            if content:
                return self._config_format_to_dict(content)
            return None
        except Exception as e:
            print(f"Error loading scene config: {e}")
            return None
    
    def save_vertex_data(self, mesh_name: str, vertices: List[List[float]], 
                        normals: Optional[List[List[float]]] = None,
                        tex_coords: Optional[List[List[float]]] = None) -> bool:
        """Save vertex data to binary file"""
        try:
            filename = f"{mesh_name}_vertices.bin"
            
            # Convert to binary format
            data = bytearray()
            
            # Write vertex count
            vertex_count = len(vertices)
            data.extend(vertex_count.to_bytes(4, byteorder='little'))
            
            # Write vertices
            for vertex in vertices:
                for component in vertex:
                    data.extend(int(component * 1000).to_bytes(4, byteorder='little'))
            
            # Write normals if provided
            if normals:
                data.extend(len(normals).to_bytes(4, byteorder='little'))
                for normal in normals:
                    for component in normal:
                        data.extend(int(component * 1000).to_bytes(4, byteorder='little'))
            else:
                data.extend((0).to_bytes(4, byteorder='little'))
            
            # Write texture coordinates if provided
            if tex_coords:
                data.extend(len(tex_coords).to_bytes(4, byteorder='little'))
                for tex_coord in tex_coords:
                    for component in tex_coord:
                        data.extend(int(component * 1000).to_bytes(4, byteorder='little'))
            else:
                data.extend((0).to_bytes(4, byteorder='little'))
            
            return self.file_ops.write_binary_file(filename, bytes(data))
            
        except Exception as e:
            print(f"Error saving vertex data: {e}")
            return False
    
    def load_vertex_data(self, mesh_name: str) -> Optional[Dict[str, Any]]:
        """Load vertex data from binary file"""
        try:
            filename = f"{mesh_name}_vertices.bin"
            data = self.file_ops.read_binary_file(filename)
            
            if not data:
                return None
            
            # Parse binary data
            offset = 0
            
            # Read vertex count
            vertex_count = int.from_bytes(data[offset:offset+4], byteorder='little')
            offset += 4
            
            # Read vertices
            vertices = []
            for i in range(vertex_count):
                vertex = []
                for j in range(3):  # x, y, z
                    component = int.from_bytes(data[offset:offset+4], byteorder='little')
                    vertex.append(component / 1000.0)
                    offset += 4
                vertices.append(vertex)
            
            # Read normals
            normal_count = int.from_bytes(data[offset:offset+4], byteorder='little')
            offset += 4
            
            normals = None
            if normal_count > 0:
                normals = []
                for i in range(normal_count):
                    normal = []
                    for j in range(3):  # x, y, z
                        component = int.from_bytes(data[offset:offset+4], byteorder='little')
                        normal.append(component / 1000.0)
                        offset += 4
                    normals.append(normal)
            
            # Read texture coordinates
            tex_count = int.from_bytes(data[offset:offset+4], byteorder='little')
            offset += 4
            
            tex_coords = None
            if tex_count > 0:
                tex_coords = []
                for i in range(tex_count):
                    tex_coord = []
                    for j in range(2):  # u, v
                        component = int.from_bytes(data[offset:offset+4], byteorder='little')
                        tex_coord.append(component / 1000.0)
                        offset += 4
                    tex_coords.append(tex_coord)
            
            return {
                'vertices': vertices,
                'normals': normals,
                'tex_coords': tex_coords
            }
            
        except Exception as e:
            print(f"Error loading vertex data: {e}")
            return None
    
    def save_material_data(self, material_name: str, material_data: Dict[str, Any]) -> bool:
        """Save material data to text file"""
        try:
            filename = f"{material_name}_material.txt"
            content = self._dict_to_config_format(material_data)
            return self.file_ops.write_text_file(filename, content)
        except Exception as e:
            print(f"Error saving material data: {e}")
            return False
    
    def load_material_data(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Load material data from text file"""
        try:
            filename = f"{material_name}_material.txt"
            content = self.file_ops.read_text_file(filename)
            if content:
                return self._config_format_to_dict(content)
            return None
        except Exception as e:
            print(f"Error loading material data: {e}")
            return None
    
    def _dict_to_config_format(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to simple config format"""
        lines = []
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value_str = ','.join(str(v) for v in value)
            else:
                value_str = str(value)
            lines.append(f"{key}={value_str}")
        return '\n'.join(lines)
    
    def _config_format_to_dict(self, content: str) -> Dict[str, Any]:
        """Convert config format to dictionary"""
        data = {}
        for line in content.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    if ',' in value:
                        # List of numbers
                        data[key] = [float(x.strip()) for x in value.split(',')]
                    elif value.lower() in ('true', 'false'):
                        data[key] = value.lower() == 'true'
                    elif '.' in value:
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value
        
        return data

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_file_operations():
    """Demonstrate basic file operations"""
    print("=== Basic File Operations Demo ===\n")
    
    # Create file operations instance
    file_ops = FileOperations("demo_data")
    
    # Write text file
    print("1. Writing text file...")
    text_content = """This is a sample text file for 3D graphics data.
It contains multiple lines of information.
Vertex data: 1.0, 2.0, 3.0
Normal data: 0.0, 1.0, 0.0
Texture coordinates: 0.5, 0.5"""
    
    file_ops.write_text_file("sample.txt", text_content)
    
    # Read text file
    print("\n2. Reading text file...")
    read_content = file_ops.read_text_file("sample.txt")
    if read_content:
        print("File content:")
        print(read_content)
    
    # Write binary file
    print("\n3. Writing binary file...")
    binary_data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
    file_ops.write_binary_file("sample.bin", binary_data)
    
    # Read binary file
    print("\n4. Reading binary file...")
    read_binary = file_ops.read_binary_file("sample.bin")
    if read_binary:
        print(f"Binary data: {read_binary}")
        print(f"Data length: {len(read_binary)} bytes")
    
    # Append to file
    print("\n5. Appending to file...")
    append_content = "\n\nAdditional data appended to the file."
    file_ops.append_text_file("sample.txt", append_content)
    
    # Check file size
    print("\n6. Checking file size...")
    size = file_ops.get_file_size("sample.txt")
    if size is not None:
        print(f"File size: {size} bytes")
    
    print()

def demonstrate_graphics_file_handling():
    """Demonstrate 3D graphics file handling"""
    print("=== 3D Graphics File Handling Demo ===\n")
    
    # Create graphics file handler
    graphics_handler = GraphicsFileHandler("graphics_demo")
    
    # Save scene configuration
    print("1. Saving scene configuration...")
    scene_config = {
        'name': 'demo_scene',
        'camera_position': [0.0, 5.0, -10.0],
        'camera_target': [0.0, 0.0, 0.0],
        'light_position': [10.0, 10.0, 10.0],
        'light_color': [1.0, 1.0, 1.0],
        'ambient_light': [0.1, 0.1, 0.1],
        'fog_enabled': True,
        'fog_density': 0.01
    }
    
    graphics_handler.save_scene_config("demo_scene", scene_config)
    
    # Load scene configuration
    print("\n2. Loading scene configuration...")
    loaded_config = graphics_handler.load_scene_config("demo_scene")
    if loaded_config:
        print("Loaded scene config:")
        for key, value in loaded_config.items():
            print(f"  {key}: {value}")
    
    # Save vertex data
    print("\n3. Saving vertex data...")
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
    
    graphics_handler.save_vertex_data("quad", vertices, normals, tex_coords)
    
    # Load vertex data
    print("\n4. Loading vertex data...")
    loaded_vertex_data = graphics_handler.load_vertex_data("quad")
    if loaded_vertex_data:
        print("Loaded vertex data:")
        print(f"  Vertices: {len(loaded_vertex_data['vertices'])}")
        print(f"  Normals: {len(loaded_vertex_data['normals']) if loaded_vertex_data['normals'] else 0}")
        print(f"  Tex coords: {len(loaded_vertex_data['tex_coords']) if loaded_vertex_data['tex_coords'] else 0}")
        
        print("  First vertex:", loaded_vertex_data['vertices'][0])
        print("  First normal:", loaded_vertex_data['normals'][0])
        print("  First tex coord:", loaded_vertex_data['tex_coords'][0])
    
    # Save material data
    print("\n5. Saving material data...")
    material_data = {
        'name': 'default_material',
        'diffuse_color': [1.0, 0.5, 0.2],
        'specular_color': [1.0, 1.0, 1.0],
        'ambient_color': [0.1, 0.05, 0.02],
        'shininess': 32.0,
        'opacity': 1.0,
        'texture_file': 'diffuse.png',
        'normal_map': 'normal.png'
    }
    
    graphics_handler.save_material_data("default", material_data)
    
    # Load material data
    print("\n6. Loading material data...")
    loaded_material = graphics_handler.load_material_data("default")
    if loaded_material:
        print("Loaded material data:")
        for key, value in loaded_material.items():
            print(f"  {key}: {value}")
    
    print()

def demonstrate_error_handling():
    """Demonstrate error handling in file operations"""
    print("=== Error Handling Demo ===\n")
    
    file_ops = FileOperations("error_demo")
    
    # Try to read non-existent file
    print("1. Reading non-existent file...")
    content = file_ops.read_text_file("nonexistent.txt")
    print(f"Result: {content}")
    
    # Try to write to invalid path
    print("\n2. Writing to invalid path...")
    # Create a file handler with invalid path
    invalid_ops = FileOperations("/invalid/path/that/does/not/exist")
    result = invalid_ops.write_text_file("test.txt", "test content")
    print(f"Result: {result}")
    
    # Try to delete non-existent file
    print("\n3. Deleting non-existent file...")
    result = file_ops.delete_file("nonexistent.txt")
    print(f"Result: {result}")
    
    # Check file existence
    print("\n4. Checking file existence...")
    exists = file_ops.file_exists("nonexistent.txt")
    print(f"File exists: {exists}")
    
    print()

def demonstrate_file_management():
    """Demonstrate file management operations"""
    print("=== File Management Demo ===\n")
    
    file_ops = FileOperations("management_demo")
    
    # Create multiple files
    print("1. Creating multiple files...")
    files_to_create = [
        ("file1.txt", "Content of file 1"),
        ("file2.txt", "Content of file 2"),
        ("file3.txt", "Content of file 3"),
        ("data.bin", b'\x01\x02\x03\x04\x05')
    ]
    
    for filename, content in files_to_create:
        if isinstance(content, str):
            file_ops.write_text_file(filename, content)
        else:
            file_ops.write_binary_file(filename, content)
    
    # Check file sizes
    print("\n2. Checking file sizes...")
    for filename, _ in files_to_create:
        size = file_ops.get_file_size(filename)
        if size is not None:
            print(f"  {filename}: {size} bytes")
    
    # Delete files
    print("\n3. Deleting files...")
    for filename, _ in files_to_create:
        file_ops.delete_file(filename)
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate file operations"""
    print("=== File I/O and Data Processing Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_file_operations()
    demonstrate_graphics_file_handling()
    demonstrate_error_handling()
    demonstrate_file_management()
    
    print("="*60)
    print("File I/O demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic file operations: Read, write, append, delete")
    print("✓ Text and binary file handling")
    print("✓ 3D graphics file formats: Scene configs, vertex data, materials")
    print("✓ Error handling and file validation")
    print("✓ File management and size checking")
    print("✓ Custom data format parsing")

if __name__ == "__main__":
    main()
