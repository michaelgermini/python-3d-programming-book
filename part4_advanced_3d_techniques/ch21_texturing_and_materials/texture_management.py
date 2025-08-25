"""
Chapter 21: Texturing and Materials - Texture Management
=======================================================

This module demonstrates texture management and loading systems for 3D graphics.

Key Concepts:
- Texture loading and caching
- Texture coordinates and mapping
- Texture filtering and mipmapping
- Texture compression and optimization
"""

import os
import hashlib
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import OpenGL.GL as gl


class TextureFormat(Enum):
    """Texture format enumeration."""
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    NORMAL_MAP = "NORMAL"
    HEIGHT_MAP = "HEIGHT"


class TextureFilter(Enum):
    """Texture filtering modes."""
    NEAREST = gl.GL_NEAREST
    LINEAR = gl.GL_LINEAR
    NEAREST_MIPMAP_NEAREST = gl.GL_NEAREST_MIPMAP_NEAREST
    LINEAR_MIPMAP_NEAREST = gl.GL_LINEAR_MIPMAP_NEAREST
    NEAREST_MIPMAP_LINEAR = gl.GL_NEAREST_MIPMAP_LINEAR
    LINEAR_MIPMAP_LINEAR = gl.GL_LINEAR_MIPMAP_LINEAR


class TextureWrap(Enum):
    """Texture wrapping modes."""
    CLAMP = gl.GL_CLAMP_TO_EDGE
    REPEAT = gl.GL_REPEAT
    MIRROR = gl.GL_MIRRORED_REPEAT


@dataclass
class TextureInfo:
    """Information about a texture."""
    width: int
    height: int
    format: TextureFormat
    channels: int
    data_size: int
    mipmap_levels: int = 1


class Texture:
    """Represents a texture with OpenGL binding."""
    
    def __init__(self, texture_id: int, info: TextureInfo, name: str = ""):
        self.texture_id = texture_id
        self.info = info
        self.name = name
        self.bound = False
        self.last_used = 0.0
        
    def bind(self, unit: int = 0):
        """Bind texture to texture unit."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        self.bound = True
        
    def unbind(self):
        """Unbind texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.bound = False
        
    def set_filtering(self, min_filter: TextureFilter, mag_filter: TextureFilter):
        """Set texture filtering modes."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter.value)
        
    def set_wrapping(self, wrap_s: TextureWrap, wrap_t: TextureWrap):
        """Set texture wrapping modes."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_s.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_t.value)
        
    def generate_mipmaps(self):
        """Generate mipmaps for the texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        self.info.mipmap_levels = int(np.log2(max(self.info.width, self.info.height))) + 1


class TextureLoader:
    """Handles loading and processing of texture files."""
    
    def __init__(self):
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tiff'}
        
    def load_image(self, filepath: str) -> Optional[Tuple[np.ndarray, TextureInfo]]:
        """Load image from file and return data with info."""
        if not os.path.exists(filepath):
            print(f"Texture file not found: {filepath}")
            return None
            
        try:
            with Image.open(filepath) as img:
                # Convert to RGBA if needed
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                    
                # Convert to numpy array
                data = np.array(img)
                
                # Create texture info
                info = TextureInfo(
                    width=img.width,
                    height=img.height,
                    format=TextureFormat.RGBA,
                    channels=4,
                    data_size=data.nbytes
                )
                
                return data, info
                
        except Exception as e:
            print(f"Error loading texture {filepath}: {e}")
            return None
            
    def create_procedural_texture(self, width: int, height: int, 
                                  texture_type: str = "checkerboard") -> Tuple[np.ndarray, TextureInfo]:
        """Create procedural texture."""
        if texture_type == "checkerboard":
            data = self._create_checkerboard(width, height)
        elif texture_type == "gradient":
            data = self._create_gradient(width, height)
        elif texture_type == "noise":
            data = self._create_noise(width, height)
        else:
            data = self._create_solid_color(width, height)
            
        info = TextureInfo(
            width=width,
            height=height,
            format=TextureFormat.RGBA,
            channels=4,
            data_size=data.nbytes
        )
        
        return data, info
        
    def _create_checkerboard(self, width: int, height: int, 
                             tile_size: int = 32) -> np.ndarray:
        """Create checkerboard pattern."""
        data = np.zeros((height, width, 4), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                tile_x = (x // tile_size) % 2
                tile_y = (y // tile_size) % 2
                
                if (tile_x + tile_y) % 2 == 0:
                    data[y, x] = [255, 255, 255, 255]  # White
                else:
                    data[y, x] = [0, 0, 0, 255]  # Black
                    
        return data
        
    def _create_gradient(self, width: int, height: int) -> np.ndarray:
        """Create gradient texture."""
        data = np.zeros((height, width, 4), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * (x + y) / (width + height))
                data[y, x] = [r, g, b, 255]
                
        return data
        
    def _create_noise(self, width: int, height: int) -> np.ndarray:
        """Create noise texture."""
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        data = np.zeros((height, width, 4), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                value = noise[y, x]
                data[y, x] = [value, value, value, 255]
                
        return data
        
    def _create_solid_color(self, width: int, height: int, 
                            color: Tuple[int, int, int, int] = (128, 128, 128, 255)) -> np.ndarray:
        """Create solid color texture."""
        data = np.full((height, width, 4), color, dtype=np.uint8)
        return data


class TextureManager:
    """Manages texture loading, caching, and OpenGL operations."""
    
    def __init__(self, max_textures: int = 100):
        self.textures: Dict[str, Texture] = {}
        self.texture_cache: Dict[str, Texture] = {}
        self.loader = TextureLoader()
        self.max_textures = max_textures
        self.texture_counter = 0
        
    def load_texture(self, filepath: str, name: str = None) -> Optional[Texture]:
        """Load texture from file."""
        if name is None:
            name = os.path.basename(filepath)
            
        # Check cache first
        if name in self.texture_cache:
            return self.texture_cache[name]
            
        # Load image data
        result = self.loader.load_image(filepath)
        if result is None:
            return None
            
        data, info = result
        
        # Create OpenGL texture
        texture_id = self._create_gl_texture(data, info)
        if texture_id is None:
            return None
            
        # Create texture object
        texture = Texture(texture_id, info, name)
        
        # Add to cache
        self.texture_cache[name] = texture
        self.textures[name] = texture
        
        return texture
        
    def create_procedural_texture(self, width: int, height: int, 
                                  texture_type: str = "checkerboard", 
                                  name: str = None) -> Optional[Texture]:
        """Create procedural texture."""
        if name is None:
            name = f"procedural_{texture_type}_{self.texture_counter}"
            self.texture_counter += 1
            
        # Create texture data
        data, info = self.loader.create_procedural_texture(width, height, texture_type)
        
        # Create OpenGL texture
        texture_id = self._create_gl_texture(data, info)
        if texture_id is None:
            return None
            
        # Create texture object
        texture = Texture(texture_id, info, name)
        
        # Add to cache
        self.texture_cache[name] = texture
        self.textures[name] = texture
        
        return texture
        
    def _create_gl_texture(self, data: np.ndarray, info: TextureInfo) -> Optional[int]:
        """Create OpenGL texture from data."""
        try:
            # Generate texture ID
            texture_id = gl.glGenTextures(1)
            
            # Bind texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            
            # Set texture data
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
                info.width, info.height, 0,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data
            )
            
            # Set default parameters
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            
            return texture_id
            
        except Exception as e:
            print(f"Error creating OpenGL texture: {e}")
            return None
            
    def get_texture(self, name: str) -> Optional[Texture]:
        """Get texture by name."""
        return self.texture_cache.get(name)
        
    def bind_texture(self, name: str, unit: int = 0) -> bool:
        """Bind texture by name to texture unit."""
        texture = self.get_texture(name)
        if texture:
            texture.bind(unit)
            return True
        return False
        
    def unbind_texture(self, name: str):
        """Unbind texture by name."""
        texture = self.get_texture(name)
        if texture:
            texture.unbind()
            
    def cleanup(self):
        """Clean up all textures."""
        for texture in self.textures.values():
            gl.glDeleteTextures([texture.texture_id])
        self.textures.clear()
        self.texture_cache.clear()


def demonstrate_texture_management():
    """Demonstrate texture management functionality."""
    print("=== Texture Management Demonstration ===\n")
    
    # Create texture manager
    manager = TextureManager()
    
    # Create procedural textures
    print("1. Creating procedural textures:")
    
    checkerboard = manager.create_procedural_texture(256, 256, "checkerboard", "checkerboard")
    print(f"   - Checkerboard texture: {checkerboard.name} ({checkerboard.info.width}x{checkerboard.info.height})")
    
    gradient = manager.create_procedural_texture(512, 512, "gradient", "gradient")
    print(f"   - Gradient texture: {gradient.name} ({gradient.info.width}x{gradient.info.height})")
    
    noise = manager.create_procedural_texture(128, 128, "noise", "noise")
    print(f"   - Noise texture: {noise.name} ({noise.info.width}x{noise.info.height})")
    
    # Test texture operations
    print("\n2. Testing texture operations:")
    
    # Set filtering
    checkerboard.set_filtering(TextureFilter.LINEAR_MIPMAP_LINEAR, TextureFilter.LINEAR)
    checkerboard.generate_mipmaps()
    print(f"   - Applied mipmapping to {checkerboard.name}")
    
    # Set wrapping
    gradient.set_wrapping(TextureWrap.REPEAT, TextureWrap.REPEAT)
    print(f"   - Applied repeat wrapping to {gradient.name}")
    
    # Test binding
    print("\n3. Testing texture binding:")
    if manager.bind_texture("checkerboard", 0):
        print("   - Successfully bound checkerboard texture to unit 0")
    if manager.bind_texture("gradient", 1):
        print("   - Successfully bound gradient texture to unit 1")
        
    # Test texture retrieval
    print("\n4. Testing texture retrieval:")
    retrieved = manager.get_texture("noise")
    if retrieved:
        print(f"   - Retrieved texture: {retrieved.name}")
        print(f"     Size: {retrieved.info.width}x{retrieved.info.height}")
        print(f"     Format: {retrieved.info.format.value}")
        print(f"     Data size: {retrieved.info.data_size} bytes")
        
    # Cleanup
    print("\n5. Cleaning up textures...")
    manager.cleanup()
    print("   - All textures cleaned up")


if __name__ == "__main__":
    demonstrate_texture_management()
