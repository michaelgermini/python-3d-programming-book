"""
Chapter 21: Texturing and Materials - Texture Optimization
========================================================

This module demonstrates texture optimization techniques for 3D graphics.

Key Concepts:
- Texture compression and formats
- Mipmap generation and management
- Texture atlasing and packing
- Performance optimization strategies
"""

import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw
import OpenGL.GL as gl


class CompressionFormat(Enum):
    """Texture compression formats."""
    NONE = "none"
    DXT1 = "dxt1"
    DXT3 = "dxt3"
    DXT5 = "dxt5"
    ETC1 = "etc1"
    ETC2 = "etc2"
    ASTC = "astc"


class TextureQuality(Enum):
    """Texture quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class TextureMetrics:
    """Texture performance metrics."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    load_time: float
    memory_usage: int
    mipmap_levels: int
    texture_units_used: int


class TextureCompressor:
    """Handles texture compression and optimization."""
    
    def __init__(self):
        self.supported_formats = {
            CompressionFormat.NONE,
            CompressionFormat.DXT1,
            CompressionFormat.DXT3,
            CompressionFormat.DXT5
        }
        
    def compress_texture(self, data: np.ndarray, format: CompressionFormat) -> Tuple[np.ndarray, float]:
        """Compress texture data."""
        start_time = time.time()
        
        if format == CompressionFormat.NONE:
            compressed_data = data
        elif format == CompressionFormat.DXT1:
            compressed_data = self._compress_dxt1(data)
        elif format == CompressionFormat.DXT3:
            compressed_data = self._compress_dxt3(data)
        elif format == CompressionFormat.DXT5:
            compressed_data = self._compress_dxt5(data)
        else:
            compressed_data = data
            
        compression_time = time.time() - start_time
        compression_ratio = 1.0 - (compressed_data.nbytes / data.nbytes)
        
        return compressed_data, compression_ratio
        
    def _compress_dxt1(self, data: np.ndarray) -> np.ndarray:
        """Compress using DXT1 format (simplified)."""
        # Simplified DXT1 compression - in practice, use a proper library
        height, width, channels = data.shape
        block_size = 4
        
        # Calculate number of blocks
        blocks_w = (width + block_size - 1) // block_size
        blocks_h = (height + block_size - 1) // block_size
        
        # Create compressed data (8 bytes per 4x4 block)
        compressed_size = blocks_w * blocks_h * 8
        compressed_data = np.zeros(compressed_size, dtype=np.uint8)
        
        # For simplicity, just copy data (real DXT1 would compress each block)
        # This is a placeholder for actual compression
        return data.flatten()[:compressed_size]
        
    def _compress_dxt3(self, data: np.ndarray) -> np.ndarray:
        """Compress using DXT3 format (simplified)."""
        # Similar to DXT1 but with alpha channel
        return self._compress_dxt1(data)
        
    def _compress_dxt5(self, data: np.ndarray) -> np.ndarray:
        """Compress using DXT5 format (simplified)."""
        # Similar to DXT1 but with interpolated alpha
        return self._compress_dxt1(data)


class MipmapGenerator:
    """Generates and manages mipmaps for textures."""
    
    def __init__(self):
        self.max_levels = 16
        
    def generate_mipmaps(self, data: np.ndarray, levels: int = None) -> List[np.ndarray]:
        """Generate mipmap chain."""
        if levels is None:
            levels = self._calculate_max_levels(data.shape[0], data.shape[1])
            
        mipmaps = [data]
        current_data = data
        
        for level in range(1, min(levels, self.max_levels)):
            current_data = self._downsample(current_data)
            mipmaps.append(current_data)
            
        return mipmaps
        
    def _calculate_max_levels(self, width: int, height: int) -> int:
        """Calculate maximum number of mipmap levels."""
        return int(math.log2(min(width, height))) + 1
        
    def _downsample(self, data: np.ndarray) -> np.ndarray:
        """Downsample image data by 2x."""
        height, width, channels = data.shape
        
        new_height = height // 2
        new_width = width // 2
        
        # Simple box filter downsampling
        downsampled = np.zeros((new_height, new_width, channels), dtype=data.dtype)
        
        for y in range(new_height):
            for x in range(new_width):
                # Average 2x2 block
                block = data[y*2:y*2+2, x*2:x*2+2]
                downsampled[y, x] = np.mean(block, axis=(0, 1)).astype(data.dtype)
                
        return downsampled
        
    def create_mipmap_chain(self, data: np.ndarray, quality: TextureQuality) -> List[np.ndarray]:
        """Create mipmap chain based on quality settings."""
        if quality == TextureQuality.LOW:
            levels = 2
        elif quality == TextureQuality.MEDIUM:
            levels = 4
        elif quality == TextureQuality.HIGH:
            levels = 8
        else:  # ULTRA
            levels = None
            
        return self.generate_mipmaps(data, levels)


class TextureAtlas:
    """Manages texture atlasing and packing."""
    
    def __init__(self, atlas_size: int = 1024):
        self.atlas_size = atlas_size
        self.textures: Dict[str, Tuple[int, int, int, int]] = {}  # name -> (x, y, width, height)
        self.atlas_data = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
        self.used_space = 0
        
    def add_texture(self, name: str, data: np.ndarray) -> bool:
        """Add texture to atlas."""
        height, width = data.shape[:2]
        
        # Find space for texture
        position = self._find_space(width, height)
        if position is None:
            return False
            
        x, y = position
        
        # Copy texture data to atlas
        self.atlas_data[y:y+height, x:x+width] = data
        
        # Record texture position
        self.textures[name] = (x, y, width, height)
        self.used_space += width * height
        
        return True
        
    def _find_space(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Find space for texture in atlas (simple algorithm)."""
        # Simple packing algorithm - in practice, use more sophisticated methods
        for y in range(0, self.atlas_size - height + 1, height):
            for x in range(0, self.atlas_size - width + 1, width):
                if self._is_space_available(x, y, width, height):
                    return (x, y)
        return None
        
    def _is_space_available(self, x: int, y: int, width: int, height: int) -> bool:
        """Check if space is available."""
        # Check if area is empty
        area = self.atlas_data[y:y+height, x:x+width]
        return np.all(area == 0)
        
    def get_texture_coords(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Get normalized texture coordinates for texture in atlas."""
        if name not in self.textures:
            return None
            
        x, y, width, height = self.textures[name]
        
        # Convert to normalized coordinates
        u1 = x / self.atlas_size
        v1 = y / self.atlas_size
        u2 = (x + width) / self.atlas_size
        v2 = (y + height) / self.atlas_size
        
        return (u1, v1, u2, v2)
        
    def get_atlas_data(self) -> np.ndarray:
        """Get atlas texture data."""
        return self.atlas_data
        
    def get_usage_ratio(self) -> float:
        """Get atlas usage ratio."""
        total_space = self.atlas_size * self.atlas_size
        return self.used_space / total_space


class TextureOptimizer:
    """Main texture optimization system."""
    
    def __init__(self):
        self.compressor = TextureCompressor()
        self.mipmap_generator = MipmapGenerator()
        self.atlases: Dict[str, TextureAtlas] = {}
        self.metrics: Dict[str, TextureMetrics] = {}
        
    def optimize_texture(self, name: str, data: np.ndarray, 
                        quality: TextureQuality = TextureQuality.MEDIUM,
                        use_compression: bool = True,
                        generate_mipmaps: bool = True) -> Dict:
        """Optimize a texture with various techniques."""
        start_time = time.time()
        original_size = data.nbytes
        
        # Generate mipmaps
        mipmap_chain = []
        mipmap_levels = 1
        if generate_mipmaps:
            mipmap_chain = self.mipmap_generator.create_mipmap_chain(data, quality)
            mipmap_levels = len(mipmap_chain)
            
        # Compress texture
        compressed_data = data
        compression_ratio = 0.0
        if use_compression:
            format = self._get_compression_format(quality)
            compressed_data, compression_ratio = self.compressor.compress_texture(data, format)
            
        # Calculate metrics
        load_time = time.time() - start_time
        memory_usage = compressed_data.nbytes
        
        # Store metrics
        metrics = TextureMetrics(
            original_size=original_size,
            compressed_size=memory_usage,
            compression_ratio=compression_ratio,
            load_time=load_time,
            memory_usage=memory_usage,
            mipmap_levels=mipmap_levels,
            texture_units_used=1
        )
        
        self.metrics[name] = metrics
        
        return {
            'data': compressed_data,
            'mipmaps': mipmap_chain,
            'metrics': metrics
        }
        
    def _get_compression_format(self, quality: TextureQuality) -> CompressionFormat:
        """Get compression format based on quality."""
        if quality == TextureQuality.LOW:
            return CompressionFormat.DXT1
        elif quality == TextureQuality.MEDIUM:
            return CompressionFormat.DXT3
        elif quality == TextureQuality.HIGH:
            return CompressionFormat.DXT5
        else:
            return CompressionFormat.NONE
            
    def create_atlas(self, name: str, atlas_size: int = 1024) -> TextureAtlas:
        """Create a new texture atlas."""
        atlas = TextureAtlas(atlas_size)
        self.atlases[name] = atlas
        return atlas
        
    def add_to_atlas(self, atlas_name: str, texture_name: str, data: np.ndarray) -> bool:
        """Add texture to atlas."""
        if atlas_name not in self.atlases:
            return False
            
        atlas = self.atlases[atlas_name]
        return atlas.add_texture(texture_name, data)
        
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        total_original_size = sum(m.original_size for m in self.metrics.values())
        total_compressed_size = sum(m.compressed_size for m in self.metrics.values())
        total_load_time = sum(m.load_time for m in self.metrics.values())
        
        return {
            'total_textures': len(self.metrics),
            'total_original_size': total_original_size,
            'total_compressed_size': total_compressed_size,
            'overall_compression_ratio': 1.0 - (total_compressed_size / total_original_size),
            'total_load_time': total_load_time,
            'average_load_time': total_load_time / len(self.metrics) if self.metrics else 0,
            'atlas_count': len(self.atlases)
        }


def demonstrate_texture_optimization():
    """Demonstrate texture optimization techniques."""
    print("=== Texture Optimization Demonstration ===\n")
    
    # Create optimizer
    optimizer = TextureOptimizer()
    
    # Create sample textures
    print("1. Creating sample textures:")
    
    # Create different sized textures
    textures = {
        'small': np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8),
        'medium': np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8),
        'large': np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
    }
    
    for name, data in textures.items():
        print(f"   - {name}: {data.shape[0]}x{data.shape[1]} ({data.nbytes} bytes)")
        
    # Optimize textures
    print("\n2. Optimizing textures:")
    
    for name, data in textures.items():
        print(f"\n   Optimizing {name} texture:")
        
        # Optimize with different quality levels
        for quality in [TextureQuality.LOW, TextureQuality.MEDIUM, TextureQuality.HIGH]:
            result = optimizer.optimize_texture(
                f"{name}_{quality.value}", 
                data, 
                quality=quality,
                use_compression=True,
                generate_mipmaps=True
            )
            
            metrics = result['metrics']
            print(f"     {quality.value}: {metrics.compressed_size} bytes "
                  f"(compression: {metrics.compression_ratio:.1%}, "
                  f"mipmaps: {metrics.mipmap_levels})")
                  
    # Create texture atlas
    print("\n3. Creating texture atlas:")
    
    atlas = optimizer.create_atlas("main_atlas", 512)
    
    # Add small textures to atlas
    for name, data in textures.items():
        if data.shape[0] <= 128:  # Only add small textures
            success = optimizer.add_to_atlas("main_atlas", name, data)
            if success:
                coords = atlas.get_texture_coords(name)
                print(f"   - Added {name} to atlas at {coords}")
                
    usage_ratio = atlas.get_usage_ratio()
    print(f"   - Atlas usage: {usage_ratio:.1%}")
    
    # Generate optimization statistics
    print("\n4. Optimization statistics:")
    
    stats = optimizer.get_optimization_stats()
    print(f"   - Total textures: {stats['total_textures']}")
    print(f"   - Original size: {stats['total_original_size']:,} bytes")
    print(f"   - Compressed size: {stats['total_compressed_size']:,} bytes")
    print(f"   - Overall compression: {stats['overall_compression_ratio']:.1%}")
    print(f"   - Average load time: {stats['average_load_time']:.3f}s")
    print(f"   - Atlas count: {stats['atlas_count']}")
    
    # Test mipmap generation
    print("\n5. Testing mipmap generation:")
    
    test_data = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
    mipmap_generator = MipmapGenerator()
    
    mipmaps = mipmap_generator.generate_mipmaps(test_data, 4)
    print(f"   - Generated {len(mipmaps)} mipmap levels:")
    
    for i, mipmap in enumerate(mipmaps):
        print(f"     Level {i}: {mipmap.shape[0]}x{mipmap.shape[1]} ({mipmap.nbytes} bytes)")
        
    print("\n6. Optimization complete!")


if __name__ == "__main__":
    demonstrate_texture_optimization()
