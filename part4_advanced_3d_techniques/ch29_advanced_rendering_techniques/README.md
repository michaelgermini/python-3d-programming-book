# Chapter 29: Advanced Rendering Techniques

This chapter demonstrates advanced rendering techniques for high-quality real-time graphics in Python.

## Overview

Advanced rendering techniques go beyond basic 3D graphics to create photorealistic, efficient, and visually stunning applications. This chapter covers:

- **Deferred Rendering**: Multi-pass rendering with G-buffer for complex lighting
- **Post-Processing**: Screen-space effects and filters for visual enhancement
- **Advanced Lighting**: Physically-based rendering and global illumination
- **Performance Optimization**: Techniques for real-time rendering
- **Modern Graphics Pipelines**: State-of-the-art rendering approaches

## Files

### 1. `deferred_rendering.py`
Deferred rendering pipeline with G-buffer and multiple render targets:
- G-buffer creation and management
- Multiple render targets (MRT)
- Screen-space lighting calculations
- Post-processing integration
- Performance optimization

**Key Classes:**
- `FramebufferObject`: Off-screen rendering framebuffer
- `GBuffer`: Geometry buffer for deferred rendering
- `DeferredRenderer`: Complete deferred rendering pipeline
- `FramebufferAttachment`: Framebuffer attachment configuration

**Usage Example:**
```python
from deferred_rendering import DeferredRenderer

# Create deferred renderer
renderer = DeferredRenderer(800, 600)

# Setup scene objects and lights
scene_objects = [...]  # Your 3D objects
lights = [...]         # Your light sources

# Render with deferred pipeline
renderer.render(scene_objects, lights, view_matrix, projection_matrix, view_pos)
```

### 2. `post_processing.py`
Comprehensive post-processing effects system:
- Bloom, SSAO, motion blur, depth of field
- Tone mapping and color grading
- Vignette and chromatic aberration
- Effect pipeline management
- Real-time performance optimization

**Key Classes:**
- `PostProcessor`: Post-processing pipeline manager
- `PostProcessEffect`: Individual effect configuration
- `EffectType`: Enumeration of available effects

**Usage Example:**
```python
from post_processing import PostProcessor, PostProcessEffect, EffectType

# Create post-processor
post_processor = PostProcessor(800, 600)

# Add effects
bloom_effect = PostProcessEffect(
    EffectType.BLOOM,
    enabled=True,
    parameters={'threshold': 1.0, 'intensity': 1.0}
)
post_processor.add_effect(bloom_effect)

# Apply post-processing
post_processor.process(input_texture, g_position, g_normal)
```

## Key Concepts

### Deferred Rendering
1. **Geometry Pass**: Render scene to G-buffer (position, normal, albedo, material)
2. **Lighting Pass**: Calculate lighting using G-buffer data
3. **Post-Processing**: Apply screen-space effects
4. **Final Output**: Composite and display

### G-Buffer Components
- **Position Buffer**: World-space positions (RGB32F)
- **Normal Buffer**: Surface normals (RGB16F)
- **Albedo Buffer**: Base colors (RGBA8)
- **Material Buffer**: Metallic, roughness, AO (RGBA8)
- **Depth Buffer**: Depth information (DEPTH24)

### Post-Processing Effects
1. **Bloom**: Bright pixel extraction and blur
2. **SSAO**: Screen-space ambient occlusion
3. **Tone Mapping**: HDR to LDR conversion
4. **Vignette**: Edge darkening effect
5. **Chromatic Aberration**: Color separation effect

## Rendering Pipeline

### Traditional Forward Rendering
```
For each object:
  For each light:
    Calculate lighting
    Render to screen
```

### Deferred Rendering
```
Geometry Pass:
  Render all objects to G-buffer

Lighting Pass:
  For each light:
    Calculate lighting using G-buffer
    Accumulate results

Post-Processing:
  Apply screen-space effects
```

## Performance Characteristics

### Deferred Rendering
- **Geometry Pass**: O(n) where n = number of objects
- **Lighting Pass**: O(m × l) where m = pixels, l = lights
- **Memory Usage**: 4 color textures + depth texture
- **Bandwidth**: Reduced for complex lighting

### Post-Processing
- **Bloom**: O(n) where n = bright pixels
- **SSAO**: O(k) where k = samples per pixel
- **Tone Mapping**: O(1) per pixel
- **Vignette**: O(1) per pixel

## Advanced Features

### Multiple Render Targets (MRT)
```python
# Setup MRT
gl.glDrawBuffers(4, [
    gl.GL_COLOR_ATTACHMENT0,  # Position
    gl.GL_COLOR_ATTACHMENT1,  # Normal
    gl.GL_COLOR_ATTACHMENT2,  # Albedo
    gl.GL_COLOR_ATTACHMENT3   # Material
])
```

### Screen-Space Effects
- **SSAO**: Ambient occlusion in screen space
- **SSR**: Screen-space reflections
- **SSGI**: Screen-space global illumination
- **Motion Blur**: Velocity-based blur

### Physically-Based Rendering (PBR)
- **Cook-Torrance BRDF**: Physically accurate materials
- **GGX Distribution**: Modern microfacet model
- **Smith Geometry**: Accurate shadowing/masking
- **Schlick Fresnel**: Efficient Fresnel approximation

## Shader Examples

### Geometry Pass Fragment Shader
```glsl
#version 330 core
layout (location = 0) out vec4 gPosition;
layout (location = 1) out vec4 gNormal;
layout (location = 2) out vec4 gAlbedo;
layout (location = 3) out vec4 gMaterial;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

void main() {
    gPosition = vec4(FragPos, 1.0);
    gNormal = vec4(normalize(Normal), 1.0);
    gAlbedo = vec4(albedo, 1.0);
    gMaterial = vec4(metallic, roughness, ao, 1.0);
}
```

### Lighting Pass Fragment Shader
```glsl
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gMaterial;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform vec3 viewPos;

void main() {
    vec3 FragPos = texture(gPosition, TexCoord).rgb;
    vec3 Normal = texture(gNormal, TexCoord).rgb;
    vec3 Albedo = texture(gAlbedo, TexCoord).rgb;
    vec3 Material = texture(gMaterial, TexCoord).rgb;
    
    float metallic = Material.r;
    float roughness = Material.g;
    float ao = Material.b;
    
    // Calculate lighting using PBR...
}
```

## Performance Optimization

### Memory Management
1. **Texture Compression**: Use compressed formats (DXT, ETC)
2. **Mipmapping**: Automatic texture level generation
3. **Texture Atlasing**: Combine multiple textures
4. **LOD Systems**: Level-of-detail for distant objects

### GPU Optimization
1. **Instanced Rendering**: Batch similar objects
2. **Frustum Culling**: Skip off-screen objects
3. **Occlusion Culling**: Skip hidden objects
4. **State Sorting**: Minimize state changes

### Bandwidth Optimization
1. **Deferred Rendering**: Reduce overdraw
2. **Tile-Based Rendering**: Process in tiles
3. **Memory Coalescing**: Optimize memory access
4. **Texture Streaming**: Load textures on demand

## Best Practices

### Deferred Rendering
1. **Use Appropriate Formats**: Balance quality vs. memory
2. **Optimize G-Buffer**: Minimize texture count and size
3. **Light Culling**: Only process relevant lights
4. **Temporal Coherence**: Reuse data between frames

### Post-Processing
1. **Effect Ordering**: Apply effects in optimal order
2. **Quality Settings**: Adjust based on performance
3. **Temporal Effects**: Use previous frame data
4. **Adaptive Quality**: Dynamic quality adjustment

### General Optimization
1. **Profile Performance**: Identify bottlenecks
2. **Use GPU Profilers**: Monitor GPU utilization
3. **Optimize Shaders**: Minimize instruction count
4. **Batch Operations**: Reduce draw calls

## Advanced Techniques

### Temporal Anti-Aliasing (TAA)
- Jittered sampling across frames
- Motion vector reprojection
- Temporal accumulation
- Edge detection and preservation

### Volumetric Effects
- Volumetric lighting
- Fog and atmospheric scattering
- Smoke and fire effects
- Cloud rendering

### Real-Time Ray Tracing
- Hardware-accelerated ray tracing
- Hybrid rendering pipelines
- Ray-traced reflections
- Ray-traced shadows

## Troubleshooting

### Common Issues
1. **G-Buffer Artifacts**: Check texture formats and precision
2. **Light Bleeding**: Adjust light culling and attenuation
3. **Performance Issues**: Profile and optimize bottlenecks
4. **Memory Problems**: Monitor texture memory usage

### Debugging Tips
1. **Visualize G-Buffer**: Render individual buffers
2. **Check Framebuffer Status**: Verify completeness
3. **Monitor GPU Usage**: Use profiling tools
4. **Validate Shaders**: Check compilation and linking

## Future Enhancements

1. **Vulkan/DirectX 12**: Modern graphics APIs
2. **Ray Tracing**: Hardware-accelerated ray tracing
3. **Machine Learning**: AI-powered effects
4. **Real-Time Path Tracing**: Advanced global illumination
5. **Holographic Rendering**: 3D display support

## Dependencies

```python
import numpy as np
import OpenGL.GL as gl
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
```

## References

- "Real-Time Rendering" by Akenine-Möller, Haines, and Hoffman
- "Physically Based Rendering" by Pharr, Jakob, and Humphreys
- "Real-Time Rendering Architecture" by Akenine-Möller and Ström
- "Advanced Graphics Programming Using OpenGL" by McReynolds and Blythe
- "GPU Gems" series by NVIDIA

This chapter provides advanced techniques for creating high-quality, performant real-time graphics applications with modern rendering pipelines and effects.
