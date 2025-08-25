# Chapter 26: Normal Mapping, Bump Mapping, and PBR

## Overview
This chapter covers advanced material rendering techniques including normal mapping, bump mapping, and Physically Based Rendering (PBR). Students will learn about surface detail enhancement, realistic material properties, and modern rendering workflows for creating visually compelling 3D graphics.

## Key Learning Objectives
- Understand normal mapping techniques and tangent space calculations
- Implement bump mapping and parallax mapping for surface detail
- Master PBR workflows and BRDF models
- Build comprehensive material systems
- Apply energy conservation and physically accurate lighting

## Core Concepts

### 1. Normal Mapping
- **Tangent Space Calculations**: Computing TBN matrices for normal mapping
- **Normal Map Generation**: Creating and sampling normal maps
- **Surface Detail Enhancement**: Adding geometric detail without additional geometry
- **Lighting Integration**: Applying normal maps in lighting calculations

### 2. Bump Mapping
- **Height Map Generation**: Creating procedural height maps
- **Normal Perturbation**: Calculating surface normal offsets
- **Parallax Mapping**: Advanced surface detail techniques
- **Parallax Occlusion Mapping**: Ray marching for realistic depth

### 3. Physically Based Rendering (PBR)
- **PBR Material Properties**: Albedo, metallic, roughness, and ambient occlusion
- **BRDF Models**: GGX, Cook-Torrance, and other reflection models
- **Energy Conservation**: Ensuring physically accurate lighting
- **Metallic-Roughness Workflow**: Standard PBR material workflow

## Example Files

### 1. `normal_mapping.py`
**Purpose**: Demonstrates normal mapping techniques and implementation.

**Key Features**:
- `TangentSpaceCalculator` for TBN matrix calculations
- `NormalMap` class for normal map generation and sampling
- `NormalMappingSystem` for managing normal maps
- Procedural normal map generation

**Learning Outcomes**:
- Calculate tangent space vectors for meshes
- Generate and sample normal maps
- Integrate normal mapping with lighting
- Apply normal maps for surface detail

### 2. `bump_mapping.py`
**Purpose**: Implements bump mapping and parallax mapping techniques.

**Key Features**:
- `BumpMap` class for height map generation
- `ParallaxMapping` for advanced surface detail
- Procedural bump map patterns (noise, ridges, craters)
- Ray marching for parallax occlusion mapping

**Learning Outcomes**:
- Generate procedural height maps
- Calculate normal perturbations from height maps
- Implement parallax mapping techniques
- Apply bump mapping for surface detail

### 3. `pbr_system.py`
**Purpose**: Demonstrates Physically Based Rendering techniques.

**Key Features**:
- `PBRMaterial` class for material properties
- `BRDFCalculator` for reflection model calculations
- `PBRRenderer` for PBR lighting calculations
- `PBRMaterialLibrary` for common materials

**Learning Outcomes**:
- Implement PBR material workflows
- Calculate BRDF functions (GGX, Cook-Torrance)
- Apply energy conservation principles
- Create realistic material appearances

## Mathematical Foundations

### Normal Mapping Mathematics
- **Tangent Space**: Understanding TBN coordinate systems
- **Normal Transformation**: Converting between coordinate spaces
- **Surface Gradients**: Calculating normal offsets from height maps
- **Lighting Integration**: Applying perturbed normals in lighting

### Bump Mapping Mathematics
- **Height Sampling**: Interpolating height values from textures
- **Gradient Calculation**: Computing surface gradients for normal perturbation
- **Parallax Ray Marching**: Ray-surface intersection for depth effects
- **Occlusion Mapping**: Calculating visibility for realistic depth

### PBR Mathematics
- **BRDF Functions**: Normal Distribution, Geometry, and Fresnel functions
- **Energy Conservation**: Ensuring lighting energy balance
- **Metallic Workflow**: Converting between material property spaces
- **Tone Mapping**: Converting HDR to displayable range

## Practical Applications

### 1. Surface Detail Enhancement
- **Normal Mapping**: Adding fine geometric detail to surfaces
- **Bump Mapping**: Creating surface roughness and texture
- **Parallax Mapping**: Adding depth perception to flat surfaces
- **Material Variation**: Creating diverse surface appearances

### 2. Realistic Material Rendering
- **PBR Workflows**: Standardized material creation processes
- **Physically Accurate Lighting**: Realistic light-material interaction
- **Material Libraries**: Reusable material collections
- **Custom Materials**: Creating specialized material properties

### 3. Performance Optimization
- **Texture Compression**: Efficient normal and height map storage
- **LOD Systems**: Level-of-detail for material complexity
- **Shader Optimization**: Efficient PBR shader implementations
- **Memory Management**: Optimizing texture memory usage

### 4. Integration Techniques
- **Rendering Pipeline Integration**: Combining with existing renderers
- **Material System Integration**: Working with material management systems
- **Asset Pipeline Integration**: Integrating with content creation tools
- **Cross-Platform Compatibility**: Ensuring consistent rendering across platforms

## Best Practices

### 1. Normal Mapping
- Use appropriate normal map resolutions for different surface types
- Ensure proper tangent space calculations for accurate lighting
- Apply normal map compression for memory efficiency
- Test normal maps under various lighting conditions

### 2. Bump Mapping
- Choose appropriate height scales for realistic surface detail
- Use parallax mapping for surfaces with significant depth variation
- Optimize ray marching parameters for performance
- Balance quality and performance for different hardware

### 3. PBR Implementation
- Follow energy conservation principles strictly
- Use standardized PBR workflows for consistency
- Implement proper tone mapping and gamma correction
- Test materials under various lighting environments

### 4. Material System Design
- Create modular material components for reusability
- Implement efficient material property management
- Provide intuitive material creation interfaces
- Support material import/export for asset pipelines

## Common Challenges

### 1. Normal Map Artifacts
- **Challenge**: Seams and discontinuities in normal maps
- **Solution**: Proper UV unwrapping and tangent space calculations
- **Tools**: Normal map debugging and visualization tools

### 2. Bump Map Performance
- **Challenge**: High computational cost of parallax mapping
- **Solution**: Adaptive quality settings and optimization techniques
- **Tools**: Performance profiling and optimization tools

### 3. PBR Consistency
- **Challenge**: Inconsistent material appearance across different lighting
- **Solution**: Proper energy conservation and standardized workflows
- **Tools**: PBR validation and testing frameworks

### 4. Material Workflow Integration
- **Challenge**: Integrating with existing content creation pipelines
- **Solution**: Standardized material formats and conversion tools
- **Tools**: Material conversion and validation utilities

## Advanced Topics

### 1. Advanced Normal Mapping
- **Object Space Normal Maps**: Alternative coordinate systems
- **World Space Normal Maps**: Global coordinate system mapping
- **Blended Normal Maps**: Combining multiple normal maps
- **Animated Normal Maps**: Dynamic surface detail

### 2. Advanced Bump Mapping
- **Displacement Mapping**: True geometric displacement
- **Tessellation**: Dynamic geometry generation
- **Multi-resolution Bump Mapping**: Adaptive detail levels
- **Procedural Bump Generation**: Runtime height map creation

### 3. Advanced PBR Techniques
- **Subsurface Scattering**: Translucent material rendering
- **Clear Coat**: Multi-layer material systems
- **Anisotropic Reflection**: Directional surface properties
- **Volume Materials**: Participating media rendering

## Integration with Other Systems

### 1. Rendering Pipeline Integration
- **Deferred Rendering**: Integration with G-buffer systems
- **Forward Rendering**: Integration with traditional pipelines
- **Hybrid Rendering**: Combining multiple rendering approaches
- **Post-Processing**: Integration with effect systems

### 2. Asset Management
- **Material Libraries**: Integration with asset management systems
- **Texture Streaming**: Dynamic texture loading and management
- **LOD Systems**: Integration with level-of-detail systems
- **Compression Systems**: Integration with texture compression

### 3. Content Creation
- **3D Modeling Tools**: Integration with modeling software
- **Texture Painting**: Integration with texture creation tools
- **Material Editors**: Integration with material editing interfaces
- **Asset Validation**: Integration with quality assurance systems

## Summary
This chapter provides a comprehensive foundation for advanced material rendering techniques. Students learn to implement normal mapping, bump mapping, and PBR systems for creating realistic and visually compelling 3D graphics with enhanced surface detail and physically accurate lighting.

## Next Steps
- **Chapter 27**: Particle Systems and Visual Effects - Create dynamic particle and visual effects
- **Chapter 28**: Simple Ray Tracing and Path Tracing - Implement basic ray tracing techniques
- **Chapter 29**: Physics Simulation and Collision Detection - Add physics-based interactions

## Exercises and Projects

### 1. Normal Mapping System
Create a complete normal mapping system that:
- Implements tangent space calculations for arbitrary meshes
- Generates procedural normal maps for different surface types
- Integrates normal mapping with existing lighting systems
- Provides normal map debugging and visualization tools

### 2. Advanced Bump Mapping Framework
Build an advanced bump mapping framework that:
- Implements multiple bump mapping techniques (height, parallax, displacement)
- Provides procedural bump map generation for various surface types
- Optimizes performance for different hardware configurations
- Includes quality settings and adaptive detail levels

### 3. PBR Material System
Develop a comprehensive PBR material system that:
- Implements multiple BRDF models (GGX, Cook-Torrance, Blinn-Phong)
- Provides a material library with common materials
- Supports custom material creation and editing
- Integrates with IBL and environment lighting systems

### 4. Material Rendering Pipeline
Create a complete material rendering pipeline that:
- Combines normal mapping, bump mapping, and PBR techniques
- Provides efficient rendering for complex material systems
- Supports multiple rendering backends and platforms
- Includes comprehensive testing and validation tools
