# Chapter 28: Simple Ray Tracing and Path Tracing

This chapter demonstrates fundamental ray tracing and path tracing techniques for realistic rendering in Python.

## Overview

Ray tracing and path tracing are rendering techniques that simulate the physical behavior of light to create photorealistic images. This chapter covers:

- **Basic Ray Tracing**: Ray generation, object intersection, and material scattering
- **Path Tracing**: Global illumination with Monte Carlo integration
- **Acceleration Structures**: Spatial partitioning for performance optimization
- **BRDF Models**: Physically-based material representations

## Files

### 1. `ray_tracing.py`
Basic ray tracing implementation with:
- Ray and camera classes
- Geometric primitives (spheres, planes, triangles)
- Material systems (Lambertian, Metal, Dielectric, Emissive)
- Basic rendering pipeline

**Key Classes:**
- `Ray`: Represents a ray in 3D space
- `Camera`: Camera with depth of field and ray generation
- `Material`: Base material class with scattering behavior
- `Hittable`: Interface for objects that can be hit by rays
- `RayTracer`: Main ray tracing engine

**Usage Example:**
```python
from ray_tracing import RayTracer, Camera, HittableList, Sphere, LambertianMaterial

# Create scene
world = HittableList()
world.add(Sphere(np.array([0, 0, -1]), 0.5, LambertianMaterial(np.array([0.7, 0.3, 0.3]))))

# Setup camera
camera = Camera(
    look_from=np.array([0, 1, 3]),
    look_at=np.array([0, 0, -1]),
    up=np.array([0, 1, 0]),
    vfov=60.0,
    aspect_ratio=16.0/9.0
)

# Render
tracer = RayTracer(world, camera)
image = tracer.render(800, 600)
```

### 2. `path_tracing.py`
Advanced path tracing with global illumination:
- BRDF models (Lambert, Phong, Cook-Torrance)
- Global illumination and indirect lighting
- Russian roulette for path termination
- Convergence analysis and denoising

**Key Classes:**
- `BRDF`: Bidirectional Reflectance Distribution Function
- `PathTracer`: Path tracing with global illumination
- `GlobalIllumination`: Progressive rendering with convergence
- `Denoiser`: Image denoising techniques

**Usage Example:**
```python
from path_tracing import PathTracer, LambertBRDF, CookTorranceBRDF

# Create BRDFs
lambert_brdf = LambertBRDF(np.array([0.7, 0.3, 0.3]))
cook_torrance_brdf = CookTorranceBRDF(np.array([0.8, 0.6, 0.2]), roughness=0.1)

# Setup path tracer
path_tracer = PathTracer(world, camera)
path_tracer.max_depth = 10
path_tracer.samples_per_pixel = 100

# Render with global illumination
image = path_tracer.render(800, 600)
```

### 3. `acceleration_structures.py`
Spatial data structures for performance optimization:
- Bounding Volume Hierarchies (BVH)
- Octree spatial partitioning
- Spatial hash for uniform grids
- Ray-object intersection optimization

**Key Classes:**
- `BoundingBox`: Axis-aligned bounding box
- `BVH`: Bounding Volume Hierarchy
- `Octree`: Octree spatial data structure
- `SpatialHash`: Uniform grid acceleration

**Usage Example:**
```python
from acceleration_structures import BVH, Octree, SpatialHash

# Create acceleration structures
bvh = BVH(objects)  # For static scenes
octree = Octree(center, size)  # For dynamic scenes
spatial_hash = SpatialHash(cell_size)  # For uniform distributions

# Query objects
nearby_objects = octree.query_range(query_bbox)
hit_objects = bvh.ray_intersect(ray_origin, ray_direction, t_min, t_max)
```

## Key Concepts

### Ray Tracing Fundamentals
1. **Ray Generation**: Create rays from camera through viewport
2. **Intersection Testing**: Find closest object hit by ray
3. **Material Scattering**: Calculate how light interacts with surfaces
4. **Recursive Tracing**: Follow scattered rays for reflections/refractions

### Path Tracing
1. **Monte Carlo Integration**: Sample light paths randomly
2. **Global Illumination**: Account for indirect lighting
3. **BRDF Evaluation**: Physically-based material models
4. **Importance Sampling**: Efficient sampling strategies

### Acceleration Structures
1. **Spatial Partitioning**: Divide space for efficient queries
2. **Bounding Volumes**: Quick rejection of distant objects
3. **Hierarchical Structures**: Log(n) search complexity
4. **Dynamic Updates**: Support for moving objects

## Material Types

### Lambertian (Diffuse)
- Perfectly diffuse reflection
- Cosine-weighted hemisphere sampling
- Used for matte surfaces

### Metal (Specular)
- Perfect reflection with roughness
- Fresnel effect for realistic metals
- Configurable surface roughness

### Dielectric (Glass)
- Refraction and reflection
- Snell's law implementation
- Schlick Fresnel approximation

### Emissive (Light)
- Self-illuminating materials
- No scattering, only emission
- Used for light sources

## BRDF Models

### Lambert BRDF
```python
f_r = albedo / π
```
Perfectly diffuse reflection with uniform scattering.

### Phong BRDF
```python
f_r = albedo * (R · V)^shininess
```
Specular reflection with configurable shininess.

### Cook-Torrance BRDF
```python
f_r = (D * G * F) / (4 * (N · L) * (N · V)) + kd * albedo / π
```
Physically-based model with:
- D: Distribution function (GGX)
- G: Geometry function (Smith)
- F: Fresnel function (Schlick)

## Performance Optimization

### Acceleration Structures
- **BVH**: Best for static scenes, O(log n) average case
- **Octree**: Good for dynamic scenes, O(log n) average case
- **Spatial Hash**: Best for uniform distributions, O(1) average case

### Sampling Strategies
- **Importance Sampling**: Sample according to BRDF
- **Russian Roulette**: Early path termination
- **Multiple Importance Sampling**: Combine different sampling strategies

### Convergence Analysis
- **Progressive Rendering**: Accumulate samples over time
- **Convergence Metrics**: Monitor image quality
- **Adaptive Sampling**: More samples in noisy regions

## Advanced Features

### Global Illumination
- Indirect lighting simulation
- Caustics and soft shadows
- Realistic light transport

### Denoising
- Gaussian filtering
- Bilateral filtering
- Edge-preserving denoising

### Progressive Rendering
- Real-time preview
- Convergence monitoring
- Adaptive quality control

## Dependencies

```python
import numpy as np
import OpenGL.GL as gl  # For rendering
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
```

## Performance Considerations

1. **Ray-Object Intersection**: Most time-consuming operation
2. **Memory Usage**: Acceleration structures require additional memory
3. **Parallelization**: Embarrassingly parallel ray tracing
4. **GPU Acceleration**: Compute shaders for massive parallelism

## Best Practices

1. **Use Acceleration Structures**: Essential for complex scenes
2. **Optimize Material Models**: Balance accuracy vs. performance
3. **Implement Progressive Rendering**: Better user experience
4. **Monitor Convergence**: Stop rendering when quality is sufficient
5. **Profile Performance**: Identify bottlenecks

## Troubleshooting

### Common Issues
1. **Noisy Images**: Increase sample count or improve sampling
2. **Slow Rendering**: Use acceleration structures
3. **Memory Issues**: Optimize data structures
4. **Artifacts**: Check material implementations

### Debugging Tips
1. **Visualize Bounding Boxes**: Check spatial partitioning
2. **Single Ray Tracing**: Debug individual rays
3. **Material Testing**: Verify BRDF implementations
4. **Performance Profiling**: Identify slow operations

## Future Enhancements

1. **GPU Ray Tracing**: CUDA/OpenCL implementation
2. **Advanced Materials**: Subsurface scattering, hair, cloth
3. **Volume Rendering**: Participating media
4. **Real-time Ray Tracing**: Hardware acceleration
5. **Machine Learning**: AI-powered denoising and sampling

## References

- "Physically Based Rendering" by Pharr, Jakob, and Humphreys
- "Realistic Ray Tracing" by Shirley
- "Ray Tracing in One Weekend" by Shirley
- "Path Tracing" by Veach
- "Monte Carlo Methods in Global Illumination" by Dutre

This chapter provides a solid foundation for understanding and implementing ray tracing and path tracing techniques in Python, with practical examples and performance optimizations.
