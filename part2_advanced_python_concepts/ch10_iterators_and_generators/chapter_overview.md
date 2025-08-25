# Chapter 10: Iterators and Generators

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand the iterator protocol and create custom iterators for 3D data
- Implement generator functions and expressions for memory-efficient data processing
- Use lazy evaluation to handle large 3D datasets without loading everything into memory
- Create streaming data pipelines for real-time 3D graphics applications
- Apply iterators and generators to 3D data traversal, generation, and processing
- Build memory-efficient solutions for large-scale 3D graphics projects

## Key Concepts

### Iterators
- **Iterator Protocol**: The `__iter__()` and `__next__()` methods
- **Custom Iterators**: Creating iterators for 3D data structures
- **Iterator Composition**: Combining multiple iterators for complex data traversal
- **Memory-Efficient Iteration**: Processing large datasets without loading everything into memory
- **3D Data Traversal**: Efficient navigation through 3D structures

### Generators
- **Generator Functions**: Using `yield` to create memory-efficient data streams
- **Generator Expressions**: Concise syntax for creating generators
- **Lazy Evaluation**: Processing data only when needed
- **3D Data Generation**: Creating 3D geometry and data on-demand
- **Animation and Time-based Generators**: Real-time data generation for graphics
- **Data Processing Generators**: Filtering, transforming, and processing 3D data

### Advanced Patterns
- **Recursive Generators**: Generating hierarchical 3D structures
- **Infinite Generators**: Creating endless data streams
- **Merge Generators**: Combining multiple data sources
- **Streaming Pipelines**: Building complex data processing workflows
- **Chunked Processing**: Managing memory usage for large datasets

## File Summaries

### `iterators.py`
Demonstrates the iterator protocol and custom iterators for efficient traversal of 3D data.

**Key Features:**
- `Vector3D` and `Triangle` iterables with custom iteration behavior
- `RangeIterator` for numeric ranges with step control
- `VectorIterator` for interpolating between 3D vectors
- `GridIterator` for traversing 3D grids efficiently
- `SphereIterator` for spherical coordinate traversal
- `MeshIterator` for mesh data traversal
- Iterator composition utilities (`FilteredIterator`, `TransformedIterator`, `ChunkedIterator`)
- Breadth-first and depth-first iterators for hierarchical structures

**Practical Applications:**
- Efficient 3D data traversal without loading entire datasets
- Memory-conscious processing of large point clouds
- Hierarchical scene graph traversal
- Real-time mesh processing

### `generators.py`
Focuses on generator functions and expressions for memory-efficient, lazy generation of 3D data.

**Key Features:**
- Mathematical generators (`fibonacci_generator`, `range_generator`)
- 3D geometry generators (`vector_interpolation_generator`, `grid_generator`, `sphere_generator`, `cylinder_generator`)
- Random data generators (`random_points_generator`, `noise_generator`)
- Color generators (`color_gradient_generator`, `rainbow_generator`)
- Animation generators (`animation_generator`, `rotation_animation_generator`)
- Data processing generators (`filter_generator`, `transform_generator`, `batch_generator`, `window_generator`)
- Advanced generators (recursive, infinite, merge generators)

**Practical Applications:**
- Real-time 3D geometry generation
- Procedural content creation
- Animation and simulation systems
- Large-scale data processing pipelines

### `3d_data_streaming.py`
Demonstrates how to use iterators and generators for streaming 3D data efficiently, handling large datasets without loading everything into memory at once.

**Key Features:**
- `PointCloudStreamer` for streaming point cloud data from various sources (grid, sphere, random, file)
- `MeshStreamer` for efficient mesh data streaming (vertices, faces, edges)
- `StreamingPipeline` for building complex data processing workflows
- `RealTimePointGenerator` for real-time data generation in simulations
- `SensorDataStreamer` for simulating streaming sensor data
- `ChunkedProcessor` for memory-efficient data processing
- `StreamingAnalyzer` for real-time 3D data analysis

**Practical Applications:**
- Large-scale 3D data processing
- Real-time graphics applications
- Sensor data processing
- Memory-constrained environments

## Code Examples

### Custom Iterator for 3D Vector Interpolation
```python
class VectorIterator:
    def __init__(self, start: Vector3D, end: Vector3D, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.steps:
            raise StopIteration
        
        t = self.current / (self.steps - 1)
        x = self.start.x + (self.end.x - self.start.x) * t
        y = self.start.y + (self.end.y - self.start.y) * t
        z = self.start.z + (self.end.z - self.start.z) * t
        
        self.current += 1
        return Vector3D(x, y, z)
```

### Generator for Sphere Surface Points
```python
def sphere_generator(center: Vector3D, radius: float, lat_steps: int, lon_steps: int):
    """Generate points on a sphere surface."""
    for lat_idx in range(lat_steps):
        for lon_idx in range(lon_steps):
            lat = math.pi * (lat_idx + 1) / (lat_steps + 1)
            lon = 2 * math.pi * lon_idx / lon_steps
            
            x = center.x + radius * math.sin(lat) * math.cos(lon)
            y = center.y + radius * math.sin(lat) * math.sin(lon)
            z = center.z + radius * math.cos(lat)
            
            yield Vector3D(x, y, z)
```

### Streaming Pipeline for 3D Data Processing
```python
class StreamingPipeline:
    def __init__(self):
        self.filters = []
        self.transforms = []
        self.sinks = []
    
    def add_filter(self, predicate):
        self.filters.append(predicate)
        return self
    
    def add_transform(self, transform_func):
        self.transforms.append(transform_func)
        return self
    
    def process_stream(self, data_stream: Iterator) -> Generator:
        for item in data_stream:
            if all(predicate(item) for predicate in self.filters):
                for transform in self.transforms:
                    item = transform(item)
                for sink in self.sinks:
                    sink(item)
                yield item
```

## Best Practices

### Iterator Design
1. **Implement the Protocol Correctly**: Always implement both `__iter__()` and `__next__()` methods
2. **Handle StopIteration**: Properly raise `StopIteration` when iteration is complete
3. **Maintain State**: Keep track of iteration state to resume correctly
4. **Memory Efficiency**: Avoid storing large datasets in memory
5. **Reusability**: Make iterators reusable by resetting state in `__iter__()`

### Generator Usage
1. **Lazy Evaluation**: Use generators for data that doesn't need to be computed immediately
2. **Memory Management**: Generators are perfect for large datasets that can't fit in memory
3. **Pipeline Processing**: Chain generators together for complex data transformations
4. **Error Handling**: Handle generator exceptions appropriately
5. **Performance**: Generators can be slower than lists for small datasets, but much more memory-efficient

### 3D Graphics Applications
1. **Large Meshes**: Use iterators for processing large 3D meshes without loading everything
2. **Point Clouds**: Stream point cloud data for real-time visualization
3. **Procedural Generation**: Use generators for creating procedural 3D content
4. **Animation**: Generate animation frames on-demand
5. **Data Processing**: Build pipelines for processing 3D sensor data

## Exercises

### Beginner Level
1. **Vector Iterator**: Create an iterator that generates points along a 3D line
2. **Grid Generator**: Write a generator that yields points in a 3D grid
3. **Color Iterator**: Create an iterator that cycles through colors in a gradient

### Intermediate Level
1. **Mesh Traversal**: Implement an iterator that traverses a mesh in different orders (vertices, faces, edges)
2. **Animation Generator**: Create a generator that produces animation frames for a rotating cube
3. **Data Pipeline**: Build a pipeline that filters and transforms 3D points

### Advanced Level
1. **Hierarchical Iterator**: Implement an iterator that traverses a scene graph
2. **Streaming Renderer**: Create a generator that yields rendered frames for a 3D scene
3. **Real-time Data Processing**: Build a system that processes streaming 3D sensor data using generators

## Common Pitfalls

1. **Infinite Loops**: Ensure iterators and generators have proper termination conditions
2. **Memory Leaks**: Be careful with generators that hold references to large objects
3. **State Management**: Don't modify iterator state during iteration
4. **Performance**: Avoid unnecessary computations in generator functions
5. **Error Propagation**: Handle exceptions properly in streaming pipelines

## Performance Considerations

- **Memory Usage**: Iterators and generators use minimal memory compared to storing all data
- **CPU Usage**: Generators can be slower than direct computation for small datasets
- **Lazy Evaluation**: Only compute what's needed, when it's needed
- **Chunking**: Process large datasets in chunks to balance memory and performance
- **Caching**: Consider caching frequently accessed generated data

## Integration with Other Chapters

- **Chapter 9 (Functional Programming)**: Combine with functional concepts for powerful data processing pipelines
- **Chapter 11 (Decorators)**: Use decorators to add functionality to iterators and generators
- **Chapter 12 (Context Managers)**: Manage resources in streaming data processing
- **Part III (3D Graphics)**: Apply these concepts to real 3D graphics applications

This chapter provides the foundation for efficient 3D data processing and real-time graphics applications, essential skills for any 3D graphics programmer working with Python.
