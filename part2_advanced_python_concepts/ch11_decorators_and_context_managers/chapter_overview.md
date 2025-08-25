# Chapter 11: Decorators and Context Managers

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand and implement function and class decorators for 3D graphics applications
- Create parameterized decorators and decorator factories
- Use context managers for resource management in 3D graphics
- Implement performance monitoring, caching, and validation decorators
- Build advanced patterns combining decorators and context managers
- Apply these concepts to OpenGL context management, file operations, and state management
- Create robust error handling and recovery mechanisms for 3D applications

## Key Concepts

### Decorators
- **Function Decorators**: Enhancing functions with additional functionality
- **Class Decorators**: Modifying classes and their behavior
- **Decorator Factories**: Creating parameterized decorators
- **Decorator Composition**: Combining multiple decorators
- **Performance Monitoring**: Timing and profiling 3D operations
- **Caching and Memoization**: Optimizing expensive calculations
- **Validation and Error Handling**: Ensuring data integrity

### Context Managers
- **Context Manager Protocol**: The `__enter__()` and `__exit__()` methods
- **Resource Management**: Automatic cleanup of OpenGL resources, files, and memory
- **Performance Profiling**: Measuring execution time and performance metrics
- **Error Handling and Recovery**: Graceful handling of exceptions and fallback strategies
- **State Management**: Preserving and restoring application state
- **Thread Safety**: Managing concurrent access to shared resources

### Advanced Patterns
- **Pipeline Processing**: Creating data processing workflows
- **Stateful Decorators**: Maintaining state across function calls
- **Resource Pools**: Managing limited resources efficiently
- **Combined Patterns**: Integrating decorators and context managers
- **Performance Tracking**: Comprehensive performance analysis
- **Error Recovery**: Robust error handling with fallback mechanisms

## File Summaries

### `decorators.py`
Demonstrates how to use decorators to enhance 3D graphics functions with additional functionality like timing, validation, caching, and logging.

**Key Features:**
- Basic decorators (`timing_decorator`, `validation_decorator`, `logging_decorator`)
- Parameterized decorators (`retry_decorator`, `cache_decorator`, `performance_monitor_decorator`)
- 3D graphics specific decorators (`bounds_check_decorator`, `normalize_result_decorator`, `matrix_validation_decorator`)
- Decorator composition (`comprehensive_3d_decorator`)
- Class decorators (`singleton_decorator`, `add_methods_decorator`)
- Example functions demonstrating various decoration patterns

**Practical Applications:**
- Performance monitoring of 3D calculations
- Input validation for 3D data structures
- Caching expensive matrix operations
- Retry logic for unreliable operations
- Automatic normalization of vector results

### `context_managers.py`
Focuses on context managers for managing resources in 3D graphics applications, including OpenGL contexts, file operations, performance monitoring, and error handling.

**Key Features:**
- Basic context managers (`PerformanceProfiler`, `ErrorHandler`, `ResourceManager`)
- OpenGL context management (`OpenGLContext`)
- File and data management (`MeshFileManager`, `TemporaryMeshStorage`)
- Threading and concurrency (`ThreadSafeOperation`, `BatchProcessor`)
- 3D graphics specific context managers (`SceneManager`, `ShaderProgram`)
- Function-based context managers using `@contextmanager`

**Practical Applications:**
- OpenGL context lifecycle management
- Automatic cleanup of 3D resources
- Thread-safe mesh operations
- Batch processing of 3D data
- Temporary file management for mesh storage

### `advanced_patterns.py`
Demonstrates advanced patterns combining decorators and context managers for sophisticated 3D graphics applications, including pipeline processing, state management, and complex resource handling.

**Key Features:**
- Advanced decorator patterns (`PipelineDecorator`, `StatefulDecorator`, `CachingDecorator`)
- Advanced context manager patterns (`RenderPipeline`, `StateManager`, `ResourcePool`)
- Combined patterns (`PerformanceTracker`)
- Pipeline processing with multiple stages
- State management with backup and restoration
- Resource pooling with thread safety

**Practical Applications:**
- Complex rendering pipelines
- Application state management
- Resource pool management
- Performance tracking across multiple operations
- Advanced caching with TTL and size limits

## Code Examples

### Parameterized Decorator for Caching
```python
def cache_decorator(max_size: int = 100):
    """Decorator factory for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            if len(cache) >= max_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        return wrapper
    return decorator
```

### OpenGL Context Manager
```python
class OpenGLContext:
    """Context manager for OpenGL context management."""
    
    def __init__(self, window_title: str = "3D Application", width: int = 800, height: int = 600):
        self.window_title = window_title
        self.width = width
        self.height = height
        self.context_active = False
    
    def __enter__(self):
        # Simulate OpenGL context creation
        print(f"Creating OpenGL context: {self.window_title}")
        self.context_active = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context_active:
            print(f"Destroying OpenGL context")
            self.context_active = False
        
        if exc_type is not None:
            print(f"OpenGL context error: {exc_type.__name__}")
            return False
        return True
```

### Pipeline Decorator
```python
class PipelineDecorator:
    """Decorator for creating processing pipelines."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stages = []
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            for stage in self.stages:
                result = stage(result)
            
            return result
        return wrapper
    
    def add_stage(self, stage_func: Callable) -> 'PipelineDecorator':
        self.stages.append(stage_func)
        return self
```

## Best Practices

### Decorator Design
1. **Use `functools.wraps`**: Preserve function metadata when creating decorators
2. **Keep Decorators Simple**: Each decorator should have a single responsibility
3. **Handle Exceptions**: Properly handle and propagate exceptions in decorators
4. **Document Decorators**: Clearly document what each decorator does
5. **Consider Performance**: Decorators add overhead, so use them judiciously

### Context Manager Design
1. **Resource Cleanup**: Always ensure resources are properly cleaned up in `__exit__`
2. **Exception Handling**: Decide whether to suppress or re-raise exceptions
3. **State Restoration**: Restore previous state when errors occur
4. **Thread Safety**: Use locks when managing shared resources
5. **Performance**: Minimize overhead in `__enter__` and `__exit__` methods

### 3D Graphics Applications
1. **OpenGL Resources**: Always use context managers for OpenGL resource management
2. **Memory Management**: Use context managers for large 3D data structures
3. **Performance Monitoring**: Decorate expensive 3D operations with timing decorators
4. **Error Recovery**: Implement fallback mechanisms for 3D operations
5. **State Management**: Use context managers to manage render state

## Exercises

### Beginner Level
1. **Timing Decorator**: Create a decorator that measures function execution time
2. **Validation Decorator**: Create a decorator that validates Vector3D inputs
3. **File Context Manager**: Create a context manager for reading 3D mesh files

### Intermediate Level
1. **Caching Decorator**: Implement a decorator that caches matrix multiplication results
2. **Retry Decorator**: Create a decorator that retries failed OpenGL operations
3. **Resource Pool**: Implement a context manager for managing texture resources

### Advanced Level
1. **Pipeline Decorator**: Create a decorator that processes 3D data through multiple stages
2. **State Manager**: Implement a context manager that manages application state with backup/restore
3. **Performance Tracker**: Create a combined decorator/context manager for comprehensive performance tracking

## Common Pitfalls

1. **Forgetting `functools.wraps`**: This can break introspection and debugging
2. **Not Handling Exceptions**: Decorators should properly handle and propagate exceptions
3. **Resource Leaks**: Context managers must ensure proper cleanup even when exceptions occur
4. **Decorator Order**: The order of decorators matters - apply them from bottom to top
5. **Performance Overhead**: Too many decorators can significantly impact performance

## Performance Considerations

- **Decorator Overhead**: Each decorator adds a small performance cost
- **Caching Benefits**: Caching decorators can dramatically improve performance for expensive operations
- **Context Manager Efficiency**: Keep `__enter__` and `__exit__` methods fast
- **Memory Management**: Context managers help prevent memory leaks in 3D applications
- **Thread Safety**: Use appropriate synchronization for shared resources

## Integration with Other Chapters

- **Chapter 9 (Functional Programming)**: Combine with functional concepts for powerful data processing
- **Chapter 10 (Iterators and Generators)**: Use context managers with generators for resource management
- **Chapter 12 (External Libraries)**: Apply decorators and context managers when working with OpenGL and other libraries
- **Part III (3D Graphics)**: Essential for building robust 3D graphics applications

This chapter provides the foundation for building robust, efficient, and maintainable 3D graphics applications using Python's powerful decorator and context manager features.
