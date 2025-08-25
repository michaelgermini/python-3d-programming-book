"""
Chapter 11: Decorators and Context Managers - Advanced Patterns
=============================================================

This module demonstrates advanced patterns combining decorators and context
managers for sophisticated 3D graphics applications, including pipeline
processing, state management, and complex resource handling.

Key Concepts:
- Decorator and context manager composition
- Pipeline processing patterns
- State management and restoration
- Advanced resource handling
- Performance optimization patterns
- Error recovery and fallback strategies
"""

import time
import math
import random
import threading
import functools
from typing import Callable, Any, Dict, List, Optional, Generator, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import json


@dataclass
class Vector3D:
    """3D vector for advanced pattern examples."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        """Normalize the vector."""
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x / mag, self.y / mag, self.z / mag)
        return Vector3D(0, 0, 0)


@dataclass
class Matrix4x4:
    """4x4 transformation matrix."""
    m: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    def __str__(self):
        return f"Matrix4x4({self.m})"


class RenderState(Enum):
    """Enumeration of render states."""
    IDLE = "idle"
    RENDERING = "rendering"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class RenderContext:
    """Context for rendering operations."""
    state: RenderState = RenderState.IDLE
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    
    def update_stats(self, operation: str, duration: float):
        """Update performance statistics."""
        if operation not in self.performance_stats:
            self.performance_stats[operation] = []
        self.performance_stats[operation].append(duration)


# Advanced Decorator Patterns
class PipelineDecorator:
    """Decorator for creating processing pipelines."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stages = []
        self.current_stage = 0
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"  🔄 Starting pipeline: {self.pipeline_name}")
            
            # Execute pipeline stages
            result = func(*args, **kwargs)
            
            for i, stage in enumerate(self.stages):
                print(f"  📋 Executing stage {i+1}: {stage.__name__}")
                result = stage(result)
            
            print(f"  ✅ Pipeline completed: {self.pipeline_name}")
            return result
        return wrapper
    
    def add_stage(self, stage_func: Callable) -> 'PipelineDecorator':
        """Add a stage to the pipeline."""
        self.stages.append(stage_func)
        return self


class StatefulDecorator:
    """Decorator that maintains state across function calls."""
    
    def __init__(self, initial_state: Dict[str, Any] = None):
        self.state = initial_state or {}
        self.call_count = 0
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.call_count += 1
            
            # Update state based on function call
            self.state['last_call'] = time.time()
            self.state['call_count'] = self.call_count
            self.state['last_args'] = args
            self.state['last_kwargs'] = kwargs
            
            # Execute function with state context
            result = func(*args, **kwargs)
            
            # Update state with result
            self.state['last_result'] = result
            
            print(f"  📊 Stateful decorator: {func.__name__} called {self.call_count} times")
            return result
        return wrapper


class CachingDecorator:
    """Advanced caching decorator with TTL and size limits."""
    
    def __init__(self, max_size: int = 100, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result is still valid
            if key in self.cache:
                if current_time - self.access_times[key] < self.ttl:
                    print(f"  💾 Cache hit for {func.__name__}")
                    self.access_times[key] = current_time
                    return self.cache[key]
                else:
                    # Expired cache entry
                    del self.cache[key]
                    del self.access_times[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = result
            self.access_times[key] = current_time
            print(f"  💾 Cache miss for {func.__name__}, stored result")
            
            return result
        return wrapper


# Advanced Context Manager Patterns
class RenderPipeline:
    """Context manager for managing a complete rendering pipeline."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.context = RenderContext()
        self.stages = []
        self.current_stage = 0
    
    def __enter__(self):
        print(f"  🎬 Initializing render pipeline: {self.pipeline_name}")
        self.context.state = RenderState.RENDERING
        self.context.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.context.start_time
        
        if exc_type is not None:
            self.context.state = RenderState.ERROR
            print(f"  ❌ Pipeline failed: {exc_type.__name__}: {exc_val}")
            return False
        else:
            self.context.state = RenderState.IDLE
            print(f"  ✅ Pipeline completed in {duration:.3f}s")
            print(f"  📊 Rendered {self.context.frame_count} frames")
        
        return True
    
    def add_stage(self, stage_name: str, stage_func: Callable):
        """Add a stage to the pipeline."""
        self.stages.append((stage_name, stage_func))
        return self
    
    def execute_stage(self, stage_name: str, data: Any) -> Any:
        """Execute a specific pipeline stage."""
        for name, func in self.stages:
            if name == stage_name:
                start_time = time.time()
                result = func(data)
                duration = time.time() - start_time
                self.context.update_stats(stage_name, duration)
                print(f"  📋 Executed stage: {stage_name} ({duration:.3f}s)")
                return result
        raise ValueError(f"Stage '{stage_name}' not found")


class StateManager:
    """Context manager for managing application state."""
    
    def __init__(self, state_name: str, initial_state: Dict[str, Any] = None):
        self.state_name = state_name
        self.initial_state = initial_state or {}
        self.backup_state = None
        self.current_state = {}
    
    def __enter__(self):
        print(f"  💾 Creating state manager: {self.state_name}")
        # Backup current state
        self.backup_state = self.current_state.copy()
        # Initialize with provided state
        self.current_state.update(self.initial_state)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"  🔄 Restoring state due to error: {exc_type.__name__}")
            # Restore backup state on error
            self.current_state = self.backup_state
            return False
        else:
            print(f"  ✅ State manager completed: {self.state_name}")
        return True
    
    def set_state(self, key: str, value: Any):
        """Set a state value."""
        self.current_state[key] = value
        print(f"  📝 Set state: {key} = {value}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.current_state.get(key, default)


class ResourcePool:
    """Context manager for managing a pool of resources."""
    
    def __init__(self, resource_type: str, pool_size: int = 5):
        self.resource_type = resource_type
        self.pool_size = pool_size
        self.available_resources = []
        self.allocated_resources = set()
        self.lock = threading.Lock()
    
    def __enter__(self):
        print(f"  🏊 Creating resource pool: {self.resource_type} (size: {self.pool_size})")
        # Initialize pool with resources
        for i in range(self.pool_size):
            resource_id = f"{self.resource_type}_{i}"
            self.available_resources.append(resource_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"  🏊 Destroying resource pool: {self.resource_type}")
        print(f"  📊 Pool stats: {len(self.allocated_resources)} allocated, {len(self.available_resources)} available")
        
        if exc_type is not None:
            print(f"  ❌ Resource pool error: {exc_type.__name__}")
            return False
        return True
    
    def acquire_resource(self) -> str:
        """Acquire a resource from the pool."""
        with self.lock:
            if self.available_resources:
                resource_id = self.available_resources.pop()
                self.allocated_resources.add(resource_id)
                print(f"  📦 Acquired resource: {resource_id}")
                return resource_id
            else:
                raise RuntimeError(f"No available {self.resource_type} resources")
    
    def release_resource(self, resource_id: str):
        """Release a resource back to the pool."""
        with self.lock:
            if resource_id in self.allocated_resources:
                self.allocated_resources.remove(resource_id)
                self.available_resources.append(resource_id)
                print(f"  📦 Released resource: {resource_id}")


# Combined Decorator and Context Manager Patterns
class PerformanceTracker:
    """Combines decorator and context manager for comprehensive performance tracking."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.call_count = 0
        self.total_time = 0.0
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.call_count += 1
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self.total_time += duration
                
                print(f"  📊 {self.operation_name}: call #{self.call_count} took {duration:.6f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                print(f"  ❌ {self.operation_name}: call #{self.call_count} failed after {duration:.6f}s")
                raise
        
        return wrapper
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"  🚀 Starting performance tracking: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        if exc_type is not None:
            print(f"  ❌ Performance tracking failed: {exc_type.__name__}")
            return False
        
        print(f"  📊 Performance summary: {self.operation_name}")
        print(f"    Total time: {total_duration:.3f}s")
        print(f"    Calls: {self.call_count}")
        if self.call_count > 0:
            print(f"    Average time: {self.total_time / self.call_count:.6f}s")
        
        return True


# Example Functions and Pipelines
@PipelineDecorator("3D Vector Processing")
def process_vector_pipeline(vector: Vector3D) -> Vector3D:
    """Process a 3D vector through multiple stages."""
    return vector


def normalize_stage(vector: Vector3D) -> Vector3D:
    """Pipeline stage: normalize vector."""
    return vector.normalize()


def scale_stage(vector: Vector3D) -> Vector3D:
    """Pipeline stage: scale vector."""
    return Vector3D(vector.x * 2, vector.y * 2, vector.z * 2)


def transform_stage(vector: Vector3D) -> Vector3D:
    """Pipeline stage: apply transformation."""
    return Vector3D(vector.x + 1, vector.y + 1, vector.z + 1)


# Add stages to the pipeline
process_vector_pipeline.add_stage(normalize_stage).add_stage(scale_stage).add_stage(transform_stage)


@StatefulDecorator({'initialized': False})
@CachingDecorator(max_size=50, ttl=60.0)
def expensive_vector_calculation(v1: Vector3D, v2: Vector3D) -> Vector3D:
    """Expensive vector calculation with state tracking and caching."""
    # Simulate expensive calculation
    time.sleep(0.1)
    return Vector3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)


@PerformanceTracker("matrix_operations")
def create_transformation_matrix(translation: Vector3D, scale: float = 1.0) -> Matrix4x4:
    """Create a transformation matrix."""
    # Simulate matrix creation
    time.sleep(0.05)
    matrix = Matrix4x4()
    matrix.m[0][3] = translation.x
    matrix.m[1][3] = translation.y
    matrix.m[2][3] = translation.z
    matrix.m[0][0] = scale
    matrix.m[1][1] = scale
    matrix.m[2][2] = scale
    return matrix


# Example Usage and Demonstration
def demonstrate_advanced_patterns():
    """Demonstrates advanced decorator and context manager patterns."""
    print("=== Advanced Decorator and Context Manager Patterns ===\n")
    
    # Pipeline processing
    print("=== Pipeline Processing ===")
    
    test_vector = Vector3D(3, 4, 5)
    print(f"Original vector: {test_vector}")
    
    result = process_vector_pipeline(test_vector)
    print(f"Processed vector: {result}")
    
    # Stateful operations
    print("\n=== Stateful Operations ===")
    
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    for i in range(3):
        result = expensive_vector_calculation(v1, v2)
        print(f"Calculation {i+1}: {result}")
    
    # Render pipeline
    print("\n=== Render Pipeline ===")
    
    with RenderPipeline("3D Scene Renderer") as pipeline:
        pipeline.add_stage("geometry", lambda data: f"Processed geometry: {data}")
        pipeline.add_stage("lighting", lambda data: f"Applied lighting: {data}")
        pipeline.add_stage("texturing", lambda data: f"Applied textures: {data}")
        
        # Execute pipeline stages
        scene_data = "cube, sphere, cylinder"
        result = pipeline.execute_stage("geometry", scene_data)
        result = pipeline.execute_stage("lighting", result)
        result = pipeline.execute_stage("texturing", result)
        
        pipeline.context.frame_count = 1
        print(f"Final result: {result}")
    
    # State management
    print("\n=== State Management ===")
    
    with StateManager("render_state", {'quality': 'high', 'resolution': '1080p'}) as state_mgr:
        state_mgr.set_state('frame_count', 0)
        state_mgr.set_state('current_scene', 'main_menu')
        
        print(f"Current quality: {state_mgr.get_state('quality')}")
        print(f"Frame count: {state_mgr.get_state('frame_count')}")
        
        # Simulate some work
        time.sleep(0.01)
    
    # Resource pool
    print("\n=== Resource Pool ===")
    
    with ResourcePool("texture", pool_size=3) as pool:
        # Acquire resources
        tex1 = pool.acquire_resource()
        tex2 = pool.acquire_resource()
        
        print(f"Using textures: {tex1}, {tex2}")
        
        # Simulate texture operations
        time.sleep(0.01)
        
        # Release resources
        pool.release_resource(tex1)
        pool.release_resource(tex2)
        
        # Try to acquire more resources
        tex3 = pool.acquire_resource()
        print(f"Acquired additional texture: {tex3}")
    
    # Performance tracking
    print("\n=== Performance Tracking ===")
    
    tracker = PerformanceTracker("matrix_operations")
    
    with tracker:
        # Create multiple matrices
        for i in range(5):
            translation = Vector3D(i, i, i)
            matrix = create_transformation_matrix(translation, scale=1.5)
            print(f"Created matrix {i+1}: {matrix}")
    
    # Combined patterns
    print("\n=== Combined Patterns ===")
    
    with StateManager("combined_state") as state:
        with ResourcePool("shader", pool_size=2) as shader_pool:
            with RenderPipeline("Combined Pipeline") as pipeline:
                # Set up pipeline stages
                pipeline.add_stage("vertex", lambda data: f"Vertex processed: {data}")
                pipeline.add_stage("fragment", lambda data: f"Fragment processed: {data}")
                
                # Acquire shader resources
                shader1 = shader_pool.acquire_resource()
                shader2 = shader_pool.acquire_resource()
                
                # Update state
                state.set_state('active_shaders', [shader1, shader2])
                state.set_state('pipeline_active', True)
                
                # Execute pipeline
                data = "triangle_data"
                result = pipeline.execute_stage("vertex", data)
                result = pipeline.execute_stage("fragment", result)
                
                pipeline.context.frame_count = 1
                
                print(f"Combined result: {result}")
                print(f"Active shaders: {state.get_state('active_shaders')}")
                
                # Release resources
                shader_pool.release_resource(shader1)
                shader_pool.release_resource(shader2)


if __name__ == "__main__":
    demonstrate_advanced_patterns()
