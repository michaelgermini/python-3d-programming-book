"""
Chapter 28: Simple Ray Tracing and Path Tracing - Optimization Techniques
====================================================================

This module demonstrates optimization techniques for ray tracing and path tracing.

Key Concepts:
- Spatial data structures for acceleration
- Parallel processing and GPU acceleration
- Noise reduction and denoising techniques
- Adaptive sampling and importance sampling
- Memory optimization and caching strategies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


class AccelerationStructure(Enum):
    """Acceleration structure type enumeration."""
    BVH = "bvh"
    OCTREE = "octree"
    KD_TREE = "kd_tree"
    GRID = "grid"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: np.ndarray
    max_point: np.ndarray
    
    def __post_init__(self):
        if self.min_point is None:
            self.min_point = np.array([-float('inf'), -float('inf'), -float('inf')])
        if self.max_point is None:
            self.max_point = np.array([float('inf'), float('inf'), float('inf')])
    
    def expand(self, point: np.ndarray):
        """Expand bounding box to include point."""
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside bounding box."""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def intersects_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> bool:
        """Test ray-bounding box intersection."""
        t_min = (self.min_point - ray_origin) / ray_direction
        t_max = (self.max_point - ray_origin) / ray_direction
        
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        
        t_near = np.max(t1)
        t_far = np.min(t2)
        
        return t_far >= t_near and t_near >= 0
    
    def get_center(self) -> np.ndarray:
        """Get center of bounding box."""
        return (self.min_point + self.max_point) / 2
    
    def get_extent(self) -> np.ndarray:
        """Get extent of bounding box."""
        return self.max_point - self.min_point


class BVHNode:
    """Node in Bounding Volume Hierarchy."""
    
    def __init__(self):
        self.bounding_box: Optional[BoundingBox] = None
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.objects: List[Any] = []
        self.is_leaf = False


class BVH:
    """Bounding Volume Hierarchy for ray tracing acceleration."""
    
    def __init__(self, objects: List[Any]):
        self.objects = objects
        self.root = self.build_bvh(objects)
    
    def build_bvh(self, objects: List[Any]) -> BVHNode:
        """Build BVH from objects."""
        if len(objects) == 0:
            return None
        
        node = BVHNode()
        
        if len(objects) == 1:
            # Leaf node
            node.is_leaf = True
            node.objects = objects
            node.bounding_box = self.compute_bounding_box(objects)
            return node
        
        # Compute bounding box for all objects
        node.bounding_box = self.compute_bounding_box(objects)
        
        # Choose split axis and position
        extent = node.bounding_box.get_extent()
        axis = np.argmax(extent)
        mid_point = len(objects) // 2
        
        # Sort objects by center along split axis
        objects.sort(key=lambda obj: self.get_object_center(obj)[axis])
        
        # Split objects
        left_objects = objects[:mid_point]
        right_objects = objects[mid_point:]
        
        # Recursively build children
        node.left = self.build_bvh(left_objects)
        node.right = self.build_bvh(right_objects)
        
        return node
    
    def compute_bounding_box(self, objects: List[Any]) -> BoundingBox:
        """Compute bounding box for list of objects."""
        if not objects:
            return BoundingBox(np.array([0, 0, 0]), np.array([0, 0, 0]))
        
        bbox = BoundingBox(np.array([float('inf'), float('inf'), float('inf')]),
                          np.array([-float('inf'), -float('inf'), -float('inf')]))
        
        for obj in objects:
            obj_bbox = self.get_object_bounding_box(obj)
            bbox.expand(obj_bbox.min_point)
            bbox.expand(obj_bbox.max_point)
        
        return bbox
    
    def get_object_center(self, obj: Any) -> np.ndarray:
        """Get center of object."""
        bbox = self.get_object_bounding_box(obj)
        return bbox.get_center()
    
    def get_object_bounding_box(self, obj: Any) -> BoundingBox:
        """Get bounding box of object."""
        # This would be implemented based on the actual object type
        # For now, return a default bounding box
        return BoundingBox(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    
    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> List[Any]:
        """Find objects intersecting with ray."""
        if self.root is None:
            return []
        
        return self.intersect_node(self.root, ray_origin, ray_direction)
    
    def intersect_node(self, node: BVHNode, ray_origin: np.ndarray, 
                      ray_direction: np.ndarray) -> List[Any]:
        """Intersect ray with BVH node."""
        if node is None:
            return []
        
        # Check if ray intersects bounding box
        if not node.bounding_box.intersects_ray(ray_origin, ray_direction):
            return []
        
        if node.is_leaf:
            return node.objects
        
        # Recursively check children
        left_objects = self.intersect_node(node.left, ray_origin, ray_direction)
        right_objects = self.intersect_node(node.right, ray_origin, ray_direction)
        
        return left_objects + right_objects


class Octree:
    """Octree spatial data structure for ray tracing acceleration."""
    
    def __init__(self, bounding_box: BoundingBox, max_depth: int = 8, max_objects: int = 10):
        self.bounding_box = bounding_box
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.objects: List[Any] = []
        self.children: List[Optional['Octree']] = [None] * 8
        self.is_subdivided = False
    
    def insert(self, obj: Any):
        """Insert object into octree."""
        if not self.bounding_box.contains(self.get_object_center(obj)):
            return
        
        if not self.is_subdivided and len(self.objects) < self.max_objects:
            self.objects.append(obj)
            return
        
        if not self.is_subdivided:
            self.subdivide()
        
        # Insert into appropriate child
        child_index = self.get_child_index(obj)
        if self.children[child_index]:
            self.children[child_index].insert(obj)
    
    def subdivide(self):
        """Subdivide octree node."""
        center = self.bounding_box.get_center()
        extent = self.bounding_box.get_extent() / 2
        
        # Create 8 children
        for i in range(8):
            child_min = center + np.array([
                -extent[0] if i & 1 else extent[0],
                -extent[1] if i & 2 else extent[1],
                -extent[2] if i & 4 else extent[2]
            ])
            child_max = center + np.array([
                extent[0] if i & 1 else -extent[0],
                extent[1] if i & 2 else -extent[1],
                extent[2] if i & 4 else -extent[2]
            ])
            
            child_bbox = BoundingBox(child_min, child_max)
            self.children[i] = Octree(child_bbox, self.max_depth - 1, self.max_objects)
        
        # Redistribute existing objects
        for obj in self.objects:
            child_index = self.get_child_index(obj)
            self.children[child_index].insert(obj)
        
        self.objects.clear()
        self.is_subdivided = True
    
    def get_child_index(self, obj: Any) -> int:
        """Get child index for object."""
        center = self.bounding_box.get_center()
        obj_center = self.get_object_center(obj)
        
        index = 0
        if obj_center[0] > center[0]:
            index |= 1
        if obj_center[1] > center[1]:
            index |= 2
        if obj_center[2] > center[2]:
            index |= 4
        
        return index
    
    def get_object_center(self, obj: Any) -> np.ndarray:
        """Get center of object."""
        # This would be implemented based on the actual object type
        return np.array([0.0, 0.0, 0.0])
    
    def query(self, query_bbox: BoundingBox) -> List[Any]:
        """Query objects in bounding box."""
        if not self.bounding_box.intersects_ray(query_bbox.get_center(), np.array([1, 0, 0])):
            return []
        
        result = []
        
        if self.is_subdivided:
            for child in self.children:
                if child:
                    result.extend(child.query(query_bbox))
        else:
            result.extend(self.objects)
        
        return result


class Denoiser:
    """Image denoising for ray traced images."""
    
    def __init__(self):
        self.kernel_size = 3
        self.sigma = 1.0
    
    def gaussian_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian denoising to image."""
        height, width, channels = image.shape
        denoised = np.zeros_like(image)
        
        # Create Gaussian kernel
        kernel = self.create_gaussian_kernel(self.kernel_size, self.sigma)
        kernel_size = len(kernel)
        pad = kernel_size // 2
        
        # Apply convolution
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    value = 0.0
                    weight_sum = 0.0
                    
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            sy = max(0, min(height - 1, y + ky - pad))
                            sx = max(0, min(width - 1, x + kx - pad))
                            
                            weight = kernel[ky, kx]
                            value += image[sy, sx, c] * weight
                            weight_sum += weight
                    
                    denoised[y, x, c] = value / weight_sum
        
        return denoised
    
    def create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create Gaussian kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                kernel[y, x] = math.exp(-(dx*dx + dy*dy) / (2*sigma*sigma))
        
        return kernel / np.sum(kernel)
    
    def bilateral_denoise(self, image: np.ndarray, color_sigma: float = 0.1, 
                         spatial_sigma: float = 1.0) -> np.ndarray:
        """Apply bilateral denoising to image."""
        height, width, channels = image.shape
        denoised = np.zeros_like(image)
        
        kernel_size = 5
        pad = kernel_size // 2
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    value = 0.0
                    weight_sum = 0.0
                    
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            sy = max(0, min(height - 1, y + ky - pad))
                            sx = max(0, min(width - 1, x + kx - pad))
                            
                            # Spatial weight
                            spatial_dist = math.sqrt((ky - pad)**2 + (kx - pad)**2)
                            spatial_weight = math.exp(-spatial_dist / (2 * spatial_sigma**2))
                            
                            # Color weight
                            color_diff = abs(image[sy, sx, c] - image[y, x, c])
                            color_weight = math.exp(-color_diff / (2 * color_sigma**2))
                            
                            weight = spatial_weight * color_weight
                            value += image[sy, sx, c] * weight
                            weight_sum += weight
                    
                    denoised[y, x, c] = value / weight_sum if weight_sum > 0 else image[y, x, c]
        
        return denoised


class AdaptiveSampler:
    """Adaptive sampling for ray tracing."""
    
    def __init__(self, min_samples: int = 4, max_samples: int = 64, 
                 variance_threshold: float = 0.01):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.variance_threshold = variance_threshold
    
    def calculate_variance(self, samples: List[np.ndarray]) -> float:
        """Calculate variance of samples."""
        if len(samples) < 2:
            return 0.0
        
        mean = np.mean(samples, axis=0)
        variance = np.mean([np.sum((sample - mean)**2) for sample in samples])
        return variance
    
    def should_continue_sampling(self, samples: List[np.ndarray]) -> bool:
        """Determine if more samples are needed."""
        if len(samples) < self.min_samples:
            return True
        
        if len(samples) >= self.max_samples:
            return False
        
        variance = self.calculate_variance(samples)
        return variance > self.variance_threshold
    
    def get_optimal_sample_count(self, samples: List[np.ndarray]) -> int:
        """Estimate optimal sample count based on variance."""
        if len(samples) < 2:
            return self.min_samples
        
        variance = self.calculate_variance(samples)
        
        # Estimate required samples based on variance
        estimated_samples = int(self.variance_threshold / variance * len(samples))
        return max(self.min_samples, min(self.max_samples, estimated_samples))


class ParallelRenderer:
    """Parallel ray tracing renderer."""
    
    def __init__(self, num_threads: int = None):
        self.num_threads = num_threads or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
    
    def render_parallel(self, render_function, width: int, height: int, 
                       chunk_size: int = 32) -> np.ndarray:
        """Render image using parallel processing."""
        image = np.zeros((height, width, 3))
        
        # Split image into chunks
        chunks = []
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                chunk_width = min(chunk_size, width - x)
                chunk_height = min(chunk_size, height - y)
                chunks.append((x, y, chunk_width, chunk_height))
        
        # Render chunks in parallel
        futures = []
        for x, y, w, h in chunks:
            future = self.thread_pool.submit(render_function, x, y, w, h)
            futures.append((future, x, y, w, h))
        
        # Collect results
        for future, x, y, w, h in futures:
            chunk_image = future.result()
            image[y:y+h, x:x+w] = chunk_image
        
        return image
    
    def render_chunk(self, render_function, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Render a chunk of the image."""
        chunk_image = np.zeros((height, width, 3))
        
        for j in range(height):
            for i in range(width):
                pixel_x = x + i
                pixel_y = y + j
                chunk_image[j, i] = render_function(pixel_x, pixel_y)
        
        return chunk_image


class MemoryOptimizer:
    """Memory optimization for ray tracing."""
    
    def __init__(self):
        self.object_pool = {}
        self.texture_cache = {}
        self.max_cache_size = 1000
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Get object from cache."""
        return self.object_pool.get(key)
    
    def cache_object(self, key: str, obj: Any):
        """Cache object."""
        if len(self.object_pool) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.object_pool))
            del self.object_pool[oldest_key]
        
        self.object_pool[key] = obj
    
    def optimize_mesh_data(self, vertices: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize mesh data for memory efficiency."""
        # Remove duplicate vertices
        unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
        optimized_indices = inverse_indices[indices]
        
        return unique_vertices, optimized_indices
    
    def compress_texture(self, texture: np.ndarray, quality: float = 0.8) -> np.ndarray:
        """Compress texture data."""
        # Simple compression by reducing precision
        compressed = (texture * 255).astype(np.uint8) / 255.0
        return compressed
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        return {
            'object_pool_size': len(self.object_pool),
            'texture_cache_size': len(self.texture_cache),
            'total_cached_objects': len(self.object_pool) + len(self.texture_cache)
        }


class PerformanceProfiler:
    """Performance profiling for ray tracing."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        if name not in self.timings:
            self.timings[name] = []
        
        self.timings[name].append(time.time())
    
    def end_timer(self, name: str):
        """End timing an operation."""
        if name in self.timings and self.timings[name]:
            start_time = self.timings[name].pop()
            duration = time.time() - start_time
            self.timings[name].append(duration)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_average_time(self, name: str) -> float:
        """Get average time for operation."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
        
        return np.mean(self.timings[name])
    
    def get_total_time(self, name: str) -> float:
        """Get total time for operation."""
        if name not in self.timings:
            return 0.0
        
        return np.sum(self.timings[name])
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def print_report(self):
        """Print performance report."""
        print("=== Performance Report ===")
        
        print("\nTimings:")
        for name, times in self.timings.items():
            if times:
                avg_time = np.mean(times)
                total_time = np.sum(times)
                print(f"  {name}: avg={avg_time:.4f}s, total={total_time:.4f}s")
        
        print("\nCounters:")
        for name, count in self.counters.items():
            print(f"  {name}: {count}")


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques."""
    print("=== Simple Ray Tracing and Path Tracing - Optimization Techniques ===\n")

    # Test BVH
    print("1. Bounding Volume Hierarchy (BVH):")
    bbox = BoundingBox(np.array([-10, -10, -10]), np.array([10, 10, 10]))
    bvh = BVH([])  # Empty BVH for demonstration
    print(f"   BVH created with bounding box: {bbox.min_point} to {bbox.max_point}")
    
    # Test Octree
    print(f"\n2. Octree spatial structure:")
    octree = Octree(bbox, max_depth=4, max_objects=5)
    print(f"   Octree created with max depth: {octree.max_depth}")
    print(f"   Max objects per node: {octree.max_objects}")
    
    # Test Denoiser
    print(f"\n3. Image denoising:")
    denoiser = Denoiser()
    test_image = np.random.random((64, 64, 3))
    denoised_image = denoiser.gaussian_denoise(test_image)
    print(f"   Original image shape: {test_image.shape}")
    print(f"   Denoised image shape: {denoised_image.shape}")
    print(f"   Noise reduction applied")
    
    # Test Adaptive Sampler
    print(f"\n4. Adaptive sampling:")
    sampler = AdaptiveSampler(min_samples=4, max_samples=32, variance_threshold=0.01)
    test_samples = [np.random.random(3) for _ in range(10)]
    should_continue = sampler.should_continue_sampling(test_samples)
    optimal_count = sampler.get_optimal_sample_count(test_samples)
    print(f"   Should continue sampling: {should_continue}")
    print(f"   Optimal sample count: {optimal_count}")
    
    # Test Parallel Renderer
    print(f"\n5. Parallel rendering:")
    renderer = ParallelRenderer(num_threads=4)
    print(f"   Number of threads: {renderer.num_threads}")
    
    # Test Memory Optimizer
    print(f"\n6. Memory optimization:")
    optimizer = MemoryOptimizer()
    test_vertices = np.random.random((1000, 3))
    test_indices = np.random.randint(0, 1000, (3000,))
    optimized_vertices, optimized_indices = optimizer.optimize_mesh_data(test_vertices, test_indices)
    print(f"   Original vertices: {len(test_vertices)}")
    print(f"   Optimized vertices: {len(optimized_vertices)}")
    print(f"   Memory reduction: {((len(test_vertices) - len(optimized_vertices)) / len(test_vertices) * 100):.1f}%")
    
    # Test Performance Profiler
    print(f"\n7. Performance profiling:")
    profiler = PerformanceProfiler()
    
    profiler.start_timer("test_operation")
    time.sleep(0.01)  # Simulate work
    profiler.end_timer("test_operation")
    
    profiler.increment_counter("ray_tests", 1000)
    profiler.increment_counter("intersections", 500)
    
    print(f"   Average time for test operation: {profiler.get_average_time('test_operation'):.4f}s")
    print(f"   Ray tests performed: {profiler.get_counter('ray_tests')}")
    print(f"   Intersections found: {profiler.get_counter('intersections')}")
    
    print(f"\n8. Features demonstrated:")
    print(f"   - Spatial acceleration structures (BVH, Octree)")
    print(f"   - Image denoising (Gaussian, Bilateral)")
    print(f"   - Adaptive sampling based on variance")
    print(f"   - Parallel rendering with thread pools")
    print(f"   - Memory optimization and caching")
    print(f"   - Performance profiling and analysis")
    print(f"   - Mesh data optimization")


if __name__ == "__main__":
    import os
    demonstrate_optimization_techniques()
