"""
Chapter 10: Iterators and Generators - 3D Data Streaming
======================================================

This module demonstrates how to use iterators and generators for streaming
3D data efficiently, handling large datasets without loading everything
into memory at once.

Key Concepts:
- Streaming large 3D datasets
- Memory-efficient data processing
- Real-time data generation
- Pipeline processing
- Lazy evaluation for 3D graphics
"""

import math
import random
import time
from typing import Iterator, Generator, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Vector3D:
    """3D vector for streaming operations."""
    x: float
    y: float
    z: float
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def distance_to(self, other: 'Vector3D') -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class PointCloud:
    """Point cloud data structure."""
    points: List[Vector3D]
    metadata: Dict[str, Any]
    
    def __iter__(self):
        """Make PointCloud iterable."""
        return iter(self.points)


@dataclass
class Mesh:
    """3D mesh data structure."""
    vertices: List[Vector3D]
    faces: List[Tuple[int, int, int]]
    metadata: Dict[str, Any]
    
    def __iter__(self):
        """Make Mesh iterable over vertices."""
        return iter(self.vertices)


# Streaming Data Sources
class PointCloudStreamer:
    """Streams point cloud data from various sources."""
    
    def __init__(self, source_type: str, **kwargs):
        self.source_type = source_type
        self.kwargs = kwargs
    
    def stream_points(self) -> Generator[Vector3D, None, None]:
        """Stream points based on source type."""
        if self.source_type == "grid":
            yield from self._stream_grid()
        elif self.source_type == "sphere":
            yield from self._stream_sphere()
        elif self.source_type == "random":
            yield from self._stream_random()
        elif self.source_type == "file":
            yield from self._stream_from_file()
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
    
    def _stream_grid(self) -> Generator[Vector3D, None, None]:
        """Stream points in a 3D grid."""
        x_range = self.kwargs.get('x_range', (0, 1, 10))
        y_range = self.kwargs.get('y_range', (0, 1, 10))
        z_range = self.kwargs.get('z_range', (0, 1, 10))
        
        x_start, x_end, x_steps = x_range
        y_start, y_end, y_steps = y_range
        z_start, z_end, z_steps = z_range
        
        for i in range(x_steps):
            for j in range(y_steps):
                for k in range(z_steps):
                    x = x_start + (x_end - x_start) * i / (x_steps - 1)
                    y = y_start + (y_end - y_start) * j / (y_steps - 1)
                    z = z_start + (z_end - z_start) * k / (z_steps - 1)
                    yield Vector3D(x, y, z)
    
    def _stream_sphere(self) -> Generator[Vector3D, None, None]:
        """Stream points on a sphere surface."""
        center = self.kwargs.get('center', Vector3D(0, 0, 0))
        radius = self.kwargs.get('radius', 1.0)
        lat_steps = self.kwargs.get('lat_steps', 20)
        lon_steps = self.kwargs.get('lon_steps', 40)
        
        for lat_idx in range(lat_steps):
            for lon_idx in range(lon_steps):
                lat = math.pi * (lat_idx + 1) / (lat_steps + 1)
                lon = 2 * math.pi * lon_idx / lon_steps
                
                x = center.x + radius * math.sin(lat) * math.cos(lon)
                y = center.y + radius * math.sin(lat) * math.sin(lon)
                z = center.z + radius * math.cos(lat)
                
                yield Vector3D(x, y, z)
    
    def _stream_random(self) -> Generator[Vector3D, None, None]:
        """Stream random points."""
        count = self.kwargs.get('count', 1000)
        bounds = self.kwargs.get('bounds', (-1, 1))
        min_bound, max_bound = bounds
        
        for _ in range(count):
            x = random.uniform(min_bound, max_bound)
            y = random.uniform(min_bound, max_bound)
            z = random.uniform(min_bound, max_bound)
            yield Vector3D(x, y, z)
    
    def _stream_from_file(self) -> Generator[Vector3D, None, None]:
        """Stream points from a file (simulated)."""
        file_path = self.kwargs.get('file_path', 'points.txt')
        
        # Simulate reading from file
        for i in range(100):  # Simulate 100 points
            # Simulate file reading delay
            time.sleep(0.001)
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)
            yield Vector3D(x, y, z)


class MeshStreamer:
    """Streams mesh data efficiently."""
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
    
    def stream_vertices(self) -> Generator[Vector3D, None, None]:
        """Stream mesh vertices."""
        for vertex in self.mesh.vertices:
            yield vertex
    
    def stream_faces(self) -> Generator[Tuple[Vector3D, Vector3D, Vector3D], None, None]:
        """Stream mesh faces as vertex triplets."""
        for face in self.mesh.faces:
            v1 = self.mesh.vertices[face[0]]
            v2 = self.mesh.vertices[face[1]]
            v3 = self.mesh.vertices[face[2]]
            yield (v1, v2, v3)
    
    def stream_edges(self) -> Generator[Tuple[Vector3D, Vector3D], None, None]:
        """Stream mesh edges."""
        for face in self.mesh.faces:
            # Yield edges of each face
            v1 = self.mesh.vertices[face[0]]
            v2 = self.mesh.vertices[face[1]]
            v3 = self.mesh.vertices[face[2]]
            
            yield (v1, v2)
            yield (v2, v3)
            yield (v3, v1)


# Data Processing Pipelines
class StreamingPipeline:
    """Pipeline for processing streaming 3D data."""
    
    def __init__(self):
        self.filters = []
        self.transforms = []
        self.sinks = []
    
    def add_filter(self, predicate):
        """Add a filter to the pipeline."""
        self.filters.append(predicate)
        return self
    
    def add_transform(self, transform_func):
        """Add a transform to the pipeline."""
        self.transforms.append(transform_func)
        return self
    
    def add_sink(self, sink_func):
        """Add a sink to the pipeline."""
        self.sinks.append(sink_func)
        return self
    
    def process_stream(self, data_stream: Iterator) -> Generator:
        """Process a data stream through the pipeline."""
        for item in data_stream:
            # Apply filters
            if all(predicate(item) for predicate in self.filters):
                # Apply transforms
                for transform in self.transforms:
                    item = transform(item)
                
                # Send to sinks
                for sink in self.sinks:
                    sink(item)
                
                yield item


# Real-time Data Generators
class RealTimePointGenerator:
    """Generates points in real-time for simulation."""
    
    def __init__(self, update_rate: float = 60.0):
        self.update_rate = update_rate
        self.time = 0.0
        self.points = []
    
    def generate_frame(self) -> Generator[Vector3D, None, None]:
        """Generate points for current frame."""
        # Simulate moving particles
        for i in range(10):
            x = math.sin(self.time + i * 0.5) * 2
            y = math.cos(self.time + i * 0.3) * 2
            z = math.sin(self.time * 0.5 + i * 0.2)
            yield Vector3D(x, y, z)
        
        self.time += 1.0 / self.update_rate
    
    def stream_frames(self, duration: float) -> Generator[List[Vector3D], None, None]:
        """Stream frames for a given duration."""
        end_time = self.time + duration
        while self.time < end_time:
            frame_points = list(self.generate_frame())
            yield frame_points


class SensorDataStreamer:
    """Simulates streaming sensor data."""
    
    def __init__(self, sensor_count: int = 5):
        self.sensor_count = sensor_count
        self.sensor_positions = [
            Vector3D(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
            for _ in range(sensor_count)
        ]
        self.time = 0.0
    
    def stream_sensor_data(self) -> Generator[Dict[str, Any], None, None]:
        """Stream sensor data."""
        while True:
            for i, position in enumerate(self.sensor_positions):
                # Simulate sensor reading
                reading = {
                    'sensor_id': i,
                    'position': position,
                    'timestamp': self.time,
                    'value': random.uniform(0, 100),
                    'temperature': random.uniform(20, 30)
                }
                yield reading
            
            self.time += 0.1
            time.sleep(0.1)  # Simulate real-time delay


# Memory-Efficient Processing
class ChunkedProcessor:
    """Processes data in chunks to manage memory usage."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def process_in_chunks(self, data_stream: Iterator, process_func) -> Generator:
        """Process data stream in chunks."""
        chunk = []
        for item in data_stream:
            chunk.append(item)
            if len(chunk) >= self.chunk_size:
                yield process_func(chunk)
                chunk = []
        
        if chunk:  # Process remaining items
            yield process_func(chunk)


class StreamingAnalyzer:
    """Analyzes streaming 3D data in real-time."""
    
    def __init__(self):
        self.stats = {
            'point_count': 0,
            'bounds': {'min': Vector3D(float('inf'), float('inf'), float('inf')),
                      'max': Vector3D(float('-inf'), float('-inf'), float('-inf'))},
            'center': Vector3D(0, 0, 0),
            'total_magnitude': 0.0
        }
    
    def update_stats(self, point: Vector3D):
        """Update statistics with new point."""
        self.stats['point_count'] += 1
        
        # Update bounds
        self.stats['bounds']['min'].x = min(self.stats['bounds']['min'].x, point.x)
        self.stats['bounds']['min'].y = min(self.stats['bounds']['min'].y, point.y)
        self.stats['bounds']['min'].z = min(self.stats['bounds']['min'].z, point.z)
        
        self.stats['bounds']['max'].x = max(self.stats['bounds']['max'].x, point.x)
        self.stats['bounds']['max'].y = max(self.stats['bounds']['max'].y, point.y)
        self.stats['bounds']['max'].z = max(self.stats['bounds']['max'].z, point.z)
        
        # Update center (running average)
        n = self.stats['point_count']
        self.stats['center'].x = (self.stats['center'].x * (n-1) + point.x) / n
        self.stats['center'].y = (self.stats['center'].y * (n-1) + point.y) / n
        self.stats['center'].z = (self.stats['center'].z * (n-1) + point.z) / n
        
        # Update total magnitude
        self.stats['total_magnitude'] += point.magnitude()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        if stats['point_count'] > 0:
            stats['average_magnitude'] = stats['total_magnitude'] / stats['point_count']
        return stats


# Utility Functions
def distance_filter(max_distance: float, reference_point: Vector3D):
    """Filter points by distance from reference point."""
    def predicate(point: Vector3D) -> bool:
        return point.distance_to(reference_point) <= max_distance
    return predicate


def magnitude_transform():
    """Transform points to their magnitudes."""
    return lambda point: point.magnitude()


def print_sink(item):
    """Print sink for pipeline."""
    print(f"  Processed: {item}")


def stats_sink(analyzer: StreamingAnalyzer):
    """Statistics sink for pipeline."""
    def sink(item):
        if isinstance(item, Vector3D):
            analyzer.update_stats(item)
    return sink


# Example Usage and Demonstration
def demonstrate_3d_data_streaming():
    """Demonstrates 3D data streaming with iterators and generators."""
    print("=== 3D Data Streaming with Iterators and Generators ===\n")
    
    # Point cloud streaming
    print("=== Point Cloud Streaming ===")
    
    print("Grid point cloud:")
    grid_streamer = PointCloudStreamer("grid", 
                                      x_range=(0, 1, 3), 
                                      y_range=(0, 1, 3), 
                                      z_range=(0, 1, 2))
    for point in grid_streamer.stream_points():
        print(f"  {point}")
    
    print("\nSphere point cloud:")
    sphere_streamer = PointCloudStreamer("sphere", 
                                        center=Vector3D(0, 0, 0), 
                                        radius=1.0, 
                                        lat_steps=3, 
                                        lon_steps=4)
    for point in sphere_streamer.stream_points():
        print(f"  {point}")
    
    # Mesh streaming
    print("\n=== Mesh Streaming ===")
    
    # Create a simple cube mesh
    vertices = [
        Vector3D(0, 0, 0), Vector3D(1, 0, 0), Vector3D(1, 1, 0), Vector3D(0, 1, 0),
        Vector3D(0, 0, 1), Vector3D(1, 0, 1), Vector3D(1, 1, 1), Vector3D(0, 1, 1)
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # Front
        (4, 6, 5), (4, 7, 6),  # Back
        (0, 4, 5), (0, 5, 1),  # Bottom
        (2, 6, 7), (2, 7, 3),  # Top
        (0, 3, 7), (0, 7, 4),  # Left
        (1, 5, 6), (1, 6, 2)   # Right
    ]
    
    mesh = Mesh(vertices, faces, {'name': 'cube'})
    mesh_streamer = MeshStreamer(mesh)
    
    print("Mesh vertices:")
    for vertex in mesh_streamer.stream_vertices():
        print(f"  {vertex}")
    
    print("\nMesh faces:")
    for face in mesh_streamer.stream_faces():
        print(f"  Face: {face[0]} - {face[1]} - {face[2]}")
    
    # Streaming pipeline
    print("\n=== Streaming Pipeline ===")
    
    # Create pipeline
    pipeline = StreamingPipeline()
    analyzer = StreamingAnalyzer()
    
    pipeline.add_filter(distance_filter(2.0, Vector3D(0, 0, 0)))
    pipeline.add_transform(magnitude_transform())
    pipeline.add_sink(print_sink)
    pipeline.add_sink(stats_sink(analyzer))
    
    print("Processing random points through pipeline:")
    random_streamer = PointCloudStreamer("random", count=10, bounds=(-2, 2))
    for result in pipeline.process_stream(random_streamer.stream_points()):
        pass  # Results are printed by sink
    
    print(f"\nFinal statistics: {analyzer.get_stats()}")
    
    # Real-time data generation
    print("\n=== Real-time Data Generation ===")
    
    print("Real-time point generation (3 frames):")
    rt_generator = RealTimePointGenerator(update_rate=10.0)
    for i, frame in enumerate(rt_generator.stream_frames(0.3)):
        print(f"  Frame {i}: {[str(p) for p in frame]}")
    
    # Chunked processing
    print("\n=== Chunked Processing ===")
    
    chunked_processor = ChunkedProcessor(chunk_size=3)
    
    def process_chunk(chunk):
        return f"Chunk with {len(chunk)} points: {[str(p) for p in chunk]}"
    
    print("Processing in chunks:")
    grid_streamer = PointCloudStreamer("grid", 
                                      x_range=(0, 1, 2), 
                                      y_range=(0, 1, 2), 
                                      z_range=(0, 1, 2))
    for chunk_result in chunked_processor.process_in_chunks(
        grid_streamer.stream_points(), process_chunk):
        print(f"  {chunk_result}")


if __name__ == "__main__":
    demonstrate_3d_data_streaming()
