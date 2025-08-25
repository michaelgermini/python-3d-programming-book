"""
Chapter 19: Scene Graphs and Object Hierarchies - Scene Management
================================================================

This module demonstrates scene management and optimization techniques.

Key Concepts:
- Scene loading and serialization
- Scene optimization and batching
- Scene state management
- Performance monitoring and profiling
"""

import json
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion
from scene_graph import SceneNode, SceneGraph, GeometryNode, LightNode, CameraNode, GroupNode
from transformation_matrices import Transform


@dataclass
class SceneStatistics:
    """Statistics about scene performance and structure."""
    total_nodes: int = 0
    visible_nodes: int = 0
    geometry_nodes: int = 0
    light_nodes: int = 0
    camera_nodes: int = 0
    render_time_ms: float = 0.0
    update_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def reset(self):
        """Reset statistics."""
        self.total_nodes = 0
        self.visible_nodes = 0
        self.geometry_nodes = 0
        self.light_nodes = 0
        self.camera_nodes = 0
        self.render_time_ms = 0.0
        self.update_time_ms = 0.0
        self.memory_usage_mb = 0.0


class SceneSerializer:
    """Handles scene serialization and deserialization."""
    
    @staticmethod
    def serialize_node(node: SceneNode) -> Dict[str, Any]:
        """Serialize a scene node to dictionary."""
        data = {
            'name': node.name,
            'type': node.node_type.value,
            'visible': node.visible,
            'transform': {
                'position': [node.transform.position.x, node.transform.position.y, node.transform.position.z],
                'rotation': [node.transform.rotation.w, node.transform.rotation.x, 
                           node.transform.rotation.y, node.transform.rotation.z],
                'scale': [node.transform.scale.x, node.transform.scale.y, node.transform.scale.z]
            },
            'children': []
        }
        
        # Add type-specific data
        if isinstance(node, GeometryNode):
            data['geometry'] = {
                'vertex_count': len(node.vertices),
                'index_count': len(node.indices),
                'material': node.material
            }
        elif isinstance(node, LightNode):
            data['light'] = {
                'type': node.light_type,
                'color': [node.color.x, node.color.y, node.color.z],
                'intensity': node.intensity,
                'range': node.range
            }
        elif isinstance(node, CameraNode):
            data['camera'] = {
                'fov': node.fov,
                'aspect_ratio': node.aspect_ratio,
                'near_plane': node.near_plane,
                'far_plane': node.far_plane
            }
        
        # Serialize children
        for child in node.children:
            data['children'].append(SceneSerializer.serialize_node(child))
        
        return data
    
    @staticmethod
    def deserialize_node(data: Dict[str, Any]) -> SceneNode:
        """Deserialize a scene node from dictionary."""
        # Create transform
        transform_data = data['transform']
        transform = Transform(
            position=Vector3D(*transform_data['position']),
            rotation=Quaternion(*transform_data['rotation']),
            scale=Vector3D(*transform_data['scale'])
        )
        
        # Create node based on type
        node_type = data['type']
        if node_type == 'group':
            node = GroupNode(data['name'])
        elif node_type == 'geometry':
            node = GeometryNode(data['name'])
            if 'geometry' in data:
                geom_data = data['geometry']
                # Note: In a real implementation, you'd load actual geometry data
                node.material = geom_data.get('material')
        elif node_type == 'light':
            node = LightNode(data['name'])
            if 'light' in data:
                light_data = data['light']
                node.light_type = light_data['type']
                node.color = Vector3D(*light_data['color'])
                node.intensity = light_data['intensity']
                node.range = light_data['range']
        elif node_type == 'camera':
            node = CameraNode(data['name'])
            if 'camera' in data:
                cam_data = data['camera']
                node.fov = cam_data['fov']
                node.aspect_ratio = cam_data['aspect_ratio']
                node.near_plane = cam_data['near_plane']
                node.far_plane = cam_data['far_plane']
        else:
            node = SceneNode(data['name'], node_type)
        
        # Set common properties
        node.transform = transform
        node.visible = data['visible']
        
        # Deserialize children
        for child_data in data['children']:
            child_node = SceneSerializer.deserialize_node(child_data)
            node.add_child(child_node)
        
        return node
    
    @staticmethod
    def save_scene(scene: SceneGraph, filename: str):
        """Save scene to JSON file."""
        scene_data = SceneSerializer.serialize_node(scene.root)
        
        with open(filename, 'w') as f:
            json.dump(scene_data, f, indent=2)
    
    @staticmethod
    def load_scene(filename: str) -> SceneGraph:
        """Load scene from JSON file."""
        with open(filename, 'r') as f:
            scene_data = json.load(f)
        
        root_node = SceneSerializer.deserialize_node(scene_data)
        scene = SceneGraph()
        scene.root = root_node
        
        return scene


class SceneOptimizer:
    """Optimizes scene for better performance."""
    
    def __init__(self):
        self.optimization_level = 1  # 0=off, 1=basic, 2=aggressive
    
    def optimize_scene(self, scene: SceneGraph) -> SceneStatistics:
        """Optimize scene and return statistics."""
        stats = SceneStatistics()
        
        # Count nodes
        def count_nodes(node: SceneNode):
            stats.total_nodes += 1
            if node.visible:
                stats.visible_nodes += 1
            
            if isinstance(node, GeometryNode):
                stats.geometry_nodes += 1
            elif isinstance(node, LightNode):
                stats.light_nodes += 1
            elif isinstance(node, CameraNode):
                stats.camera_nodes += 1
        
        scene.root.traverse(count_nodes)
        
        # Apply optimizations based on level
        if self.optimization_level >= 1:
            self._basic_optimizations(scene)
        
        if self.optimization_level >= 2:
            self._aggressive_optimizations(scene)
        
        return stats
    
    def _basic_optimizations(self, scene: SceneGraph):
        """Apply basic optimizations."""
        # Remove invisible nodes from rendering
        def optimize_node(node: SceneNode):
            if not node.visible:
                # Mark children as not visible for rendering
                for child in node.children:
                    child.visible = False
        
        scene.root.traverse(optimize_node)
    
    def _aggressive_optimizations(self, scene: SceneGraph):
        """Apply aggressive optimizations."""
        # Combine static geometry
        self._combine_static_geometry(scene)
        
        # Optimize LOD
        self._optimize_lod(scene)
    
    def _combine_static_geometry(self, scene: SceneGraph):
        """Combine static geometry nodes."""
        # This is a simplified implementation
        # In practice, you'd analyze geometry and combine compatible meshes
        pass
    
    def _optimize_lod(self, scene: SceneGraph):
        """Optimize level-of-detail settings."""
        # This is a simplified implementation
        # In practice, you'd analyze distance and adjust LOD levels
        pass


class SceneProfiler:
    """Profiles scene performance."""
    
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start a performance timer."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """End a performance timer."""
        if name in self.start_times:
            elapsed = (time.time() - self.start_times[name]) * 1000  # Convert to ms
            self.timers[name] = elapsed
            del self.start_times[name]
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_timer(self, name: str) -> float:
        """Get timer value."""
        return self.timers.get(name, 0.0)
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def reset(self):
        """Reset all timers and counters."""
        self.timers.clear()
        self.counters.clear()
        self.start_times.clear()
    
    def print_report(self):
        """Print performance report."""
        print("=== Performance Report ===")
        print("Timers:")
        for name, value in self.timers.items():
            print(f"  {name}: {value:.2f}ms")
        
        print("Counters:")
        for name, value in self.counters.items():
            print(f"  {name}: {value}")


class SceneManager:
    """Manages scene loading, optimization, and performance."""
    
    def __init__(self):
        self.scene: Optional[SceneGraph] = None
        self.optimizer = SceneOptimizer()
        self.profiler = SceneProfiler()
        self.statistics = SceneStatistics()
        self.render_callbacks: List[Callable[[SceneGraph], None]] = []
    
    def load_scene(self, filename: str) -> SceneGraph:
        """Load scene from file."""
        self.profiler.start_timer("scene_load")
        
        try:
            self.scene = SceneSerializer.load_scene(filename)
            self.profiler.increment_counter("scenes_loaded")
        except Exception as e:
            print(f"Error loading scene: {e}")
            self.scene = SceneGraph()
        
        self.profiler.end_timer("scene_load")
        return self.scene
    
    def save_scene(self, filename: str):
        """Save scene to file."""
        if not self.scene:
            return
        
        self.profiler.start_timer("scene_save")
        SceneSerializer.save_scene(self.scene, filename)
        self.profiler.end_timer("scene_save")
    
    def optimize_scene(self, level: int = 1):
        """Optimize the current scene."""
        if not self.scene:
            return
        
        self.profiler.start_timer("scene_optimization")
        self.optimizer.optimization_level = level
        self.statistics = self.optimizer.optimize_scene(self.scene)
        self.profiler.end_timer("scene_optimization")
    
    def update_scene(self):
        """Update the scene."""
        if not self.scene:
            return
        
        self.profiler.start_timer("scene_update")
        self.scene.update()
        self.profiler.end_timer("scene_update")
    
    def render_scene(self):
        """Render the scene."""
        if not self.scene:
            return
        
        self.profiler.start_timer("scene_render")
        
        # Call render callbacks
        for callback in self.render_callbacks:
            callback(self.scene)
        
        # Render scene
        self.scene.render()
        
        self.profiler.end_timer("scene_render")
        self.profiler.increment_counter("frames_rendered")
    
    def add_render_callback(self, callback: Callable[[SceneGraph], None]):
        """Add a render callback."""
        self.render_callbacks.append(callback)
    
    def get_statistics(self) -> SceneStatistics:
        """Get current scene statistics."""
        if self.scene:
            # Update statistics
            self.statistics.reset()
            
            def count_nodes(node: SceneNode):
                self.statistics.total_nodes += 1
                if node.visible:
                    self.statistics.visible_nodes += 1
                
                if isinstance(node, GeometryNode):
                    self.statistics.geometry_nodes += 1
                elif isinstance(node, LightNode):
                    self.statistics.light_nodes += 1
                elif isinstance(node, CameraNode):
                    self.statistics.camera_nodes += 1
            
            self.scene.root.traverse(count_nodes)
            
            # Add timing information
            self.statistics.render_time_ms = self.profiler.get_timer("scene_render")
            self.statistics.update_time_ms = self.profiler.get_timer("scene_update")
        
        return self.statistics
    
    def print_performance_report(self):
        """Print comprehensive performance report."""
        stats = self.get_statistics()
        
        print("=== Scene Performance Report ===")
        print(f"Total nodes: {stats.total_nodes}")
        print(f"Visible nodes: {stats.visible_nodes}")
        print(f"Geometry nodes: {stats.geometry_nodes}")
        print(f"Light nodes: {stats.light_nodes}")
        print(f"Camera nodes: {stats.camera_nodes}")
        print(f"Render time: {stats.render_time_ms:.2f}ms")
        print(f"Update time: {stats.update_time_ms:.2f}ms")
        print()
        
        self.profiler.print_report()


def demonstrate_scene_management():
    """Demonstrate scene management functionality."""
    print("=== Scene Management Demonstration ===\n")
    
    # Create scene manager
    manager = SceneManager()
    
    # Create a simple scene
    scene = SceneGraph()
    
    # Add camera
    camera = CameraNode("MainCamera")
    camera.transform.position = Vector3D(0, 0, 5)
    scene.add_node(camera)
    
    # Add light
    light = LightNode("MainLight", "directional")
    light.transform.position = Vector3D(5, 5, 5)
    scene.add_node(light)
    
    # Add geometry group
    geometry_group = GroupNode("GeometryGroup")
    scene.add_node(geometry_group)
    
    # Add some geometry
    for i in range(5):
        cube = GeometryNode(f"Cube_{i}")
        cube.transform.position = Vector3D(i * 2 - 4, 0, 0)
        cube.vertices = [Vector3D(0, 0, 0)]  # Simplified
        cube.indices = []
        geometry_group.add_child(cube)
    
    manager.scene = scene
    
    print("1. Scene Statistics:")
    stats = manager.get_statistics()
    print(f"Total nodes: {stats.total_nodes}")
    print(f"Visible nodes: {stats.visible_nodes}")
    print(f"Geometry nodes: {stats.geometry_nodes}")
    print()
    
    # Test optimization
    print("2. Scene Optimization:")
    manager.optimize_scene(level=1)
    stats = manager.get_statistics()
    print(f"After optimization - Total nodes: {stats.total_nodes}")
    print()
    
    # Test performance profiling
    print("3. Performance Profiling:")
    manager.update_scene()
    manager.render_scene()
    manager.print_performance_report()
    print()
    
    # Test serialization
    print("4. Scene Serialization:")
    try:
        manager.save_scene("test_scene.json")
        print("Scene saved to test_scene.json")
        
        # Load scene
        loaded_scene = manager.load_scene("test_scene.json")
        print("Scene loaded from test_scene.json")
        
        # Compare statistics
        original_stats = manager.get_statistics()
        manager.scene = loaded_scene
        loaded_stats = manager.get_statistics()
        
        print(f"Original nodes: {original_stats.total_nodes}")
        print(f"Loaded nodes: {loaded_stats.total_nodes}")
        
    except Exception as e:
        print(f"Serialization test failed: {e}")
    
    print("\n5. Scene Management Complete!")


if __name__ == "__main__":
    demonstrate_scene_management()
