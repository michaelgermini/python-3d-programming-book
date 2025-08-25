"""
Chapter 19: Scene Graphs and Object Hierarchies - Scene Graph
============================================================

This module demonstrates scene graph implementation and management.

Key Concepts:
- Scene graph structure and organization
- Node types and hierarchies
- Scene traversal and rendering
- Spatial organization and culling
"""

import math
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion
from transformation_matrices import Transform


class NodeType(Enum):
    """Types of scene graph nodes."""
    GROUP = "group"
    GEOMETRY = "geometry"
    LIGHT = "light"
    CAMERA = "camera"
    TRANSFORM = "transform"
    MATERIAL = "material"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: Vector3D
    max_point: Vector3D
    
    def __post_init__(self):
        """Ensure min and max are properly ordered."""
        self.min_point = Vector3D(
            min(self.min_point.x, self.max_point.x),
            min(self.min_point.y, self.max_point.y),
            min(self.min_point.z, self.max_point.z)
        )
        self.max_point = Vector3D(
            max(self.min_point.x, self.max_point.x),
            max(self.min_point.y, self.max_point.y),
            max(self.min_point.z, self.max_point.z)
        )
    
    def get_center(self) -> Vector3D:
        """Get center point of bounding box."""
        return (self.min_point + self.max_point) * 0.5
    
    def get_size(self) -> Vector3D:
        """Get size of bounding box."""
        return self.max_point - self.min_point
    
    def get_radius(self) -> float:
        """Get radius of bounding sphere."""
        size = self.get_size()
        return size.magnitude() * 0.5
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if point is inside bounding box."""
        return (self.min_point.x <= point.x <= self.max_point.x and
                self.min_point.y <= point.y <= self.max_point.y and
                self.min_point.z <= point.z <= self.max_point.z)
    
    def intersects_box(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return not (self.max_point.x < other.min_point.x or
                   self.min_point.x > other.max_point.x or
                   self.max_point.y < other.min_point.y or
                   self.min_point.y > other.max_point.y or
                   self.max_point.z < other.min_point.z or
                   self.min_point.z > other.max_point.z)
    
    def expand(self, point: Vector3D):
        """Expand bounding box to include point."""
        self.min_point = Vector3D(
            min(self.min_point.x, point.x),
            min(self.min_point.y, point.y),
            min(self.min_point.z, point.z)
        )
        self.max_point = Vector3D(
            max(self.max_point.x, point.x),
            max(self.max_point.y, point.y),
            max(self.max_point.z, point.z)
        )
    
    def merge(self, other: 'BoundingBox'):
        """Merge this bounding box with another."""
        self.expand(other.min_point)
        self.expand(other.max_point)


class SceneNode:
    """Base class for scene graph nodes."""
    
    def __init__(self, name: str, node_type: NodeType):
        self.name = name
        self.node_type = node_type
        self.parent: Optional[SceneNode] = None
        self.children: List[SceneNode] = []
        self.transform = Transform()
        self.visible = True
        self.bounding_box: Optional[BoundingBox] = None
        self.user_data: Dict[str, Any] = {}
    
    def add_child(self, child: 'SceneNode'):
        """Add a child node."""
        if child.parent:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'SceneNode'):
        """Remove a child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def get_world_transform(self) -> Transform:
        """Get world space transformation."""
        if self.parent is None:
            return self.transform
        
        parent_world = self.parent.get_world_transform()
        return parent_world.combine(self.transform)
    
    def get_world_matrix(self) -> Matrix4x4:
        """Get world space transformation matrix."""
        return self.get_world_transform().get_matrix()
    
    def find_child_by_name(self, name: str) -> Optional['SceneNode']:
        """Find child node by name."""
        for child in self.children:
            if child.name == name:
                return child
            result = child.find_child_by_name(name)
            if result:
                return result
        return None
    
    def get_all_children(self) -> List['SceneNode']:
        """Get all descendant nodes."""
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_children())
        return result
    
    def update_bounding_box(self):
        """Update bounding box for this node and children."""
        if not self.children:
            return
        
        # Update children first
        for child in self.children:
            child.update_bounding_box()
        
        # Merge children bounding boxes
        if self.children[0].bounding_box:
            self.bounding_box = BoundingBox(
                self.children[0].bounding_box.min_point,
                self.children[0].bounding_box.max_point
            )
            
            for child in self.children[1:]:
                if child.bounding_box:
                    self.bounding_box.merge(child.bounding_box)
    
    def traverse(self, visitor: Callable[['SceneNode'], None]):
        """Traverse the scene graph with a visitor function."""
        if not self.visible:
            return
        
        visitor(self)
        
        for child in self.children:
            child.traverse(visitor)
    
    def render(self, render_context: Dict[str, Any]):
        """Render this node (to be overridden by subclasses)."""
        pass


class GroupNode(SceneNode):
    """Group node that can contain other nodes."""
    
    def __init__(self, name: str):
        super().__init__(name, NodeType.GROUP)


class GeometryNode(SceneNode):
    """Node containing 3D geometry."""
    
    def __init__(self, name: str):
        super().__init__(name, NodeType.GEOMETRY)
        self.vertices: List[Vector3D] = []
        self.indices: List[int] = []
        self.material: Optional[str] = None
    
    def set_geometry(self, vertices: List[Vector3D], indices: List[int]):
        """Set geometry data."""
        self.vertices = vertices
        self.indices = indices
        self._update_bounding_box_from_geometry()
    
    def _update_bounding_box_from_geometry(self):
        """Update bounding box from geometry vertices."""
        if not self.vertices:
            return
        
        min_point = self.vertices[0]
        max_point = self.vertices[0]
        
        for vertex in self.vertices[1:]:
            min_point = Vector3D(
                min(min_point.x, vertex.x),
                min(min_point.y, vertex.y),
                min(min_point.z, vertex.z)
            )
            max_point = Vector3D(
                max(max_point.x, vertex.x),
                max(max_point.y, vertex.y),
                max(max_point.z, vertex.z)
            )
        
        self.bounding_box = BoundingBox(min_point, max_point)
    
    def render(self, render_context: Dict[str, Any]):
        """Render geometry."""
        if not self.visible or not self.vertices:
            return
        
        # Apply transformation
        world_matrix = self.get_world_matrix()
        
        # Transform vertices to world space
        world_vertices = []
        for vertex in self.vertices:
            world_vertex = world_matrix.transform_point(vertex)
            world_vertices.append(world_vertex)
        
        # Render geometry (simplified)
        print(f"Rendering geometry '{self.name}' with {len(self.vertices)} vertices")


class LightNode(SceneNode):
    """Node representing a light source."""
    
    def __init__(self, name: str, light_type: str = "point"):
        super().__init__(name, NodeType.LIGHT)
        self.light_type = light_type
        self.color = Vector3D(1, 1, 1)
        self.intensity = 1.0
        self.range = 10.0
    
    def render(self, render_context: Dict[str, Any]):
        """Setup light for rendering."""
        if not self.visible:
            return
        
        world_position = self.get_world_transform().position
        print(f"Setting up {self.light_type} light '{self.name}' at {world_position}")


class CameraNode(SceneNode):
    """Node representing a camera."""
    
    def __init__(self, name: str):
        super().__init__(name, NodeType.CAMERA)
        self.fov = math.pi / 4
        self.aspect_ratio = 16.0 / 9.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
    
    def get_view_matrix(self) -> Matrix4x4:
        """Get view matrix for this camera."""
        world_transform = self.get_world_transform()
        position = world_transform.position
        target = position + world_transform.rotation.rotate_vector(Vector3D(0, 0, -1))
        up = world_transform.rotation.rotate_vector(Vector3D(0, 1, 0))
        
        return Matrix4x4.look_at(position, target, up)
    
    def get_projection_matrix(self) -> Matrix4x4:
        """Get projection matrix for this camera."""
        return Matrix4x4.perspective(self.fov, self.aspect_ratio, self.near_plane, self.far_plane)
    
    def render(self, render_context: Dict[str, Any]):
        """Setup camera for rendering."""
        if not self.visible:
            return
        
        view_matrix = self.get_view_matrix()
        projection_matrix = self.get_projection_matrix()
        
        render_context['view_matrix'] = view_matrix
        render_context['projection_matrix'] = projection_matrix
        
        print(f"Setting up camera '{self.name}'")


class SceneGraph:
    """Main scene graph class."""
    
    def __init__(self):
        self.root = GroupNode("Root")
        self.active_camera: Optional[CameraNode] = None
        self.lights: List[LightNode] = []
        self.render_callbacks: List[Callable[[SceneNode, Dict[str, Any]], None]] = []
    
    def add_node(self, node: SceneNode, parent: Optional[SceneNode] = None):
        """Add a node to the scene graph."""
        if parent is None:
            parent = self.root
        
        parent.add_child(node)
        
        # Track special node types
        if isinstance(node, CameraNode):
            if not self.active_camera:
                self.active_camera = node
        elif isinstance(node, LightNode):
            self.lights.append(node)
    
    def remove_node(self, node: SceneNode):
        """Remove a node from the scene graph."""
        if node.parent:
            node.parent.remove_child(node)
        
        # Remove from special tracking
        if isinstance(node, CameraNode) and node == self.active_camera:
            self.active_camera = None
        elif isinstance(node, LightNode) and node in self.lights:
            self.lights.remove(node)
    
    def find_node(self, name: str) -> Optional[SceneNode]:
        """Find a node by name."""
        return self.root.find_child_by_name(name)
    
    def update(self):
        """Update the scene graph."""
        self.root.update_bounding_box()
    
    def render(self):
        """Render the entire scene."""
        render_context = {
            'view_matrix': None,
            'projection_matrix': None,
            'lights': []
        }
        
        # Setup camera
        if self.active_camera:
            self.active_camera.render(render_context)
        
        # Collect lights
        for light in self.lights:
            if light.visible:
                render_context['lights'].append(light)
        
        # Render scene
        self.root.traverse(lambda node: node.render(render_context))
        
        # Call render callbacks
        for callback in self.render_callbacks:
            self.root.traverse(lambda node: callback(node, render_context))
    
    def add_render_callback(self, callback: Callable[[SceneNode, Dict[str, Any]], None]):
        """Add a custom render callback."""
        self.render_callbacks.append(callback)
    
    def cull_frustum(self, frustum) -> List[SceneNode]:
        """Cull nodes outside the view frustum."""
        visible_nodes = []
        
        def cull_visitor(node: SceneNode):
            if node.bounding_box and frustum.is_box_visible(node.bounding_box.min_point, node.bounding_box.max_point):
                visible_nodes.append(node)
        
        self.root.traverse(cull_visitor)
        return visible_nodes


def demonstrate_scene_graph():
    """Demonstrate scene graph functionality."""
    print("=== Scene Graph Demonstration ===\n")
    
    # Create scene graph
    scene = SceneGraph()
    
    # Create camera
    camera = CameraNode("MainCamera")
    camera.transform.position = Vector3D(0, 0, 5)
    scene.add_node(camera)
    
    # Create light
    light = LightNode("MainLight", "directional")
    light.transform.position = Vector3D(5, 5, 5)
    scene.add_node(light)
    
    # Create geometry group
    geometry_group = GroupNode("GeometryGroup")
    geometry_group.transform.position = Vector3D(0, 0, 0)
    scene.add_node(geometry_group)
    
    # Create cube geometry
    cube = GeometryNode("Cube")
    cube_vertices = [
        Vector3D(-1, -1, -1), Vector3D(1, -1, -1), Vector3D(1, 1, -1), Vector3D(-1, 1, -1),
        Vector3D(-1, -1, 1), Vector3D(1, -1, 1), Vector3D(1, 1, 1), Vector3D(-1, 1, 1)
    ]
    cube_indices = [
        0, 1, 2, 2, 3, 0,  # Front
        1, 5, 6, 6, 2, 1,  # Right
        5, 4, 7, 7, 6, 5,  # Back
        4, 0, 3, 3, 7, 4,  # Left
        3, 2, 6, 6, 7, 3,  # Top
        4, 5, 1, 1, 0, 4   # Bottom
    ]
    cube.set_geometry(cube_vertices, cube_indices)
    geometry_group.add_child(cube)
    
    # Create sphere geometry
    sphere = GeometryNode("Sphere")
    sphere.transform.position = Vector3D(3, 0, 0)
    sphere_vertices = [Vector3D(0, 0, 0)]  # Simplified
    sphere_indices = []
    sphere.set_geometry(sphere_vertices, sphere_indices)
    geometry_group.add_child(sphere)
    
    print("1. Scene Graph Structure:")
    print(f"Root node: {scene.root.name}")
    print(f"Active camera: {scene.active_camera.name if scene.active_camera else 'None'}")
    print(f"Number of lights: {len(scene.lights)}")
    print(f"Total nodes: {len(scene.root.get_all_children()) + 1}")
    print()
    
    # Update scene
    scene.update()
    
    print("2. Bounding Boxes:")
    for node in scene.root.get_all_children():
        if node.bounding_box:
            center = node.bounding_box.get_center()
            size = node.bounding_box.get_size()
            print(f"{node.name}: center={center}, size={size}")
    print()
    
    # Render scene
    print("3. Scene Rendering:")
    scene.render()
    print()
    
    # Find nodes
    print("4. Node Search:")
    found_cube = scene.find_node("Cube")
    found_sphere = scene.find_node("Sphere")
    print(f"Found cube: {found_cube is not None}")
    print(f"Found sphere: {found_sphere is not None}")
    
    # Test transformations
    print("\n5. Transformations:")
    world_transform = cube.get_world_transform()
    print(f"Cube world position: {world_transform.position}")
    
    # Move geometry group
    geometry_group.transform.position = Vector3D(0, 2, 0)
    world_transform = cube.get_world_transform()
    print(f"Cube world position after moving group: {world_transform.position}")


if __name__ == "__main__":
    demonstrate_scene_graph()
