"""
Chapter 19: Scene Graphs and Object Hierarchies - Spatial Organization
====================================================================

This module demonstrates spatial organization and culling techniques for scene graphs.

Key Concepts:
- Spatial partitioning and octrees
- Frustum culling and visibility testing
- Level-of-detail systems
- Spatial queries and collision detection
"""

import math
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from scene_graph import SceneNode, BoundingBox


@dataclass
class OctreeNode:
    """Node in an octree spatial partitioning structure."""
    bounds: BoundingBox
    children: List[Optional['OctreeNode']] = None
    objects: List[SceneNode] = None
    max_objects: int = 8
    max_depth: int = 8
    
    def __post_init__(self):
        """Initialize octree node."""
        if self.children is None:
            self.children = [None] * 8
        if self.objects is None:
            self.objects = []
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return all(child is None for child in self.children)
    
    def subdivide(self):
        """Subdivide this node into 8 children."""
        center = self.bounds.get_center()
        size = self.bounds.get_size() * 0.5
        
        # Create 8 octants
        octants = [
            BoundingBox(
                Vector3D(center.x - size.x, center.y - size.y, center.z - size.z),
                Vector3D(center.x, center.y, center.z)
            ),
            BoundingBox(
                Vector3D(center.x, center.y - size.y, center.z - size.z),
                Vector3D(center.x + size.x, center.y, center.z)
            ),
            BoundingBox(
                Vector3D(center.x - size.x, center.y, center.z - size.z),
                Vector3D(center.x, center.y + size.y, center.z)
            ),
            BoundingBox(
                Vector3D(center.x, center.y, center.z - size.z),
                Vector3D(center.x + size.x, center.y + size.y, center.z)
            ),
            BoundingBox(
                Vector3D(center.x - size.x, center.y - size.y, center.z),
                Vector3D(center.x, center.y, center.z + size.z)
            ),
            BoundingBox(
                Vector3D(center.x, center.y - size.y, center.z),
                Vector3D(center.x + size.x, center.y, center.z + size.z)
            ),
            BoundingBox(
                Vector3D(center.x - size.x, center.y, center.z),
                Vector3D(center.x, center.y + size.y, center.z + size.z)
            ),
            BoundingBox(
                Vector3D(center.x, center.y, center.z),
                Vector3D(center.x + size.x, center.y + size.y, center.z + size.z)
            )
        ]
        
        for i, octant in enumerate(octants):
            self.children[i] = OctreeNode(octant, max_objects=self.max_objects, max_depth=self.max_depth - 1)


class Octree:
    """Octree spatial partitioning structure."""
    
    def __init__(self, bounds: BoundingBox, max_objects: int = 8, max_depth: int = 8):
        self.root = OctreeNode(bounds, max_objects=max_objects, max_depth=max_depth)
        self.max_objects = max_objects
        self.max_depth = max_depth
    
    def insert(self, node: SceneNode, octree_node: Optional[OctreeNode] = None, depth: int = 0):
        """Insert a scene node into the octree."""
        if octree_node is None:
            octree_node = self.root
        
        if not node.bounding_box or not octree_node.bounds.intersects_box(node.bounding_box):
            return False
        
        # If this is a leaf node and has space, add the object
        if octree_node.is_leaf() and len(octree_node.objects) < self.max_objects:
            octree_node.objects.append(node)
            return True
        
        # If we need to subdivide
        if octree_node.is_leaf() and depth < self.max_depth:
            octree_node.subdivide()
            
            # Redistribute existing objects
            old_objects = octree_node.objects.copy()
            octree_node.objects.clear()
            
            for obj in old_objects:
                self.insert(obj, octree_node, depth)
        
        # Insert into appropriate child
        if not octree_node.is_leaf():
            for child in octree_node.children:
                if child and self.insert(node, child, depth + 1):
                    return True
        
        # If we couldn't insert into children, add to this node
        if len(octree_node.objects) < self.max_objects:
            octree_node.objects.append(node)
            return True
        
        return False
    
    def query(self, bounds: BoundingBox, octree_node: Optional[OctreeNode] = None) -> List[SceneNode]:
        """Query objects within a bounding box."""
        if octree_node is None:
            octree_node = self.root
        
        result = []
        
        if not octree_node.bounds.intersects_box(bounds):
            return result
        
        # Add objects in this node
        for obj in octree_node.objects:
            if obj.bounding_box and obj.bounding_box.intersects_box(bounds):
                result.append(obj)
        
        # Query children
        if not octree_node.is_leaf():
            for child in octree_node.children:
                if child:
                    result.extend(self.query(bounds, child))
        
        return result
    
    def query_sphere(self, center: Vector3D, radius: float, octree_node: Optional[OctreeNode] = None) -> List[SceneNode]:
        """Query objects within a sphere."""
        if octree_node is None:
            octree_node = self.root
        
        result = []
        
        # Check if sphere intersects with this node's bounds
        node_center = octree_node.bounds.get_center()
        node_radius = octree_node.bounds.get_radius()
        
        if center.distance_to(node_center) > radius + node_radius:
            return result
        
        # Add objects in this node
        for obj in octree_node.objects:
            if obj.bounding_box:
                obj_center = obj.bounding_box.get_center()
                obj_radius = obj.bounding_box.get_radius()
                if center.distance_to(obj_center) <= radius + obj_radius:
                    result.append(obj)
        
        # Query children
        if not octree_node.is_leaf():
            for child in octree_node.children:
                if child:
                    result.extend(self.query_sphere(center, radius, child))
        
        return result
    
    def clear(self, octree_node: Optional[OctreeNode] = None):
        """Clear all objects from the octree."""
        if octree_node is None:
            octree_node = self.root
        
        octree_node.objects.clear()
        
        if not octree_node.is_leaf():
            for child in octree_node.children:
                if child:
                    self.clear(child)


class FrustumCuller:
    """Frustum culling for visibility testing."""
    
    def __init__(self):
        self.planes = []  # 6 frustum planes
    
    def set_frustum(self, view_matrix: Matrix4x4, projection_matrix: Matrix4x4):
        """Set frustum from view and projection matrices."""
        # Combine matrices
        vp_matrix = projection_matrix * view_matrix
        
        # Extract frustum planes
        self.planes = []
        
        # Left plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] + vp_matrix.data[0][0],
                    vp_matrix.data[1][3] + vp_matrix.data[1][0],
                    vp_matrix.data[2][3] + vp_matrix.data[2][0]),
            vp_matrix.data[3][3] + vp_matrix.data[3][0]
        ))
        
        # Right plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] - vp_matrix.data[0][0],
                    vp_matrix.data[1][3] - vp_matrix.data[1][0],
                    vp_matrix.data[2][3] - vp_matrix.data[2][0]),
            vp_matrix.data[3][3] - vp_matrix.data[3][0]
        ))
        
        # Bottom plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] + vp_matrix.data[0][1],
                    vp_matrix.data[1][3] + vp_matrix.data[1][1],
                    vp_matrix.data[2][3] + vp_matrix.data[2][1]),
            vp_matrix.data[3][3] + vp_matrix.data[3][1]
        ))
        
        # Top plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] - vp_matrix.data[0][1],
                    vp_matrix.data[1][3] - vp_matrix.data[1][1],
                    vp_matrix.data[2][3] - vp_matrix.data[2][1]),
            vp_matrix.data[3][3] - vp_matrix.data[3][1]
        ))
        
        # Near plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] + vp_matrix.data[0][2],
                    vp_matrix.data[1][3] + vp_matrix.data[1][2],
                    vp_matrix.data[2][3] + vp_matrix.data[2][2]),
            vp_matrix.data[3][3] + vp_matrix.data[3][2]
        ))
        
        # Far plane
        self.planes.append((
            Vector3D(vp_matrix.data[0][3] - vp_matrix.data[0][2],
                    vp_matrix.data[1][3] - vp_matrix.data[1][2],
                    vp_matrix.data[2][3] - vp_matrix.data[2][2]),
            vp_matrix.data[3][3] - vp_matrix.data[3][2]
        ))
        
        # Normalize planes
        for i in range(len(self.planes)):
            normal, distance = self.planes[i]
            length = normal.magnitude()
            if length > 0:
                self.planes[i] = (normal / length, distance / length)
    
    def is_point_visible(self, point: Vector3D) -> bool:
        """Check if a point is visible within the frustum."""
        for normal, distance in self.planes:
            if point.dot(normal) + distance < 0:
                return False
        return True
    
    def is_sphere_visible(self, center: Vector3D, radius: float) -> bool:
        """Check if a sphere is visible within the frustum."""
        for normal, distance in self.planes:
            if center.dot(normal) + distance < -radius:
                return False
        return True
    
    def is_box_visible(self, min_point: Vector3D, max_point: Vector3D) -> bool:
        """Check if an axis-aligned bounding box is visible."""
        for normal, distance in self.planes:
            # Find the most negative vertex
            test_point = Vector3D(
                min_point.x if normal.x >= 0 else max_point.x,
                min_point.y if normal.y >= 0 else max_point.y,
                min_point.z if normal.z >= 0 else max_point.z
            )
            
            if test_point.dot(normal) + distance < 0:
                return False
        return True


class LevelOfDetail:
    """Level-of-detail system for scene optimization."""
    
    def __init__(self):
        self.lod_levels: Dict[str, List[Tuple[float, SceneNode]]] = {}
    
    def add_lod_level(self, object_name: str, distance: float, node: SceneNode):
        """Add a LOD level for an object."""
        if object_name not in self.lod_levels:
            self.lod_levels[object_name] = []
        
        self.lod_levels[object_name].append((distance, node))
        # Sort by distance (closest first)
        self.lod_levels[object_name].sort(key=lambda x: x[0])
    
    def get_appropriate_lod(self, object_name: str, distance: float) -> Optional[SceneNode]:
        """Get the appropriate LOD level for a given distance."""
        if object_name not in self.lod_levels:
            return None
        
        levels = self.lod_levels[object_name]
        
        # Find the highest quality LOD within range
        for lod_distance, node in reversed(levels):
            if distance <= lod_distance:
                return node
        
        return None


class SpatialManager:
    """Manages spatial organization and culling for a scene."""
    
    def __init__(self, world_bounds: BoundingBox):
        self.octree = Octree(world_bounds)
        self.frustum_culler = FrustumCuller()
        self.lod_system = LevelOfDetail()
        self.objects: Dict[str, SceneNode] = {}
    
    def add_object(self, node: SceneNode):
        """Add an object to the spatial manager."""
        if node.bounding_box:
            self.octree.insert(node)
            self.objects[node.name] = node
    
    def remove_object(self, node: SceneNode):
        """Remove an object from the spatial manager."""
        if node.name in self.objects:
            del self.objects[node.name]
            # Rebuild octree
            self.octree.clear()
            for obj in self.objects.values():
                if obj.bounding_box:
                    self.octree.insert(obj)
    
    def update_frustum(self, view_matrix: Matrix4x4, projection_matrix: Matrix4x4):
        """Update frustum for culling."""
        self.frustum_culler.set_frustum(view_matrix, projection_matrix)
    
    def get_visible_objects(self, camera_position: Vector3D) -> List[SceneNode]:
        """Get objects visible from camera position."""
        visible_objects = []
        
        # Use frustum culling
        for node in self.objects.values():
            if node.bounding_box and self.frustum_culler.is_box_visible(
                node.bounding_box.min_point, node.bounding_box.max_point
            ):
                visible_objects.append(node)
        
        return visible_objects
    
    def query_nearby_objects(self, position: Vector3D, radius: float) -> List[SceneNode]:
        """Query objects within a radius of a position."""
        return self.octree.query_sphere(position, radius)
    
    def query_bounds(self, bounds: BoundingBox) -> List[SceneNode]:
        """Query objects within a bounding box."""
        return self.octree.query(bounds)
    
    def get_lod_object(self, object_name: str, distance: float) -> Optional[SceneNode]:
        """Get appropriate LOD level for an object."""
        return self.lod_system.get_appropriate_lod(object_name, distance)


def demonstrate_spatial_organization():
    """Demonstrate spatial organization and culling."""
    print("=== Spatial Organization Demonstration ===\n")
    
    # Create world bounds
    world_bounds = BoundingBox(
        Vector3D(-100, -100, -100),
        Vector3D(100, 100, 100)
    )
    
    # Create spatial manager
    spatial_manager = SpatialManager(world_bounds)
    
    # Create some test objects
    from scene_graph import GeometryNode
    
    objects = []
    for i in range(10):
        obj = GeometryNode(f"Object_{i}")
        obj.transform.position = Vector3D(
            (i - 5) * 10,
            (i % 3) * 5,
            (i // 3) * 8
        )
        
        # Create bounding box
        obj.bounding_box = BoundingBox(
            obj.transform.position + Vector3D(-1, -1, -1),
            obj.transform.position + Vector3D(1, 1, 1)
        )
        
        objects.append(obj)
        spatial_manager.add_object(obj)
    
    print("1. Spatial Organization:")
    print(f"Added {len(objects)} objects to spatial manager")
    print(f"World bounds: {world_bounds.min_point} to {world_bounds.max_point}")
    print()
    
    # Test spatial queries
    print("2. Spatial Queries:")
    query_center = Vector3D(0, 0, 0)
    query_radius = 20.0
    
    nearby_objects = spatial_manager.query_nearby_objects(query_center, query_radius)
    print(f"Objects within {query_radius} units of {query_center}: {len(nearby_objects)}")
    for obj in nearby_objects:
        print(f"  - {obj.name} at {obj.transform.position}")
    print()
    
    # Test bounding box query
    query_bounds = BoundingBox(
        Vector3D(-15, -5, -10),
        Vector3D(15, 5, 10)
    )
    
    objects_in_bounds = spatial_manager.query_bounds(query_bounds)
    print(f"Objects in bounds {query_bounds.min_point} to {query_bounds.max_point}: {len(objects_in_bounds)}")
    for obj in objects_in_bounds:
        print(f"  - {obj.name} at {obj.transform.position}")
    print()
    
    # Test frustum culling
    print("3. Frustum Culling:")
    from matrix_operations import Matrix4x4
    
    # Create simple view and projection matrices
    view_matrix = Matrix4x4.look_at(
        Vector3D(0, 0, 50),
        Vector3D(0, 0, 0),
        Vector3D(0, 1, 0)
    )
    projection_matrix = Matrix4x4.perspective(math.pi/4, 16/9, 0.1, 1000)
    
    spatial_manager.update_frustum(view_matrix, projection_matrix)
    
    camera_position = Vector3D(0, 0, 50)
    visible_objects = spatial_manager.get_visible_objects(camera_position)
    print(f"Visible objects from camera at {camera_position}: {len(visible_objects)}")
    for obj in visible_objects:
        print(f"  - {obj.name} at {obj.transform.position}")
    print()
    
    # Test LOD system
    print("4. Level of Detail:")
    lod_object = GeometryNode("LOD_Test")
    lod_object.transform.position = Vector3D(0, 0, 0)
    
    # Add LOD levels
    spatial_manager.lod_system.add_lod_level("LOD_Test", 10.0, lod_object)  # High detail
    spatial_manager.lod_system.add_lod_level("LOD_Test", 50.0, lod_object)  # Medium detail
    spatial_manager.lod_system.add_lod_level("LOD_Test", 100.0, lod_object)  # Low detail
    
    # Test different distances
    test_distances = [5.0, 25.0, 75.0, 150.0]
    for distance in test_distances:
        lod_node = spatial_manager.get_lod_object("LOD_Test", distance)
        print(f"Distance {distance}: LOD node = {lod_node.name if lod_node else 'None'}")


if __name__ == "__main__":
    demonstrate_spatial_organization()
