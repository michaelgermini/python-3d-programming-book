"""
Chapter 28: Simple Ray Tracing and Path Tracing - Acceleration Structures
=======================================================================

This module demonstrates acceleration structures for efficient ray tracing.

Key Concepts:
- Bounding Volume Hierarchies (BVH) for spatial partitioning
- Octree data structures for scene organization
- Ray-object intersection optimization
- Spatial data structures and algorithms
- Performance optimization for ray tracing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
import random


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_point: np.ndarray
    max_point: np.ndarray
    
    def __post_init__(self):
        if self.min_point is None:
            self.min_point = np.array([-1.0, -1.0, -1.0])
        if self.max_point is None:
            self.max_point = np.array([1.0, 1.0, 1.0])
    
    def expand(self, point: np.ndarray):
        """Expand bounding box to include point."""
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def expand_box(self, other: 'BoundingBox'):
        """Expand bounding box to include other bounding box."""
        self.min_point = np.minimum(self.min_point, other.min_point)
        self.max_point = np.maximum(self.max_point, other.max_point)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside bounding box."""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return np.all(self.min_point <= other.max_point) and np.all(self.max_point >= other.min_point)
    
    def ray_intersects(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> bool:
        """Check if ray intersects with bounding box."""
        # Slab method for ray-box intersection
        t_min = (self.min_point - ray_origin) / ray_direction
        t_max = (self.max_point - ray_origin) / ray_direction
        
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        
        t_near = np.max(t1)
        t_far = np.min(t2)
        
        return t_far >= t_near and t_far >= 0
    
    def get_center(self) -> np.ndarray:
        """Get center point of bounding box."""
        return (self.min_point + self.max_point) / 2.0
    
    def get_size(self) -> np.ndarray:
        """Get size of bounding box."""
        return self.max_point - self.min_point
    
    def get_surface_area(self) -> float:
        """Get surface area of bounding box."""
        size = self.get_size()
        return 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])


class BVHNode:
    """Node in Bounding Volume Hierarchy."""
    
    def __init__(self):
        self.left: Optional['BVHNode'] = None
        self.right: Optional['BVHNode'] = None
        self.bounding_box: Optional[BoundingBox] = None
        self.object: Optional[Any] = None  # Leaf node contains object
        self.is_leaf: bool = False
    
    def is_leaf_node(self) -> bool:
        """Check if this is a leaf node."""
        return self.is_leaf


class BVH:
    """Bounding Volume Hierarchy for spatial partitioning."""
    
    def __init__(self, objects: List[Any]):
        self.objects = objects
        self.root: Optional[BVHNode] = None
        self.build_bvh()
    
    def build_bvh(self):
        """Build BVH from objects."""
        if not self.objects:
            return
        
        # Create leaf nodes for all objects
        nodes = []
        for obj in self.objects:
            node = BVHNode()
            node.bounding_box = self._compute_bounding_box(obj)
            node.object = obj
            node.is_leaf = True
            nodes.append(node)
        
        # Build tree recursively
        self.root = self._build_bvh_recursive(nodes)
    
    def _build_bvh_recursive(self, nodes: List[BVHNode]) -> BVHNode:
        """Recursively build BVH from nodes."""
        if len(nodes) == 1:
            return nodes[0]
        
        if len(nodes) == 2:
            # Create internal node with two children
            internal_node = BVHNode()
            internal_node.left = nodes[0]
            internal_node.right = nodes[1]
            internal_node.bounding_box = BoundingBox(
                np.minimum(nodes[0].bounding_box.min_point, nodes[1].bounding_box.min_point),
                np.maximum(nodes[0].bounding_box.max_point, nodes[1].bounding_box.max_point)
            )
            return internal_node
        
        # Find best split axis and position
        split_axis, split_pos = self._find_best_split(nodes)
        
        # Partition nodes
        left_nodes, right_nodes = self._partition_nodes(nodes, split_axis, split_pos)
        
        # Create internal node
        internal_node = BVHNode()
        internal_node.left = self._build_bvh_recursive(left_nodes)
        internal_node.right = self._build_bvh_recursive(right_nodes)
        
        # Compute bounding box
        internal_node.bounding_box = BoundingBox(
            np.minimum(internal_node.left.bounding_box.min_point, internal_node.right.bounding_box.min_point),
            np.maximum(internal_node.left.bounding_box.max_point, internal_node.right.bounding_box.max_point)
        )
        
        return internal_node
    
    def _find_best_split(self, nodes: List[BVHNode]) -> Tuple[int, float]:
        """Find best split axis and position using SAH (Surface Area Heuristic)."""
        best_axis = 0
        best_pos = 0.0
        best_cost = float('inf')
        
        # Try each axis
        for axis in range(3):
            # Sort nodes by center along this axis
            sorted_nodes = sorted(nodes, key=lambda n: n.bounding_box.get_center()[axis])
            
            # Try different split positions
            for i in range(1, len(sorted_nodes)):
                left_nodes = sorted_nodes[:i]
                right_nodes = sorted_nodes[i:]
                
                # Compute bounding boxes
                left_bbox = self._compute_combined_bbox(left_nodes)
                right_bbox = self._compute_combined_bbox(right_nodes)
                
                # Compute SAH cost
                left_area = left_bbox.get_surface_area()
                right_area = right_bbox.get_surface_area()
                total_area = left_area + right_area
                
                if total_area > 0:
                    cost = (left_area * len(left_nodes) + right_area * len(right_nodes)) / total_area
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_pos = (left_bbox.max_point[axis] + right_bbox.min_point[axis]) / 2.0
        
        return best_axis, best_pos
    
    def _partition_nodes(self, nodes: List[BVHNode], axis: int, pos: float) -> Tuple[List[BVHNode], List[BVHNode]]:
        """Partition nodes based on split axis and position."""
        left_nodes = []
        right_nodes = []
        
        for node in nodes:
            center = node.bounding_box.get_center()[axis]
            if center < pos:
                left_nodes.append(node)
            else:
                right_nodes.append(node)
        
        # Ensure both partitions have at least one node
        if not left_nodes:
            left_nodes = [nodes[0]]
            right_nodes = nodes[1:]
        elif not right_nodes:
            right_nodes = [nodes[-1]]
            left_nodes = nodes[:-1]
        
        return left_nodes, right_nodes
    
    def _compute_combined_bbox(self, nodes: List[BVHNode]) -> BoundingBox:
        """Compute combined bounding box for a list of nodes."""
        if not nodes:
            return BoundingBox(np.array([0, 0, 0]), np.array([0, 0, 0]))
        
        min_point = nodes[0].bounding_box.min_point.copy()
        max_point = nodes[0].bounding_box.max_point.copy()
        
        for node in nodes[1:]:
            min_point = np.minimum(min_point, node.bounding_box.min_point)
            max_point = np.maximum(max_point, node.bounding_box.max_point)
        
        return BoundingBox(min_point, max_point)
    
    def _compute_bounding_box(self, obj: Any) -> BoundingBox:
        """Compute bounding box for an object."""
        # This should be implemented based on the object type
        # For now, assume objects have a bounding_box property
        if hasattr(obj, 'bounding_box'):
            return obj.bounding_box
        else:
            # Fallback: create a unit bounding box
            return BoundingBox(np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5]))
    
    def ray_intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray, 
                     t_min: float, t_max: float) -> Optional[Any]:
        """Find closest object hit by ray using BVH."""
        if not self.root:
            return None
        
        return self._ray_intersect_recursive(self.root, ray_origin, ray_direction, t_min, t_max)
    
    def _ray_intersect_recursive(self, node: BVHNode, ray_origin: np.ndarray, 
                                ray_direction: np.ndarray, t_min: float, t_max: float) -> Optional[Any]:
        """Recursively traverse BVH for ray intersection."""
        if not node.bounding_box.ray_intersects(ray_origin, ray_direction):
            return None
        
        if node.is_leaf:
            # Test intersection with object
            if hasattr(node.object, 'hit'):
                hit_record = node.object.hit(ray_origin, ray_direction, t_min, t_max)
                return hit_record
            return None
        
        # Test children
        left_hit = self._ray_intersect_recursive(node.left, ray_origin, ray_direction, t_min, t_max)
        right_hit = self._ray_intersect_recursive(node.right, ray_origin, ray_direction, t_min, t_max)
        
        # Return closest hit
        if left_hit and right_hit:
            return left_hit if left_hit.t < right_hit.t else right_hit
        elif left_hit:
            return left_hit
        else:
            return right_hit


class OctreeNode:
    """Node in octree spatial data structure."""
    
    def __init__(self, center: np.ndarray, size: float):
        self.center = center
        self.size = size
        self.children: List[Optional['OctreeNode']] = [None] * 8
        self.objects: List[Any] = []
        self.is_leaf = True
        self.max_objects = 10  # Maximum objects per leaf node
    
    def get_child_index(self, point: np.ndarray) -> int:
        """Get child index for a point."""
        index = 0
        for i in range(3):
            if point[i] > self.center[i]:
                index |= (1 << i)
        return index
    
    def get_child_center(self, child_index: int) -> np.ndarray:
        """Get center of child node."""
        child_center = self.center.copy()
        child_size = self.size / 2.0
        
        for i in range(3):
            if child_index & (1 << i):
                child_center[i] += child_size / 2.0
            else:
                child_center[i] -= child_size / 2.0
        
        return child_center
    
    def subdivide(self):
        """Subdivide octree node into 8 children."""
        if not self.is_leaf:
            return
        
        child_size = self.size / 2.0
        
        # Create children
        for i in range(8):
            child_center = self.get_child_center(i)
            self.children[i] = OctreeNode(child_center, child_size)
        
        # Distribute objects to children
        for obj in self.objects:
            # Compute bounding box for object
            if hasattr(obj, 'bounding_box'):
                bbox = obj.bounding_box
                obj_center = bbox.get_center()
                child_index = self.get_child_index(obj_center)
                self.children[child_index].objects.append(obj)
        
        # Clear objects from this node
        self.objects.clear()
        self.is_leaf = False


class Octree:
    """Octree spatial data structure."""
    
    def __init__(self, center: np.ndarray, size: float, max_depth: int = 8):
        self.root = OctreeNode(center, size)
        self.max_depth = max_depth
    
    def insert(self, obj: Any, depth: int = 0):
        """Insert object into octree."""
        self._insert_recursive(self.root, obj, depth)
    
    def _insert_recursive(self, node: OctreeNode, obj: Any, depth: int):
        """Recursively insert object into octree."""
        if node.is_leaf:
            node.objects.append(obj)
            
            # Subdivide if necessary
            if len(node.objects) > node.max_objects and depth < self.max_depth:
                node.subdivide()
                # Redistribute objects
                for obj in node.objects:
                    if hasattr(obj, 'bounding_box'):
                        bbox = obj.bounding_box
                        obj_center = bbox.get_center()
                        child_index = node.get_child_index(obj_center)
                        node.children[child_index].objects.append(obj)
                node.objects.clear()
        else:
            # Find appropriate child
            if hasattr(obj, 'bounding_box'):
                bbox = obj.bounding_box
                obj_center = bbox.get_center()
                child_index = node.get_child_index(obj_center)
                self._insert_recursive(node.children[child_index], obj, depth + 1)
    
    def query_range(self, query_bbox: BoundingBox) -> List[Any]:
        """Query objects within bounding box."""
        result = []
        self._query_range_recursive(self.root, query_bbox, result)
        return result
    
    def _query_range_recursive(self, node: OctreeNode, query_bbox: BoundingBox, result: List[Any]):
        """Recursively query objects within range."""
        # Check if node bounding box intersects query
        node_bbox = BoundingBox(
            node.center - node.size / 2.0,
            node.center + node.size / 2.0
        )
        
        if not node_bbox.intersects(query_bbox):
            return
        
        if node.is_leaf:
            # Add objects that intersect with query
            for obj in node.objects:
                if hasattr(obj, 'bounding_box'):
                    if obj.bounding_box.intersects(query_bbox):
                        result.append(obj)
        else:
            # Recursively query children
            for child in node.children:
                if child:
                    self._query_range_recursive(child, query_bbox, result)
    
    def ray_query(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> List[Any]:
        """Query objects potentially hit by ray."""
        result = []
        self._ray_query_recursive(self.root, ray_origin, ray_direction, result)
        return result
    
    def _ray_query_recursive(self, node: OctreeNode, ray_origin: np.ndarray, 
                           ray_direction: np.ndarray, result: List[Any]):
        """Recursively query objects hit by ray."""
        # Check if ray intersects node bounding box
        node_bbox = BoundingBox(
            node.center - node.size / 2.0,
            node.center + node.size / 2.0
        )
        
        if not node_bbox.ray_intersects(ray_origin, ray_direction):
            return
        
        if node.is_leaf:
            # Add all objects in leaf node
            result.extend(node.objects)
        else:
            # Recursively query children
            for child in node.children:
                if child:
                    self._ray_query_recursive(child, ray_origin, ray_direction, result)


class SpatialHash:
    """Spatial hash for uniform grid acceleration."""
    
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[Any]] = {}
    
    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Get cell coordinates for a point."""
        return (
            int(point[0] // self.cell_size),
            int(point[1] // self.cell_size),
            int(point[2] // self.cell_size)
        )
    
    def insert(self, obj: Any):
        """Insert object into spatial hash."""
        if hasattr(obj, 'bounding_box'):
            bbox = obj.bounding_box
            
            # Get all cells that the object might occupy
            min_cell = self._get_cell_coords(bbox.min_point)
            max_cell = self._get_cell_coords(bbox.max_point)
            
            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    for z in range(min_cell[2], max_cell[2] + 1):
                        cell = (x, y, z)
                        if cell not in self.grid:
                            self.grid[cell] = []
                        self.grid[cell].append(obj)
    
    def query_range(self, query_bbox: BoundingBox) -> List[Any]:
        """Query objects within bounding box."""
        result = []
        seen_objects = set()
        
        min_cell = self._get_cell_coords(query_bbox.min_point)
        max_cell = self._get_cell_coords(query_bbox.max_point)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    if cell in self.grid:
                        for obj in self.grid[cell]:
                            if id(obj) not in seen_objects:
                                if hasattr(obj, 'bounding_box'):
                                    if obj.bounding_box.intersects(query_bbox):
                                        result.append(obj)
                                        seen_objects.add(id(obj))
        
        return result


def demonstrate_acceleration_structures():
    """Demonstrate acceleration structures functionality."""
    print("=== Simple Ray Tracing and Path Tracing - Acceleration Structures ===\n")

    # Test bounding box
    print("1. Testing bounding box...")
    
    bbox = BoundingBox(np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0]))
    test_point = np.array([1.0, 1.0, 1.0])
    
    print(f"   - Bounding box center: {bbox.get_center()}")
    print(f"   - Bounding box size: {bbox.get_size()}")
    print(f"   - Surface area: {bbox.get_surface_area():.2f}")
    print(f"   - Contains test point: {bbox.contains(test_point)}")

    # Test BVH
    print("\n2. Testing BVH...")
    
    # Create mock objects with bounding boxes
    class MockObject:
        def __init__(self, center, size):
            self.bounding_box = BoundingBox(center - size/2, center + size/2)
        
        def hit(self, origin, direction, t_min, t_max):
            return None  # Simplified for demo
    
    objects = [
        MockObject(np.array([0.0, 0.0, 0.0]), 1.0),
        MockObject(np.array([2.0, 0.0, 0.0]), 1.0),
        MockObject(np.array([0.0, 2.0, 0.0]), 1.0),
        MockObject(np.array([2.0, 2.0, 0.0]), 1.0)
    ]
    
    bvh = BVH(objects)
    print(f"   - Created BVH with {len(objects)} objects")
    print(f"   - Root bounding box: {bvh.root.bounding_box.get_center()}")

    # Test Octree
    print("\n3. Testing Octree...")
    
    octree = Octree(np.array([0.0, 0.0, 0.0]), 10.0)
    
    for obj in objects:
        octree.insert(obj)
    
    print(f"   - Created octree with {len(objects)} objects")
    
    # Test spatial query
    query_bbox = BoundingBox(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    query_result = octree.query_range(query_bbox)
    print(f"   - Objects in query range: {len(query_result)}")

    # Test Spatial Hash
    print("\n4. Testing Spatial Hash...")
    
    spatial_hash = SpatialHash(1.0)
    
    for obj in objects:
        spatial_hash.insert(obj)
    
    print(f"   - Created spatial hash with {len(objects)} objects")
    print(f"   - Number of occupied cells: {len(spatial_hash.grid)}")
    
    # Test spatial hash query
    hash_result = spatial_hash.query_range(query_bbox)
    print(f"   - Objects in hash query range: {len(hash_result)}")

    # Performance comparison
    print("\n5. Performance characteristics:")
    print("   - BVH: O(log n) average case, good for static scenes")
    print("   - Octree: O(log n) average case, good for dynamic scenes")
    print("   - Spatial Hash: O(1) average case, good for uniform distributions")

    print("\n6. Features demonstrated:")
    print("   - Bounding box operations and ray intersection")
    print("   - BVH construction and traversal")
    print("   - Octree spatial partitioning")
    print("   - Spatial hash for uniform grid")
    print("   - Range queries and ray queries")
    print("   - Spatial data structure optimization")


if __name__ == "__main__":
    demonstrate_acceleration_structures()
