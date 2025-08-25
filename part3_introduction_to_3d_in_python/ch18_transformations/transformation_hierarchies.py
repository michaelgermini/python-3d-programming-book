"""
Chapter 18: Transformations - Transformation Hierarchies
=======================================================

This module demonstrates transformation hierarchies and parent-child relationships.

Key Concepts:
- Parent-child transformation relationships
- Local and world space transformations
- Transformation inheritance
"""

import math
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from vector_operations import Vector3D
from matrix_operations import Matrix4x4
from quaternions import Quaternion
from transformation_matrices import Transform


@dataclass
class TransformNode:
    """A node in a transformation hierarchy."""
    name: str
    transform: Transform
    parent: Optional['TransformNode'] = None
    children: List['TransformNode'] = field(default_factory=list)
    
    def add_child(self, child: 'TransformNode'):
        """Add a child node."""
        if child.parent:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'TransformNode'):
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
    
    def get_world_position(self) -> Vector3D:
        """Get world position."""
        return self.get_world_transform().position
    
    def set_world_position(self, position: Vector3D):
        """Set world position."""
        if self.parent is None:
            self.transform.position = position
        else:
            parent_world = self.parent.get_world_transform()
            local_pos = position - parent_world.position
            self.transform.position = local_pos
    
    def get_child_by_name(self, name: str) -> Optional['TransformNode']:
        """Find child node by name."""
        for child in self.children:
            if child.name == name:
                return child
            result = child.get_child_by_name(name)
            if result:
                return result
        return None


class TransformHierarchy:
    """Manages a hierarchy of transformation nodes."""
    
    def __init__(self):
        self.root_nodes: List[TransformNode] = []
        self.node_map: Dict[str, TransformNode] = {}
    
    def add_root_node(self, node: TransformNode):
        """Add a root node to the hierarchy."""
        self.root_nodes.append(node)
        self._register_node(node)
    
    def create_node(self, name: str, transform: Transform = None) -> TransformNode:
        """Create a new node."""
        if transform is None:
            transform = Transform()
        
        node = TransformNode(name, transform)
        self._register_node(node)
        return node
    
    def _register_node(self, node: TransformNode):
        """Register a node in the node map."""
        self.node_map[node.name] = node
        for child in node.children:
            self._register_node(child)
    
    def get_node(self, name: str) -> Optional[TransformNode]:
        """Get node by name."""
        return self.node_map.get(name)


def demonstrate_transformation_hierarchies():
    """Demonstrate transformation hierarchies."""
    print("=== Transformation Hierarchies Demonstration ===\n")
    
    # Create hierarchy
    hierarchy = TransformHierarchy()
    
    # Create root node (character)
    character = hierarchy.create_node("Character", Transform(
        position=Vector3D(0, 0, 0)
    ))
    hierarchy.add_root_node(character)
    
    # Create body part nodes
    torso = hierarchy.create_node("Torso", Transform(
        position=Vector3D(0, 1, 0)
    ))
    character.add_child(torso)
    
    head = hierarchy.create_node("Head", Transform(
        position=Vector3D(0, 0.5, 0)
    ))
    torso.add_child(head)
    
    print("1. Hierarchy Structure:")
    print(f"Character world position: {character.get_world_position()}")
    print(f"Torso world position: {torso.get_world_position()}")
    print(f"Head world position: {head.get_world_position()}")
    
    # Move character
    character.set_world_position(Vector3D(5, 0, 3))
    print("\n2. After Moving Character:")
    print(f"Character world position: {character.get_world_position()}")
    print(f"Torso world position: {torso.get_world_position()}")
    print(f"Head world position: {head.get_world_position()}")
    
    # Find nodes
    found_head = character.get_child_by_name("Head")
    print(f"\n3. Node Search:")
    print(f"Found head node: {found_head is not None}")


if __name__ == "__main__":
    demonstrate_transformation_hierarchies()
