#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Nested Structures Example

Demonstrates complex nested data structures for 3D graphics applications.
"""

import math

def demonstrate_nested_lists():
    """Demonstrate nested list structures"""
    print("=== Nested Lists ===\n")
    
    # 3D mesh data structure
    mesh_data = {
        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "normals": [[0, 0, -1], [0, 0, -1]]
    }
    
    print(f"Mesh vertices: {len(mesh_data['vertices'])}")
    print(f"Mesh faces: {len(mesh_data['faces'])}")
    
    # Animation keyframe data
    animation_data = {
        "position": [
            [0.0, [0, 0, 0]],
            [1.0, [2, 0, 0]],
            [2.0, [2, 2, 0]]
        ]
    }
    
    print(f"Position keyframes: {len(animation_data['position'])}")

def demonstrate_nested_dictionaries():
    """Demonstrate nested dictionary structures"""
    print("\n=== Nested Dictionaries ===\n")
    
    # Scene graph hierarchy
    scene_graph = {
        "root": {
            "name": "root",
            "type": "group",
            "children": {
                "world": {
                    "name": "world",
                    "type": "group",
                    "children": {
                        "cube": {
                            "name": "cube",
                            "type": "mesh",
                            "children": {}
                        }
                    }
                }
            }
        }
    }
    
    print(f"Scene graph depth: {calculate_graph_depth(scene_graph)}")
    
    # Component-based object system
    game_object = {
        "id": "player1",
        "components": {
            "transform": {
                "type": "transform",
                "enabled": True,
                "data": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}
            },
            "renderer": {
                "type": "renderer",
                "enabled": True,
                "data": {"mesh": "player_mesh.obj", "material": "player_material"}
            }
        }
    }
    
    print(f"Active components: {count_active_components(game_object)}")

def demonstrate_mixed_structures():
    """Demonstrate mixed data structures"""
    print("\n=== Mixed Data Structures ===\n")
    
    # Level data structure
    level_data = {
        "metadata": {"name": "Level 1", "version": "1.0"},
        "objects": [
            {
                "id": "player_spawn",
                "type": "spawn_point",
                "position": [0, 1, 0],
                "properties": {"team": "player", "respawn_time": 3.0}
            }
        ],
        "ai": {
            "patrol_routes": [
                {
                    "id": "route_1",
                    "waypoints": [
                        {"position": [0, 0, 0], "wait_time": 2.0},
                        {"position": [10, 0, 0], "wait_time": 1.0}
                    ]
                }
            ]
        }
    }
    
    print(f"Level: {level_data['metadata']['name']}")
    print(f"Objects: {len(level_data['objects'])}")

def demonstrate_data_traversal():
    """Demonstrate traversing complex nested structures"""
    print("\n=== Data Traversal ===\n")
    
    def traverse_scene_graph(node, depth=0):
        """Recursively traverse scene graph"""
        indent = "  " * depth
        print(f"{indent}- {node['name']} ({node['type']})")
        
        if 'children' in node:
            for child_name, child_node in node['children'].items():
                traverse_scene_graph(child_node, depth + 1)
    
    simple_scene = {
        "name": "root",
        "type": "group",
        "children": {
            "world": {
                "name": "world",
                "type": "group",
                "children": {
                    "cube": {"name": "cube", "type": "mesh", "children": {}}
                }
            }
        }
    }
    
    print("Scene graph structure:")
    traverse_scene_graph(simple_scene)

def demonstrate_utility_functions():
    """Demonstrate utility functions for nested structures"""
    print("\n=== Utility Functions ===\n")
    
    def deep_copy_dict(data):
        """Create a deep copy of a nested dictionary"""
        if isinstance(data, dict):
            return {key: deep_copy_dict(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [deep_copy_dict(item) for item in data]
        else:
            return data
    
    def flatten_dict(data, prefix=""):
        """Flatten a nested dictionary into key-value pairs"""
        flattened = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(flatten_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
    
    # Test deep copy
    original_data = {"position": [1, 2, 3], "children": {"child1": {"position": [4, 5, 6]}}}
    copied_data = deep_copy_dict(original_data)
    copied_data["position"][0] = 999
    
    print(f"Original position: {original_data['position']}")
    print(f"Copied position: {copied_data['position']}")
    
    # Test flatten
    nested_data = {
        "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0]},
        "renderer": {"mesh": "cube.obj", "material": "metal"}
    }
    
    flattened = flatten_dict(nested_data)
    print("Flattened structure:")
    for key, value in flattened.items():
        print(f"  {key}: {value}")

def calculate_graph_depth(node, current_depth=0):
    """Calculate the maximum depth of a scene graph"""
    max_depth = current_depth
    if 'children' in node:
        for child in node['children'].values():
            child_depth = calculate_graph_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
    return max_depth

def count_active_components(game_object):
    """Count active components in a game object"""
    return sum(1 for comp in game_object['components'].values() if comp['enabled'])

def main():
    """Main function to run all nested structure demonstrations"""
    print("=== Python Nested Structures for 3D Graphics ===\n")
    
    demonstrate_nested_lists()
    demonstrate_nested_dictionaries()
    demonstrate_mixed_structures()
    demonstrate_data_traversal()
    demonstrate_utility_functions()
    
    print("\n=== Summary ===")
    print("Nested structures are essential for:")
    print("- Representing complex hierarchical data")
    print("- Building component-based systems")
    print("- Managing scene graphs and object hierarchies")
    print("- Storing configuration and game data")

if __name__ == "__main__":
    main()
