#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Dictionaries for Scene Data Example

This example demonstrates how to use Python dictionaries for managing scene data,
object properties, and associative data in 3D graphics applications.
"""

import math

def demonstrate_basic_dictionaries():
    """Demonstrate basic dictionary operations with 3D data"""
    print("=== Basic Dictionary Operations ===\n")
    
    # 1. Creating dictionaries
    print("1. Creating Dictionaries:")
    
    # 3D object properties
    cube_properties = {
        "name": "cube1",
        "position": [0, 0, 0],
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1],
        "visible": True,
        "material": "metal"
    }
    
    print(f"   Cube properties: {cube_properties}")
    print(f"   Number of properties: {len(cube_properties)}")
    print(f"   Cube name: {cube_properties['name']}")
    print(f"   Cube position: {cube_properties['position']}")
    
    # 2. Accessing and modifying values
    print("\n2. Accessing and Modifying Values:")
    
    # Access with get() (safe access)
    material = cube_properties.get("material", "default")
    color = cube_properties.get("color", [255, 255, 255])  # Default white
    print(f"   Material: {material}")
    print(f"   Color: {color}")
    
    # Modify values
    cube_properties["position"] = [1, 2, 3]
    cube_properties["scale"] = [2, 2, 2]
    print(f"   Updated position: {cube_properties['position']}")
    print(f"   Updated scale: {cube_properties['scale']}")
    
    # Add new properties
    cube_properties["color"] = [255, 0, 0]  # Red
    cube_properties["opacity"] = 1.0
    print(f"   Added color: {cube_properties['color']}")
    print(f"   Added opacity: {cube_properties['opacity']}")
    
    # 3. Dictionary methods
    print("\n3. Dictionary Methods:")
    
    # Get all keys
    keys = list(cube_properties.keys())
    print(f"   Keys: {keys}")
    
    # Get all values
    values = list(cube_properties.values())
    print(f"   Values: {values}")
    
    # Get all items
    items = list(cube_properties.items())
    print(f"   Items: {items}")
    
    # Check if key exists
    print(f"   Has 'name' key? {'name' in cube_properties}")
    print(f"   Has 'velocity' key? {'velocity' in cube_properties}")

def demonstrate_scene_management():
    """Demonstrate scene management with dictionaries"""
    print("\n=== Scene Management ===\n")
    
    # 1. Scene object registry
    print("1. Scene Object Registry:")
    
    # Dictionary of scene objects
    scene_objects = {
        "cube1": {
            "type": "cube",
            "position": [0, 0, 0],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1],
            "visible": True,
            "material": "metal"
        },
        "sphere1": {
            "type": "sphere",
            "position": [2, 0, 0],
            "rotation": [0, 0, 0],
            "scale": [0.5, 0.5, 0.5],
            "visible": True,
            "material": "plastic"
        },
        "light1": {
            "type": "light",
            "position": [5, 5, 5],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1],
            "visible": True,
            "intensity": 1.0,
            "color": [255, 255, 255]
        },
        "camera1": {
            "type": "camera",
            "position": [0, 0, 10],
            "rotation": [0, 0, 0],
            "scale": [1, 1, 1],
            "visible": False,
            "fov": 60,
            "near": 0.1,
            "far": 1000
        }
    }
    
    print(f"   Scene objects: {len(scene_objects)}")
    for obj_name, obj_data in scene_objects.items():
        print(f"     {obj_name}: {obj_data['type']} at {obj_data['position']}")
    
    # 2. Object type filtering
    print("\n2. Object Type Filtering:")
    
    # Filter objects by type
    geometry_objects = {name: data for name, data in scene_objects.items() 
                       if data["type"] in ["cube", "sphere", "cylinder"]}
    light_objects = {name: data for name, data in scene_objects.items() 
                    if data["type"] == "light"}
    camera_objects = {name: data for name, data in scene_objects.items() 
                     if data["type"] == "camera"}
    
    print(f"   Geometry objects: {len(geometry_objects)}")
    for name in geometry_objects:
        print(f"     {name}")
    
    print(f"   Light objects: {len(light_objects)}")
    for name in light_objects:
        print(f"     {name}")
    
    print(f"   Camera objects: {len(camera_objects)}")
    for name in camera_objects:
        print(f"     {name}")
    
    # 3. Object property queries
    print("\n3. Object Property Queries:")
    
    # Find objects by material
    metal_objects = {name: data for name, data in scene_objects.items() 
                    if data.get("material") == "metal"}
    print(f"   Metal objects: {list(metal_objects.keys())}")
    
    # Find visible objects
    visible_objects = {name: data for name, data in scene_objects.items() 
                      if data.get("visible", False)}
    print(f"   Visible objects: {list(visible_objects.keys())}")
    
    # Find objects within distance
    def distance_from_origin(obj_data):
        pos = obj_data["position"]
        return math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    
    nearby_objects = {name: data for name, data in scene_objects.items() 
                     if distance_from_origin(data) < 5}
    print(f"   Objects within 5 units of origin: {list(nearby_objects.keys())}")

def demonstrate_configuration_management():
    """Demonstrate configuration management with dictionaries"""
    print("\n=== Configuration Management ===\n")
    
    # 1. Graphics settings
    print("1. Graphics Settings:")
    
    graphics_config = {
        "resolution": {
            "width": 1920,
            "height": 1080,
            "fullscreen": False
        },
        "quality": {
            "antialiasing": True,
            "shadows": True,
            "reflections": False,
            "texture_quality": "high",
            "model_detail": "medium"
        },
        "performance": {
            "vsync": True,
            "max_fps": 60,
            "render_distance": 1000,
            "particle_limit": 1000
        },
        "camera": {
            "fov": 60,
            "near_clip": 0.1,
            "far_clip": 1000,
            "sensitivity": 1.0
        }
    }
    
    print(f"   Resolution: {graphics_config['resolution']['width']}x{graphics_config['resolution']['height']}")
    print(f"   Fullscreen: {graphics_config['resolution']['fullscreen']}")
    print(f"   Antialiasing: {graphics_config['quality']['antialiasing']}")
    print(f"   Max FPS: {graphics_config['performance']['max_fps']}")
    
    # 2. Material definitions
    print("\n2. Material Definitions:")
    
    materials = {
        "metal": {
            "diffuse_color": [0.8, 0.8, 0.8],
            "specular_color": [1.0, 1.0, 1.0],
            "shininess": 100,
            "reflectivity": 0.8,
            "texture": "metal_diffuse.png",
            "normal_map": "metal_normal.png"
        },
        "plastic": {
            "diffuse_color": [0.2, 0.2, 0.2],
            "specular_color": [0.5, 0.5, 0.5],
            "shininess": 50,
            "reflectivity": 0.1,
            "texture": "plastic_diffuse.png",
            "normal_map": None
        },
        "glass": {
            "diffuse_color": [0.9, 0.9, 0.9],
            "specular_color": [1.0, 1.0, 1.0],
            "shininess": 200,
            "reflectivity": 0.9,
            "transparency": 0.8,
            "refraction_index": 1.5,
            "texture": None,
            "normal_map": None
        }
    }
    
    print(f"   Available materials: {list(materials.keys())}")
    for mat_name, mat_props in materials.items():
        print(f"     {mat_name}: shininess={mat_props['shininess']}, reflectivity={mat_props['reflectivity']}")
    
    # 3. Animation data
    print("\n3. Animation Data:")
    
    animations = {
        "cube_rotation": {
            "target": "cube1",
            "type": "rotation",
            "duration": 2.0,
            "easing": "linear",
            "keyframes": {
                0.0: [0, 0, 0],
                1.0: [0, 360, 0],
                2.0: [0, 720, 0]
            }
        },
        "sphere_bounce": {
            "target": "sphere1",
            "type": "position",
            "duration": 1.5,
            "easing": "ease_in_out",
            "keyframes": {
                0.0: [2, 0, 0],
                0.5: [2, 2, 0],
                1.0: [2, 0, 0],
                1.5: [2, 2, 0]
            }
        }
    }
    
    print(f"   Available animations: {list(animations.keys())}")
    for anim_name, anim_data in animations.items():
        print(f"     {anim_name}: {anim_data['type']} for {anim_data['duration']}s")

def demonstrate_data_structures():
    """Demonstrate complex data structures with dictionaries"""
    print("\n=== Complex Data Structures ===\n")
    
    # 1. Hierarchical scene graph
    print("1. Hierarchical Scene Graph:")
    
    scene_graph = {
        "root": {
            "name": "root",
            "type": "group",
            "children": ["world", "ui"],
            "transform": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            }
        },
        "world": {
            "name": "world",
            "type": "group",
            "children": ["terrain", "buildings", "vehicles"],
            "transform": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            }
        },
        "terrain": {
            "name": "terrain",
            "type": "mesh",
            "children": [],
            "transform": {
                "position": [0, -1, 0],
                "rotation": [0, 0, 0],
                "scale": [100, 1, 100]
            },
            "geometry": {
                "vertices": [[-50, 0, -50], [50, 0, -50], [50, 0, 50], [-50, 0, 50]],
                "faces": [[0, 1, 2], [0, 2, 3]]
            },
            "material": "grass"
        },
        "buildings": {
            "name": "buildings",
            "type": "group",
            "children": ["building1", "building2"],
            "transform": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            }
        }
    }
    
    print(f"   Scene graph nodes: {len(scene_graph)}")
    for node_name, node_data in scene_graph.items():
        print(f"     {node_name}: {node_data['type']} with {len(node_data['children'])} children")
    
    # 2. Component-based object system
    print("\n2. Component-based Object System:")
    
    game_object = {
        "id": "player1",
        "name": "Player",
        "components": {
            "transform": {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            },
            "renderer": {
                "mesh": "player_mesh.obj",
                "material": "player_material",
                "visible": True,
                "cast_shadows": True,
                "receive_shadows": True
            },
            "physics": {
                "mass": 70.0,
                "collision_shape": "capsule",
                "collision_size": [0.5, 1.8],
                "friction": 0.8,
                "restitution": 0.2
            },
            "input": {
                "enabled": True,
                "sensitivity": 1.0,
                "key_bindings": {
                    "move_forward": "W",
                    "move_backward": "S",
                    "move_left": "A",
                    "move_right": "D",
                    "jump": "SPACE"
                }
            },
            "health": {
                "current": 100,
                "maximum": 100,
                "regeneration_rate": 1.0
            }
        }
    }
    
    print(f"   Game object: {game_object['name']}")
    print(f"   Components: {list(game_object['components'].keys())}")
    for comp_name, comp_data in game_object['components'].items():
        print(f"     {comp_name}: {type(comp_data).__name__}")
    
    # 3. Spatial partitioning
    print("\n3. Spatial Partitioning:")
    
    spatial_grid = {
        "cell_size": 10,
        "bounds": {
            "min": [-100, -100, -100],
            "max": [100, 100, 100]
        },
        "cells": {
            "0,0,0": ["object1", "object2"],
            "0,0,1": ["object3"],
            "1,0,0": ["object4", "object5", "object6"],
            "1,1,0": ["object7"],
            "0,1,1": ["object8", "object9"]
        }
    }
    
    print(f"   Grid cell size: {spatial_grid['cell_size']}")
    print(f"   Grid bounds: {spatial_grid['bounds']['min']} to {spatial_grid['bounds']['max']}")
    print(f"   Occupied cells: {len(spatial_grid['cells'])}")
    
    # Count total objects
    total_objects = sum(len(objects) for objects in spatial_grid['cells'].values())
    print(f"   Total objects in grid: {total_objects}")

def demonstrate_performance_optimization():
    """Demonstrate performance optimization with dictionaries"""
    print("\n=== Performance Optimization ===\n")
    
    import time
    
    # 1. Lookup performance
    print("1. Lookup Performance:")
    
    # Create large object registry
    large_registry = {}
    for i in range(10000):
        large_registry[f"object_{i}"] = {
            "id": i,
            "position": [i % 100, (i // 100) % 100, (i // 10000) % 100],
            "visible": i % 2 == 0
        }
    
    # Test lookup performance
    test_key = "object_5000"
    
    start_time = time.time()
    for _ in range(100000):
        obj = large_registry.get(test_key)
    dict_lookup_time = time.time() - start_time
    
    # Compare with list lookup
    large_list = list(large_registry.items())
    
    start_time = time.time()
    for _ in range(100000):
        obj = next((item for item in large_list if item[0] == test_key), None)
    list_lookup_time = time.time() - start_time
    
    print(f"   Dictionary lookup: {dict_lookup_time*1000:.2f} ms")
    print(f"   List lookup: {list_lookup_time*1000:.2f} ms")
    print(f"   Dictionary is {list_lookup_time/dict_lookup_time:.1f}x faster")
    
    # 2. Memory usage comparison
    print("\n2. Memory Usage Comparison:")
    
    import sys
    
    # Dictionary storage
    dict_size = sys.getsizeof(large_registry)
    
    # Equivalent list storage
    list_size = sys.getsizeof(large_list)
    
    print(f"   Dictionary size: {dict_size} bytes")
    print(f"   List size: {list_size} bytes")
    print(f"   Dictionary uses {dict_size/list_size:.1f}x more memory")
    
    # 3. Batch operations
    print("\n3. Batch Operations:")
    
    # Batch update positions
    start_time = time.time()
    for key, obj in large_registry.items():
        obj["position"][0] += 1
    batch_update_time = time.time() - start_time
    
    print(f"   Batch update time: {batch_update_time*1000:.2f} ms")
    
    # Batch filtering
    start_time = time.time()
    visible_objects = {key: obj for key, obj in large_registry.items() if obj["visible"]}
    filter_time = time.time() - start_time
    
    print(f"   Filter time: {filter_time*1000:.2f} ms")
    print(f"   Visible objects: {len(visible_objects)}")

def demonstrate_practical_applications():
    """Demonstrate practical applications of dictionaries"""
    print("\n=== Practical Applications ===\n")
    
    # 1. Asset management system
    print("1. Asset Management System:")
    
    asset_database = {
        "meshes": {
            "cube": {
                "file": "models/cube.obj",
                "size": 1024,
                "vertices": 8,
                "faces": 12,
                "loaded": False,
                "references": 5
            },
            "sphere": {
                "file": "models/sphere.obj",
                "size": 2048,
                "vertices": 256,
                "faces": 512,
                "loaded": False,
                "references": 3
            },
            "character": {
                "file": "models/character.obj",
                "size": 10240,
                "vertices": 2048,
                "faces": 4096,
                "loaded": True,
                "references": 1
            }
        },
        "textures": {
            "metal_diffuse": {
                "file": "textures/metal_diffuse.png",
                "size": 512,
                "width": 512,
                "height": 512,
                "loaded": True,
                "references": 2
            },
            "grass_diffuse": {
                "file": "textures/grass_diffuse.png",
                "size": 1024,
                "width": 1024,
                "height": 1024,
                "loaded": False,
                "references": 1
            }
        }
    }
    
    print(f"   Asset database: {len(asset_database['meshes'])} meshes, {len(asset_database['textures'])} textures")
    
    # Calculate total memory usage
    total_memory = 0
    for category in asset_database.values():
        for asset in category.values():
            if asset["loaded"]:
                total_memory += asset["size"]
    
    print(f"   Total loaded memory: {total_memory} bytes")
    
    # 2. Event system
    print("\n2. Event System:")
    
    event_handlers = {
        "object_created": [
            {"function": "log_event", "priority": 1},
            {"function": "update_statistics", "priority": 2},
            {"function": "notify_ui", "priority": 3}
        ],
        "collision_detected": [
            {"function": "play_sound", "priority": 1},
            {"function": "apply_damage", "priority": 2},
            {"function": "create_particles", "priority": 3}
        ],
        "level_completed": [
            {"function": "save_progress", "priority": 1},
            {"function": "show_ui", "priority": 2},
            {"function": "play_music", "priority": 3}
        ]
    }
    
    print(f"   Event types: {list(event_handlers.keys())}")
    for event_type, handlers in event_handlers.items():
        print(f"     {event_type}: {len(handlers)} handlers")
    
    # 3. State machine
    print("\n3. State Machine:")
    
    player_state_machine = {
        "current_state": "idle",
        "states": {
            "idle": {
                "transitions": {
                    "move": "walking",
                    "jump": "jumping",
                    "attack": "attacking"
                },
                "actions": ["play_idle_animation", "check_input"]
            },
            "walking": {
                "transitions": {
                    "stop": "idle",
                    "jump": "jumping",
                    "attack": "attacking",
                    "run": "running"
                },
                "actions": ["play_walk_animation", "update_position", "check_collisions"]
            },
            "jumping": {
                "transitions": {
                    "land": "idle",
                    "attack": "attacking"
                },
                "actions": ["play_jump_animation", "apply_gravity", "check_landing"]
            },
            "attacking": {
                "transitions": {
                    "finish": "idle"
                },
                "actions": ["play_attack_animation", "check_hit", "apply_damage"]
            }
        }
    }
    
    current_state = player_state_machine["current_state"]
    available_transitions = player_state_machine["states"][current_state]["transitions"]
    current_actions = player_state_machine["states"][current_state]["actions"]
    
    print(f"   Current state: {current_state}")
    print(f"   Available transitions: {list(available_transitions.keys())}")
    print(f"   Current actions: {current_actions}")

def main():
    """Main function to run all dictionary demonstrations"""
    print("=== Python Dictionaries for Scene Data ===\n")
    
    # Run all demonstrations
    demonstrate_basic_dictionaries()
    demonstrate_scene_management()
    demonstrate_configuration_management()
    demonstrate_data_structures()
    demonstrate_performance_optimization()
    demonstrate_practical_applications()
    
    print("\n=== Summary ===")
    print("This chapter covered dictionary data structures:")
    print("✓ Basic dictionary operations and methods")
    print("✓ Scene management and object registries")
    print("✓ Configuration management and settings")
    print("✓ Complex data structures and hierarchies")
    print("✓ Performance optimization and efficiency")
    print("✓ Practical applications in graphics")
    
    print("\nDictionaries are essential for:")
    print("- Managing object properties and attributes")
    print("- Creating efficient lookup tables")
    print("- Storing configuration and settings")
    print("- Building hierarchical data structures")
    print("- Implementing component-based systems")
    print("- Creating flexible and extensible data models")

if __name__ == "__main__":
    main()
