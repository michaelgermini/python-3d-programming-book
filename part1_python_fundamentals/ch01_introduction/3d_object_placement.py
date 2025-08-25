#!/usr/bin/env python3
"""
Chapter 1: Introduction to Python
3D Object Placement Example

This example demonstrates how Python can be used to automate 3D object placement
in a scene, showing practical applications of Python in 3D graphics.
"""

import math
import random

# Simulating a 3D scene with objects
class Scene3D:
    """A simple 3D scene class to demonstrate Python concepts"""
    
    def __init__(self, name="My 3D Scene"):
        self.name = name
        self.objects = []  # List to store 3D objects
        self.camera_position = [0, 0, 5]  # Camera position [x, y, z]
        
    def add_object(self, obj_type, position, size=1.0):
        """Add an object to the scene"""
        object_data = {
            'type': obj_type,
            'position': position,  # [x, y, z]
            'size': size,
            'id': len(self.objects) + 1
        }
        self.objects.append(object_data)
        print(f"Added {obj_type} at position {position}")
        
    def list_objects(self):
        """List all objects in the scene"""
        print(f"\n--- Objects in {self.name} ---")
        for obj in self.objects:
            print(f"ID {obj['id']}: {obj['type']} at {obj['position']} (size: {obj['size']})")
            
    def find_objects_by_type(self, obj_type):
        """Find all objects of a specific type"""
        found = [obj for obj in self.objects if obj['type'] == obj_type]
        print(f"\nFound {len(found)} {obj_type}(s):")
        for obj in found:
            print(f"  - {obj['type']} at {obj['position']}")
        return found

def calculate_distance(pos1, pos2):
    """Calculate distance between two 3D points"""
    return math.sqrt((pos1[0] - pos2[0])**2 + 
                    (pos1[1] - pos2[1])**2 + 
                    (pos1[2] - pos2[2])**2)

def create_random_scene():
    """Create a scene with randomly placed objects"""
    scene = Scene3D("Random 3D Scene")
    
    # Object types we can place
    object_types = ["cube", "sphere", "cylinder", "pyramid"]
    
    # Add random objects
    for i in range(5):
        obj_type = random.choice(object_types)
        position = [
            random.uniform(-10, 10),  # x coordinate
            random.uniform(-10, 10),  # y coordinate
            random.uniform(-10, 10)   # z coordinate
        ]
        size = random.uniform(0.5, 2.0)
        scene.add_object(obj_type, position, size)
    
    return scene

def demonstrate_python_concepts():
    """Demonstrate various Python concepts in 3D context"""
    print("=== Python & 3D Programming Demo ===\n")
    
    # 1. Creating a scene (object creation)
    scene = Scene3D("Python Learning Scene")
    
    # 2. Adding objects (method calls)
    scene.add_object("cube", [0, 0, 0], 1.0)
    scene.add_object("sphere", [2, 0, 0], 0.8)
    scene.add_object("cylinder", [0, 2, 0], 1.2)
    scene.add_object("cube", [-2, 0, 0], 0.6)
    
    # 3. Listing objects (data access)
    scene.list_objects()
    
    # 4. Finding specific objects (filtering)
    cubes = scene.find_objects_by_type("cube")
    
    # 5. Calculating distances (mathematical operations)
    print(f"\n--- Distance Calculations ---")
    if len(scene.objects) >= 2:
        obj1 = scene.objects[0]
        obj2 = scene.objects[1]
        distance = calculate_distance(obj1['position'], obj2['position'])
        print(f"Distance between {obj1['type']} and {obj2['type']}: {distance:.2f}")
    
    # 6. Random scene generation (random operations)
    print(f"\n--- Random Scene Generation ---")
    random_scene = create_random_scene()
    random_scene.list_objects()
    
    # 7. String formatting and f-strings
    print(f"\n--- Scene Statistics ---")
    total_objects = len(scene.objects)
    object_types = set(obj['type'] for obj in scene.objects)
    print(f"Total objects: {total_objects}")
    print(f"Unique object types: {', '.join(object_types)}")
    
    # 8. List comprehensions
    positions = [obj['position'] for obj in scene.objects]
    print(f"All object positions: {positions}")
    
    # 9. Dictionary operations
    scene_info = {
        'name': scene.name,
        'object_count': len(scene.objects),
        'camera_position': scene.camera_position
    }
    print(f"Scene info: {scene_info}")

if __name__ == "__main__":
    demonstrate_python_concepts()
    
    print("\n=== Key Python Concepts Demonstrated ===")
    print("✓ Object-oriented programming (Scene3D class)")
    print("✓ Functions and method calls")
    print("✓ Lists and dictionaries")
    print("✓ String formatting (f-strings)")
    print("✓ Mathematical operations")
    print("✓ Random number generation")
    print("✓ List comprehensions")
    print("✓ Data filtering and searching")
    print("✓ File organization and structure")
    
    print("\nThis example shows how Python can be used to:")
    print("- Manage 3D scenes and objects")
    print("- Automate object placement")
    print("- Calculate 3D distances and positions")
    print("- Process and filter object data")
    print("- Generate procedural content")
