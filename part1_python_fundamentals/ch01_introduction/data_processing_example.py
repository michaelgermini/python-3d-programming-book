#!/usr/bin/env python3
"""
Chapter 1: Introduction to Python
Data Processing Example

This example demonstrates how Python can efficiently process large sets of 3D object data,
showing practical applications in 3D graphics and game development.
"""

import time
import random

# Simulating 3D object data
class Object3D:
    """Represents a 3D object with various properties"""
    
    def __init__(self, obj_id, obj_type, position, rotation, scale, material):
        self.id = obj_id
        self.type = obj_type
        self.position = position  # [x, y, z]
        self.rotation = rotation  # [rx, ry, rz] in degrees
        self.scale = scale        # [sx, sy, sz]
        self.material = material
        self.visible = True
        self.selected = False
        
    def __str__(self):
        return f"Object3D(id={self.id}, type={self.type}, pos={self.position})"
    
    def get_bounding_box(self):
        """Calculate bounding box for the object"""
        # Simplified bounding box calculation
        size = 1.0  # Assume base size
        half_size = size * 0.5
        return {
            'min': [self.position[0] - half_size, 
                   self.position[1] - half_size, 
                   self.position[2] - half_size],
            'max': [self.position[0] + half_size, 
                   self.position[1] + half_size, 
                   self.position[2] + half_size]
        }

def generate_sample_data(num_objects=1000):
    """Generate sample 3D object data for processing"""
    print(f"Generating {num_objects} sample 3D objects...")
    
    object_types = ["cube", "sphere", "cylinder", "pyramid", "cone"]
    materials = ["metal", "wood", "plastic", "glass", "stone"]
    
    objects = []
    for i in range(num_objects):
        obj = Object3D(
            obj_id=i + 1,
            obj_type=random.choice(object_types),
            position=[
                random.uniform(-100, 100),  # x
                random.uniform(-100, 100),  # y
                random.uniform(-100, 100)   # z
            ],
            rotation=[
                random.uniform(0, 360),     # rx
                random.uniform(0, 360),     # ry
                random.uniform(0, 360)      # rz
            ],
            scale=[
                random.uniform(0.1, 5.0),   # sx
                random.uniform(0.1, 5.0),   # sy
                random.uniform(0.1, 5.0)    # sz
            ],
            material=random.choice(materials)
        )
        objects.append(obj)
    
    print(f"Generated {len(objects)} objects successfully!")
    return objects

def demonstrate_data_processing(objects):
    """Demonstrate various data processing techniques"""
    print("\n=== Data Processing Demonstrations ===\n")
    
    # 1. Basic statistics
    print("1. Basic Statistics:")
    print(f"   Total objects: {len(objects)}")
    
    # Count objects by type
    type_counts = {}
    for obj in objects:
        type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
    
    print("   Objects by type:")
    for obj_type, count in type_counts.items():
        print(f"     {obj_type}: {count}")
    
    # 2. Filtering objects
    print("\n2. Filtering Objects:")
    
    # Find all metal objects
    metal_objects = [obj for obj in objects if obj.material == "metal"]
    print(f"   Metal objects: {len(metal_objects)}")
    
    # Find large objects (scale > 3.0)
    large_objects = [obj for obj in objects if max(obj.scale) > 3.0]
    print(f"   Large objects (scale > 3.0): {len(large_objects)}")
    
    # 3. Spatial queries
    print("\n3. Spatial Queries:")
    
    # Find objects in a specific region
    center_region_objects = [
        obj for obj in objects 
        if abs(obj.position[0]) < 20 and abs(obj.position[1]) < 20 and abs(obj.position[2]) < 20
    ]
    print(f"   Objects in center region (±20 units): {len(center_region_objects)}")
    
    # 4. Data transformation
    print("\n4. Data Transformation:")
    
    # Create a summary of object positions
    positions_summary = {
        'x_range': (min(obj.position[0] for obj in objects), 
                   max(obj.position[0] for obj in objects)),
        'y_range': (min(obj.position[1] for obj in objects), 
                   max(obj.position[1] for obj in objects)),
        'z_range': (min(obj.position[2] for obj in objects), 
                   max(obj.position[2] for obj in objects))
    }
    print(f"   Position ranges: {positions_summary}")
    
    # 5. Performance measurement
    print("\n5. Performance Measurement:")
    
    # Time the filtering operation
    start_time = time.time()
    filtered_objects = [obj for obj in objects if obj.type == "cube"]
    end_time = time.time()
    
    print(f"   Filtered {len(filtered_objects)} cubes in {end_time - start_time:.4f} seconds")
    
    # 6. Data grouping
    print("\n6. Data Grouping:")
    
    # Group objects by material
    material_groups = {}
    for obj in objects:
        if obj.material not in material_groups:
            material_groups[obj.material] = []
        material_groups[obj.material].append(obj)
    
    print("   Objects grouped by material:")
    for material, obj_list in material_groups.items():
        print(f"     {material}: {len(obj_list)} objects")
    
    # 7. Advanced filtering with multiple conditions
    print("\n7. Advanced Filtering:")
    
    # Find visible metal spheres in the positive x region
    special_objects = [
        obj for obj in objects
        if (obj.visible and 
            obj.material == "metal" and 
            obj.type == "sphere" and 
            obj.position[0] > 0)
    ]
    print(f"   Visible metal spheres in positive X region: {len(special_objects)}")
    
    return {
        'total_objects': len(objects),
        'type_counts': type_counts,
        'material_groups': material_groups,
        'center_region_count': len(center_region_objects),
        'filtered_cubes': len(filtered_objects)
    }

def demonstrate_memory_efficiency():
    """Demonstrate memory-efficient processing techniques"""
    print("\n=== Memory Efficiency Techniques ===\n")
    
    # 1. Generator expressions (memory efficient)
    print("1. Generator Expressions:")
    
    # Instead of creating a full list, use a generator
    large_objects_gen = (obj for obj in generate_sample_data(100) if max(obj.scale) > 3.0)
    large_count = sum(1 for _ in large_objects_gen)
    print(f"   Large objects found (using generator): {large_count}")
    
    # 2. Batch processing
    print("\n2. Batch Processing:")
    
    def process_batch(objects, batch_size=100):
        """Process objects in batches to manage memory"""
        total_processed = 0
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            # Process batch here
            total_processed += len(batch)
        return total_processed
    
    sample_objects = generate_sample_data(500)
    processed = process_batch(sample_objects, batch_size=50)
    print(f"   Processed {processed} objects in batches of 50")

def main():
    """Main function to run the data processing demonstration"""
    print("=== Python Data Processing for 3D Objects ===\n")
    
    # Generate sample data
    objects = generate_sample_data(500)  # Generate 500 objects for demonstration
    
    # Demonstrate processing techniques
    results = demonstrate_data_processing(objects)
    
    # Demonstrate memory efficiency
    demonstrate_memory_efficiency()
    
    # Summary
    print("\n=== Summary ===")
    print("Python excels at data processing because it provides:")
    print("✓ List comprehensions for efficient filtering")
    print("✓ Built-in data structures (lists, dictionaries, sets)")
    print("✓ High-level operations that are easy to read and write")
    print("✓ Memory-efficient generators for large datasets")
    print("✓ Fast iteration and processing capabilities")
    print("✓ Rich ecosystem of data processing libraries")
    
    print("\nThis makes Python perfect for:")
    print("- Processing large 3D scenes")
    print("- Filtering and querying object data")
    print("- Batch operations on multiple objects")
    print("- Real-time data analysis in games and simulations")
    print("- Automated content generation and management")

if __name__ == "__main__":
    main()
