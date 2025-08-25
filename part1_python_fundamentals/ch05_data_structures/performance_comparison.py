#!/usr/bin/env python3
"""
Chapter 5: Data Structures
Performance Comparison Example

Demonstrates performance characteristics of different data structures.
"""

import time
import sys
import random

def measure_time(func, *args, iterations=1000):
    """Measure execution time of a function"""
    start = time.time()
    for _ in range(iterations):
        func(*args)
    return (time.time() - start) / iterations

def list_performance_tests():
    """Test list performance"""
    print("=== List Performance ===\n")
    
    # Creation and access
    sizes = [100, 1000, 10000]
    for size in sizes:
        data = list(range(size))
        indices = random.sample(range(size), min(100, size))
        
        access_time = measure_time(lambda: [data[i] for i in indices], iterations=1000)
        memory = sys.getsizeof(data)
        
        print(f"Size {size:5d}: Access={access_time*1000000:6.2f}μs, Memory={memory:6d} bytes")
    
    # Operations
    base = list(range(1000))
    items = list(range(100))
    
    append_time = measure_time(lambda: base + items, iterations=100)
    print(f"Append: {append_time*1000:6.2f}ms")

def set_performance_tests():
    """Test set performance"""
    print("\n=== Set Performance ===\n")
    
    # Membership testing
    sizes = [100, 1000, 10000]
    for size in sizes:
        list_data = list(range(size))
        set_data = set(range(size))
        test_values = list(range(0, size * 2, 2))
        
        list_time = measure_time(lambda: [v in list_data for v in test_values], iterations=10)
        set_time = measure_time(lambda: [v in set_data for v in test_values], iterations=10)
        
        print(f"Size {size:5d}: List={list_time*1000:6.2f}ms, Set={set_time*1000:6.2f}ms, Speedup={list_time/set_time:6.2f}x")

def dict_performance_tests():
    """Test dictionary performance"""
    print("\n=== Dictionary Performance ===\n")
    
    # Lookup performance
    sizes = [100, 1000, 10000]
    for size in sizes:
        dict_data = {f"key_{i}": f"value_{i}" for i in range(size)}
        list_data = [f"value_{i}" for i in range(size)]
        
        test_keys = [f"key_{i}" for i in random.sample(range(size), min(100, size))]
        test_indices = random.sample(range(size), min(100, size))
        
        dict_time = measure_time(lambda: [dict_data.get(k, None) for k in test_keys], iterations=100)
        list_time = measure_time(lambda: [list_data[i] if i < len(list_data) else None for i in test_indices], iterations=100)
        
        print(f"Size {size:5d}: Dict={dict_time*1000:6.2f}ms, List={list_time*1000:6.2f}ms, Speedup={list_time/dict_time:6.2f}x")

def real_world_tests():
    """Test real-world 3D graphics scenarios"""
    print("\n=== Real-World 3D Graphics Tests ===\n")
    
    # Object visibility culling
    num_objects = 10000
    objects = [
        {
            'id': f'obj_{i}',
            'position': [random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)]
        }
        for i in range(num_objects)
    ]
    
    camera_pos = [0, 0, 0]
    max_dist = 50
    
    def cull_objects():
        visible = []
        for obj in objects:
            dist = ((obj['position'][0] - camera_pos[0])**2 + 
                   (obj['position'][1] - camera_pos[1])**2 + 
                   (obj['position'][2] - camera_pos[2])**2)**0.5
            if dist <= max_dist:
                visible.append(obj)
        return visible
    
    cull_time = measure_time(cull_objects, iterations=10)
    print(f"Visibility culling: {cull_time*1000:6.2f}ms")
    
    # Material lookup
    materials = {f'material_{i}': {'albedo': [0.8, 0.8, 0.8]} for i in range(1000)}
    test_material = f'material_{random.randint(0, 999)}'
    
    lookup_time = measure_time(lambda: materials.get(test_material, {}).get('albedo'), iterations=1000)
    print(f"Material lookup: {lookup_time*1000000:6.2f}μs")

def memory_comparison():
    """Compare memory usage"""
    print("\n=== Memory Usage Comparison ===\n")
    
    size = 10000
    data = list(range(size))
    
    list_mem = sys.getsizeof(data)
    tuple_mem = sys.getsizeof(tuple(data))
    set_mem = sys.getsizeof(set(data))
    dict_mem = sys.getsizeof({i: i for i in data})
    
    print(f"Data size: {size} elements")
    print(f"List:  {list_mem:8d} bytes ({list_mem/size:6.2f} bytes/element)")
    print(f"Tuple: {tuple_mem:8d} bytes ({tuple_mem/size:6.2f} bytes/element)")
    print(f"Set:   {set_mem:8d} bytes ({set_mem/size:6.2f} bytes/element)")
    print(f"Dict:  {dict_mem:8d} bytes ({dict_mem/size:6.2f} bytes/element)")

def main():
    """Main function"""
    print("=== Python Data Structure Performance ===\n")
    
    list_performance_tests()
    set_performance_tests()
    dict_performance_tests()
    real_world_tests()
    memory_comparison()
    
    print("\n=== Performance Guidelines ===")
    print("✓ Use lists for ordered, dynamic collections")
    print("✓ Use tuples for immutable, fixed data")
    print("✓ Use sets for unique elements and fast membership testing")
    print("✓ Use dictionaries for key-value mappings and fast lookups")
    print("✓ Consider memory usage for large datasets")

if __name__ == "__main__":
    main()
