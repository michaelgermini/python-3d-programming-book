#!/usr/bin/env python3
"""
Chapter 4: Functions
Recursion Example

This example demonstrates recursive functions and algorithms, focusing on
applications in 3D graphics programming.
"""

import math

def demonstrate_basic_recursion():
    """Demonstrate basic recursive function concepts"""
    print("=== Basic Recursion ===\n")
    
    # 1. Factorial function
    print("1. Factorial Function:")
    
    def factorial(n):
        """Calculate factorial using recursion"""
        if n <= 1:
            return 1
        else:
            return n * factorial(n - 1)
    
    # Test factorial
    for i in range(6):
        print(f"   {i}! = {factorial(i)}")
    
    # 2. Fibonacci sequence
    print("\n2. Fibonacci Sequence:")
    
    def fibonacci(n):
        """Calculate nth Fibonacci number using recursion"""
        if n <= 1:
            return n
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)
    
    # Test Fibonacci
    print("   First 10 Fibonacci numbers:")
    for i in range(10):
        print(f"     F({i}) = {fibonacci(i)}")
    
    # 3. Sum of list
    print("\n3. Sum of List:")
    
    def sum_list(numbers):
        """Calculate sum of list using recursion"""
        if not numbers:
            return 0
        else:
            return numbers[0] + sum_list(numbers[1:])
    
    # Test sum
    test_numbers = [1, 2, 3, 4, 5]
    print(f"   Sum of {test_numbers} = {sum_list(test_numbers)}")

def demonstrate_3d_recursion():
    """Demonstrate recursion in 3D graphics contexts"""
    print("\n=== 3D Graphics Recursion ===\n")
    
    # 1. Fractal tree generation
    print("1. Fractal Tree Generation:")
    
    def generate_tree_branches(start_point, direction, length, angle, depth, max_depth):
        """Generate fractal tree branches recursively"""
        if depth >= max_depth or length < 1:
            return []
        
        # Calculate end point of current branch
        end_point = [
            start_point[0] + direction[0] * length,
            start_point[1] + direction[1] * length,
            start_point[2] + direction[2] * length
        ]
        
        # Current branch
        branches = [{
            "start": start_point,
            "end": end_point,
            "length": length,
            "depth": depth
        }]
        
        # Calculate new directions for child branches
        # Rotate around Z-axis
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        left_direction = [
            direction[0] * cos_a - direction[1] * sin_a,
            direction[0] * sin_a + direction[1] * cos_a,
            direction[2]
        ]
        
        right_direction = [
            direction[0] * cos_a + direction[1] * sin_a,
            -direction[0] * sin_a + direction[1] * cos_a,
            direction[2]
        ]
        
        # Recursive calls for child branches
        new_length = length * 0.7
        branches.extend(generate_tree_branches(end_point, left_direction, new_length, angle, depth + 1, max_depth))
        branches.extend(generate_tree_branches(end_point, right_direction, new_length, angle, depth + 1, max_depth))
        
        return branches
    
    # Generate a simple tree
    tree_branches = generate_tree_branches(
        start_point=[0, 0, 0],
        direction=[0, 1, 0],  # Growing upward
        length=10,
        angle=math.pi / 6,  # 30 degrees
        depth=0,
        max_depth=3
    )
    
    print(f"   Generated {len(tree_branches)} branches")
    print("   Branch structure:")
    for i, branch in enumerate(tree_branches[:5]):  # Show first 5 branches
        print(f"     Branch {i}: depth {branch['depth']}, length {branch['length']:.1f}")
    
    # 2. Recursive subdivision of 3D objects
    print("\n2. 3D Object Subdivision:")
    
    def subdivide_cube(center, size, depth, max_depth):
        """Recursively subdivide a cube into smaller cubes"""
        if depth >= max_depth:
            return [{"center": center, "size": size, "depth": depth}]
        
        # Calculate new size
        new_size = size / 2
        
        # Generate 8 sub-cubes
        sub_cubes = []
        for x in [-new_size/2, new_size/2]:
            for y in [-new_size/2, new_size/2]:
                for z in [-new_size/2, new_size/2]:
                    new_center = [
                        center[0] + x,
                        center[1] + y,
                        center[2] + z
                    ]
                    sub_cubes.extend(subdivide_cube(new_center, new_size, depth + 1, max_depth))
        
        return sub_cubes
    
    # Subdivide a cube
    subdivided_cubes = subdivide_cube([0, 0, 0], 10, 0, 2)
    print(f"   Generated {len(subdivided_cubes)} sub-cubes")
    print("   Subdivision levels:")
    for depth in range(3):
        count = sum(1 for cube in subdivided_cubes if cube["depth"] == depth)
        print(f"     Depth {depth}: {count} cubes")
    
    # 3. Recursive mesh generation
    print("\n3. Recursive Mesh Generation:")
    
    def generate_sphere_mesh(center, radius, segments, depth, max_depth):
        """Generate sphere mesh using recursive subdivision"""
        if depth >= max_depth:
            return []
        
        # Generate base icosahedron vertices (simplified)
        vertices = []
        for i in range(segments):
            for j in range(segments):
                # Spherical coordinates
                phi = (i / segments) * 2 * math.pi
                theta = (j / segments) * math.pi
                
                x = center[0] + radius * math.sin(theta) * math.cos(phi)
                y = center[1] + radius * math.sin(theta) * math.sin(phi)
                z = center[2] + radius * math.cos(theta)
                
                vertices.append([x, y, z])
        
        # Recursive subdivision
        if depth < max_depth - 1:
            new_radius = radius * 0.8
            vertices.extend(generate_sphere_mesh(center, new_radius, segments, depth + 1, max_depth))
        
        return vertices
    
    # Generate sphere mesh
    sphere_vertices = generate_sphere_mesh([0, 0, 0], 5, 4, 0, 2)
    print(f"   Generated {len(sphere_vertices)} sphere vertices")
    print(f"   First few vertices: {sphere_vertices[:3]}")

def demonstrate_divide_and_conquer():
    """Demonstrate divide and conquer algorithms"""
    print("\n=== Divide and Conquer ===\n")
    
    # 1. Merge sort for 3D points
    print("1. Merge Sort for 3D Points:")
    
    def merge_sort_3d_points(points, axis=0):
        """Sort 3D points by specified axis using merge sort"""
        if len(points) <= 1:
            return points
        
        # Divide
        mid = len(points) // 2
        left = merge_sort_3d_points(points[:mid], axis)
        right = merge_sort_3d_points(points[mid:], axis)
        
        # Conquer (merge)
        return merge_3d_points(left, right, axis)
    
    def merge_3d_points(left, right, axis):
        """Merge two sorted lists of 3D points"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i][axis] <= right[j][axis]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    # Test merge sort
    test_points = [[3, 1, 4], [1, 5, 9], [2, 6, 5], [3, 5, 8], [9, 7, 9]]
    sorted_points = merge_sort_3d_points(test_points, axis=0)
    print(f"   Original points: {test_points}")
    print(f"   Sorted by X-axis: {sorted_points}")
    
    # 2. Closest pair of points
    print("\n2. Closest Pair of Points:")
    
    def closest_pair_2d(points):
        """Find closest pair of 2D points using divide and conquer"""
        if len(points) <= 3:
            return brute_force_closest_pair(points)
        
        # Sort by x-coordinate
        points_sorted_x = sorted(points, key=lambda p: p[0])
        
        # Divide
        mid = len(points_sorted_x) // 2
        left_points = points_sorted_x[:mid]
        right_points = points_sorted_x[mid:]
        
        # Conquer
        left_closest = closest_pair_2d(left_points)
        right_closest = closest_pair_2d(right_points)
        
        # Find minimum distance
        min_distance = min(left_closest[0], right_closest[0])
        closest_pair = left_closest[1] if left_closest[0] < right_closest[0] else right_closest[1]
        
        # Check for pairs across the divide
        mid_x = points_sorted_x[mid][0]
        strip_points = [p for p in points_sorted_x if abs(p[0] - mid_x) < min_distance]
        
        # Sort strip by y-coordinate
        strip_points.sort(key=lambda p: p[1])
        
        # Check pairs in strip
        for i in range(len(strip_points)):
            for j in range(i + 1, min(i + 7, len(strip_points))):
                distance = math.sqrt((strip_points[i][0] - strip_points[j][0])**2 + 
                                   (strip_points[i][1] - strip_points[j][1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (strip_points[i], strip_points[j])
        
        return min_distance, closest_pair
    
    def brute_force_closest_pair(points):
        """Brute force method for small sets"""
        min_distance = float('inf')
        closest_pair = None
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = math.sqrt((points[i][0] - points[j][0])**2 + 
                                   (points[i][1] - points[j][1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (points[i], points[j])
        
        return min_distance, closest_pair
    
    # Test closest pair
    test_points_2d = [[0, 0], [1, 1], [2, 2], [3, 3], [1, 0], [0, 1]]
    distance, pair = closest_pair_2d(test_points_2d)
    print(f"   Points: {test_points_2d}")
    print(f"   Closest pair: {pair} with distance {distance:.3f}")
    
    # 3. Recursive bounding box calculation
    print("\n3. Recursive Bounding Box:")
    
    def calculate_bounding_box_recursive(objects, depth=0, max_depth=3):
        """Calculate bounding box recursively for object hierarchy"""
        if not objects:
            return None
        
        if len(objects) == 1 or depth >= max_depth:
            # Base case: calculate bounding box for single object or group
            positions = [obj["position"] for obj in objects]
            min_coords = [min(pos[i] for pos in positions) for i in range(3)]
            max_coords = [max(pos[i] for pos in positions) for i in range(3)]
            return {"min": min_coords, "max": max_coords, "depth": depth}
        
        # Divide objects into groups
        mid = len(objects) // 2
        left_objects = objects[:mid]
        right_objects = objects[mid:]
        
        # Recursive calls
        left_bbox = calculate_bounding_box_recursive(left_objects, depth + 1, max_depth)
        right_bbox = calculate_bounding_box_recursive(right_objects, depth + 1, max_depth)
        
        # Combine bounding boxes
        if left_bbox and right_bbox:
            combined_min = [min(left_bbox["min"][i], right_bbox["min"][i]) for i in range(3)]
            combined_max = [max(left_bbox["max"][i], right_bbox["max"][i]) for i in range(3)]
            return {"min": combined_min, "max": combined_max, "depth": depth}
        elif left_bbox:
            return left_bbox
        else:
            return right_bbox
    
    # Test bounding box calculation
    test_objects = [
        {"position": [0, 0, 0]},
        {"position": [1, 1, 1]},
        {"position": [-1, -1, -1]},
        {"position": [2, 2, 2]},
        {"position": [-2, -2, -2]}
    ]
    
    bbox = calculate_bounding_box_recursive(test_objects)
    print(f"   Objects: {[obj['position'] for obj in test_objects]}")
    print(f"   Bounding box: {bbox}")

def demonstrate_recursive_algorithms():
    """Demonstrate specific recursive algorithms for 3D graphics"""
    print("\n=== Recursive Algorithms ===\n")
    
    # 1. Recursive pathfinding
    print("1. Recursive Pathfinding:")
    
    def find_path_recursive(grid, start, end, visited=None, path=None):
        """Find path in 2D grid using recursion"""
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        # Base cases
        if start == end:
            return path + [start]
        
        if start in visited:
            return None
        
        # Mark current position as visited
        visited.add(start)
        
        # Try all four directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dx, dy in directions:
            new_x, new_y = start[0] + dx, start[1] + dy
            
            # Check bounds and obstacles
            if (0 <= new_x < len(grid) and 
                0 <= new_y < len(grid[0]) and 
                grid[new_x][new_y] == 0):  # 0 = walkable
                
                result = find_path_recursive(grid, (new_x, new_y), end, visited, path + [start])
                if result:
                    return result
        
        return None
    
    # Test pathfinding
    test_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],  # 1 = obstacle
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start_pos = (0, 0)
    end_pos = (4, 4)
    path = find_path_recursive(test_grid, start_pos, end_pos)
    
    print(f"   Grid size: {len(test_grid)}x{len(test_grid[0])}")
    print(f"   Start: {start_pos}, End: {end_pos}")
    print(f"   Path found: {path}")
    print(f"   Path length: {len(path) if path else 'No path'}")
    
    # 2. Recursive flood fill
    print("\n2. Recursive Flood Fill:")
    
    def flood_fill_recursive(grid, x, y, target_color, replacement_color):
        """Fill connected area with new color using recursion"""
        # Base cases
        if (x < 0 or x >= len(grid) or 
            y < 0 or y >= len(grid[0]) or 
            grid[x][y] != target_color):
            return
        
        # Fill current pixel
        grid[x][y] = replacement_color
        
        # Recursively fill neighbors
        flood_fill_recursive(grid, x + 1, y, target_color, replacement_color)
        flood_fill_recursive(grid, x - 1, y, target_color, replacement_color)
        flood_fill_recursive(grid, x, y + 1, target_color, replacement_color)
        flood_fill_recursive(grid, x, y - 1, target_color, replacement_color)
    
    # Test flood fill
    test_grid_fill = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    
    print("   Before flood fill:")
    for row in test_grid_fill:
        print(f"     {row}")
    
    flood_fill_recursive(test_grid_fill, 2, 2, 0, 2)
    
    print("   After flood fill:")
    for row in test_grid_fill:
        print(f"     {row}")
    
    # 3. Recursive quadtree for spatial partitioning
    print("\n3. Recursive Quadtree:")
    
    class Quadtree:
        def __init__(self, bounds, max_objects=4, max_depth=4, depth=0):
            self.bounds = bounds  # [x, y, width, height]
            self.max_objects = max_objects
            self.max_depth = max_depth
            self.depth = depth
            self.objects = []
            self.children = []
        
        def subdivide(self):
            """Split node into four children"""
            x, y, w, h = self.bounds
            half_w, half_h = w / 2, h / 2
            
            self.children = [
                Quadtree([x, y, half_w, half_h], self.max_objects, self.max_depth, self.depth + 1),
                Quadtree([x + half_w, y, half_w, half_h], self.max_objects, self.max_depth, self.depth + 1),
                Quadtree([x, y + half_h, half_w, half_h], self.max_objects, self.max_depth, self.depth + 1),
                Quadtree([x + half_w, y + half_h, half_w, half_h], self.max_objects, self.max_depth, self.depth + 1)
            ]
        
        def insert(self, obj):
            """Insert object into quadtree"""
            if len(self.children) == 0:
                if len(self.objects) < self.max_objects or self.depth >= self.max_depth:
                    self.objects.append(obj)
                else:
                    self.subdivide()
                    # Redistribute existing objects
                    for existing_obj in self.objects:
                        self._insert_into_children(existing_obj)
                    self.objects = []
                    # Insert new object
                    self._insert_into_children(obj)
            else:
                self._insert_into_children(obj)
        
        def _insert_into_children(self, obj):
            """Insert object into appropriate child"""
            for child in self.children:
                if self._intersects(child.bounds, obj["bounds"]):
                    child.insert(obj)
        
        def _intersects(self, bounds1, bounds2):
            """Check if two bounding boxes intersect"""
            return not (bounds1[0] + bounds1[2] < bounds2[0] or
                       bounds2[0] + bounds2[2] < bounds1[0] or
                       bounds1[1] + bounds1[3] < bounds2[1] or
                       bounds2[1] + bounds2[3] < bounds1[1])
        
        def query(self, query_bounds):
            """Query objects in given bounds"""
            result = []
            
            if not self._intersects(self.bounds, query_bounds):
                return result
            
            # Add objects in this node
            for obj in self.objects:
                if self._intersects(query_bounds, obj["bounds"]):
                    result.append(obj)
            
            # Query children
            for child in self.children:
                result.extend(child.query(query_bounds))
            
            return result
    
    # Test quadtree
    quadtree = Quadtree([0, 0, 100, 100])
    test_objects_quad = [
        {"id": 1, "bounds": [10, 10, 5, 5]},
        {"id": 2, "bounds": [20, 20, 5, 5]},
        {"id": 3, "bounds": [30, 30, 5, 5]},
        {"id": 4, "bounds": [40, 40, 5, 5]},
        {"id": 5, "bounds": [50, 50, 5, 5]}
    ]
    
    for obj in test_objects_quad:
        quadtree.insert(obj)
    
    query_bounds = [15, 15, 20, 20]
    found_objects = quadtree.query(query_bounds)
    
    print(f"   Inserted {len(test_objects_quad)} objects")
    print(f"   Query bounds: {query_bounds}")
    print(f"   Found objects: {[obj['id'] for obj in found_objects]}")

def demonstrate_recursion_optimization():
    """Demonstrate optimization techniques for recursion"""
    print("\n=== Recursion Optimization ===\n")
    
    # 1. Memoization (caching)
    print("1. Memoization:")
    
    def fibonacci_memoized(n, memo=None):
        """Fibonacci with memoization to avoid redundant calculations"""
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 1:
            result = n
        else:
            result = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
        
        memo[n] = result
        return result
    
    # Compare performance
    import time
    
    start_time = time.time()
    fib_30_memo = fibonacci_memoized(30)
    memo_time = time.time() - start_time
    
    print(f"   Fibonacci(30) with memoization: {fib_30_memo}")
    print(f"   Time with memoization: {memo_time:.4f} seconds")
    
    # 2. Tail recursion optimization (simulated)
    print("\n2. Tail Recursion Optimization:")
    
    def factorial_tail_recursive(n, accumulator=1):
        """Factorial using tail recursion"""
        if n <= 1:
            return accumulator
        else:
            return factorial_tail_recursive(n - 1, n * accumulator)
    
    # Test tail recursive factorial
    result = factorial_tail_recursive(5)
    print(f"   5! using tail recursion: {result}")
    
    # 3. Iterative conversion
    print("\n3. Iterative Conversion:")
    
    def fibonacci_iterative(n):
        """Convert recursive Fibonacci to iterative"""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    # Compare iterative vs recursive
    start_time = time.time()
    fib_30_iter = fibonacci_iterative(30)
    iter_time = time.time() - start_time
    
    print(f"   Fibonacci(30) iterative: {fib_30_iter}")
    print(f"   Time iterative: {iter_time:.4f} seconds")
    print(f"   Speedup: {memo_time/iter_time:.1f}x faster")

def main():
    """Main function to run all recursion demonstrations"""
    print("=== Python Recursion for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_recursion()
    demonstrate_3d_recursion()
    demonstrate_divide_and_conquer()
    demonstrate_recursive_algorithms()
    demonstrate_recursion_optimization()
    
    print("\n=== Summary ===")
    print("This chapter covered recursive functions and algorithms:")
    print("✓ Basic recursive function concepts")
    print("✓ Recursion in 3D graphics contexts")
    print("✓ Divide and conquer algorithms")
    print("✓ Specific recursive algorithms")
    print("✓ Recursion optimization techniques")
    print("✓ Performance considerations")
    
    print("\nRecursion is essential for:")
    print("- Fractal generation and procedural content")
    print("- Tree and hierarchy traversal")
    print("- Divide and conquer algorithms")
    print("- Spatial partitioning and optimization")
    print("- Pathfinding and AI algorithms")
    print("- Complex geometric calculations")

if __name__ == "__main__":
    main()
