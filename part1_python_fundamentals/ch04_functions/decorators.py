#!/usr/bin/env python3
"""
Chapter 4: Functions
Decorators Example

This example demonstrates function decorators and advanced patterns, focusing on
applications in 3D graphics programming.
"""

import time
import math
import functools

def demonstrate_basic_decorators():
    """Demonstrate basic decorator concepts"""
    print("=== Basic Decorators ===\n")
    
    # 1. Simple decorator function
    print("1. Simple Decorator Function:")
    
    def timer_decorator(func):
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"   {func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    
    # Use the decorator
    @timer_decorator
    def calculate_distance(p1, p2):
        """Calculate distance between two 3D points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
    # Test the decorated function
    point1 = [0, 0, 0]
    point2 = [3, 4, 0]
    distance = calculate_distance(point1, point2)
    print(f"   Distance: {distance}")
    
    # 2. Decorator with arguments
    print("\n2. Decorator with Arguments:")
    
    def repeat_decorator(times):
        """Decorator that repeats function execution"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                results = []
                for i in range(times):
                    result = func(*args, **kwargs)
                    results.append(result)
                return results
            return wrapper
        return decorator
    
    @repeat_decorator(3)
    def generate_random_position():
        """Generate a random 3D position"""
        import random
        return [random.uniform(-10, 10) for _ in range(3)]
    
    positions = generate_random_position()
    print(f"   Generated positions: {positions}")
    
    # 3. Decorator that modifies function behavior
    print("\n3. Behavior Modification Decorator:")
    
    def validate_3d_point(func):
        """Decorator to validate 3D point arguments"""
        def wrapper(*args, **kwargs):
            # Check if first argument is a 3D point
            if args and len(args[0]) == 3:
                x, y, z = args[0]
                if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
                    raise ValueError("All coordinates must be numbers")
                if not all(-1000 <= coord <= 1000 for coord in [x, y, z]):
                    raise ValueError("Coordinates must be within [-1000, 1000]")
            return func(*args, **kwargs)
        return wrapper
    
    @validate_3d_point
    def process_3d_point(point):
        """Process a 3D point"""
        return f"Processed point: {point}"
    
    # Test validation
    try:
        result = process_3d_point([1, 2, 3])
        print(f"   {result}")
        
        # This should raise an error
        # result = process_3d_point([10000, 0, 0])
    except ValueError as e:
        print(f"   Validation error: {e}")

def demonstrate_decorator_syntax():
    """Demonstrate different ways to use decorators"""
    print("\n=== Decorator Syntax ===\n")
    
    # 1. Function decorator syntax
    print("1. Function Decorator Syntax:")
    
    def log_calls(func):
        """Log function calls"""
        def wrapper(*args, **kwargs):
            print(f"   Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            print(f"   {func.__name__} returned: {result}")
            return result
        return wrapper
    
    # Method 1: Using @ syntax
    @log_calls
    def add_vectors(v1, v2):
        return [v1[i] + v2[i] for i in range(3)]
    
    # Method 2: Manual decoration
    def scale_vector(v, s):
        return [v[i] * s for i in range(3)]
    
    scale_vector = log_calls(scale_vector)
    
    # Test both methods
    result1 = add_vectors([1, 2, 3], [4, 5, 6])
    result2 = scale_vector([1, 2, 3], 2)
    
    # 2. Multiple decorators
    print("\n2. Multiple Decorators:")
    
    def cache_result(func):
        """Simple caching decorator"""
        cache = {}
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return wrapper
    
    def debug_info(func):
        """Add debug information"""
        def wrapper(*args, **kwargs):
            print(f"   DEBUG: Entering {func.__name__}")
            result = func(*args, **kwargs)
            print(f"   DEBUG: Exiting {func.__name__}")
            return result
        return wrapper
    
    @debug_info
    @cache_result
    @timer_decorator
    def expensive_calculation(x, y, z):
        """Simulate expensive calculation"""
        time.sleep(0.1)  # Simulate work
        return math.sqrt(x**2 + y**2 + z**2)
    
    # Test multiple decorators
    result = expensive_calculation(3, 4, 0)
    result2 = expensive_calculation(3, 4, 0)  # Should use cache
    
    # 3. Decorator with functools.wraps
    print("\n3. Using functools.wraps:")
    
    def preserve_metadata(func):
        """Decorator that preserves function metadata"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"   Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    
    @preserve_metadata
    def normalize_vector(vector):
        """Normalize a 3D vector"""
        magnitude = math.sqrt(sum(x**2 for x in vector))
        return [x/magnitude for x in vector] if magnitude > 0 else vector
    
    # Check metadata preservation
    print(f"   Function name: {normalize_vector.__name__}")
    print(f"   Function docstring: {normalize_vector.__doc__}")
    
    result = normalize_vector([3, 4, 0])

def demonstrate_3d_graphics_decorators():
    """Demonstrate decorators specific to 3D graphics"""
    print("\n=== 3D Graphics Decorators ===\n")
    
    # 1. Performance monitoring decorator
    print("1. Performance Monitoring:")
    
    def performance_monitor(func):
        """Monitor performance of 3D graphics functions"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = get_memory_usage()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"   {func.__name__}: {execution_time:.4f}s, {memory_used:.2f}MB")
            return result
        return wrapper
    
    def get_memory_usage():
        """Get current memory usage (simplified)"""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    @performance_monitor
    def generate_mesh_vertices(segments):
        """Generate vertices for a sphere mesh"""
        vertices = []
        for i in range(segments):
            for j in range(segments):
                phi = (i / segments) * 2 * math.pi
                theta = (j / segments) * math.pi
                
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                
                vertices.append([x, y, z])
        return vertices
    
    # Test performance monitoring
    try:
        vertices = generate_mesh_vertices(50)
        print(f"   Generated {len(vertices)} vertices")
    except ImportError:
        print("   psutil not available, skipping memory monitoring")
        vertices = generate_mesh_vertices(50)
        print(f"   Generated {len(vertices)} vertices")
    
    # 2. Validation decorator for 3D objects
    print("\n2. 3D Object Validation:")
    
    def validate_3d_object(func):
        """Validate 3D object properties"""
        def wrapper(*args, **kwargs):
            # Check if first argument is a 3D object
            if args and isinstance(args[0], dict):
                obj = args[0]
                
                # Validate required properties
                required_props = ['position', 'rotation', 'scale']
                for prop in required_props:
                    if prop not in obj:
                        raise ValueError(f"Missing required property: {prop}")
                
                # Validate position
                if len(obj['position']) != 3:
                    raise ValueError("Position must be a 3D vector")
                
                # Validate scale (must be positive)
                if any(s <= 0 for s in obj['scale']):
                    raise ValueError("Scale values must be positive")
            
            return func(*args, **kwargs)
        return wrapper
    
    @validate_3d_object
    def transform_object(obj):
        """Transform a 3D object"""
        return f"Transformed object at {obj['position']}"
    
    # Test validation
    valid_object = {
        'position': [1, 2, 3],
        'rotation': [0, 0, 0],
        'scale': [1, 1, 1]
    }
    
    try:
        result = transform_object(valid_object)
        print(f"   {result}")
        
        # This should raise an error
        invalid_object = {'position': [1, 2], 'rotation': [0, 0, 0], 'scale': [1, 1, 1]}
        # result = transform_object(invalid_object)
    except ValueError as e:
        print(f"   Validation error: {e}")
    
    # 3. Caching decorator for expensive calculations
    print("\n3. Caching for Expensive Calculations:")
    
    def cache_3d_calculations(func):
        """Cache results of expensive 3D calculations"""
        cache = {}
        
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                print(f"   Cache miss for {func.__name__}")
            else:
                print(f"   Cache hit for {func.__name__}")
            
            return cache[key]
        
        return wrapper
    
    @cache_3d_calculations
    def calculate_bounding_sphere(points):
        """Calculate bounding sphere for a set of points"""
        if not points:
            return None
        
        # Find center (average of all points)
        center = [sum(p[i] for p in points) / len(points) for i in range(3)]
        
        # Find maximum distance from center
        max_distance = max(math.sqrt(sum((p[i] - center[i])**2 for i in range(3))) 
                          for p in points)
        
        return {'center': center, 'radius': max_distance}
    
    # Test caching
    test_points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    sphere1 = calculate_bounding_sphere(test_points)
    sphere2 = calculate_bounding_sphere(test_points)  # Should use cache
    
    print(f"   Bounding sphere: {sphere1}")

def demonstrate_class_decorators():
    """Demonstrate class-based decorators"""
    print("\n=== Class Decorators ===\n")
    
    # 1. Class decorator for singleton pattern
    print("1. Singleton Class Decorator:")
    
    def singleton(cls):
        """Make a class a singleton"""
        instances = {}
        
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        
        return get_instance
    
    @singleton
    class SceneManager:
        """Singleton scene manager"""
        def __init__(self):
            self.objects = []
            self.lights = []
            self.camera = None
        
        def add_object(self, obj):
            self.objects.append(obj)
            return len(self.objects)
        
        def get_object_count(self):
            return len(self.objects)
    
    # Test singleton
    manager1 = SceneManager()
    manager2 = SceneManager()
    
    print(f"   Same instance: {manager1 is manager2}")
    print(f"   Object count: {manager1.get_object_count()}")
    
    manager1.add_object({'name': 'cube'})
    print(f"   Object count after adding: {manager2.get_object_count()}")
    
    # 2. Class decorator for method registration
    print("\n2. Method Registration Decorator:")
    
    class EventSystem:
        """Event system with method registration"""
        def __init__(self):
            self.handlers = {}
        
        def register(self, event_type):
            """Decorator to register event handlers"""
            def decorator(func):
                if event_type not in self.handlers:
                    self.handlers[event_type] = []
                self.handlers[event_type].append(func)
                return func
            return decorator
        
        def trigger(self, event_type, *args, **kwargs):
            """Trigger all handlers for an event"""
            if event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    handler(*args, **kwargs)
    
    # Create event system
    events = EventSystem()
    
    @events.register('object_created')
    def handle_object_created(obj):
        print(f"   Object created: {obj}")
    
    @events.register('object_created')
    def log_object_created(obj):
        print(f"   Logging: New object added to scene")
    
    @events.register('collision_detected')
    def handle_collision(obj1, obj2):
        print(f"   Collision between {obj1} and {obj2}")
    
    # Test event system
    events.trigger('object_created', {'name': 'sphere', 'position': [0, 0, 0]})
    events.trigger('collision_detected', 'player', 'enemy')
    
    # 3. Class decorator for property validation
    print("\n3. Property Validation Decorator:")
    
    def validate_properties(cls):
        """Add property validation to a class"""
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Validate properties after initialization
            for attr_name, attr_value in self.__dict__.items():
                if hasattr(self, f'validate_{attr_name}'):
                    validator = getattr(self, f'validate_{attr_name}')
                    validator(attr_value)
        
        cls.__init__ = new_init
        return cls
    
    @validate_properties
    class GameObject:
        def __init__(self, position, scale, health):
            self.position = position
            self.scale = scale
            self.health = health
        
        def validate_position(self, position):
            if len(position) != 3:
                raise ValueError("Position must be a 3D vector")
        
        def validate_scale(self, scale):
            if any(s <= 0 for s in scale):
                raise ValueError("Scale values must be positive")
        
        def validate_health(self, health):
            if not 0 <= health <= 100:
                raise ValueError("Health must be between 0 and 100")
    
    # Test property validation
    try:
        obj = GameObject([1, 2, 3], [1, 1, 1], 50)
        print(f"   Created valid object: {obj.position}")
        
        # This should raise an error
        # obj2 = GameObject([1, 2], [1, 1, 1], 50)
    except ValueError as e:
        print(f"   Validation error: {e}")

def demonstrate_advanced_decorator_patterns():
    """Demonstrate advanced decorator patterns"""
    print("\n=== Advanced Decorator Patterns ===\n")
    
    # 1. Decorator with state
    print("1. Decorator with State:")
    
    def call_counter(func):
        """Decorator that counts function calls"""
        def wrapper(*args, **kwargs):
            wrapper.call_count += 1
            print(f"   {func.__name__} called {wrapper.call_count} times")
            return func(*args, **kwargs)
        
        wrapper.call_count = 0
        return wrapper
    
    @call_counter
    def calculate_normal(vertices):
        """Calculate normal vector for a triangle"""
        if len(vertices) < 3:
            return [0, 0, 1]
        
        v1 = [vertices[1][i] - vertices[0][i] for i in range(3)]
        v2 = [vertices[2][i] - vertices[0][i] for i in range(3)]
        
        # Cross product
        normal = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        ]
        
        # Normalize
        magnitude = math.sqrt(sum(x**2 for x in normal))
        return [x/magnitude for x in normal] if magnitude > 0 else [0, 0, 1]
    
    # Test call counter
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    normal1 = calculate_normal(vertices)
    normal2 = calculate_normal(vertices)
    
    # 2. Decorator factory with configuration
    print("\n2. Decorator Factory:")
    
    def retry_on_error(max_attempts=3, delay=1):
        """Decorator factory for retrying failed operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            print(f"   Attempt {attempt + 1} failed: {e}")
                            time.sleep(delay)
                
                print(f"   All {max_attempts} attempts failed")
                raise last_exception
            
            return wrapper
        return decorator
    
    @retry_on_error(max_attempts=3, delay=0.1)
    def load_3d_model(filename):
        """Simulate loading a 3D model with potential failures"""
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise FileNotFoundError(f"Failed to load {filename}")
        return f"Loaded model: {filename}"
    
    # Test retry decorator
    try:
        result = load_3d_model("character.obj")
        print(f"   {result}")
    except Exception as e:
        print(f"   Final error: {e}")
    
    # 3. Decorator for method chaining
    print("\n3. Method Chaining Decorator:")
    
    def chainable(func):
        """Make a method chainable by returning self"""
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            return self
        return wrapper
    
    class TransformableObject:
        def __init__(self, position=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1]):
            self.position = list(position)
            self.rotation = list(rotation)
            self.scale = list(scale)
        
        @chainable
        def translate(self, offset):
            """Translate the object"""
            for i in range(3):
                self.position[i] += offset[i]
        
        @chainable
        def rotate(self, angles):
            """Rotate the object"""
            for i in range(3):
                self.rotation[i] += angles[i]
        
        @chainable
        def scale_by(self, factors):
            """Scale the object"""
            for i in range(3):
                self.scale[i] *= factors[i]
        
        def get_transform(self):
            """Get current transform"""
            return {
                'position': self.position,
                'rotation': self.rotation,
                'scale': self.scale
            }
    
    # Test method chaining
    obj = TransformableObject()
    obj.translate([1, 0, 0]).rotate([0, 0, 45]).scale_by([2, 1, 1])
    
    transform = obj.get_transform()
    print(f"   Final transform: {transform}")

def main():
    """Main function to run all decorator demonstrations"""
    print("=== Python Decorators for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_decorators()
    demonstrate_decorator_syntax()
    demonstrate_3d_graphics_decorators()
    demonstrate_class_decorators()
    demonstrate_advanced_decorator_patterns()
    
    print("\n=== Summary ===")
    print("This chapter covered function decorators and advanced patterns:")
    print("✓ Basic decorator concepts and syntax")
    print("✓ Decorator arguments and multiple decorators")
    print("✓ 3D graphics specific decorators")
    print("✓ Class-based decorators")
    print("✓ Advanced decorator patterns")
    print("✓ Performance monitoring and caching")
    
    print("\nDecorators are essential for:")
    print("- Adding functionality without modifying original code")
    print("- Performance monitoring and optimization")
    print("- Input validation and error handling")
    print("- Caching and memoization")
    print("- Logging and debugging")
    print("- Method registration and event handling")
    print("- Code reuse and modularity")

if __name__ == "__main__":
    main()
