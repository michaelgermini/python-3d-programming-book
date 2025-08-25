#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
Object Properties Example

This example demonstrates how to manage 3D object properties using Python
variables and data structures in a practical 3D graphics context.
"""

def demonstrate_object_properties():
    """Demonstrate managing 3D object properties"""
    print("=== 3D Object Properties Management ===\n")
    
    # 1. Basic object properties
    print("1. Basic Object Properties:")
    
    # Create a 3D object with various properties
    cube = {
        'name': 'player_cube',
        'type': 'cube',
        'position': [0, 0, 0],
        'rotation': [0, 0, 0],
        'scale': [1.0, 1.0, 1.0],
        'color': (255, 128, 64),
        'material': 'metal',
        'visible': True,
        'selected': False,
        'collision_enabled': True
    }
    
    print(f"   Object: {cube['name']}")
    print(f"   Type: {cube['type']}")
    print(f"   Position: {cube['position']}")
    print(f"   Color: {cube['color']}")
    print(f"   Material: {cube['material']}")
    print(f"   Visible: {cube['visible']}")
    
    # 2. Modifying object properties
    print("\n2. Modifying Object Properties:")
    
    # Move the object
    cube['position'] = [5, 2, 0]
    print(f"   New position: {cube['position']}")
    
    # Rotate the object
    cube['rotation'] = [45, 0, 90]
    print(f"   New rotation: {cube['rotation']}")
    
    # Scale the object
    cube['scale'] = [2.0, 1.5, 1.0]
    print(f"   New scale: {cube['scale']}")
    
    # Change color
    cube['color'] = (64, 128, 255)  # Blue
    print(f"   New color: {cube['color']}")
    
    # 3. Multiple objects management
    print("\n3. Multiple Objects Management:")
    
    # Create multiple objects
    objects = {
        'player': {
            'name': 'player',
            'type': 'sphere',
            'position': [0, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [1.0, 1.0, 1.0],
            'color': (255, 255, 255),
            'material': 'player',
            'visible': True,
            'health': 100,
            'speed': 5.0
        },
        'enemy_1': {
            'name': 'enemy_1',
            'type': 'cube',
            'position': [10, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [1.0, 1.0, 1.0],
            'color': (255, 0, 0),
            'material': 'enemy',
            'visible': True,
            'health': 50,
            'speed': 2.0
        },
        'collectible': {
            'name': 'collectible',
            'type': 'sphere',
            'position': [5, 2, 0],
            'rotation': [0, 0, 0],
            'scale': [0.5, 0.5, 0.5],
            'color': (255, 255, 0),
            'material': 'collectible',
            'visible': True,
            'collected': False,
            'value': 10
        }
    }
    
    print("   Objects in scene:")
    for obj_id, obj in objects.items():
        print(f"     {obj_id}: {obj['name']} at {obj['position']}")
    
    # 4. Property validation and constraints
    print("\n4. Property Validation and Constraints:")
    
    def validate_position(position):
        """Validate position coordinates"""
        x, y, z = position
        return -100 <= x <= 100 and -100 <= y <= 100 and -100 <= z <= 100
    
    def validate_scale(scale):
        """Validate scale factors"""
        sx, sy, sz = scale
        return 0.1 <= sx <= 10 and 0.1 <= sy <= 10 and 0.1 <= sz <= 10
    
    def validate_color(color):
        """Validate RGB color values"""
        r, g, b = color
        return 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    
    # Test validation
    test_position = [150, 0, 0]  # Invalid (x > 100)
    test_scale = [0.05, 1.0, 1.0]  # Invalid (sx < 0.1)
    test_color = (300, 128, 64)  # Invalid (r > 255)
    
    print(f"   Position {test_position} valid: {validate_position(test_position)}")
    print(f"   Scale {test_scale} valid: {validate_scale(test_scale)}")
    print(f"   Color {test_color} valid: {validate_color(test_color)}")
    
    # 5. Property inheritance and defaults
    print("\n5. Property Inheritance and Defaults:")
    
    # Default properties for different object types
    default_properties = {
        'cube': {
            'type': 'cube',
            'scale': [1.0, 1.0, 1.0],
            'material': 'default',
            'visible': True,
            'collision_enabled': True
        },
        'sphere': {
            'type': 'sphere',
            'scale': [1.0, 1.0, 1.0],
            'material': 'default',
            'visible': True,
            'collision_enabled': True
        },
        'cylinder': {
            'type': 'cylinder',
            'scale': [1.0, 1.0, 1.0],
            'material': 'default',
            'visible': True,
            'collision_enabled': True
        }
    }
    
    def create_object(obj_type, name, position, **custom_properties):
        """Create an object with default properties and custom overrides"""
        # Start with default properties for the object type
        obj = default_properties[obj_type].copy()
        
        # Add required properties
        obj['name'] = name
        obj['position'] = position
        obj['rotation'] = [0, 0, 0]
        obj['color'] = (128, 128, 128)  # Default gray
        
        # Override with custom properties
        obj.update(custom_properties)
        
        return obj
    
    # Create objects using the factory function
    new_cube = create_object('cube', 'test_cube', [0, 0, 0], 
                           color=(255, 0, 0), material='metal')
    new_sphere = create_object('sphere', 'test_sphere', [5, 0, 0],
                             scale=[2.0, 2.0, 2.0], color=(0, 255, 0))
    
    print(f"   Created cube: {new_cube['name']} at {new_cube['position']}")
    print(f"   Created sphere: {new_sphere['name']} at {new_sphere['position']}")

def demonstrate_property_operations():
    """Demonstrate operations on object properties"""
    print("\n=== Property Operations ===\n")
    
    # 1. Property queries and filtering
    print("1. Property Queries and Filtering:")
    
    # Sample objects
    scene_objects = [
        {'name': 'player', 'type': 'sphere', 'position': [0, 0, 0], 'visible': True},
        {'name': 'enemy1', 'type': 'cube', 'position': [10, 0, 0], 'visible': True},
        {'name': 'enemy2', 'type': 'cube', 'position': [-10, 0, 0], 'visible': False},
        {'name': 'collectible', 'type': 'sphere', 'position': [5, 2, 0], 'visible': True},
        {'name': 'obstacle', 'type': 'cylinder', 'position': [0, 5, 0], 'visible': True}
    ]
    
    # Find all visible objects
    visible_objects = [obj for obj in scene_objects if obj['visible']]
    print(f"   Visible objects: {len(visible_objects)}")
    for obj in visible_objects:
        print(f"     {obj['name']} ({obj['type']})")
    
    # Find all cubes
    cubes = [obj for obj in scene_objects if obj['type'] == 'cube']
    print(f"   Cubes: {len(cubes)}")
    for obj in cubes:
        print(f"     {obj['name']} at {obj['position']}")
    
    # 2. Property calculations
    print("\n2. Property Calculations:")
    
    def calculate_object_volume(obj):
        """Calculate approximate volume of an object"""
        scale = obj.get('scale', [1.0, 1.0, 1.0])
        base_volume = 1.0  # Assume base volume of 1
        
        if obj['type'] == 'cube':
            return base_volume * scale[0] * scale[1] * scale[2]
        elif obj['type'] == 'sphere':
            radius = scale[0] / 2  # Assume scale[0] is diameter
            return (4/3) * 3.14159 * radius**3
        elif obj['type'] == 'cylinder':
            radius = scale[0] / 2
            height = scale[1]
            return 3.14159 * radius**2 * height
        else:
            return base_volume
    
    # Calculate volumes
    for obj in scene_objects:
        volume = calculate_object_volume(obj)
        print(f"   {obj['name']} volume: {volume:.2f}")
    
    # 3. Property transformations
    print("\n3. Property Transformations:")
    
    def transform_object(obj, translation=None, rotation=None, scale=None):
        """Transform an object's properties"""
        new_obj = obj.copy()
        
        if translation:
            current_pos = new_obj['position']
            new_obj['position'] = [
                current_pos[0] + translation[0],
                current_pos[1] + translation[1],
                current_pos[2] + translation[2]
            ]
        
        if rotation:
            current_rot = new_obj.get('rotation', [0, 0, 0])
            new_obj['rotation'] = [
                current_rot[0] + rotation[0],
                current_rot[1] + rotation[1],
                current_rot[2] + rotation[2]
            ]
        
        if scale:
            current_scale = new_obj.get('scale', [1.0, 1.0, 1.0])
            new_obj['scale'] = [
                current_scale[0] * scale[0],
                current_scale[1] * scale[1],
                current_scale[2] * scale[2]
            ]
        
        return new_obj
    
    # Transform the player object
    player = scene_objects[0]
    transformed_player = transform_object(
        player,
        translation=[5, 0, 0],
        rotation=[0, 45, 0],
        scale=[1.5, 1.5, 1.5]
    )
    
    print(f"   Original player position: {player['position']}")
    print(f"   Transformed player position: {transformed_player['position']}")
    print(f"   Transformed player rotation: {transformed_player['rotation']}")
    print(f"   Transformed player scale: {transformed_player['scale']}")

def demonstrate_property_serialization():
    """Demonstrate serializing and deserializing object properties"""
    print("\n=== Property Serialization ===\n")
    
    # 1. Converting to different formats
    print("1. Property Format Conversion:")
    
    # Complex object with nested properties
    complex_object = {
        'name': 'advanced_cube',
        'transform': {
            'position': [10, 5, -3],
            'rotation': [30, 45, 60],
            'scale': [2.0, 1.5, 0.8]
        },
        'appearance': {
            'color': (255, 128, 64),
            'material': 'metal',
            'texture': 'brick.jpg',
            'shininess': 0.8
        },
        'physics': {
            'mass': 10.0,
            'friction': 0.3,
            'bounciness': 0.1,
            'collision_shape': 'box'
        },
        'behavior': {
            'visible': True,
            'interactive': True,
            'destructible': False,
            'tags': ['environment', 'obstacle']
        }
    }
    
    # Flatten the object for simple storage
    def flatten_object(obj, prefix=''):
        """Flatten nested object properties"""
        flattened = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested = flatten_object(value, f"{prefix}{key}_")
                flattened.update(nested)
            else:
                flattened[f"{prefix}{key}"] = value
        return flattened
    
    flattened = flatten_object(complex_object)
    print("   Flattened properties:")
    for key, value in flattened.items():
        print(f"     {key}: {value}")
    
    # 2. Property validation and type checking
    print("\n2. Property Validation and Type Checking:")
    
    def validate_object_properties(obj):
        """Validate object properties and their types"""
        errors = []
        
        # Check required properties
        required_props = ['name', 'transform', 'appearance']
        for prop in required_props:
            if prop not in obj:
                errors.append(f"Missing required property: {prop}")
        
        # Check transform properties
        if 'transform' in obj:
            transform = obj['transform']
            if not isinstance(transform.get('position'), list) or len(transform['position']) != 3:
                errors.append("Transform position must be a list of 3 numbers")
            
            if not isinstance(transform.get('scale'), list) or len(transform['scale']) != 3:
                errors.append("Transform scale must be a list of 3 numbers")
        
        # Check appearance properties
        if 'appearance' in obj:
            appearance = obj['appearance']
            if not isinstance(appearance.get('color'), tuple) or len(appearance['color']) != 3:
                errors.append("Appearance color must be a tuple of 3 integers")
        
        return errors
    
    # Validate the complex object
    validation_errors = validate_object_properties(complex_object)
    if validation_errors:
        print("   Validation errors:")
        for error in validation_errors:
            print(f"     ❌ {error}")
    else:
        print("   ✅ Object properties are valid")

def main():
    """Main function to run all property management demonstrations"""
    print("=== Python 3D Object Properties Management ===\n")
    
    # Run all demonstrations
    demonstrate_object_properties()
    demonstrate_property_operations()
    demonstrate_property_serialization()
    
    print("\n=== Summary ===")
    print("This chapter covered object property management:")
    print("✓ Creating and modifying object properties")
    print("✓ Managing multiple objects efficiently")
    print("✓ Property validation and constraints")
    print("✓ Property inheritance and defaults")
    print("✓ Property queries and filtering")
    print("✓ Property transformations and calculations")
    print("✓ Property serialization and validation")
    
    print("\nThese concepts are essential for:")
    print("- Managing complex 3D scenes with many objects")
    print("- Implementing object behaviors and interactions")
    print("- Creating reusable object templates and factories")
    print("- Validating and maintaining data integrity")
    print("- Optimizing object storage and retrieval")

if __name__ == "__main__":
    main()
