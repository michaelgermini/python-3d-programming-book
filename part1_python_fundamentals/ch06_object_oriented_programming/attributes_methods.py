#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Attributes and Methods Example

This example demonstrates different types of attributes (instance, class) and
methods (instance, class, static) in object-oriented programming, applied to
3D graphics and game development scenarios.
"""

import math
import time

class GameObject:
    """Base class for all game objects"""
    
    # Class variables (shared across all instances)
    total_objects = 0
    object_types = set()
    
    def __init__(self, name, object_type="generic"):
        """Constructor - initialize a game object"""
        # Instance variables (unique to each instance)
        self.name = name
        self.object_type = object_type
        self.active = True
        self.visible = True
        self.created_time = time.time()
        
        # Update class variables
        GameObject.total_objects += 1
        GameObject.object_types.add(object_type)
    
    def __del__(self):
        """Destructor - called when object is destroyed"""
        GameObject.total_objects -= 1
    
    # Instance method - operates on instance data
    def update(self, delta_time):
        """Update the game object"""
        if not self.active:
            return
        print(f"   Updating {self.name} ({self.object_type})")
    
    def render(self):
        """Render the game object"""
        if not self.active or not self.visible:
            return
        print(f"   Rendering {self.name}")
    
    # Class method - operates on class data
    @classmethod
    def get_total_objects(cls):
        """Get the total number of game objects"""
        return cls.total_objects
    
    @classmethod
    def get_object_types(cls):
        """Get all object types that have been created"""
        return list(cls.object_types)
    
    @classmethod
    def create_default_object(cls, name):
        """Create a default game object"""
        return cls(name, "default")
    
    # Static method - doesn't use instance or class data
    @staticmethod
    def calculate_distance(pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    @staticmethod
    def is_valid_name(name):
        """Check if a name is valid for a game object"""
        return isinstance(name, str) and len(name) > 0 and name.isalnum()

class Transform:
    """Transform component for 3D objects"""
    
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        """Initialize transform with position, rotation, and scale"""
        # Instance variables
        self._position = list(position)  # Private by convention
        self._rotation = list(rotation)
        self._scale = list(scale)
        self._dirty = True  # Flag for optimization
    
    # Property decorators for controlled access
    @property
    def position(self):
        """Get the position"""
        return tuple(self._position)
    
    @position.setter
    def position(self, value):
        """Set the position"""
        self._position = list(value)
        self._dirty = True
    
    @property
    def rotation(self):
        """Get the rotation"""
        return tuple(self._rotation)
    
    @rotation.setter
    def rotation(self, value):
        """Set the rotation"""
        self._rotation = list(value)
        self._dirty = True
    
    @property
    def scale(self):
        """Get the scale"""
        return tuple(self._scale)
    
    @scale.setter
    def scale(self, value):
        """Set the scale"""
        self._scale = list(value)
        self._dirty = True
    
    # Instance methods
    def translate(self, dx, dy, dz):
        """Translate the transform"""
        self._position[0] += dx
        self._position[1] += dy
        self._position[2] += dz
        self._dirty = True
    
    def rotate(self, dx, dy, dz):
        """Rotate the transform"""
        self._rotation[0] += dx
        self._rotation[1] += dy
        self._rotation[2] += dz
        self._dirty = True
    
    def scale_by(self, sx, sy, sz):
        """Scale the transform"""
        self._scale[0] *= sx
        self._scale[1] *= sy
        self._scale[2] *= sz
        self._dirty = True
    
    def get_matrix(self):
        """Get transformation matrix (simplified)"""
        if self._dirty:
            # In a real implementation, this would calculate the actual matrix
            self._dirty = False
        
        return {
            'position': self._position.copy(),
            'rotation': self._rotation.copy(),
            'scale': self._scale.copy()
        }
    
    # Class method
    @classmethod
    def identity(cls):
        """Create an identity transform"""
        return cls((0, 0, 0), (0, 0, 0), (1, 1, 1))
    
    # Static method
    @staticmethod
    def lerp(transform1, transform2, t):
        """Linear interpolation between two transforms"""
        pos1, pos2 = transform1.position, transform2.position
        rot1, rot2 = transform1.rotation, transform2.rotation
        scale1, scale2 = transform1.scale, transform2.scale
        
        new_pos = [pos1[i] + (pos2[i] - pos1[i]) * t for i in range(3)]
        new_rot = [rot1[i] + (rot2[i] - rot1[i]) * t for i in range(3)]
        new_scale = [scale1[i] + (scale2[i] - scale1[i]) * t for i in range(3)]
        
        return Transform(new_pos, new_rot, new_scale)

class Material:
    """Material class for 3D objects"""
    
    # Class variables for default materials
    default_materials = {
        "default": {"albedo": [0.8, 0.8, 0.8], "metallic": 0.0, "roughness": 0.5},
        "metal": {"albedo": [0.8, 0.8, 0.8], "metallic": 1.0, "roughness": 0.2},
        "plastic": {"albedo": [0.2, 0.2, 0.2], "metallic": 0.0, "roughness": 0.8},
        "glass": {"albedo": [0.9, 0.9, 0.9], "metallic": 0.0, "roughness": 0.1}
    }
    
    def __init__(self, name, properties=None):
        """Initialize material with name and properties"""
        self.name = name
        self.properties = properties or Material.default_materials["default"].copy()
        self.textures = {}
        self.shader = "pbr"
    
    # Instance methods
    def set_property(self, property_name, value):
        """Set a material property"""
        self.properties[property_name] = value
    
    def get_property(self, property_name, default=None):
        """Get a material property"""
        return self.properties.get(property_name, default)
    
    def add_texture(self, texture_type, texture_path):
        """Add a texture to the material"""
        self.textures[texture_type] = texture_path
    
    def apply_to_object(self, game_object):
        """Apply this material to a game object"""
        print(f"   Applying material '{self.name}' to '{game_object.name}'")
    
    # Class methods
    @classmethod
    def create_default(cls, material_type):
        """Create a default material of the specified type"""
        if material_type in cls.default_materials:
            return cls(material_type, cls.default_materials[material_type].copy())
        else:
            return cls(material_type)
    
    @classmethod
    def get_available_types(cls):
        """Get all available material types"""
        return list(cls.default_materials.keys())
    
    # Static methods
    @staticmethod
    def validate_properties(properties):
        """Validate material properties"""
        required = ["albedo", "metallic", "roughness"]
        return all(prop in properties for prop in required)
    
    @staticmethod
    def blend_materials(material1, material2, blend_factor):
        """Blend two materials together"""
        blended_props = {}
        for key in material1.properties:
            if key in material2.properties:
                val1 = material1.properties[key]
                val2 = material2.properties[key]
                if isinstance(val1, list):
                    blended_props[key] = [val1[i] + (val2[i] - val1[i]) * blend_factor for i in range(len(val1))]
                else:
                    blended_props[key] = val1 + (val2 - val1) * blend_factor
        
        return Material(f"blended_{material1.name}_{material2.name}", blended_props)

class Component:
    """Base class for all components"""
    
    def __init__(self, name):
        """Initialize component"""
        self.name = name
        self.enabled = True
        self.game_object = None
    
    def attach_to(self, game_object):
        """Attach this component to a game object"""
        self.game_object = game_object
        print(f"   Attached component '{self.name}' to '{game_object.name}'")
    
    def detach(self):
        """Detach this component from its game object"""
        if self.game_object:
            print(f"   Detached component '{self.name}' from '{self.game_object.name}'")
            self.game_object = None
    
    def update(self, delta_time):
        """Update the component"""
        if self.enabled and self.game_object:
            print(f"   Updating component '{self.name}' on '{self.game_object.name}'")
    
    def render(self):
        """Render the component"""
        if self.enabled and self.game_object:
            print(f"   Rendering component '{self.name}' on '{self.game_object.name}'")

class Renderer(Component):
    """Renderer component for 3D objects"""
    
    def __init__(self, mesh_data=None, material=None):
        """Initialize renderer"""
        super().__init__("Renderer")
        self.mesh_data = mesh_data or {}
        self.material = material or Material.create_default("default")
        self.visible = True
        self.cast_shadows = True
        self.receive_shadows = True
    
    def set_mesh(self, mesh_data):
        """Set the mesh data"""
        self.mesh_data = mesh_data
    
    def set_material(self, material):
        """Set the material"""
        self.material = material
    
    def render(self):
        """Render the object"""
        if self.enabled and self.visible and self.game_object:
            print(f"   Rendering mesh with material '{self.material.name}' for '{self.game_object.name}'")

class Collider(Component):
    """Collider component for physics"""
    
    def __init__(self, collider_type="box", size=(1, 1, 1)):
        """Initialize collider"""
        super().__init__("Collider")
        self.collider_type = collider_type
        self.size = list(size)
        self.enabled = True
    
    def check_collision(self, other_collider):
        """Check collision with another collider"""
        if not self.enabled or not other_collider.enabled:
            return False
        
        # Simplified collision detection
        return True
    
    def get_bounds(self):
        """Get the bounding box"""
        return {
            'type': self.collider_type,
            'size': self.size.copy()
        }

def demonstrate_instance_attributes():
    """Demonstrate instance attributes"""
    print("=== Instance Attributes ===\n")
    
    # Create game objects with different instance attributes
    player = GameObject("Player", "character")
    enemy = GameObject("Enemy", "character")
    cube = GameObject("Cube", "geometry")
    
    print(f"Player name: {player.name}")
    print(f"Player type: {player.object_type}")
    print(f"Player active: {player.active}")
    print(f"Player created: {player.created_time}")
    
    print(f"\nEnemy name: {enemy.name}")
    print(f"Enemy type: {enemy.object_type}")
    
    # Modify instance attributes
    player.active = False
    enemy.visible = False
    
    print(f"\nAfter modification:")
    print(f"Player active: {player.active}")
    print(f"Enemy visible: {enemy.visible}")

def demonstrate_class_attributes():
    """Demonstrate class attributes"""
    print("\n=== Class Attributes ===\n")
    
    # Access class attributes
    print(f"Total objects: {GameObject.get_total_objects()}")
    print(f"Object types: {GameObject.get_object_types()}")
    
    # Create more objects to see class attributes change
    light = GameObject("Light", "light")
    camera = GameObject("Camera", "camera")
    
    print(f"\nAfter creating more objects:")
    print(f"Total objects: {GameObject.get_total_objects()}")
    print(f"Object types: {GameObject.get_object_types()}")

def demonstrate_instance_methods():
    """Demonstrate instance methods"""
    print("\n=== Instance Methods ===\n")
    
    # Create objects and call instance methods
    player = GameObject("Player", "character")
    enemy = GameObject("Enemy", "character")
    
    print("Calling instance methods:")
    player.update(0.016)
    player.render()
    enemy.update(0.016)
    enemy.render()
    
    # Transform instance methods
    transform = Transform((1, 2, 3), (45, 0, 0), (2, 1, 1))
    print(f"\nTransform: {transform.get_matrix()}")
    
    transform.translate(1, 0, 0)
    transform.rotate(0, 90, 0)
    print(f"After transformations: {transform.get_matrix()}")

def demonstrate_class_methods():
    """Demonstrate class methods"""
    print("\n=== Class Methods ===\n")
    
    # Use class methods
    print(f"Total objects: {GameObject.get_total_objects()}")
    print(f"Available material types: {Material.get_available_types()}")
    
    # Create objects using class methods
    default_obj = GameObject.create_default_object("DefaultObject")
    metal_material = Material.create_default("metal")
    
    print(f"\nCreated default object: {default_obj.name}")
    print(f"Created metal material: {metal_material.name}")
    print(f"Metal properties: {metal_material.properties}")

def demonstrate_static_methods():
    """Demonstrate static methods"""
    print("\n=== Static Methods ===\n")
    
    # Use static methods
    pos1 = (0, 0, 0)
    pos2 = (3, 4, 0)
    distance = GameObject.calculate_distance(pos1, pos2)
    print(f"Distance between {pos1} and {pos2}: {distance:.2f}")
    
    # Validate names
    valid_names = ["Player", "Enemy", "123", ""]
    for name in valid_names:
        is_valid = GameObject.is_valid_name(name)
        print(f"Name '{name}' is valid: {is_valid}")
    
    # Validate material properties
    good_props = {"albedo": [0.8, 0.8, 0.8], "metallic": 0.0, "roughness": 0.5}
    bad_props = {"albedo": [0.8, 0.8, 0.8]}
    
    print(f"\nGood properties valid: {Material.validate_properties(good_props)}")
    print(f"Bad properties valid: {Material.validate_properties(bad_props)}")

def demonstrate_properties():
    """Demonstrate property decorators"""
    print("\n=== Properties ===\n")
    
    # Create transform and use properties
    transform = Transform((1, 2, 3), (45, 0, 0), (2, 1, 1))
    
    print(f"Initial position: {transform.position}")
    print(f"Initial rotation: {transform.rotation}")
    print(f"Initial scale: {transform.scale}")
    
    # Modify using properties
    transform.position = (5, 10, 15)
    transform.rotation = (90, 0, 45)
    transform.scale = (3, 2, 1)
    
    print(f"\nAfter modification:")
    print(f"Position: {transform.position}")
    print(f"Rotation: {transform.rotation}")
    print(f"Scale: {transform.scale}")

def demonstrate_component_system():
    """Demonstrate component system"""
    print("\n=== Component System ===\n")
    
    # Create game object with components
    player = GameObject("Player", "character")
    
    # Create and attach components
    renderer = Renderer(material=Material.create_default("metal"))
    collider = Collider("capsule", (0.5, 1.8, 0.5))
    
    renderer.attach_to(player)
    collider.attach_to(player)
    
    # Update and render
    print("\nUpdating components:")
    renderer.update(0.016)
    collider.update(0.016)
    
    print("\nRendering components:")
    renderer.render()
    
    # Check collision
    enemy = GameObject("Enemy", "character")
    enemy_collider = Collider("box", (1, 1, 1))
    enemy_collider.attach_to(enemy)
    
    collision = collider.check_collision(enemy_collider)
    print(f"\nCollision between player and enemy: {collision}")

def demonstrate_material_system():
    """Demonstrate material system"""
    print("\n=== Material System ===\n")
    
    # Create different materials
    metal = Material.create_default("metal")
    plastic = Material.create_default("plastic")
    glass = Material.create_default("glass")
    
    print("Material properties:")
    print(f"Metal: {metal.properties}")
    print(f"Plastic: {plastic.properties}")
    print(f"Glass: {glass.properties}")
    
    # Blend materials
    blended = Material.blend_materials(metal, plastic, 0.5)
    print(f"\nBlended material: {blended.properties}")
    
    # Apply materials to objects
    cube = GameObject("Cube", "geometry")
    metal.apply_to_object(cube)

def main():
    """Main function to run all demonstrations"""
    print("=== Python Attributes and Methods for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_instance_attributes()
    demonstrate_class_attributes()
    demonstrate_instance_methods()
    demonstrate_class_methods()
    demonstrate_static_methods()
    demonstrate_properties()
    demonstrate_component_system()
    demonstrate_material_system()
    
    print("\n=== Summary ===")
    print("This chapter covered different types of attributes and methods:")
    print("✓ Instance attributes: Unique data for each object")
    print("✓ Class attributes: Shared data across all instances")
    print("✓ Instance methods: Operate on instance data")
    print("✓ Class methods: Operate on class data")
    print("✓ Static methods: Don't use instance or class data")
    print("✓ Properties: Controlled access to attributes")
    print("✓ Component system: Modular object architecture")
    print("✓ Material system: Flexible material management")

if __name__ == "__main__":
    main()
