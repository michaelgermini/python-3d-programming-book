#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Basic Classes Example

This example demonstrates fundamental OOP concepts including class definition,
object creation, constructors, and methods, applied to 3D graphics programming.
"""

import math

class Point3D:
    """A simple 3D point class"""
    
    def __init__(self, x=0, y=0, z=0):
        """Constructor - initialize a 3D point"""
        self.x = x
        self.y = y
        self.z = z
    
    def distance_to(self, other_point):
        """Calculate distance to another point"""
        dx = self.x - other_point.x
        dy = self.y - other_point.y
        dz = self.z - other_point.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def move(self, dx, dy, dz):
        """Move the point by the given offsets"""
        self.x += dx
        self.y += dy
        self.z += dz
    
    def __str__(self):
        """String representation of the point"""
        return f"Point3D({self.x}, {self.y}, {self.z})"

class Vector3D:
    """A 3D vector class for mathematical operations"""
    
    def __init__(self, x=0, y=0, z=0):
        """Constructor - initialize a 3D vector"""
        self.x = x
        self.y = y
        self.z = z
    
    def magnitude(self):
        """Calculate the magnitude (length) of the vector"""
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def normalize(self):
        """Return a normalized (unit) vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)
    
    def dot_product(self, other):
        """Calculate dot product with another vector"""
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def cross_product(self, other):
        """Calculate cross product with another vector"""
        return Vector3D(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
    
    def __add__(self, other):
        """Add two vectors"""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """Subtract two vectors"""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        """Multiply vector by scalar"""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __str__(self):
        """String representation of the vector"""
        return f"Vector3D({self.x}, {self.y}, {self.z})"

class Transform3D:
    """A 3D transformation class for position, rotation, and scale"""
    
    def __init__(self, position=None, rotation=None, scale=None):
        """Constructor - initialize a 3D transform"""
        self.position = position or Point3D(0, 0, 0)
        self.rotation = rotation or Vector3D(0, 0, 0)  # Euler angles in degrees
        self.scale = scale or Vector3D(1, 1, 1)
    
    def translate(self, dx, dy, dz):
        """Translate the transform by the given offsets"""
        self.position.move(dx, dy, dz)
    
    def rotate(self, dx, dy, dz):
        """Rotate the transform by the given angles (in degrees)"""
        self.rotation.x += dx
        self.rotation.y += dy
        self.rotation.z += dz
    
    def scale_by(self, sx, sy, sz):
        """Scale the transform by the given factors"""
        self.scale.x *= sx
        self.scale.y *= sy
        self.scale.z *= sz
    
    def get_matrix(self):
        """Get transformation matrix (simplified version)"""
        # This is a simplified transformation matrix
        # In a real implementation, you'd use proper matrix math
        return {
            'position': [self.position.x, self.position.y, self.position.z],
            'rotation': [self.rotation.x, self.rotation.y, self.rotation.z],
            'scale': [self.scale.x, self.scale.y, self.scale.z]
        }
    
    def __str__(self):
        """String representation of the transform"""
        return f"Transform3D(pos={self.position}, rot={self.rotation}, scale={self.scale})"

class GameObject3D:
    """A basic 3D game object class"""
    
    def __init__(self, name, transform=None):
        """Constructor - initialize a 3D game object"""
        self.name = name
        self.transform = transform or Transform3D()
        self.active = True
        self.visible = True
        self.components = {}
    
    def add_component(self, component_name, component):
        """Add a component to the game object"""
        self.components[component_name] = component
    
    def get_component(self, component_name):
        """Get a component by name"""
        return self.components.get(component_name)
    
    def remove_component(self, component_name):
        """Remove a component by name"""
        if component_name in self.components:
            del self.components[component_name]
    
    def update(self, delta_time):
        """Update the game object (called each frame)"""
        if not self.active:
            return
        
        # Update all components
        for component in self.components.values():
            if hasattr(component, 'update'):
                component.update(delta_time)
    
    def render(self):
        """Render the game object (called each frame)"""
        if not self.active or not self.visible:
            return
        
        # Render all components
        for component in self.components.values():
            if hasattr(component, 'render'):
                component.render()
    
    def __str__(self):
        """String representation of the game object"""
        return f"GameObject3D('{self.name}', active={self.active}, visible={self.visible})"

class MeshRenderer:
    """A component for rendering 3D meshes"""
    
    def __init__(self, mesh_data=None, material=None):
        """Constructor - initialize a mesh renderer"""
        self.mesh_data = mesh_data or {}
        self.material = material or "default"
        self.visible = True
    
    def set_mesh(self, mesh_data):
        """Set the mesh data"""
        self.mesh_data = mesh_data
    
    def set_material(self, material):
        """Set the material"""
        self.material = material
    
    def render(self):
        """Render the mesh"""
        if not self.visible:
            return
        
        # In a real implementation, this would render the mesh
        print(f"   Rendering mesh with material: {self.material}")
    
    def __str__(self):
        """String representation of the mesh renderer"""
        return f"MeshRenderer(material='{self.material}', visible={self.visible})"

class Collider:
    """A component for collision detection"""
    
    def __init__(self, collider_type="box", size=None):
        """Constructor - initialize a collider"""
        self.collider_type = collider_type
        self.size = size or Vector3D(1, 1, 1)
        self.enabled = True
    
    def check_collision(self, other_collider):
        """Check collision with another collider"""
        if not self.enabled or not other_collider.enabled:
            return False
        
        # Simplified collision detection
        # In a real implementation, this would be more complex
        return True
    
    def get_bounds(self):
        """Get the bounding box of the collider"""
        return {
            'type': self.collider_type,
            'size': [self.size.x, self.size.y, self.size.z]
        }
    
    def __str__(self):
        """String representation of the collider"""
        return f"Collider(type='{self.collider_type}', size={self.size}, enabled={self.enabled})"

def demonstrate_basic_classes():
    """Demonstrate basic class usage"""
    print("=== Basic Classes Demonstration ===\n")
    
    # 1. Point3D class
    print("1. Point3D Class:")
    point1 = Point3D(1, 2, 3)
    point2 = Point3D(4, 5, 6)
    
    print(f"   Point 1: {point1}")
    print(f"   Point 2: {point2}")
    print(f"   Distance between points: {point1.distance_to(point2):.2f}")
    
    point1.move(1, 1, 1)
    print(f"   After moving point1: {point1}")
    
    # 2. Vector3D class
    print("\n2. Vector3D Class:")
    vec1 = Vector3D(3, 4, 0)
    vec2 = Vector3D(1, 0, 0)
    
    print(f"   Vector 1: {vec1}")
    print(f"   Vector 2: {vec2}")
    print(f"   Magnitude of vec1: {vec1.magnitude():.2f}")
    print(f"   Normalized vec1: {vec1.normalize()}")
    print(f"   Dot product: {vec1.dot_product(vec2):.2f}")
    print(f"   Cross product: {vec1.cross_product(vec2)}")
    print(f"   Vector addition: {vec1 + vec2}")
    print(f"   Vector subtraction: {vec1 - vec2}")
    print(f"   Scalar multiplication: {vec1 * 2}")
    
    # 3. Transform3D class
    print("\n3. Transform3D Class:")
    transform = Transform3D(
        position=Point3D(10, 20, 30),
        rotation=Vector3D(45, 90, 0),
        scale=Vector3D(2, 1, 1)
    )
    
    print(f"   Transform: {transform}")
    print(f"   Matrix: {transform.get_matrix()}")
    
    transform.translate(5, 0, 0)
    transform.rotate(0, 45, 0)
    transform.scale_by(1.5, 1, 1)
    
    print(f"   After transformations: {transform}")
    
    # 4. GameObject3D class
    print("\n4. GameObject3D Class:")
    game_object = GameObject3D("Player")
    
    # Add components
    mesh_renderer = MeshRenderer(material="player_material")
    collider = Collider("capsule", Vector3D(0.5, 1.8, 0.5))
    
    game_object.add_component("renderer", mesh_renderer)
    game_object.add_component("collider", collider)
    
    print(f"   Game Object: {game_object}")
    print(f"   Components: {list(game_object.components.keys())}")
    
    # Update and render
    game_object.update(0.016)  # 60 FPS
    game_object.render()
    
    # Get component
    renderer = game_object.get_component("renderer")
    print(f"   Renderer component: {renderer}")

def demonstrate_object_creation():
    """Demonstrate different ways to create objects"""
    print("\n=== Object Creation Patterns ===\n")
    
    # 1. Basic object creation
    print("1. Basic Object Creation:")
    point = Point3D(1, 2, 3)
    print(f"   Created point: {point}")
    
    # 2. Object with default values
    print("\n2. Object with Default Values:")
    default_point = Point3D()
    print(f"   Default point: {default_point}")
    
    # 3. Object creation with keyword arguments
    print("\n3. Object Creation with Keywords:")
    transform = Transform3D(
        position=Point3D(0, 0, 0),
        rotation=Vector3D(0, 0, 0),
        scale=Vector3D(1, 1, 1)
    )
    print(f"   Transform: {transform}")
    
    # 4. Object creation and modification
    print("\n4. Object Creation and Modification:")
    game_object = GameObject3D("Enemy")
    game_object.transform.translate(10, 0, 0)
    game_object.transform.rotate(0, 180, 0)
    print(f"   Modified game object: {game_object}")

def demonstrate_method_calls():
    """Demonstrate method calls and object interaction"""
    print("\n=== Method Calls and Object Interaction ===\n")
    
    # 1. Method chaining simulation
    print("1. Method Chaining:")
    point = Point3D(0, 0, 0)
    point.move(1, 0, 0)
    point.move(0, 1, 0)
    point.move(0, 0, 1)
    print(f"   Final position: {point}")
    
    # 2. Object interaction
    print("\n2. Object Interaction:")
    player = GameObject3D("Player")
    enemy = GameObject3D("Enemy")
    
    player.transform.translate(0, 0, 0)
    enemy.transform.translate(5, 0, 0)
    
    # Calculate distance between objects
    distance = player.transform.position.distance_to(enemy.transform.position)
    print(f"   Distance between player and enemy: {distance:.2f}")
    
    # 3. Component interaction
    print("\n3. Component Interaction:")
    player.add_component("renderer", MeshRenderer("player_mesh", "player_material"))
    player.add_component("collider", Collider("capsule"))
    
    renderer = player.get_component("renderer")
    collider = player.get_component("collider")
    
    print(f"   Player renderer: {renderer}")
    print(f"   Player collider: {collider}")
    
    # Check collision between objects
    enemy.add_component("collider", Collider("box"))
    enemy_collider = enemy.get_component("collider")
    
    collision = collider.check_collision(enemy_collider)
    print(f"   Collision detected: {collision}")

def demonstrate_practical_examples():
    """Demonstrate practical 3D graphics examples"""
    print("\n=== Practical 3D Graphics Examples ===\n")
    
    # 1. Camera system
    print("1. Camera System:")
    camera = GameObject3D("MainCamera")
    camera.transform.position = Point3D(0, 5, -10)
    camera.transform.rotation = Vector3D(15, 0, 0)
    
    print(f"   Camera: {camera}")
    print(f"   Camera position: {camera.transform.position}")
    print(f"   Camera rotation: {camera.transform.rotation}")
    
    # 2. Light system
    print("\n2. Light System:")
    light = GameObject3D("DirectionalLight")
    light.transform.position = Point3D(0, 10, 0)
    light.transform.rotation = Vector3D(-45, 0, 0)
    
    light.add_component("light", {
        "type": "directional",
        "color": [1, 1, 1],
        "intensity": 1.0
    })
    
    print(f"   Light: {light}")
    print(f"   Light position: {light.transform.position}")
    
    # 3. Scene setup
    print("\n3. Scene Setup:")
    scene_objects = []
    
    # Create a simple scene
    for i in range(3):
        cube = GameObject3D(f"Cube_{i}")
        cube.transform.position = Point3D(i * 2, 0, 0)
        cube.add_component("renderer", MeshRenderer("cube_mesh", "cube_material"))
        cube.add_component("collider", Collider("box"))
        scene_objects.append(cube)
    
    print(f"   Created {len(scene_objects)} scene objects:")
    for obj in scene_objects:
        print(f"     {obj}")
    
    # 4. Scene update loop simulation
    print("\n4. Scene Update Loop:")
    delta_time = 0.016  # 60 FPS
    
    for i in range(3):
        print(f"   Frame {i+1}:")
        for obj in scene_objects:
            obj.update(delta_time)
            obj.render()

def main():
    """Main function to run all demonstrations"""
    print("=== Python Basic Classes for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_classes()
    demonstrate_object_creation()
    demonstrate_method_calls()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered basic OOP concepts:")
    print("✓ Class definition and object creation")
    print("✓ Constructors and instance methods")
    print("✓ Object attributes and state management")
    print("✓ Method calls and object interaction")
    print("✓ Practical 3D graphics applications")
    
    print("\nKey concepts demonstrated:")
    print("- Point3D: Basic 3D coordinate representation")
    print("- Vector3D: Mathematical vector operations")
    print("- Transform3D: Position, rotation, and scale management")
    print("- GameObject3D: Base class for 3D objects")
    print("- Component system: Modular object architecture")
    print("- Scene management: Multiple object coordination")

if __name__ == "__main__":
    main()
