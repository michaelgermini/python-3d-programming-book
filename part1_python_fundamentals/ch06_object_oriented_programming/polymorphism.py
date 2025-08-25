#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Polymorphism Example

Demonstrates polymorphic behavior, duck typing, and abstract base classes
with 3D graphics applications.
"""

from abc import ABC, abstractmethod
import math

# Abstract Base Classes

class Renderable(ABC):
    """Abstract base class for renderable objects"""
    
    @abstractmethod
    def render(self):
        """Render the object"""
        pass
    
    @abstractmethod
    def get_bounds(self):
        """Get the bounding box of the object"""
        pass

class Updatable(ABC):
    """Abstract base class for updatable objects"""
    
    @abstractmethod
    def update(self, delta_time):
        """Update the object"""
        pass

class Collidable(ABC):
    """Abstract base class for collidable objects"""
    
    @abstractmethod
    def check_collision(self, other):
        """Check collision with another object"""
        pass
    
    @abstractmethod
    def get_collision_bounds(self):
        """Get collision bounds"""
        pass

# Concrete Classes

class GameObject:
    """Base class for all game objects"""
    
    def __init__(self, name, position=(0, 0, 0)):
        self.name = name
        self.position = list(position)
        self.active = True
        self.visible = True
    
    def __str__(self):
        return f"{self.__class__.__name__}('{self.name}')"

class Cube(GameObject, Renderable, Collidable):
    """Cube class implementing multiple interfaces"""
    
    def __init__(self, name, position=(0, 0, 0), size=1.0):
        GameObject.__init__(self, name, position)
        self.size = size
        self.material = "default"
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering cube '{self.name}' with size {self.size} and material '{self.material}'")
    
    def get_bounds(self):
        half_size = self.size / 2
        return {
            'min': [self.position[0] - half_size, self.position[1] - half_size, self.position[2] - half_size],
            'max': [self.position[0] + half_size, self.position[1] + half_size, self.position[2] + half_size]
        }
    
    def check_collision(self, other):
        if not hasattr(other, 'get_collision_bounds'):
            return False
        
        bounds1 = self.get_collision_bounds()
        bounds2 = other.get_collision_bounds()
        
        # Simple AABB collision detection
        return (bounds1['min'][0] < bounds2['max'][0] and bounds1['max'][0] > bounds2['min'][0] and
                bounds1['min'][1] < bounds2['max'][1] and bounds1['max'][1] > bounds2['min'][1] and
                bounds1['min'][2] < bounds2['max'][2] and bounds1['max'][2] > bounds2['min'][2])
    
    def get_collision_bounds(self):
        return self.get_bounds()

class Sphere(GameObject, Renderable, Collidable):
    """Sphere class implementing multiple interfaces"""
    
    def __init__(self, name, position=(0, 0, 0), radius=1.0):
        GameObject.__init__(self, name, position)
        self.radius = radius
        self.material = "default"
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering sphere '{self.name}' with radius {self.radius} and material '{self.material}'")
    
    def get_bounds(self):
        return {
            'min': [self.position[0] - self.radius, self.position[1] - self.radius, self.position[2] - self.radius],
            'max': [self.position[0] + self.radius, self.position[1] + self.radius, self.position[2] + self.radius]
        }
    
    def check_collision(self, other):
        if not hasattr(other, 'position'):
            return False
        
        # Distance-based collision for spheres
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        other_radius = getattr(other, 'radius', 0)
        return distance < (self.radius + other_radius)
    
    def get_collision_bounds(self):
        return self.get_bounds()

class Character(GameObject, Renderable, Updatable, Collidable):
    """Character class implementing multiple interfaces"""
    
    def __init__(self, name, position=(0, 0, 0), health=100):
        GameObject.__init__(self, name, position)
        self.health = health
        self.max_health = health
        self.speed = 5.0
        self.mesh = "character_mesh"
        self.material = "character_material"
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering character '{self.name}' with mesh '{self.mesh}' and material '{self.material}'")
    
    def update(self, delta_time):
        if not self.active:
            return
        print(f"   Updating character '{self.name}' (health: {self.health}/{self.max_health})")
    
    def get_bounds(self):
        return {
            'min': [self.position[0] - 0.5, self.position[1] - 0.5, self.position[2] - 0.5],
            'max': [self.position[0] + 0.5, self.position[1] + 0.5, self.position[2] + 0.5]
        }
    
    def check_collision(self, other):
        if not hasattr(other, 'get_collision_bounds'):
            return False
        
        bounds1 = self.get_collision_bounds()
        bounds2 = other.get_collision_bounds()
        
        return (bounds1['min'][0] < bounds2['max'][0] and bounds1['max'][0] > bounds2['min'][0] and
                bounds1['min'][1] < bounds2['max'][1] and bounds1['max'][1] > bounds2['min'][1] and
                bounds1['min'][2] < bounds2['max'][2] and bounds1['max'][2] > bounds2['min'][2])
    
    def get_collision_bounds(self):
        return self.get_bounds()
    
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
        print(f"   {self.name} took {damage} damage. Health: {self.health}")

class Light(GameObject, Renderable):
    """Light class implementing renderable interface"""
    
    def __init__(self, name, position=(0, 0, 0), light_type="point", intensity=1.0):
        GameObject.__init__(self, name, position)
        self.light_type = light_type
        self.intensity = intensity
        self.color = [1.0, 1.0, 1.0]
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering {self.light_type} light '{self.name}' with intensity {self.intensity}")
    
    def get_bounds(self):
        # Lights don't have physical bounds
        return {'min': [0, 0, 0], 'max': [0, 0, 0]}

class Camera(GameObject, Updatable):
    """Camera class implementing updatable interface"""
    
    def __init__(self, name, position=(0, 0, 0), target=(0, 0, 0)):
        GameObject.__init__(self, name, position)
        self.target = list(target)
        self.fov = 60.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
    
    def update(self, delta_time):
        if not self.active:
            return
        print(f"   Updating camera '{self.name}' (FOV: {self.fov}°)")
    
    def look_at(self, target):
        self.target = list(target)
        print(f"   Camera '{self.name}' looking at {target}")

# Duck Typing Examples

class Particle:
    """Particle class with duck typing - no explicit interface inheritance"""
    
    def __init__(self, name, position=(0, 0, 0), lifetime=1.0):
        self.name = name
        self.position = list(position)
        self.lifetime = lifetime
        self.age = 0.0
        self.active = True
        self.visible = True
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering particle '{self.name}' (age: {self.age:.2f}/{self.lifetime:.2f})")
    
    def update(self, delta_time):
        if not self.active:
            return
        self.age += delta_time
        if self.age >= self.lifetime:
            self.active = False
        print(f"   Updating particle '{self.name}' (age: {self.age:.2f})")
    
    def get_bounds(self):
        return {
            'min': [self.position[0] - 0.1, self.position[1] - 0.1, self.position[2] - 0.1],
            'max': [self.position[0] + 0.1, self.position[1] + 0.1, self.position[2] + 0.1]
        }

class Effect:
    """Effect class with duck typing"""
    
    def __init__(self, name, position=(0, 0, 0)):
        self.name = name
        self.position = list(position)
        self.active = True
        self.visible = True
    
    def render(self):
        if not self.visible:
            return
        print(f"   Rendering effect '{self.name}'")
    
    def update(self, delta_time):
        if not self.active:
            return
        print(f"   Updating effect '{self.name}'")

def demonstrate_polymorphic_rendering():
    """Demonstrate polymorphic rendering"""
    print("=== Polymorphic Rendering ===\n")
    
    # Create different types of renderable objects
    renderable_objects = [
        Cube("Box1", (0, 0, 0), 2.0),
        Sphere("Ball1", (3, 0, 0), 1.5),
        Character("Hero", (6, 0, 0), 100),
        Light("MainLight", (0, 5, 0), "directional", 1.0),
        Particle("Sparkle", (9, 0, 0), 2.0),
        Effect("Explosion", (12, 0, 0))
    ]
    
    print("Rendering all objects polymorphically:")
    for obj in renderable_objects:
        obj.render()
    
    print(f"\nTotal renderable objects: {len(renderable_objects)}")

def demonstrate_polymorphic_updating():
    """Demonstrate polymorphic updating"""
    print("\n=== Polymorphic Updating ===\n")
    
    # Create different types of updatable objects
    updatable_objects = [
        Character("Hero", (0, 0, 0), 100),
        Camera("MainCamera", (0, 5, -10)),
        Particle("Sparkle", (3, 0, 0), 2.0),
        Effect("Explosion", (6, 0, 0))
    ]
    
    delta_time = 0.016  # 60 FPS
    
    print("Updating all objects polymorphically:")
    for obj in updatable_objects:
        obj.update(delta_time)
    
    print(f"\nTotal updatable objects: {len(updatable_objects)}")

def demonstrate_polymorphic_collision():
    """Demonstrate polymorphic collision detection"""
    print("\n=== Polymorphic Collision Detection ===\n")
    
    # Create different types of collidable objects
    collidable_objects = [
        Cube("Box1", (0, 0, 0), 2.0),
        Sphere("Ball1", (3, 0, 0), 1.5),
        Character("Hero", (6, 0, 0), 100)
    ]
    
    print("Checking collisions between all objects:")
    for i, obj1 in enumerate(collidable_objects):
        for j, obj2 in enumerate(collidable_objects):
            if i != j:
                collision = obj1.check_collision(obj2)
                print(f"   {obj1.name} vs {obj2.name}: {'Collision' if collision else 'No collision'}")

def demonstrate_duck_typing():
    """Demonstrate duck typing"""
    print("\n=== Duck Typing ===\n")
    
    # Objects that have render() and update() methods but don't inherit from abstract classes
    duck_objects = [
        Particle("Particle1", (0, 0, 0), 1.0),
        Effect("Effect1", (2, 0, 0)),
        Cube("Cube1", (4, 0, 0), 1.0),  # This one does inherit, but we treat it the same
        Sphere("Sphere1", (6, 0, 0), 1.0)  # This one too
    ]
    
    print("Duck typing - treating all objects the same way:")
    for obj in duck_objects:
        # We don't care about the type, only that it has the methods we need
        if hasattr(obj, 'render'):
            obj.render()
        if hasattr(obj, 'update'):
            obj.update(0.016)

def demonstrate_interface_consistency():
    """Demonstrate interface consistency"""
    print("\n=== Interface Consistency ===\n")
    
    # Test that all renderable objects have the required methods
    renderable_objects = [
        Cube("Box1", (0, 0, 0), 1.0),
        Sphere("Ball1", (2, 0, 0), 1.0),
        Character("Hero", (4, 0, 0), 100),
        Light("Light1", (6, 0, 0), "point", 1.0)
    ]
    
    print("Checking interface consistency:")
    for obj in renderable_objects:
        if isinstance(obj, Renderable):
            bounds = obj.get_bounds()
            print(f"   {obj.name}: {type(obj).__name__} implements Renderable interface")
            print(f"     Bounds: {bounds}")

def demonstrate_polymorphic_behavior():
    """Demonstrate polymorphic behavior in game scenarios"""
    print("\n=== Polymorphic Behavior in Game Scenarios ===\n")
    
    # Create a game scene with different object types
    scene_objects = [
        Cube("Ground", (0, -1, 0), 10.0),
        Sphere("Ball", (0, 1, 0), 0.5),
        Character("Player", (2, 0, 0), 100),
        Character("Enemy", (-2, 0, 0), 80),
        Light("Sun", (0, 10, 0), "directional", 1.0),
        Camera("MainCamera", (0, 5, -10)),
        Particle("Dust", (1, 0, 0), 1.0)
    ]
    
    # Simulate a game frame
    delta_time = 0.016
    
    print("Game frame simulation:")
    print("1. Updating all objects:")
    for obj in scene_objects:
        if hasattr(obj, 'update'):
            obj.update(delta_time)
    
    print("\n2. Rendering all visible objects:")
    for obj in scene_objects:
        if hasattr(obj, 'render'):
            obj.render()
    
    print("\n3. Checking collisions:")
    for i, obj1 in enumerate(scene_objects):
        for j, obj2 in enumerate(scene_objects):
            if i != j and hasattr(obj1, 'check_collision') and hasattr(obj2, 'check_collision'):
                collision = obj1.check_collision(obj2)
                if collision:
                    print(f"   Collision detected: {obj1.name} vs {obj2.name}")

def demonstrate_abstract_base_classes():
    """Demonstrate abstract base class usage"""
    print("\n=== Abstract Base Classes ===\n")
    
    # Try to instantiate abstract classes (this would raise an error)
    print("Abstract base classes cannot be instantiated:")
    print("   Renderable() - Would raise TypeError")
    print("   Updatable() - Would raise TypeError")
    print("   Collidable() - Would raise TypeError")
    
    print("\nConcrete classes that implement abstract interfaces:")
    concrete_objects = [
        Cube("Box", (0, 0, 0), 1.0),
        Sphere("Sphere", (2, 0, 0), 1.0),
        Character("Hero", (4, 0, 0), 100)
    ]
    
    for obj in concrete_objects:
        print(f"   {type(obj).__name__} implements:")
        if isinstance(obj, Renderable):
            print("     - Renderable")
        if isinstance(obj, Updatable):
            print("     - Updatable")
        if isinstance(obj, Collidable):
            print("     - Collidable")

def main():
    """Main function to run all demonstrations"""
    print("=== Python Polymorphism for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_polymorphic_rendering()
    demonstrate_polymorphic_updating()
    demonstrate_polymorphic_collision()
    demonstrate_duck_typing()
    demonstrate_interface_consistency()
    demonstrate_polymorphic_behavior()
    demonstrate_abstract_base_classes()
    
    print("\n=== Summary ===")
    print("This chapter covered polymorphism concepts:")
    print("✓ Polymorphic rendering: Different objects with same render() interface")
    print("✓ Polymorphic updating: Different objects with same update() interface")
    print("✓ Polymorphic collision: Different objects with same collision interface")
    print("✓ Duck typing: Objects with same methods regardless of inheritance")
    print("✓ Abstract base classes: Defining interfaces without implementation")
    print("✓ Interface consistency: Ensuring objects follow expected contracts")
    print("✓ Practical applications: Game scene management with polymorphic behavior")
    
    print("\nKey benefits of polymorphism:")
    print("- Code reusability: Same code works with different object types")
    print("- Extensibility: Easy to add new object types")
    print("- Maintainability: Changes to interface affect all implementations")
    print("- Flexibility: Objects can be treated uniformly")

if __name__ == "__main__":
    main()
