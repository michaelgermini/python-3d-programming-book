#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Encapsulation Example

Demonstrates encapsulation concepts including private attributes, properties,
and data hiding with 3D graphics applications.
"""

import math

class GameObject:
    """Base class demonstrating encapsulation"""
    
    def __init__(self, name, position=(0, 0, 0)):
        # Public attributes
        self.name = name
        
        # Private attributes (convention with underscore)
        self._position = list(position)
        self._active = True
        self._visible = True
        self._dirty = True  # Flag for optimization
    
    # Property decorators for controlled access
    @property
    def position(self):
        """Get the position"""
        return tuple(self._position)
    
    @position.setter
    def position(self, value):
        """Set the position with validation"""
        if len(value) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        self._position = list(value)
        self._dirty = True
    
    @property
    def active(self):
        """Get the active state"""
        return self._active
    
    @active.setter
    def active(self, value):
        """Set the active state"""
        if not isinstance(value, bool):
            raise TypeError("Active must be a boolean")
        self._active = value
    
    @property
    def visible(self):
        """Get the visible state"""
        return self._visible
    
    @visible.setter
    def visible(self, value):
        """Set the visible state"""
        if not isinstance(value, bool):
            raise TypeError("Visible must be a boolean")
        self._visible = value
    
    # Public methods that use private data
    def move(self, dx, dy, dz):
        """Move the object by the given offsets"""
        self._position[0] += dx
        self._position[1] += dy
        self._position[2] += dz
        self._dirty = True
    
    def get_transform_matrix(self):
        """Get transformation matrix (simplified)"""
        if self._dirty:
            # In a real implementation, this would calculate the actual matrix
            self._dirty = False
        return {
            'position': self._position.copy(),
            'active': self._active,
            'visible': self._visible
        }
    
    # Private method (convention with underscore)
    def _update_internal_state(self):
        """Update internal state - should not be called from outside"""
        if self._dirty:
            # Perform internal updates
            pass

class Transform:
    """Transform class with strong encapsulation"""
    
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        # Private attributes
        self._position = list(position)
        self._rotation = list(rotation)
        self._scale = list(scale)
        self._matrix = None
        self._dirty = True
    
    # Properties with validation
    @property
    def position(self):
        """Get position as immutable tuple"""
        return tuple(self._position)
    
    @position.setter
    def position(self, value):
        """Set position with validation"""
        if len(value) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        for coord in value:
            if not isinstance(coord, (int, float)):
                raise TypeError("Position coordinates must be numbers")
        self._position = list(value)
        self._dirty = True
    
    @property
    def rotation(self):
        """Get rotation as immutable tuple"""
        return tuple(self._rotation)
    
    @rotation.setter
    def rotation(self, value):
        """Set rotation with validation"""
        if len(value) != 3:
            raise ValueError("Rotation must have exactly 3 coordinates")
        for angle in value:
            if not isinstance(angle, (int, float)):
                raise TypeError("Rotation angles must be numbers")
        self._rotation = list(value)
        self._dirty = True
    
    @property
    def scale(self):
        """Get scale as immutable tuple"""
        return tuple(self._scale)
    
    @scale.setter
    def scale(self, value):
        """Set scale with validation"""
        if len(value) != 3:
            raise ValueError("Scale must have exactly 3 coordinates")
        for factor in value:
            if not isinstance(factor, (int, float)) or factor <= 0:
                raise TypeError("Scale factors must be positive numbers")
        self._scale = list(value)
        self._dirty = True
    
    # Public methods
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
        """Get the transformation matrix"""
        if self._dirty:
            self._update_matrix()
        return self._matrix.copy() if self._matrix else None
    
    # Private method
    def _update_matrix(self):
        """Update the internal matrix - private method"""
        # Simplified matrix calculation
        self._matrix = {
            'position': self._position.copy(),
            'rotation': self._rotation.copy(),
            'scale': self._scale.copy()
        }
        self._dirty = False

class Material:
    """Material class with encapsulation"""
    
    def __init__(self, name, albedo=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5):
        # Private attributes
        self._name = name
        self._albedo = list(albedo)
        self._metallic = max(0.0, min(1.0, metallic))  # Clamp to [0, 1]
        self._roughness = max(0.0, min(1.0, roughness))  # Clamp to [0, 1]
        self._textures = {}
        self._shader = "pbr"
    
    # Properties with validation
    @property
    def name(self):
        """Get material name"""
        return self._name
    
    @name.setter
    def name(self, value):
        """Set material name"""
        if not isinstance(value, str) or len(value) == 0:
            raise ValueError("Material name must be a non-empty string")
        self._name = value
    
    @property
    def albedo(self):
        """Get albedo color"""
        return tuple(self._albedo)
    
    @albedo.setter
    def albedo(self, value):
        """Set albedo color with validation"""
        if len(value) != 3:
            raise ValueError("Albedo must have exactly 3 components (RGB)")
        for component in value:
            if not isinstance(component, (int, float)) or component < 0 or component > 1:
                raise ValueError("Albedo components must be numbers between 0 and 1")
        self._albedo = list(value)
    
    @property
    def metallic(self):
        """Get metallic value"""
        return self._metallic
    
    @metallic.setter
    def metallic(self, value):
        """Set metallic value with validation"""
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError("Metallic must be a number between 0 and 1")
        self._metallic = value
    
    @property
    def roughness(self):
        """Get roughness value"""
        return self._roughness
    
    @roughness.setter
    def roughness(self, value):
        """Set roughness value with validation"""
        if not isinstance(value, (int, float)) or value < 0 or value > 1:
            raise ValueError("Roughness must be a number between 0 and 1")
        self._roughness = value
    
    # Public methods
    def add_texture(self, texture_type, texture_path):
        """Add a texture to the material"""
        if not isinstance(texture_type, str) or len(texture_type) == 0:
            raise ValueError("Texture type must be a non-empty string")
        if not isinstance(texture_path, str) or len(texture_path) == 0:
            raise ValueError("Texture path must be a non-empty string")
        self._textures[texture_type] = texture_path
    
    def get_texture(self, texture_type):
        """Get a texture by type"""
        return self._textures.get(texture_type)
    
    def get_properties(self):
        """Get all material properties"""
        return {
            'name': self._name,
            'albedo': self._albedo.copy(),
            'metallic': self._metallic,
            'roughness': self._roughness,
            'textures': self._textures.copy(),
            'shader': self._shader
        }

class Character(GameObject):
    """Character class with encapsulation"""
    
    def __init__(self, name, position=(0, 0, 0), health=100):
        super().__init__(name, position)
        
        # Private attributes
        self._health = max(0, health)
        self._max_health = self._health
        self._speed = 5.0
        self._inventory = []
        self._experience = 0
        self._level = 1
    
    # Properties with validation
    @property
    def health(self):
        """Get current health"""
        return self._health
    
    @health.setter
    def health(self, value):
        """Set health with validation"""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Health must be a non-negative number")
        self._health = min(value, self._max_health)
    
    @property
    def max_health(self):
        """Get maximum health"""
        return self._max_health
    
    @max_health.setter
    def max_health(self, value):
        """Set maximum health with validation"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Max health must be a positive number")
        self._max_health = value
        self._health = min(self._health, self._max_health)
    
    @property
    def speed(self):
        """Get movement speed"""
        return self._speed
    
    @speed.setter
    def speed(self, value):
        """Set movement speed with validation"""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Speed must be a non-negative number")
        self._speed = value
    
    @property
    def level(self):
        """Get character level (read-only)"""
        return self._level
    
    @property
    def experience(self):
        """Get experience points (read-only)"""
        return self._experience
    
    # Public methods
    def take_damage(self, damage):
        """Take damage with validation"""
        if not isinstance(damage, (int, float)) or damage < 0:
            raise ValueError("Damage must be a non-negative number")
        
        old_health = self._health
        self._health = max(0, self._health - damage)
        actual_damage = old_health - self._health
        
        print(f"   {self.name} took {actual_damage} damage. Health: {self._health}/{self._max_health}")
        return actual_damage
    
    def heal(self, amount):
        """Heal the character with validation"""
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ValueError("Heal amount must be a non-negative number")
        
        old_health = self._health
        self._health = min(self._max_health, self._health + amount)
        actual_heal = self._health - old_health
        
        print(f"   {self.name} healed {actual_heal}. Health: {self._health}/{self._max_health}")
        return actual_heal
    
    def gain_experience(self, amount):
        """Gain experience points"""
        if not isinstance(amount, (int, float)) or amount < 0:
            raise ValueError("Experience amount must be a non-negative number")
        
        self._experience += amount
        print(f"   {self.name} gained {amount} experience. Total: {self._experience}")
        
        # Check for level up
        required_exp = self._level * 100
        while self._experience >= required_exp:
            self._level_up()
            required_exp = self._level * 100
    
    def add_to_inventory(self, item):
        """Add item to inventory"""
        if not isinstance(item, str) or len(item) == 0:
            raise ValueError("Item must be a non-empty string")
        self._inventory.append(item)
        print(f"   {self.name} picked up {item}")
    
    def get_inventory(self):
        """Get inventory (read-only copy)"""
        return self._inventory.copy()
    
    # Private method
    def _level_up(self):
        """Level up the character - private method"""
        self._level += 1
        self._max_health += 20
        self._health = self._max_health  # Full heal on level up
        print(f"   {self.name} leveled up to level {self._level}!")

class PhysicsBody:
    """Physics body with encapsulation"""
    
    def __init__(self, mass=1.0, collider_type="box"):
        # Private attributes
        self._mass = max(0.1, mass)  # Minimum mass
        self._collider_type = collider_type
        self._velocity = [0.0, 0.0, 0.0]
        self._acceleration = [0.0, 0.0, 0.0]
        self._enabled = True
        self._gravity_affected = True
    
    # Properties
    @property
    def mass(self):
        """Get mass"""
        return self._mass
    
    @mass.setter
    def mass(self, value):
        """Set mass with validation"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Mass must be a positive number")
        self._mass = value
    
    @property
    def velocity(self):
        """Get velocity (read-only)"""
        return tuple(self._velocity)
    
    @property
    def acceleration(self):
        """Get acceleration (read-only)"""
        return tuple(self._acceleration)
    
    @property
    def enabled(self):
        """Get enabled state"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        """Set enabled state"""
        if not isinstance(value, bool):
            raise TypeError("Enabled must be a boolean")
        self._enabled = value
    
    # Public methods
    def apply_force(self, force):
        """Apply force to the body"""
        if len(force) != 3:
            raise ValueError("Force must have exactly 3 components")
        
        if self._enabled:
            # F = ma, so a = F/m
            self._acceleration[0] += force[0] / self._mass
            self._acceleration[1] += force[1] / self._mass
            self._acceleration[2] += force[2] / self._mass
    
    def update(self, delta_time):
        """Update physics body"""
        if not self._enabled:
            return
        
        # Apply gravity if affected
        if self._gravity_affected:
            self._acceleration[1] -= 9.81  # Gravity
        
        # Update velocity: v = v + a * dt
        self._velocity[0] += self._acceleration[0] * delta_time
        self._velocity[1] += self._acceleration[1] * delta_time
        self._velocity[2] += self._acceleration[2] * delta_time
        
        # Reset acceleration
        self._acceleration = [0.0, 0.0, 0.0]
    
    def get_kinetic_energy(self):
        """Calculate kinetic energy: KE = 0.5 * m * v^2"""
        v_squared = sum(v * v for v in self._velocity)
        return 0.5 * self._mass * v_squared

def demonstrate_basic_encapsulation():
    """Demonstrate basic encapsulation concepts"""
    print("=== Basic Encapsulation ===\n")
    
    # Create game object with encapsulated data
    obj = GameObject("TestObject", (1, 2, 3))
    
    print(f"Object name: {obj.name}")
    print(f"Object position: {obj.position}")
    print(f"Object active: {obj.active}")
    print(f"Object visible: {obj.visible}")
    
    # Modify using properties
    obj.position = (5, 10, 15)
    obj.active = False
    obj.visible = False
    
    print(f"\nAfter modification:")
    print(f"Object position: {obj.position}")
    print(f"Object active: {obj.active}")
    print(f"Object visible: {obj.visible}")
    
    # Try to access private attributes (convention only)
    print(f"\nPrivate attributes (convention only):")
    print(f"obj._position: {obj._position}")
    print(f"obj._active: {obj._active}")

def demonstrate_validation():
    """Demonstrate validation in properties"""
    print("\n=== Property Validation ===\n")
    
    # Create transform with validation
    transform = Transform((0, 0, 0), (0, 0, 0), (1, 1, 1))
    
    print("Setting valid values:")
    transform.position = (1, 2, 3)
    transform.rotation = (45, 90, 0)
    transform.scale = (2, 2, 2)
    
    print(f"Position: {transform.position}")
    print(f"Rotation: {transform.rotation}")
    print(f"Scale: {transform.scale}")
    
    # Try invalid values
    print("\nTrying invalid values:")
    try:
        transform.position = (1, 2)  # Wrong number of coordinates
    except ValueError as e:
        print(f"   Error: {e}")
    
    try:
        transform.scale = (0, 1, 1)  # Non-positive scale
    except ValueError as e:
        print(f"   Error: {e}")

def demonstrate_material_encapsulation():
    """Demonstrate material encapsulation"""
    print("\n=== Material Encapsulation ===\n")
    
    # Create material with validation
    material = Material("Metal", (0.8, 0.8, 0.8), 1.0, 0.2)
    
    print(f"Material: {material.name}")
    print(f"Albedo: {material.albedo}")
    print(f"Metallic: {material.metallic}")
    print(f"Roughness: {material.roughness}")
    
    # Modify properties
    material.albedo = (0.9, 0.9, 0.9)
    material.metallic = 0.8
    material.roughness = 0.3
    
    print(f"\nAfter modification:")
    print(f"Albedo: {material.albedo}")
    print(f"Metallic: {material.metallic}")
    print(f"Roughness: {material.roughness}")
    
    # Add textures
    material.add_texture("albedo", "metal_albedo.png")
    material.add_texture("normal", "metal_normal.png")
    
    print(f"\nTextures:")
    print(f"Albedo texture: {material.get_texture('albedo')}")
    print(f"Normal texture: {material.get_texture('normal')}")

def demonstrate_character_encapsulation():
    """Demonstrate character encapsulation"""
    print("\n=== Character Encapsulation ===\n")
    
    # Create character
    character = Character("Hero", (0, 0, 0), 100)
    
    print(f"Character: {character.name}")
    print(f"Health: {character.health}/{character.max_health}")
    print(f"Level: {character.level}")
    print(f"Experience: {character.experience}")
    
    # Take damage and heal
    character.take_damage(30)
    character.heal(20)
    
    # Gain experience and level up
    character.gain_experience(150)
    
    print(f"\nAfter actions:")
    print(f"Health: {character.health}/{character.max_health}")
    print(f"Level: {character.level}")
    print(f"Experience: {character.experience}")
    
    # Add items to inventory
    character.add_to_inventory("Sword")
    character.add_to_inventory("Shield")
    
    print(f"Inventory: {character.get_inventory()}")

def demonstrate_physics_encapsulation():
    """Demonstrate physics body encapsulation"""
    print("\n=== Physics Body Encapsulation ===\n")
    
    # Create physics body
    body = PhysicsBody(mass=10.0, collider_type="sphere")
    
    print(f"Mass: {body.mass}")
    print(f"Velocity: {body.velocity}")
    print(f"Acceleration: {body.acceleration}")
    print(f"Enabled: {body.enabled}")
    
    # Apply force
    body.apply_force((10, 0, 0))  # Force in X direction
    print(f"\nAfter applying force (10, 0, 0):")
    print(f"Acceleration: {body.acceleration}")
    
    # Update physics
    delta_time = 0.016
    body.update(delta_time)
    
    print(f"\nAfter physics update:")
    print(f"Velocity: {body.velocity}")
    print(f"Kinetic energy: {body.get_kinetic_energy():.2f}")

def demonstrate_read_only_properties():
    """Demonstrate read-only properties"""
    print("\n=== Read-Only Properties ===\n")
    
    character = Character("Hero", (0, 0, 0), 100)
    body = PhysicsBody(mass=5.0)
    
    print("Read-only properties:")
    print(f"Character level: {character.level}")
    print(f"Character experience: {character.experience}")
    print(f"Body velocity: {body.velocity}")
    print(f"Body acceleration: {body.acceleration}")
    
    # Try to set read-only properties
    print("\nTrying to set read-only properties:")
    try:
        character.level = 10
    except AttributeError as e:
        print(f"   Error: {e}")
    
    try:
        body.velocity = (1, 0, 0)
    except AttributeError as e:
        print(f"   Error: {e}")

def demonstrate_private_methods():
    """Demonstrate private methods"""
    print("\n=== Private Methods ===\n")
    
    transform = Transform((1, 2, 3), (45, 0, 0), (2, 1, 1))
    
    print("Public interface:")
    print(f"Matrix: {transform.get_matrix()}")
    
    # Try to access private method (convention only)
    print("\nPrivate methods (convention only):")
    try:
        transform._update_matrix()
        print("   Private method called successfully")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Main function to run all demonstrations"""
    print("=== Python Encapsulation for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_encapsulation()
    demonstrate_validation()
    demonstrate_material_encapsulation()
    demonstrate_character_encapsulation()
    demonstrate_physics_encapsulation()
    demonstrate_read_only_properties()
    demonstrate_private_methods()
    
    print("\n=== Summary ===")
    print("This chapter covered encapsulation concepts:")
    print("✓ Private attributes: Data hiding with underscore convention")
    print("✓ Properties: Controlled access to attributes")
    print("✓ Validation: Ensuring data integrity")
    print("✓ Read-only properties: Immutable data access")
    print("✓ Private methods: Internal implementation details")
    print("✓ Public interfaces: Controlled external access")
    print("✓ Data protection: Preventing invalid state")
    
    print("\nKey benefits of encapsulation:")
    print("- Data integrity: Validation prevents invalid states")
    print("- Implementation hiding: Internal details are protected")
    print("- Controlled access: Properties provide validation and logic")
    print("- Maintainability: Changes to internal structure don't affect external code")
    print("- Reusability: Well-defined interfaces make classes easier to use")

if __name__ == "__main__":
    main()
