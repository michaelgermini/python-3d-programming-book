#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Special Methods Example

Demonstrates magic methods and operator overloading with 3D graphics applications.
"""

import math

class Vector3D:
    """3D Vector class with operator overloading"""
    
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    # String representation methods
    def __str__(self):
        """String representation for display"""
        return f"Vector3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def __repr__(self):
        """Detailed string representation for debugging"""
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
    
    # Comparison methods
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Vector3D):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __lt__(self, other):
        """Less than comparison (by magnitude)"""
        if not isinstance(other, Vector3D):
            return NotImplemented
        return self.magnitude() < other.magnitude()
    
    def __le__(self, other):
        """Less than or equal comparison"""
        return self < other or self == other
    
    # Arithmetic operators
    def __add__(self, other):
        """Vector addition"""
        if not isinstance(other, Vector3D):
            return NotImplemented
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """Vector subtraction"""
        if not isinstance(other, Vector3D):
            return NotImplemented
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        """Vector multiplication by scalar"""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        """Right multiplication (scalar * vector)"""
        return self * scalar
    
    def __truediv__(self, scalar):
        """Vector division by scalar"""
        if not isinstance(scalar, (int, float)) or scalar == 0:
            return NotImplemented
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self):
        """Unary negation"""
        return Vector3D(-self.x, -self.y, -self.z)
    
    def __abs__(self):
        """Absolute value (magnitude)"""
        return self.magnitude()
    
    # Container-like methods
    def __len__(self):
        """Length (always 3 for 3D vector)"""
        return 3
    
    def __getitem__(self, index):
        """Index access"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError("Vector3D index out of range")
    
    def __setitem__(self, index, value):
        """Index assignment"""
        if index == 0:
            self.x = float(value)
        elif index == 1:
            self.y = float(value)
        elif index == 2:
            self.z = float(value)
        else:
            raise IndexError("Vector3D index out of range")
    
    def __iter__(self):
        """Iterator support"""
        yield self.x
        yield self.y
        yield self.z
    
    # Callable object
    def __call__(self, *args):
        """Make vector callable for operations"""
        if len(args) == 0:
            return self.magnitude()
        elif len(args) == 1 and isinstance(args[0], Vector3D):
            return self.dot(args[0])
        else:
            raise TypeError("Vector3D() takes 0 or 1 arguments")
    
    # Utility methods
    def magnitude(self):
        """Calculate vector magnitude"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Return normalized vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return self / mag
    
    def dot(self, other):
        """Dot product with another vector"""
        if not isinstance(other, Vector3D):
            raise TypeError("Dot product requires Vector3D")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Cross product with another vector"""
        if not isinstance(other, Vector3D):
            raise TypeError("Cross product requires Vector3D")
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Transform3D:
    """3D Transform class with special methods"""
    
    def __init__(self, position=None, rotation=None, scale=None):
        self.position = position or Vector3D(0, 0, 0)
        self.rotation = rotation or Vector3D(0, 0, 0)
        self.scale = scale or Vector3D(1, 1, 1)
        self._dirty = True
    
    def __str__(self):
        """String representation"""
        return f"Transform3D(pos={self.position}, rot={self.rotation}, scale={self.scale})"
    
    def __repr__(self):
        """Detailed representation"""
        return f"Transform3D(position={repr(self.position)}, rotation={repr(self.rotation)}, scale={repr(self.scale)})"
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Transform3D):
            return False
        return (self.position == other.position and 
                self.rotation == other.rotation and 
                self.scale == other.scale)
    
    def __mul__(self, other):
        """Transform multiplication (composition)"""
        if not isinstance(other, Transform3D):
            return NotImplemented
        
        # Simplified transform composition
        new_pos = self.position + other.position
        new_rot = self.rotation + other.rotation
        new_scale = Vector3D(
            self.scale.x * other.scale.x,
            self.scale.y * other.scale.y,
            self.scale.z * other.scale.z
        )
        
        return Transform3D(new_pos, new_rot, new_scale)
    
    def __getitem__(self, key):
        """Index access to transform components"""
        if key == 0 or key == "position":
            return self.position
        elif key == 1 or key == "rotation":
            return self.rotation
        elif key == 2 or key == "scale":
            return self.scale
        else:
            raise KeyError("Transform3D key must be 0-2 or 'position', 'rotation', 'scale'")
    
    def __setitem__(self, key, value):
        """Index assignment to transform components"""
        if not isinstance(value, Vector3D):
            raise TypeError("Transform3D values must be Vector3D")
        
        if key == 0 or key == "position":
            self.position = value
        elif key == 1 or key == "rotation":
            self.rotation = value
        elif key == 2 or key == "scale":
            self.scale = value
        else:
            raise KeyError("Transform3D key must be 0-2 or 'position', 'rotation', 'scale'")
        
        self._dirty = True
    
    def __len__(self):
        """Length (always 3 for transform components)"""
        return 3
    
    def __iter__(self):
        """Iterator over transform components"""
        yield self.position
        yield self.rotation
        yield self.scale

class GameObject:
    """Game object with special methods"""
    
    def __init__(self, name, position=(0, 0, 0)):
        self.name = name
        self.transform = Transform3D(position=Vector3D(*position))
        self.active = True
        self.visible = True
    
    def __str__(self):
        """String representation"""
        return f"GameObject('{self.name}', pos={self.transform.position})"
    
    def __repr__(self):
        """Detailed representation"""
        return f"GameObject(name='{self.name}', transform={repr(self.transform)})"
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, GameObject):
            return False
        return (self.name == other.name and 
                self.transform == other.transform and
                self.active == other.active and
                self.visible == other.visible)
    
    def __hash__(self):
        """Hash based on name (for use in sets/dicts)"""
        return hash(self.name)
    
    def __bool__(self):
        """Boolean conversion (True if active)"""
        return self.active
    
    def __len__(self):
        """Length (number of properties)"""
        return len(self.__dict__)
    
    def __getattr__(self, name):
        """Attribute access for transform properties"""
        if name in ['position', 'rotation', 'scale']:
            return getattr(self.transform, name)
        raise AttributeError(f"GameObject has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Attribute assignment for transform properties"""
        if name in ['position', 'rotation', 'scale']:
            if not isinstance(value, Vector3D):
                value = Vector3D(*value)
            setattr(self.transform, name, value)
        else:
            super().__setattr__(name, value)

class Material:
    """Material class with special methods"""
    
    def __init__(self, name, albedo=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5):
        self.name = name
        self.albedo = Vector3D(*albedo)
        self.metallic = float(metallic)
        self.roughness = float(roughness)
    
    def __str__(self):
        """String representation"""
        return f"Material('{self.name}', albedo={self.albedo})"
    
    def __repr__(self):
        """Detailed representation"""
        return f"Material(name='{self.name}', albedo={repr(self.albedo)}, metallic={self.metallic}, roughness={self.roughness})"
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Material):
            return False
        return (self.name == other.name and 
                self.albedo == other.albedo and
                self.metallic == other.metallic and
                self.roughness == other.roughness)
    
    def __hash__(self):
        """Hash based on name"""
        return hash(self.name)
    
    def __add__(self, other):
        """Material blending"""
        if not isinstance(other, Material):
            return NotImplemented
        
        # Blend materials (simplified)
        blended_albedo = (self.albedo + other.albedo) * 0.5
        blended_metallic = (self.metallic + other.metallic) * 0.5
        blended_roughness = (self.roughness + other.roughness) * 0.5
        
        return Material(
            f"blended_{self.name}_{other.name}",
            albedo=blended_albedo,
            metallic=blended_metallic,
            roughness=blended_roughness
        )
    
    def __mul__(self, factor):
        """Material scaling"""
        if not isinstance(factor, (int, float)):
            return NotImplemented
        
        scaled_albedo = self.albedo * factor
        scaled_metallic = self.metallic * factor
        scaled_roughness = self.roughness * factor
        
        return Material(
            f"scaled_{self.name}",
            albedo=scaled_albedo,
            metallic=scaled_metallic,
            roughness=scaled_roughness
        )

def demonstrate_vector_operations():
    """Demonstrate vector operations with special methods"""
    print("=== Vector Operations ===\n")
    
    # Create vectors
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"repr(v1) = {repr(v1)}")
    
    # Arithmetic operations
    print(f"\nArithmetic operations:")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"2 * v1 = {2 * v1}")
    print(f"v1 / 2 = {v1 / 2}")
    print(f"-v1 = {-v1}")
    
    # Comparison operations
    print(f"\nComparison operations:")
    print(f"v1 == v2: {v1 == v2}")
    print(f"v1 < v2: {v1 < v2}")
    print(f"abs(v1) = {abs(v1)}")
    
    # Container-like operations
    print(f"\nContainer-like operations:")
    print(f"len(v1) = {len(v1)}")
    print(f"v1[0] = {v1[0]}")
    print(f"v1[1] = {v1[1]}")
    print(f"v1[2] = {v1[2]}")
    
    # Iterator
    print(f"Vector components: {[x for x in v1]}")
    
    # Callable
    print(f"v1() = {v1()}")
    print(f"v1.dot(v2) = {v1.dot(v2)}")

def demonstrate_transform_operations():
    """Demonstrate transform operations with special methods"""
    print("\n=== Transform Operations ===\n")
    
    # Create transforms
    t1 = Transform3D(
        position=Vector3D(1, 2, 3),
        rotation=Vector3D(45, 0, 0),
        scale=Vector3D(2, 1, 1)
    )
    t2 = Transform3D(
        position=Vector3D(10, 0, 0),
        rotation=Vector3D(0, 90, 0),
        scale=Vector3D(1, 1, 1)
    )
    
    print(f"t1 = {t1}")
    print(f"t2 = {t2}")
    print(f"repr(t1) = {repr(t1)}")
    
    # Transform composition
    print(f"\nTransform composition:")
    print(f"t1 * t2 = {t1 * t2}")
    
    # Index access
    print(f"\nIndex access:")
    print(f"t1[0] = {t1[0]}")
    print(f"t1['position'] = {t1['position']}")
    print(f"t1['rotation'] = {t1['rotation']}")
    print(f"t1['scale'] = {t1['scale']}")
    
    # Iterator
    print(f"Transform components: {[comp for comp in t1]}")

def demonstrate_game_object_operations():
    """Demonstrate game object operations with special methods"""
    print("\n=== Game Object Operations ===\n")
    
    # Create game objects
    obj1 = GameObject("Player", (0, 0, 0))
    obj2 = GameObject("Enemy", (5, 0, 0))
    
    print(f"obj1 = {obj1}")
    print(f"obj2 = {obj2}")
    print(f"repr(obj1) = {repr(obj1)}")
    
    # Boolean conversion
    print(f"\nBoolean conversion:")
    print(f"bool(obj1) = {bool(obj1)}")
    obj1.active = False
    print(f"bool(obj1) after deactivation = {bool(obj1)}")
    obj1.active = True
    
    # Attribute access
    print(f"\nAttribute access:")
    print(f"obj1.position = {obj1.position}")
    print(f"obj1.rotation = {obj1.rotation}")
    print(f"obj1.scale = {obj1.scale}")
    
    # Attribute assignment
    obj1.position = Vector3D(10, 20, 30)
    print(f"obj1.position after assignment = {obj1.position}")
    
    # Length
    print(f"len(obj1) = {len(obj1)}")
    
    # Hash and equality
    obj3 = GameObject("Player", (0, 0, 0))
    print(f"obj1 == obj3: {obj1 == obj3}")
    print(f"hash(obj1) = {hash(obj1)}")

def demonstrate_material_operations():
    """Demonstrate material operations with special methods"""
    print("\n=== Material Operations ===\n")
    
    # Create materials
    metal = Material("Metal", (0.8, 0.8, 0.8), 1.0, 0.2)
    plastic = Material("Plastic", (0.2, 0.2, 0.2), 0.0, 0.8)
    
    print(f"metal = {metal}")
    print(f"plastic = {plastic}")
    print(f"repr(metal) = {repr(metal)}")
    
    # Material blending
    print(f"\nMaterial blending:")
    blended = metal + plastic
    print(f"metal + plastic = {blended}")
    
    # Material scaling
    print(f"\nMaterial scaling:")
    scaled = metal * 0.5
    print(f"metal * 0.5 = {scaled}")
    
    # Equality and hash
    metal2 = Material("Metal", (0.8, 0.8, 0.8), 1.0, 0.2)
    print(f"metal == metal2: {metal == metal2}")
    print(f"hash(metal) = {hash(metal)}")

def demonstrate_collection_operations():
    """Demonstrate objects in collections"""
    print("\n=== Collection Operations ===\n")
    
    # Vectors in a list
    vectors = [Vector3D(1, 0, 0), Vector3D(0, 1, 0), Vector3D(0, 0, 1)]
    print(f"Vectors: {vectors}")
    
    # Sum of vectors
    total = Vector3D(0, 0, 0)
    for v in vectors:
        total += v
    print(f"Sum of vectors: {total}")
    
    # Transforms in a dictionary
    transforms = {
        "player": Transform3D(position=Vector3D(0, 0, 0)),
        "camera": Transform3D(position=Vector3D(0, 5, -10)),
        "light": Transform3D(position=Vector3D(0, 10, 0))
    }
    print(f"Transforms: {transforms}")
    
    # Game objects in a set
    objects = {
        GameObject("Player", (0, 0, 0)),
        GameObject("Enemy", (5, 0, 0)),
        GameObject("Item", (2, 0, 0))
    }
    print(f"Game objects: {[str(obj) for obj in objects]}")

def demonstrate_context_managers():
    """Demonstrate context manager methods"""
    print("\n=== Context Manager Methods ===\n")
    
    class TransformContext:
        """Context manager for transform operations"""
        
        def __init__(self, transform):
            self.transform = transform
            self.original_position = None
        
        def __enter__(self):
            """Enter context - save original state"""
            self.original_position = self.transform.position
            print(f"   Entering context, saved position: {self.original_position}")
            return self.transform
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit context - restore original state"""
            self.transform.position = self.original_position
            print(f"   Exiting context, restored position: {self.transform.position}")
            return False  # Don't suppress exceptions
    
    # Use context manager
    transform = Transform3D(position=Vector3D(0, 0, 0))
    print(f"Original position: {transform.position}")
    
    with TransformContext(transform) as t:
        t.position = Vector3D(10, 20, 30)
        print(f"   Modified position: {t.position}")
    
    print(f"Final position: {transform.position}")

def demonstrate_descriptor_protocol():
    """Demonstrate descriptor protocol"""
    print("\n=== Descriptor Protocol ===\n")
    
    class ValidatedProperty:
        """Descriptor for validated properties"""
        
        def __init__(self, min_val=0, max_val=1):
            self.min_val = min_val
            self.max_val = max_val
            self.name = None
        
        def __set_name__(self, owner, name):
            """Set the name of the attribute"""
            self.name = name
        
        def __get__(self, instance, owner):
            """Get the attribute value"""
            if instance is None:
                return self
            return instance.__dict__.get(self.name, 0)
        
        def __set__(self, instance, value):
            """Set the attribute value with validation"""
            if not isinstance(value, (int, float)):
                raise TypeError(f"{self.name} must be a number")
            if value < self.min_val or value > self.max_val:
                raise ValueError(f"{self.name} must be between {self.min_val} and {self.max_val}")
            instance.__dict__[self.name] = value
    
    class ValidatedMaterial:
        """Material with validated properties using descriptors"""
        
        metallic = ValidatedProperty(0, 1)
        roughness = ValidatedProperty(0, 1)
        
        def __init__(self, name, metallic=0, roughness=0.5):
            self.name = name
            self.metallic = metallic
            self.roughness = roughness
    
    # Use validated material
    material = ValidatedMaterial("Test", metallic=0.8, roughness=0.3)
    print(f"Material: {material.name}, metallic: {material.metallic}, roughness: {material.roughness}")
    
    try:
        material.metallic = 1.5  # Invalid value
    except ValueError as e:
        print(f"   Error: {e}")

def main():
    """Main function to run all demonstrations"""
    print("=== Python Special Methods for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_vector_operations()
    demonstrate_transform_operations()
    demonstrate_game_object_operations()
    demonstrate_material_operations()
    demonstrate_collection_operations()
    demonstrate_context_managers()
    demonstrate_descriptor_protocol()
    
    print("\n=== Summary ===")
    print("This chapter covered special methods:")
    print("✓ String representation: __str__, __repr__")
    print("✓ Comparison operators: __eq__, __lt__, __le__")
    print("✓ Arithmetic operators: __add__, __sub__, __mul__, __truediv__")
    print("✓ Container operations: __len__, __getitem__, __setitem__, __iter__")
    print("✓ Callable objects: __call__")
    print("✓ Context managers: __enter__, __exit__")
    print("✓ Descriptors: __get__, __set__, __set_name__")
    print("✓ Boolean conversion: __bool__")
    print("✓ Hash and equality: __hash__, __eq__")
    
    print("\nKey benefits of special methods:")
    print("- Natural syntax: Objects behave like built-in types")
    print("- Operator overloading: Custom behavior for operators")
    print("- Collection support: Objects work in lists, sets, dicts")
    print("- Context management: Safe resource handling")
    print("- Descriptor protocol: Advanced attribute access control")
    print("- Intuitive interfaces: Objects feel native to Python")

if __name__ == "__main__":
    main()
