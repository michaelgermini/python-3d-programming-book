#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Exception Handling Example

Demonstrates exception handling, custom exceptions, and error management
with 3D graphics applications.
"""

import math
from typing import List, Dict, Any, Optional

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class GraphicsError(Exception):
    """Base exception for graphics-related errors"""
    pass

class InvalidVectorError(GraphicsError):
    """Raised when vector operations are invalid"""
    pass

class ResourceNotFoundError(GraphicsError):
    """Raised when a resource (texture, mesh, etc.) is not found"""
    pass

class ValidationError(GraphicsError):
    """Raised when data validation fails"""
    pass

# ============================================================================
# VECTOR CLASS WITH EXCEPTION HANDLING
# ============================================================================

class Vector3D:
    """3D Vector class with comprehensive exception handling"""
    
    def __init__(self, x=0, y=0, z=0):
        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Vector coordinates must be numbers: {e}")
    
    def __str__(self):
        return f"Vector3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def magnitude(self):
        """Calculate vector magnitude"""
        try:
            return math.sqrt(self.x**2 + self.y**2 + self.z**2)
        except Exception as e:
            raise GraphicsError(f"Failed to calculate magnitude: {e}")
    
    def normalize(self):
        """Return normalized vector"""
        try:
            mag = self.magnitude()
            if mag == 0:
                raise InvalidVectorError("Cannot normalize zero vector")
            return Vector3D(self.x / mag, self.y / mag, self.z / mag)
        except InvalidVectorError:
            raise
        except Exception as e:
            raise GraphicsError(f"Failed to normalize vector: {e}")
    
    def add(self, other):
        """Add another vector"""
        if not isinstance(other, Vector3D):
            raise InvalidVectorError("Can only add Vector3D objects")
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def multiply(self, scalar):
        """Multiply by scalar"""
        try:
            scalar = float(scalar)
            return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
        except (ValueError, TypeError):
            raise InvalidVectorError("Scalar must be a number")

# ============================================================================
# RESOURCE MANAGER WITH EXCEPTION HANDLING
# ============================================================================

class ResourceManager:
    """Resource manager with exception handling"""
    
    def __init__(self):
        self.textures = {}
        self.load_attempts = 0
        self.max_retries = 3
    
    def load_texture(self, texture_path: str) -> Dict[str, Any]:
        """Load a texture with error handling"""
        if texture_path in self.textures:
            return self.textures[texture_path]
        
        for attempt in range(self.max_retries):
            try:
                # Simulate texture loading
                if not texture_path.endswith(('.png', '.jpg', '.jpeg')):
                    raise ResourceNotFoundError(f"Unsupported texture format: {texture_path}")
                
                if 'error' in texture_path.lower():
                    raise ResourceNotFoundError(f"Texture file not found: {texture_path}")
                
                # Simulate successful loading
                texture_data = {
                    'path': texture_path,
                    'width': 512,
                    'height': 512,
                    'format': 'RGBA8',
                    'is_fallback': False
                }
                
                self.textures[texture_path] = texture_data
                print(f"   Successfully loaded texture: {texture_path}")
                return texture_data
                
            except ResourceNotFoundError as e:
                print(f"   Attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    # Use fallback texture
                    fallback_data = {
                        'path': 'fallback.png',
                        'width': 256,
                        'height': 256,
                        'format': 'RGBA8',
                        'is_fallback': True
                    }
                    self.textures[texture_path] = fallback_data
                    print(f"   Using fallback texture for: {texture_path}")
                    return fallback_data
                
            except Exception as e:
                raise GraphicsError(f"Unexpected error loading texture {texture_path}: {e}")

# ============================================================================
# TRANSFORM CLASS WITH EXCEPTION HANDLING
# ============================================================================

class Transform3D:
    """3D Transform class with exception handling"""
    
    def __init__(self, position=None, rotation=None, scale=None):
        try:
            self.position = position or Vector3D(0, 0, 0)
            self.rotation = rotation or Vector3D(0, 0, 0)
            self.scale = scale or Vector3D(1, 1, 1)
            
            # Validate scale components
            if self.scale.x <= 0 or self.scale.y <= 0 or self.scale.z <= 0:
                raise ValidationError("Scale components must be positive")
                
        except Exception as e:
            raise ValidationError(f"Failed to initialize transform: {e}")
    
    def translate(self, dx, dy, dz):
        """Translate the transform"""
        try:
            dx, dy, dz = float(dx), float(dy), float(dz)
            self.position = self.position.add(Vector3D(dx, dy, dz))
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid translation values: {e}")
        except Exception as e:
            raise GraphicsError(f"Translation failed: {e}")
    
    def scale_by(self, sx, sy, sz):
        """Scale the transform"""
        try:
            sx, sy, sz = float(sx), float(sy), float(sz)
            if sx <= 0 or sy <= 0 or sz <= 0:
                raise ValidationError("Scale factors must be positive")
            
            self.scale = Vector3D(
                self.scale.x * sx,
                self.scale.y * sy,
                self.scale.z * sz
            )
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid scale values: {e}")
        except Exception as e:
            raise GraphicsError(f"Scaling failed: {e}")

# ============================================================================
# GAME OBJECT WITH EXCEPTION HANDLING
# ============================================================================

class GameObject:
    """Game object with comprehensive exception handling"""
    
    def __init__(self, name: str, position=None):
        try:
            if not name or not isinstance(name, str):
                raise ValidationError("Name must be a non-empty string")
            
            self.name = name
            self.transform = Transform3D(position=position or Vector3D(0, 0, 0))
            self.resources = {}
            self.active = True
            self.visible = True
            
        except Exception as e:
            raise GraphicsError(f"Failed to create game object '{name}': {e}")
    
    def load_resource(self, resource_manager: ResourceManager, resource_path: str):
        """Load a resource for this object"""
        try:
            resource_data = resource_manager.load_texture(resource_path)
            self.resources[resource_path] = resource_data
            print(f"   Loaded resource {resource_path} for {self.name}")
        except Exception as e:
            print(f"   Warning: Failed to load resource {resource_path} for {self.name}: {e}")
    
    def update(self, delta_time):
        """Update the game object"""
        try:
            if not self.active:
                return
            
            # Simple update logic
            print(f"   Updating {self.name}")
            
        except Exception as e:
            print(f"   Error updating {self.name}: {e}")
    
    def render(self):
        """Render the game object"""
        try:
            if not self.visible:
                return
            
            print(f"   Rendering {self.name} at {self.transform.position}")
            
        except Exception as e:
            print(f"   Error rendering {self.name}: {e}")

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_vector_exceptions():
    """Demonstrate vector exception handling"""
    print("=== Vector Exception Handling ===\n")
    
    # Valid vector creation
    try:
        v1 = Vector3D(1, 2, 3)
        print(f"Created vector: {v1}")
    except Exception as e:
        print(f"Error creating vector: {e}")
    
    # Invalid vector creation
    try:
        v2 = Vector3D("invalid", "data", "here")
        print(f"Created vector: {v2}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Vector operations
    try:
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        
        # Valid operations
        result = v1.add(v2)
        print(f"Vector addition: {result}")
        
        normalized = v1.normalize()
        print(f"Normalized vector: {normalized}")
        
        # Invalid operations
        try:
            v1.add("not a vector")
        except InvalidVectorError as e:
            print(f"Vector operation error: {e}")
        
        # Zero vector normalization
        zero_vector = Vector3D(0, 0, 0)
        try:
            zero_vector.normalize()
        except InvalidVectorError as e:
            print(f"Normalization error: {e}")
            
    except Exception as e:
        print(f"Vector operation error: {e}")

def demonstrate_resource_exceptions():
    """Demonstrate resource loading exception handling"""
    print("\n=== Resource Exception Handling ===\n")
    
    resource_manager = ResourceManager()
    
    # Valid texture loading
    try:
        texture1 = resource_manager.load_texture("player.png")
        print(f"Loaded texture: {texture1['path']}")
    except Exception as e:
        print(f"Error loading texture: {e}")
    
    # Missing texture (will use fallback)
    try:
        texture2 = resource_manager.load_texture("missing_texture.png")
        print(f"Loaded texture: {texture2['path']}")
        if texture2.get('is_fallback'):
            print("   (Using fallback texture)")
    except Exception as e:
        print(f"Error loading texture: {e}")
    
    # Invalid format
    try:
        texture3 = resource_manager.load_texture("document.txt")
    except ResourceNotFoundError as e:
        print(f"Resource error: {e}")

def demonstrate_transform_exceptions():
    """Demonstrate transform exception handling"""
    print("\n=== Transform Exception Handling ===\n")
    
    # Valid transform
    try:
        transform = Transform3D(
            position=Vector3D(0, 0, 0),
            rotation=Vector3D(0, 0, 0),
            scale=Vector3D(1, 1, 1)
        )
        print("Created valid transform")
    except Exception as e:
        print(f"Error creating transform: {e}")
    
    # Invalid scale
    try:
        transform = Transform3D(scale=Vector3D(0, 1, 1))
    except ValidationError as e:
        print(f"Transform error: {e}")
    
    # Transform operations
    try:
        transform = Transform3D()
        
        # Valid operations
        transform.translate(1, 2, 3)
        transform.scale_by(2, 2, 2)
        print("Transform operations completed successfully")
        
        # Invalid operations
        try:
            transform.translate("invalid", "data", "here")
        except ValidationError as e:
            print(f"Translation error: {e}")
        
        try:
            transform.scale_by(0, 1, 1)  # Zero scale
        except ValidationError as e:
            print(f"Scaling error: {e}")
            
    except Exception as e:
        print(f"Transform operation error: {e}")

def demonstrate_game_object_exceptions():
    """Demonstrate game object exception handling"""
    print("\n=== Game Object Exception Handling ===\n")
    
    # Valid game object
    try:
        obj = GameObject("Player", Vector3D(0, 0, 0))
        print(f"Created game object: {obj.name}")
    except Exception as e:
        print(f"Error creating game object: {e}")
    
    # Invalid name
    try:
        obj = GameObject("", Vector3D(0, 0, 0))
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Game object operations
    try:
        obj = GameObject("TestObject")
        
        # Load resources
        resource_manager = ResourceManager()
        obj.load_resource(resource_manager, "texture.png")
        obj.load_resource(resource_manager, "missing.png")  # Will use fallback
        
        # Update and render
        obj.update(0.016)
        obj.render()
        
    except Exception as e:
        print(f"Game object operation error: {e}")

def demonstrate_exception_hierarchy():
    """Demonstrate exception hierarchy and handling"""
    print("\n=== Exception Hierarchy ===\n")
    
    def test_operation(operation_name: str, operation_func):
        """Test an operation and handle different exception types"""
        try:
            result = operation_func()
            print(f"   {operation_name}: Success")
            return result
        except ValidationError as e:
            print(f"   {operation_name}: Validation error - {e}")
        except ResourceNotFoundError as e:
            print(f"   {operation_name}: Resource error - {e}")
        except InvalidVectorError as e:
            print(f"   {operation_name}: Vector error - {e}")
        except GraphicsError as e:
            print(f"   {operation_name}: Graphics error - {e}")
        except Exception as e:
            print(f"   {operation_name}: Unexpected error - {e}")
        return None
    
    # Test various operations
    test_operation("Valid Vector", lambda: Vector3D(1, 2, 3))
    test_operation("Invalid Vector", lambda: Vector3D("a", "b", "c"))
    test_operation("Zero Vector Normalize", lambda: Vector3D(0, 0, 0).normalize())
    test_operation("Valid Transform", lambda: Transform3D())
    test_operation("Invalid Transform", lambda: Transform3D(scale=Vector3D(0, 1, 1)))
    test_operation("Valid GameObject", lambda: GameObject("Test"))
    test_operation("Invalid GameObject", lambda: GameObject(""))

def main():
    """Main function to run all demonstrations"""
    print("=== Python Exception Handling for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_vector_exceptions()
    demonstrate_resource_exceptions()
    demonstrate_transform_exceptions()
    demonstrate_game_object_exceptions()
    demonstrate_exception_hierarchy()
    
    print("\n=== Summary ===")
    print("This chapter covered exception handling:")
    print("✓ Custom exceptions: GraphicsError hierarchy")
    print("✓ Try-except blocks: Proper error handling")
    print("✓ Exception types: Specific vs general handling")
    print("✓ Resource management: Graceful failure handling")
    print("✓ Validation: Input validation with exceptions")
    print("✓ Error recovery: Fallback mechanisms")
    print("✓ Error propagation: Proper exception flow")
    
    print("\nKey benefits of exception handling:")
    print("- Robust applications: Graceful handling of errors")
    print("- Debugging: Clear error messages and stack traces")
    print("- User experience: Applications don't crash unexpectedly")
    print("- Resource management: Proper cleanup on errors")
    print("- Validation: Input validation with meaningful feedback")
    print("- Recovery: Fallback mechanisms for common failures")

if __name__ == "__main__":
    main()
