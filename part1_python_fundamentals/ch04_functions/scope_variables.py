#!/usr/bin/env python3
"""
Chapter 4: Functions
Scope and Variables Example

This example demonstrates variable scope and lifetime in Python functions,
focusing on applications in 3D graphics programming.
"""

import math

# Global variables
GLOBAL_CONSTANTS = {
    "PI": 3.14159,
    "MAX_OBJECTS": 1000,
    "DEFAULT_COLOR": [255, 255, 255],
    "WORLD_SIZE": 1000.0
}

game_settings = {
    "resolution": [1920, 1080],
    "fullscreen": False,
    "vsync": True
}

def demonstrate_basic_scope():
    """Demonstrate basic variable scope concepts"""
    print("=== Basic Variable Scope ===\n")
    
    # 1. Local variables
    print("1. Local Variables:")
    
    def create_cube(position, size):
        """Create a cube with local variables"""
        # Local variables - only accessible within this function
        vertices = [
            [position[0] - size/2, position[1] - size/2, position[2] - size/2],
            [position[0] + size/2, position[1] - size/2, position[2] - size/2],
            [position[0] + size/2, position[1] + size/2, position[2] - size/2],
            [position[0] - size/2, position[1] + size/2, position[2] - size/2],
            [position[0] - size/2, position[1] - size/2, position[2] + size/2],
            [position[0] + size/2, position[1] - size/2, position[2] + size/2],
            [position[0] + size/2, position[1] + size/2, position[2] + size/2],
            [position[0] - size/2, position[1] + size/2, position[2] + size/2]
        ]
        
        # Calculate volume locally
        volume = size ** 3
        
        return {
            "vertices": vertices,
            "volume": volume,
            "position": position,
            "size": size
        }
    
    cube = create_cube([0, 0, 0], 2.0)
    print(f"   Created cube: volume = {cube['volume']}, vertices = {len(cube['vertices'])}")
    
    # 2. Global variable access
    print("\n2. Global Variable Access:")
    
    def get_world_bounds():
        """Access global constants"""
        # Can read global constants
        world_size = GLOBAL_CONSTANTS["WORLD_SIZE"]
        max_objects = GLOBAL_CONSTANTS["MAX_OBJECTS"]
        
        return {
            "min": [-world_size/2, -world_size/2, -world_size/2],
            "max": [world_size/2, world_size/2, world_size/2],
            "max_objects": max_objects
        }
    
    bounds = get_world_bounds()
    print(f"   World bounds: {bounds}")
    
    # 3. Local vs global variable names
    print("\n3. Local vs Global Variable Names:")
    
    def process_color(color):
        """Demonstrate local variable shadowing global"""
        # Local 'color' shadows any global 'color' variable
        if color is None:
            color = GLOBAL_CONSTANTS["DEFAULT_COLOR"]  # Use global default
        
        # Create a local copy
        processed_color = [c for c in color]
        
        # Apply gamma correction (simplified)
        for i in range(3):
            processed_color[i] = min(255, int(processed_color[i] * 1.2))
        
        return processed_color
    
    original_color = [200, 150, 100]
    processed = process_color(original_color)
    print(f"   Original color: {original_color}")
    print(f"   Processed color: {processed}")

def demonstrate_global_keyword():
    """Demonstrate the global keyword"""
    print("\n=== Global Keyword ===\n")
    
    # 1. Modifying global variables
    print("1. Modifying Global Variables:")
    
    def update_game_settings(new_resolution, fullscreen=None):
        """Update global game settings"""
        global game_settings  # Declare we want to modify the global variable
        
        # Modify global variable
        game_settings["resolution"] = new_resolution
        
        if fullscreen is not None:
            game_settings["fullscreen"] = fullscreen
        
        print(f"   Updated settings: {game_settings}")
    
    # Before modification
    print(f"   Original settings: {game_settings}")
    
    # Modify global settings
    update_game_settings([2560, 1440], True)
    
    # After modification
    print(f"   Current settings: {game_settings}")
    
    # 2. Global counter example
    print("\n2. Global Counter Example:")
    
    # Global counter
    object_counter = 0
    
    def create_object(object_type):
        """Create an object with global counter"""
        global object_counter
        
        object_counter += 1
        object_id = f"{object_type}_{object_counter}"
        
        return {
            "id": object_id,
            "type": object_type,
            "counter": object_counter
        }
    
    # Create several objects
    objects = []
    for obj_type in ["cube", "sphere", "cylinder", "light"]:
        obj = create_object(obj_type)
        objects.append(obj)
        print(f"   Created {obj['id']} (total: {obj['counter']})")
    
    # 3. Global constants modification (not recommended)
    print("\n3. Global Constants Modification (Not Recommended):")
    
    def add_constant(name, value):
        """Add a new constant to global constants (demonstration only)"""
        global GLOBAL_CONSTANTS
        
        if name not in GLOBAL_CONSTANTS:
            GLOBAL_CONSTANTS[name] = value
            print(f"   Added constant: {name} = {value}")
        else:
            print(f"   Constant {name} already exists")
    
    # Add new constants
    add_constant("GRAVITY", 9.81)
    add_constant("LIGHT_SPEED", 299792458)
    
    print(f"   Updated constants: {GLOBAL_CONSTANTS}")

def demonstrate_nonlocal_keyword():
    """Demonstrate the nonlocal keyword"""
    print("\n=== Nonlocal Keyword ===\n")
    
    # 1. Nested function with nonlocal
    print("1. Nested Function with Nonlocal:")
    
    def create_animation_system():
        """Create an animation system with nested functions"""
        # Enclosing scope variables
        animation_count = 0
        active_animations = []
        
        def add_animation(name, duration, start_value, end_value):
            """Add a new animation"""
            nonlocal animation_count, active_animations
            
            animation_count += 1
            animation = {
                "id": animation_count,
                "name": name,
                "duration": duration,
                "start": start_value,
                "end": end_value,
                "progress": 0.0
            }
            
            active_animations.append(animation)
            print(f"   Added animation: {name} (ID: {animation_count})")
            return animation_count
        
        def update_animations(delta_time):
            """Update all active animations"""
            nonlocal active_animations
            
            completed_animations = []
            
            for animation in active_animations:
                animation["progress"] += delta_time / animation["duration"]
                
                if animation["progress"] >= 1.0:
                    animation["progress"] = 1.0
                    completed_animations.append(animation["id"])
                    print(f"   Completed animation: {animation['name']}")
            
            # Remove completed animations
            active_animations = [anim for anim in active_animations 
                               if anim["id"] not in completed_animations]
            
            return len(completed_animations)
        
        def get_animation_count():
            """Get current animation count"""
            return animation_count
        
        # Return the nested functions
        return {
            "add": add_animation,
            "update": update_animations,
            "count": get_animation_count
        }
    
    # Use the animation system
    anim_system = create_animation_system()
    
    # Add animations
    anim_system["add"]("fade_in", 2.0, 0.0, 1.0)
    anim_system["add"]("scale_up", 1.5, 1.0, 2.0)
    
    print(f"   Total animations: {anim_system['count']()}")
    
    # Update animations
    completed = anim_system["update"](1.0)  # Update by 1 second
    print(f"   Completed {completed} animations")
    
    # 2. Closure with nonlocal
    print("\n2. Closure with Nonlocal:")
    
    def create_counter(initial_value=0):
        """Create a counter with nonlocal variable"""
        count = initial_value
        
        def increment(amount=1):
            nonlocal count
            count += amount
            return count
        
        def decrement(amount=1):
            nonlocal count
            count -= amount
            return count
        
        def get_value():
            return count
        
        def reset():
            nonlocal count
            count = initial_value
            return count
        
        return {
            "increment": increment,
            "decrement": decrement,
            "get": get_value,
            "reset": reset
        }
    
    # Use the counter
    counter = create_counter(10)
    
    print(f"   Initial value: {counter['get']()}")
    print(f"   After increment: {counter['increment'](5)}")
    print(f"   After decrement: {counter['decrement'](2)}")
    print(f"   After reset: {counter['reset']()}")

def demonstrate_scope_pitfalls():
    """Demonstrate common scope pitfalls"""
    print("\n=== Scope Pitfalls ===\n")
    
    # 1. Mutable default arguments
    print("1. Mutable Default Arguments:")
    
    def create_object_list(name, position, tags=None):
        """Create object with tags (problematic with mutable default)"""
        if tags is None:  # Better approach
            tags = []
        
        tags.append(name)  # Modifies the list
        
        return {
            "name": name,
            "position": position,
            "tags": tags
        }
    
    # This would be problematic with tags=[] as default
    obj1 = create_object_list("Cube", [0, 0, 0], ["geometry"])
    obj2 = create_object_list("Sphere", [5, 0, 0], ["geometry", "round"])
    
    print(f"   Object 1: {obj1}")
    print(f"   Object 2: {obj2}")
    
    # 2. Late binding in closures
    print("\n2. Late Binding in Closures:")
    
    def create_multipliers():
        """Create multiplier functions (demonstrates late binding issue)"""
        multipliers = []
        
        for i in range(3):
            def multiplier(x, factor=i):  # Capture current value of i
                return x * factor
            multipliers.append(multiplier)
        
        return multipliers
    
    # Create multipliers
    mults = create_multipliers()
    
    print("   Multipliers:")
    for i, mult in enumerate(mults):
        result = mult(5)
        print(f"     multiplier[{i}](5) = {result}")
    
    # 3. Global variable shadowing
    print("\n3. Global Variable Shadowing:")
    
    def process_settings():
        """Demonstrate global variable shadowing"""
        # This creates a local variable, doesn't modify global
        game_settings = {"local": True}
        print(f"   Local game_settings: {game_settings}")
        
        # To modify global, need global keyword
        global game_settings
        game_settings["processed"] = True
        print(f"   Modified global game_settings: {game_settings}")
    
    print(f"   Before processing: {game_settings}")
    process_settings()
    print(f"   After processing: {game_settings}")

def demonstrate_best_practices():
    """Demonstrate scope best practices"""
    print("\n=== Scope Best Practices ===\n")
    
    # 1. Use function parameters instead of globals
    print("1. Function Parameters vs Globals:")
    
    def calculate_physics(position, velocity, gravity=9.81, delta_time=0.016):
        """Calculate physics with parameters instead of globals"""
        # Use parameters instead of accessing globals
        new_velocity = [v + gravity * delta_time for v in velocity]
        new_position = [p + v * delta_time for p, v in zip(position, new_velocity)]
        
        return new_position, new_velocity
    
    # Test physics calculation
    pos = [0, 10, 0]  # Start at height 10
    vel = [5, 0, 0]   # Moving horizontally
    
    for i in range(5):
        pos, vel = calculate_physics(pos, vel)
        print(f"   Step {i+1}: Position = {[f'{x:.2f}' for x in pos]}")
    
    # 2. Use return values instead of modifying globals
    print("\n2. Return Values vs Global Modification:")
    
    def create_scene_manager():
        """Create a scene manager with proper encapsulation"""
        # Private state (not global)
        objects = []
        lights = []
        camera = None
        
        def add_object(obj):
            """Add object to scene"""
            objects.append(obj)
            return len(objects)
        
        def add_light(light):
            """Add light to scene"""
            lights.append(light)
            return len(lights)
        
        def set_camera(cam):
            """Set scene camera"""
            nonlocal camera
            camera = cam
        
        def get_scene_info():
            """Get scene information"""
            return {
                "object_count": len(objects),
                "light_count": len(lights),
                "has_camera": camera is not None
            }
        
        return {
            "add_object": add_object,
            "add_light": add_light,
            "set_camera": set_camera,
            "get_info": get_scene_info
        }
    
    # Use the scene manager
    scene = create_scene_manager()
    
    scene["add_object"]({"name": "Cube", "type": "geometry"})
    scene["add_object"]({"name": "Sphere", "type": "geometry"})
    scene["add_light"]({"type": "point", "position": [0, 10, 0]})
    scene["set_camera"]({"position": [0, 0, 5], "target": [0, 0, 0]})
    
    info = scene["get_info"]()
    print(f"   Scene info: {info}")
    
    # 3. Use constants for configuration
    print("\n3. Constants for Configuration:")
    
    # Define constants at module level
    RENDER_SETTINGS = {
        "MAX_LIGHTS": 8,
        "SHADOW_RESOLUTION": 1024,
        "ANTIALIASING": True,
        "VSYNC": True
    }
    
    def create_renderer(settings=None):
        """Create renderer with settings"""
        if settings is None:
            settings = RENDER_SETTINGS.copy()  # Use copy of constants
        
        return {
            "settings": settings,
            "light_count": 0,
            "shadow_enabled": settings["SHADOW_RESOLUTION"] > 0
        }
    
    renderer = create_renderer()
    print(f"   Renderer: {renderer}")

def main():
    """Main function to run all scope demonstrations"""
    print("=== Python Scope and Variables for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_scope()
    demonstrate_global_keyword()
    demonstrate_nonlocal_keyword()
    demonstrate_scope_pitfalls()
    demonstrate_best_practices()
    
    print("\n=== Summary ===")
    print("This chapter covered variable scope concepts:")
    print("✓ Local vs global variable scope")
    print("✓ Global and nonlocal keywords")
    print("✓ Nested functions and closures")
    print("✓ Common scope pitfalls and solutions")
    print("✓ Best practices for variable management")
    print("✓ Encapsulation and data hiding")
    
    print("\nVariable scope is essential for:")
    print("- Creating modular and maintainable code")
    print("- Avoiding naming conflicts and bugs")
    print("- Implementing proper encapsulation")
    print("- Managing state in complex applications")
    print("- Building robust and reusable functions")

if __name__ == "__main__":
    main()
