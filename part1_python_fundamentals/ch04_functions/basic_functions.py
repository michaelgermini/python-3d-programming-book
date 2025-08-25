#!/usr/bin/env python3
"""
Chapter 4: Functions
Basic Functions Example

This example demonstrates basic function definition and usage in Python,
focusing on applications in 3D graphics programming.
"""

import math

def demonstrate_function_basics():
    """Demonstrate basic function concepts"""
    print("=== Basic Function Concepts ===\n")
    
    # 1. Simple function definition
    print("1. Simple Function Definition:")
    
    def greet_player(player_name):
        """Simple function to greet a player"""
        print(f"   Welcome, {player_name}!")
    
    # Call the function
    greet_player("Alice")
    greet_player("Bob")
    
    # 2. Function with return value
    print("\n2. Function with Return Value:")
    
    def calculate_area(length, width):
        """Calculate the area of a rectangle"""
        area = length * width
        return area
    
    # Use the function
    room_area = calculate_area(10, 5)
    print(f"   Room area: {room_area} square units")
    
    # 3. Function with multiple operations
    print("\n3. Function with Multiple Operations:")
    
    def process_object_data(object_name, position, scale):
        """Process 3D object data and return formatted information"""
        # Calculate volume (assuming cube for simplicity)
        volume = scale ** 3
        
        # Create formatted string
        info = f"Object: {object_name}"
        info += f", Position: {position}"
        info += f", Scale: {scale}"
        info += f", Volume: {volume}"
        
        return info
    
    # Test the function
    object_info = process_object_data("Cube", [0, 0, 0], 2.0)
    print(f"   {object_info}")

def demonstrate_3d_functions():
    """Demonstrate functions for 3D graphics"""
    print("\n=== 3D Graphics Functions ===\n")
    
    # 1. Distance calculation function
    print("1. Distance Calculation:")
    
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        dz = point2[2] - point1[2]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        return distance
    
    # Test distance calculation
    camera_pos = [0, 0, 5]
    object_pos = [3, 4, 0]
    
    distance = calculate_distance(camera_pos, object_pos)
    print(f"   Distance from camera to object: {distance:.2f}")
    
    # 2. Vector operations
    print("\n2. Vector Operations:")
    
    def add_vectors(v1, v2):
        """Add two 3D vectors"""
        return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
    
    def scale_vector(vector, scalar):
        """Scale a 3D vector by a scalar"""
        return [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
    
    def normalize_vector(vector):
        """Normalize a 3D vector (make it unit length)"""
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        if magnitude == 0:
            return [0, 0, 0]
        return [vector[0]/magnitude, vector[1]/magnitude, vector[2]/magnitude]
    
    # Test vector operations
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    
    sum_vector = add_vectors(v1, v2)
    scaled_vector = scale_vector(v1, 2.0)
    normalized_vector = normalize_vector(v1)
    
    print(f"   Vector 1: {v1}")
    print(f"   Vector 2: {v2}")
    print(f"   Sum: {sum_vector}")
    print(f"   Scaled (2x): {scaled_vector}")
    print(f"   Normalized: {[f'{x:.3f}' for x in normalized_vector]}")
    
    # 3. Object transformation functions
    print("\n3. Object Transformations:")
    
    def translate_object(position, translation):
        """Translate an object by a given vector"""
        new_position = add_vectors(position, translation)
        return new_position
    
    def rotate_point_2d(point, angle_degrees, center=[0, 0]):
        """Rotate a 2D point around a center (simplified)"""
        angle_radians = math.radians(angle_degrees)
        
        # Translate to origin
        x = point[0] - center[0]
        y = point[1] - center[1]
        
        # Rotate
        new_x = x * math.cos(angle_radians) - y * math.sin(angle_radians)
        new_y = x * math.sin(angle_radians) + y * math.cos(angle_radians)
        
        # Translate back
        return [new_x + center[0], new_y + center[1]]
    
    def scale_object(position, scale_factor, center=[0, 0, 0]):
        """Scale an object from a center point"""
        # For 3D, we'll use the center as reference
        offset = [position[0] - center[0], position[1] - center[1], position[2] - center[2]]
        scaled_offset = scale_vector(offset, scale_factor)
        return add_vectors(center, scaled_offset)
    
    # Test transformations
    original_pos = [5, 3, 0]
    
    # Translate
    translated_pos = translate_object(original_pos, [2, -1, 0])
    print(f"   Original position: {original_pos}")
    print(f"   Translated: {translated_pos}")
    
    # Rotate (2D)
    rotated_pos = rotate_point_2d([5, 3], 45)
    print(f"   Rotated 45°: {[f'{x:.2f}' for x in rotated_pos]}")
    
    # Scale
    scaled_pos = scale_object(original_pos, 2.0)
    print(f"   Scaled 2x: {scaled_pos}")

def demonstrate_game_functions():
    """Demonstrate functions for game development"""
    print("\n=== Game Development Functions ===\n")
    
    # 1. Player status functions
    print("1. Player Status Functions:")
    
    def check_player_health(health, max_health=100):
        """Check player health status"""
        if health <= 0:
            return "dead"
        elif health < max_health * 0.25:
            return "critical"
        elif health < max_health * 0.5:
            return "wounded"
        elif health < max_health * 0.75:
            return "injured"
        else:
            return "healthy"
    
    def calculate_experience_needed(level):
        """Calculate experience needed for next level"""
        return level * 100
    
    def can_level_up(current_exp, current_level):
        """Check if player can level up"""
        exp_needed = calculate_experience_needed(current_level)
        return current_exp >= exp_needed
    
    # Test player functions
    player_health = 35
    player_exp = 250
    player_level = 2
    
    health_status = check_player_health(player_health)
    exp_needed = calculate_experience_needed(player_level)
    can_level = can_level_up(player_exp, player_level)
    
    print(f"   Player health: {player_health} ({health_status})")
    print(f"   Experience: {player_exp}/{exp_needed}")
    print(f"   Can level up: {can_level}")
    
    # 2. Combat functions
    print("\n2. Combat Functions:")
    
    def calculate_damage(base_damage, weapon_multiplier, armor_reduction):
        """Calculate final damage after weapon and armor modifiers"""
        weapon_damage = base_damage * weapon_multiplier
        final_damage = weapon_damage * (1 - armor_reduction)
        return max(1, int(final_damage))  # Minimum 1 damage
    
    def calculate_hit_chance(attacker_accuracy, defender_evasion, distance):
        """Calculate hit chance based on accuracy, evasion, and distance"""
        base_chance = attacker_accuracy - defender_evasion
        distance_penalty = distance * 0.1  # 10% penalty per unit distance
        final_chance = base_chance - distance_penalty
        return max(5, min(95, final_chance))  # Between 5% and 95%
    
    def is_critical_hit(critical_chance):
        """Determine if an attack is a critical hit"""
        import random
        return random.random() < critical_chance
    
    # Test combat functions
    damage = calculate_damage(20, 1.5, 0.3)  # Base 20, sword 1.5x, armor 30%
    hit_chance = calculate_hit_chance(80, 20, 5)  # 80 acc, 20 eva, 5 distance
    is_critical = is_critical_hit(0.15)  # 15% crit chance
    
    print(f"   Damage: {damage}")
    print(f"   Hit chance: {hit_chance}%")
    print(f"   Critical hit: {is_critical}")
    
    # 3. Inventory functions
    print("\n3. Inventory Functions:")
    
    def add_item_to_inventory(inventory, item_name, quantity=1):
        """Add item to inventory"""
        if item_name in inventory:
            inventory[item_name] += quantity
        else:
            inventory[item_name] = quantity
        return inventory
    
    def remove_item_from_inventory(inventory, item_name, quantity=1):
        """Remove item from inventory"""
        if item_name in inventory:
            if inventory[item_name] >= quantity:
                inventory[item_name] -= quantity
                if inventory[item_name] == 0:
                    del inventory[item_name]
                return True
        return False
    
    def has_item(inventory, item_name, quantity=1):
        """Check if inventory has enough of an item"""
        return inventory.get(item_name, 0) >= quantity
    
    # Test inventory functions
    player_inventory = {}
    
    add_item_to_inventory(player_inventory, "health_potion", 3)
    add_item_to_inventory(player_inventory, "sword", 1)
    add_item_to_inventory(player_inventory, "gold", 100)
    
    print(f"   Inventory: {player_inventory}")
    print(f"   Has health potion: {has_item(player_inventory, 'health_potion')}")
    print(f"   Has 5 health potions: {has_item(player_inventory, 'health_potion', 5)}")
    
    remove_item_from_inventory(player_inventory, "health_potion", 1)
    print(f"   After using potion: {player_inventory}")

def demonstrate_utility_functions():
    """Demonstrate utility functions"""
    print("\n=== Utility Functions ===\n")
    
    # 1. String formatting functions
    print("1. String Formatting Functions:")
    
    def format_vector(vector, precision=2):
        """Format a vector for display"""
        formatted = [f"{x:.{precision}f}" for x in vector]
        return f"[{', '.join(formatted)}]"
    
    def format_time(seconds):
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    
    def format_file_size(bytes_size):
        """Format file size in human-readable format"""
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024**2:
            return f"{bytes_size/1024:.1f} KB"
        elif bytes_size < 1024**3:
            return f"{bytes_size/1024**2:.1f} MB"
        else:
            return f"{bytes_size/1024**3:.1f} GB"
    
    # Test formatting functions
    test_vector = [1.234567, 2.345678, 3.456789]
    test_time = 125.7
    test_size = 2048576
    
    print(f"   Vector: {format_vector(test_vector)}")
    print(f"   Time: {format_time(test_time)}")
    print(f"   File size: {format_file_size(test_size)}")
    
    # 2. Validation functions
    print("\n2. Validation Functions:")
    
    def is_valid_position(position):
        """Validate if position is within reasonable bounds"""
        if len(position) != 3:
            return False
        
        x, y, z = position
        return -1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000
    
    def is_valid_color(color):
        """Validate if color values are in valid range"""
        if len(color) != 3:
            return False
        
        r, g, b = color
        return 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    
    def is_valid_filename(filename):
        """Validate filename for common restrictions"""
        if not filename or len(filename) > 255:
            return False
        
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        return not any(char in filename for char in invalid_chars)
    
    # Test validation functions
    test_positions = [[0, 0, 0], [1500, 0, 0], [1, 2]]
    test_colors = [[255, 128, 64], [300, 0, 0], [128, 64]]
    test_filenames = ["texture.png", "file<name.txt", ""]
    
    print("   Position validation:")
    for pos in test_positions:
        print(f"     {pos}: {is_valid_position(pos)}")
    
    print("   Color validation:")
    for color in test_colors:
        print(f"     {color}: {is_valid_color(color)}")
    
    print("   Filename validation:")
    for filename in test_filenames:
        print(f"     '{filename}': {is_valid_filename(filename)}")
    
    # 3. Helper functions
    print("\n3. Helper Functions:")
    
    def clamp(value, min_val, max_val):
        """Clamp a value between min and max"""
        return max(min_val, min(value, max_val))
    
    def lerp(start, end, t):
        """Linear interpolation between start and end values"""
        return start + t * (end - start)
    
    def smooth_step(t):
        """Smooth step function for easing"""
        return t * t * (3 - 2 * t)
    
    # Test helper functions
    test_values = [-5, 0, 50, 150, 200]
    min_val, max_val = 0, 100
    
    print("   Clamping values:")
    for value in test_values:
        clamped = clamp(value, min_val, max_val)
        print(f"     clamp({value}, {min_val}, {max_val}) = {clamped}")
    
    print("   Linear interpolation:")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = lerp(0, 100, t)
        print(f"     lerp(0, 100, {t}) = {interpolated}")
    
    print("   Smooth step:")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        smoothed = smooth_step(t)
        print(f"     smooth_step({t}) = {smoothed:.3f}")

def main():
    """Main function to run all basic function demonstrations"""
    print("=== Python Basic Functions for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_function_basics()
    demonstrate_3d_functions()
    demonstrate_game_functions()
    demonstrate_utility_functions()
    
    print("\n=== Summary ===")
    print("This chapter covered basic function concepts:")
    print("✓ Function definition and calling")
    print("✓ Return values and data processing")
    print("✓ 3D math and vector operations")
    print("✓ Game development utilities")
    print("✓ String formatting and validation")
    print("✓ Helper functions for common tasks")
    
    print("\nFunctions are essential for:")
    print("- Organizing code into reusable modules")
    print("- Creating 3D math and graphics utilities")
    print("- Building game systems and mechanics")
    print("- Data validation and formatting")
    print("- Improving code readability and maintainability")

if __name__ == "__main__":
    main()
