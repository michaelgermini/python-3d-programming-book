#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
Basic Conditionals Example

This example demonstrates basic conditional statements (if/elif/else)
in Python, focusing on applications in 3D graphics programming.
"""

def demonstrate_basic_if():
    """Demonstrate basic if statements"""
    print("=== Basic If Statements ===\n")
    
    # 1. Simple if statement
    print("1. Simple If Statement:")
    
    object_distance = 50
    
    if object_distance < 100:
        print(f"   Object at distance {object_distance} is close enough to render")
    
    # 2. If with else
    print("\n2. If-Else Statement:")
    
    player_health = 75
    
    if player_health > 50:
        print(f"   Player health ({player_health}) is good")
    else:
        print(f"   Player health ({player_health}) is low - warning!")
    
    # 3. If-elif-else chain
    print("\n3. If-Elif-Else Chain:")
    
    object_type = "sphere"
    
    if object_type == "cube":
        print("   Rendering cube geometry")
    elif object_type == "sphere":
        print("   Rendering sphere geometry")
    elif object_type == "cylinder":
        print("   Rendering cylinder geometry")
    else:
        print("   Unknown object type - using default geometry")

def demonstrate_comparison_operators():
    """Demonstrate comparison operators"""
    print("\n=== Comparison Operators ===\n")
    
    # 1. Numeric comparisons
    print("1. Numeric Comparisons:")
    
    camera_x = 10
    camera_y = 5
    camera_z = 15
    
    if camera_x > 0:
        print(f"   Camera is on the positive X side (x={camera_x})")
    
    if camera_y <= 10:
        print(f"   Camera Y position is within bounds (y={camera_y})")
    
    if camera_z != 0:
        print(f"   Camera is not at ground level (z={camera_z})")
    
    # 2. String comparisons
    print("\n2. String Comparisons:")
    
    material_name = "metal"
    
    if material_name == "metal":
        print("   Applying metallic shader")
    elif material_name == "wood":
        print("   Applying wooden texture")
    elif material_name == "glass":
        print("   Applying transparent shader")
    
    # 3. List membership
    print("\n3. List Membership:")
    
    visible_objects = ["player", "enemy1", "enemy2", "item1"]
    current_object = "enemy1"
    
    if current_object in visible_objects:
        print(f"   Object '{current_object}' is visible")
    else:
        print(f"   Object '{current_object}' is not visible")

def demonstrate_3d_conditionals():
    """Demonstrate conditionals in 3D context"""
    print("\n=== 3D Graphics Conditionals ===\n")
    
    # 1. Object visibility based on distance
    print("1. Distance-Based Visibility:")
    
    def check_object_visibility(object_position, camera_position, max_distance=100):
        """Check if object should be visible based on distance"""
        # Calculate distance (simplified 2D for example)
        dx = object_position[0] - camera_position[0]
        dy = object_position[1] - camera_position[1]
        distance = (dx**2 + dy**2)**0.5
        
        if distance <= max_distance:
            return True, distance
        else:
            return False, distance
    
    # Test with different objects
    camera_pos = [0, 0]
    objects = [
        {"name": "tree1", "position": [50, 30]},
        {"name": "building1", "position": [150, 80]},
        {"name": "car1", "position": [25, 15]}
    ]
    
    for obj in objects:
        visible, distance = check_object_visibility(obj["position"], camera_pos)
        if visible:
            print(f"   {obj['name']} is visible (distance: {distance:.1f})")
        else:
            print(f"   {obj['name']} is too far (distance: {distance:.1f})")
    
    # 2. Level of detail based on distance
    print("\n2. Level of Detail (LOD):")
    
    def determine_lod_level(distance):
        """Determine level of detail based on distance"""
        if distance < 10:
            return "high"
        elif distance < 50:
            return "medium"
        elif distance < 100:
            return "low"
        else:
            return "none"
    
    test_distances = [5, 25, 75, 150]
    for distance in test_distances:
        lod = determine_lod_level(distance)
        print(f"   Distance {distance}: {lod} detail level")
    
    # 3. Collision detection
    print("\n3. Simple Collision Detection:")
    
    def check_collision(obj1_pos, obj1_size, obj2_pos, obj2_size):
        """Check if two objects are colliding"""
        # Check X axis overlap
        x_overlap = (obj1_pos[0] - obj1_size/2 < obj2_pos[0] + obj2_size/2 and
                    obj1_pos[0] + obj1_size/2 > obj2_pos[0] - obj2_size/2)
        
        # Check Y axis overlap
        y_overlap = (obj1_pos[1] - obj1_size/2 < obj2_pos[1] + obj2_size/2 and
                    obj1_pos[1] + obj1_size/2 > obj2_pos[1] - obj2_size/2)
        
        return x_overlap and y_overlap
    
    # Test collision detection
    player = {"position": [10, 10], "size": 2}
    enemies = [
        {"position": [12, 11], "size": 1.5},
        {"position": [20, 20], "size": 2},
        {"position": [8, 8], "size": 1}
    ]
    
    for i, enemy in enumerate(enemies):
        if check_collision(player["position"], player["size"], 
                         enemy["position"], enemy["size"]):
            print(f"   Collision detected with enemy {i+1}!")
        else:
            print(f"   No collision with enemy {i+1}")

def demonstrate_game_state_conditionals():
    """Demonstrate conditionals for game state management"""
    print("\n=== Game State Conditionals ===\n")
    
    # 1. Player state management
    print("1. Player State Management:")
    
    player_state = {
        "health": 75,
        "energy": 30,
        "position": [10, 5],
        "inventory": ["sword", "shield", "potion"]
    }
    
    # Health status
    if player_state["health"] > 80:
        print("   Player is in excellent condition")
    elif player_state["health"] > 50:
        print("   Player is in good condition")
    elif player_state["health"] > 20:
        print("   Player is wounded - seek healing!")
    else:
        print("   Player is critically injured!")
    
    # Energy check
    if player_state["energy"] < 20:
        print("   Low energy - consider resting")
    
    # Inventory check
    if "potion" in player_state["inventory"]:
        print("   Player has healing potion available")
    
    # 2. Game difficulty adjustment
    print("\n2. Dynamic Difficulty Adjustment:")
    
    def adjust_difficulty(player_score, player_deaths, game_time):
        """Adjust game difficulty based on player performance"""
        if player_score > 1000 and player_deaths < 3:
            return "hard"
        elif player_score > 500 and player_deaths < 5:
            return "medium"
        elif player_deaths > 10:
            return "easy"
        else:
            return "normal"
    
    # Test different scenarios
    scenarios = [
        {"score": 1200, "deaths": 2, "time": 300},
        {"score": 600, "deaths": 4, "time": 200},
        {"score": 200, "deaths": 12, "time": 150}
    ]
    
    for i, scenario in enumerate(scenarios):
        difficulty = adjust_difficulty(scenario["score"], scenario["deaths"], scenario["time"])
        print(f"   Scenario {i+1}: {difficulty} difficulty")

def demonstrate_animation_conditionals():
    """Demonstrate conditionals for animation control"""
    print("\n=== Animation Conditionals ===\n")
    
    # 1. Animation state machine
    print("1. Animation State Machine:")
    
    def determine_animation_state(player_speed, is_jumping, is_attacking):
        """Determine which animation to play"""
        if is_attacking:
            return "attack"
        elif is_jumping:
            return "jump"
        elif player_speed > 0.5:
            return "run"
        elif player_speed > 0:
            return "walk"
        else:
            return "idle"
    
    # Test different states
    states = [
        {"speed": 0, "jumping": False, "attacking": False},
        {"speed": 0.3, "jumping": False, "attacking": False},
        {"speed": 0.8, "jumping": False, "attacking": False},
        {"speed": 0.2, "jumping": True, "attacking": False},
        {"speed": 0, "jumping": False, "attacking": True}
    ]
    
    for i, state in enumerate(states):
        animation = determine_animation_state(state["speed"], state["jumping"], state["attacking"])
        print(f"   State {i+1}: {animation} animation")
    
    # 2. Transition conditions
    print("\n2. Animation Transition Conditions:")
    
    def can_transition_to_animation(current_anim, target_anim, animation_progress):
        """Check if animation transition is allowed"""
        # Some animations can't be interrupted
        uninterruptible = ["attack", "death"]
        
        if current_anim in uninterruptible and animation_progress < 0.8:
            return False
        
        # Some transitions are not allowed
        forbidden_transitions = [
            ("death", "idle"),
            ("death", "walk"),
            ("attack", "jump")
        ]
        
        if (current_anim, target_anim) in forbidden_transitions:
            return False
        
        return True
    
    # Test transitions
    transitions = [
        ("idle", "walk", 0.5),
        ("attack", "idle", 0.3),
        ("attack", "idle", 0.9),
        ("death", "idle", 1.0)
    ]
    
    for current, target, progress in transitions:
        allowed = can_transition_to_animation(current, target, progress)
        print(f"   {current} -> {target} (progress: {progress}): {'Allowed' if allowed else 'Blocked'}")

def main():
    """Main function to run all conditional demonstrations"""
    print("=== Python Basic Conditionals for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_if()
    demonstrate_comparison_operators()
    demonstrate_3d_conditionals()
    demonstrate_game_state_conditionals()
    demonstrate_animation_conditionals()
    
    print("\n=== Summary ===")
    print("This chapter covered basic conditional statements:")
    print("✓ if, elif, else statements for decision making")
    print("✓ Comparison operators (==, !=, <, >, <=, >=, in, not in)")
    print("✓ 3D-specific conditionals (visibility, LOD, collision)")
    print("✓ Game state management with conditionals")
    print("✓ Animation control using conditionals")
    
    print("\nConditional statements are essential for:")
    print("- Making decisions based on game state")
    print("- Controlling object visibility and rendering")
    print("- Implementing collision detection and physics")
    print("- Managing animation states and transitions")
    print("- Creating interactive and responsive applications")

if __name__ == "__main__":
    main()
