#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
Complex Conditions Example

This example demonstrates complex conditional logic using logical operators
(and, or, not) in Python, focusing on applications in 3D graphics programming.
"""

def demonstrate_logical_operators():
    """Demonstrate logical operators"""
    print("=== Logical Operators ===\n")
    
    # 1. AND operator
    print("1. AND Operator:")
    
    player_position = [10, 5, 0]
    player_health = 75
    player_energy = 30
    
    # Check if player is in safe zone AND has sufficient health
    in_safe_zone = (0 <= player_position[0] <= 20 and 
                   0 <= player_position[1] <= 10 and 
                   player_position[2] == 0)
    
    if in_safe_zone and player_health > 50:
        print("   Player is safe and healthy")
    else:
        print("   Player needs attention")
    
    # 2. OR operator
    print("\n2. OR Operator:")
    
    object_type = "sphere"
    object_size = 5
    
    # Check if object is special (large OR specific type)
    is_special = (object_type in ["sphere", "cube"] or object_size > 10)
    
    if is_special:
        print(f"   {object_type} (size: {object_size}) is special")
    else:
        print(f"   {object_type} (size: {object_size}) is normal")
    
    # 3. NOT operator
    print("\n3. NOT Operator:")
    
    is_visible = True
    is_selected = False
    
    # Check if object should be highlighted (visible but not selected)
    should_highlight = is_visible and not is_selected
    
    if should_highlight:
        print("   Object should be highlighted")
    else:
        print("   Object should not be highlighted")

def demonstrate_complex_conditions():
    """Demonstrate complex conditional logic"""
    print("\n=== Complex Conditional Logic ===\n")
    
    # 1. Multiple conditions with parentheses
    print("1. Multiple Conditions with Parentheses:")
    
    def check_render_conditions(object_data, camera_data, settings):
        """Check complex conditions for object rendering"""
        # Extract data
        distance = object_data["distance"]
        is_visible = object_data["visible"]
        object_type = object_data["type"]
        camera_fov = camera_data["fov"]
        max_distance = settings["max_render_distance"]
        
        # Complex condition: object should be rendered if:
        # - it's visible AND
        # - it's within render distance AND
        # - (it's close OR it's an important object type)
        should_render = (is_visible and 
                        distance <= max_distance and
                        (distance < 50 or object_type in ["player", "enemy", "item"]))
        
        return should_render
    
    # Test the function
    test_objects = [
        {"distance": 30, "visible": True, "type": "tree"},
        {"distance": 80, "visible": True, "type": "player"},
        {"distance": 120, "visible": True, "type": "rock"},
        {"distance": 40, "visible": False, "type": "enemy"}
    ]
    
    camera_data = {"fov": 60}
    settings = {"max_render_distance": 100}
    
    for i, obj in enumerate(test_objects):
        should_render = check_render_conditions(obj, camera_data, settings)
        print(f"   Object {i+1} ({obj['type']}): {'Render' if should_render else 'Skip'}")
    
    # 2. Nested conditions
    print("\n2. Nested Conditions:")
    
    def evaluate_collision_response(obj1, obj2, game_state):
        """Evaluate how to respond to a collision"""
        obj1_type = obj1["type"]
        obj2_type = obj2["type"]
        obj1_health = obj1.get("health", 100)
        obj2_health = obj2.get("health", 100)
        
        # Check if either object is invulnerable
        if obj1.get("invulnerable", False) or obj2.get("invulnerable", False):
            return "no_damage"
        
        # Check if it's a player-enemy collision
        if (obj1_type == "player" and obj2_type == "enemy") or \
           (obj1_type == "enemy" and obj2_type == "player"):
            
            # Check if player has shield
            if obj1_type == "player" and obj1.get("has_shield", False):
                return "player_blocked"
            elif obj2_type == "player" and obj2.get("has_shield", False):
                return "player_blocked"
            else:
                return "damage_both"
        
        # Check if it's an item pickup
        elif obj1_type == "item" or obj2_type == "item":
            return "pickup_item"
        
        # Default collision response
        else:
            return "bounce"
    
    # Test collision responses
    collisions = [
        ({"type": "player", "health": 80, "has_shield": True}, 
         {"type": "enemy", "health": 50}),
        ({"type": "player", "health": 60, "has_shield": False}, 
         {"type": "enemy", "health": 40}),
        ({"type": "player", "health": 100}, 
         {"type": "item", "health": 0}),
        ({"type": "rock", "health": 100}, 
         {"type": "tree", "health": 100})
    ]
    
    game_state = {"level": 1}
    
    for i, (obj1, obj2) in enumerate(collisions):
        response = evaluate_collision_response(obj1, obj2, game_state)
        print(f"   Collision {i+1}: {response}")

def demonstrate_3d_complex_conditions():
    """Demonstrate complex conditions in 3D context"""
    print("\n=== 3D Complex Conditions ===\n")
    
    # 1. Frustum culling with multiple conditions
    print("1. Frustum Culling:")
    
    def is_in_frustum(object_pos, camera_pos, camera_rotation, fov, max_distance):
        """Check if object is within camera frustum"""
        # Calculate distance
        dx = object_pos[0] - camera_pos[0]
        dy = object_pos[1] - camera_pos[1]
        dz = object_pos[2] - camera_pos[2]
        distance = (dx**2 + dy**2 + dz**2)**0.5
        
        # Check distance
        if distance > max_distance:
            return False
        
        # Check if object is behind camera (simplified)
        if dz < 0:
            return False
        
        # Check horizontal field of view (simplified)
        horizontal_angle = abs(dx / dz) if dz != 0 else 0
        max_horizontal = math.tan(math.radians(fov / 2))
        
        if horizontal_angle > max_horizontal:
            return False
        
        return True
    
    import math
    
    # Test frustum culling
    camera_data = {
        "position": [0, 0, 0],
        "rotation": 0,
        "fov": 60,
        "max_distance": 100
    }
    
    test_objects = [
        {"name": "near_object", "position": [10, 5, 20]},
        {"name": "far_object", "position": [50, 25, 100]},
        {"name": "behind_camera", "position": [0, 0, -10]},
        {"name": "outside_fov", "position": [100, 0, 50]}
    ]
    
    for obj in test_objects:
        in_frustum = is_in_frustum(obj["position"], 
                                  camera_data["position"],
                                  camera_data["rotation"],
                                  camera_data["fov"],
                                  camera_data["max_distance"])
        print(f"   {obj['name']}: {'In frustum' if in_frustum else 'Culled'}")
    
    # 2. Lighting conditions
    print("\n2. Lighting Conditions:")
    
    def determine_lighting_level(object_pos, light_sources, time_of_day):
        """Determine lighting level for an object"""
        total_light = 0
        
        for light in light_sources:
            # Calculate distance to light
            dx = object_pos[0] - light["position"][0]
            dy = object_pos[1] - light["position"][1]
            dz = object_pos[2] - light["position"][2]
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            # Check if light is on and in range
            if (light["enabled"] and 
                distance <= light["range"] and
                (light["type"] == "ambient" or distance > 0)):
                
                # Calculate light intensity (inverse square law)
                intensity = light["intensity"] / (distance**2 + 1)
                total_light += intensity
        
        # Add ambient lighting based on time of day
        if 6 <= time_of_day <= 18:  # Daytime
            total_light += 0.3
        elif 18 < time_of_day <= 22:  # Evening
            total_light += 0.1
        else:  # Night
            total_light += 0.05
        
        # Determine lighting level
        if total_light > 1.0:
            return "bright"
        elif total_light > 0.5:
            return "normal"
        elif total_light > 0.2:
            return "dim"
        else:
            return "dark"
    
    # Test lighting
    object_position = [10, 5, 0]
    lights = [
        {"position": [0, 10, 0], "intensity": 1.0, "range": 20, "enabled": True, "type": "point"},
        {"position": [20, 0, 0], "intensity": 0.5, "range": 15, "enabled": False, "type": "point"},
        {"position": [0, 0, 0], "intensity": 0.2, "range": 100, "enabled": True, "type": "ambient"}
    ]
    
    times = [12, 20, 2]  # Noon, evening, night
    
    for time in times:
        lighting = determine_lighting_level(object_position, lights, time)
        print(f"   Time {time}:00 - Lighting level: {lighting}")

def demonstrate_game_logic_conditions():
    """Demonstrate complex game logic conditions"""
    print("\n=== Game Logic Conditions ===\n")
    
    # 1. Combat system conditions
    print("1. Combat System:")
    
    def evaluate_attack_success(attacker, defender, weapon, environment):
        """Evaluate if an attack is successful"""
        # Base hit chance
        hit_chance = attacker["accuracy"]
        
        # Modify based on weapon
        if weapon["type"] == "melee":
            # Melee weapons are more accurate at close range
            distance = abs(attacker["position"][0] - defender["position"][0])
            if distance <= 2:
                hit_chance += 0.2
            else:
                hit_chance -= 0.3
        elif weapon["type"] == "ranged":
            # Ranged weapons are less accurate at close range
            distance = abs(attacker["position"][0] - defender["position"][0])
            if distance < 1:
                hit_chance -= 0.4
            elif distance > 10:
                hit_chance -= 0.2
        
        # Modify based on defender's status
        if defender.get("dodging", False):
            hit_chance -= 0.3
        
        if defender.get("shield_up", False):
            hit_chance -= 0.2
        
        # Modify based on environment
        if environment.get("low_visibility", False):
            hit_chance -= 0.1
        
        if environment.get("windy", False) and weapon["type"] == "ranged":
            hit_chance -= 0.15
        
        # Ensure hit chance is within bounds
        hit_chance = max(0.05, min(0.95, hit_chance))
        
        return hit_chance > 0.5  # 50% threshold
    
    # Test combat scenarios
    scenarios = [
        {
            "attacker": {"accuracy": 0.7, "position": [0, 0]},
            "defender": {"position": [1, 0], "dodging": False, "shield_up": False},
            "weapon": {"type": "melee"},
            "environment": {"low_visibility": False, "windy": False}
        },
        {
            "attacker": {"accuracy": 0.6, "position": [0, 0]},
            "defender": {"position": [15, 0], "dodging": True, "shield_up": False},
            "weapon": {"type": "ranged"},
            "environment": {"low_visibility": True, "windy": True}
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        success = evaluate_attack_success(**scenario)
        print(f"   Attack {i+1}: {'Hit' if success else 'Miss'}")
    
    # 2. Quest completion conditions
    print("\n2. Quest Completion:")
    
    def check_quest_completion(quest, player_state, world_state):
        """Check if a quest is completed"""
        if quest["type"] == "collect":
            # Check if player has collected required items
            required_items = quest["requirements"]["items"]
            player_inventory = player_state["inventory"]
            
            for item, count in required_items.items():
                if player_inventory.get(item, 0) < count:
                    return False
            return True
        
        elif quest["type"] == "defeat":
            # Check if required enemies are defeated
            required_enemies = quest["requirements"]["enemies"]
            defeated_enemies = world_state.get("defeated_enemies", [])
            
            for enemy in required_enemies:
                if enemy not in defeated_enemies:
                    return False
            return True
        
        elif quest["type"] == "explore":
            # Check if player has visited required locations
            required_locations = quest["requirements"]["locations"]
            visited_locations = player_state.get("visited_locations", [])
            
            for location in required_locations:
                if location not in visited_locations:
                    return False
            return True
        
        return False
    
    # Test quest completion
    quests = [
        {
            "type": "collect",
            "requirements": {"items": {"gold": 100, "gems": 5}}
        },
        {
            "type": "defeat",
            "requirements": {"enemies": ["dragon", "goblin_chief"]}
        },
        {
            "type": "explore",
            "requirements": {"locations": ["cave", "temple", "village"]}
        }
    ]
    
    player_state = {
        "inventory": {"gold": 150, "gems": 3},
        "visited_locations": ["cave", "temple"]
    }
    
    world_state = {
        "defeated_enemies": ["dragon"]
    }
    
    for i, quest in enumerate(quests):
        completed = check_quest_completion(quest, player_state, world_state)
        print(f"   Quest {i+1} ({quest['type']}): {'Completed' if completed else 'Incomplete'}")

def demonstrate_optimization_conditions():
    """Demonstrate conditions for performance optimization"""
    print("\n=== Performance Optimization Conditions ===\n")
    
    # 1. LOD (Level of Detail) system
    print("1. LOD System:")
    
    def determine_lod_level(object_data, camera_data, performance_settings):
        """Determine appropriate level of detail for an object"""
        distance = object_data["distance"]
        object_type = object_data["type"]
        object_size = object_data["size"]
        
        # Base LOD on distance
        if distance < 10:
            base_lod = "high"
        elif distance < 50:
            base_lod = "medium"
        elif distance < 100:
            base_lod = "low"
        else:
            base_lod = "none"
        
        # Adjust based on object importance
        if object_type in ["player", "enemy", "item"]:
            # Important objects get higher detail
            if base_lod == "low":
                base_lod = "medium"
            elif base_lod == "none":
                base_lod = "low"
        
        # Adjust based on performance settings
        if performance_settings["quality"] == "low":
            if base_lod == "high":
                base_lod = "medium"
            elif base_lod == "medium":
                base_lod = "low"
        
        # Adjust based on object size
        if object_size > 10 and base_lod != "none":
            # Large objects get lower detail to save performance
            if base_lod == "high":
                base_lod = "medium"
            elif base_lod == "medium":
                base_lod = "low"
        
        return base_lod
    
    # Test LOD system
    objects = [
        {"distance": 5, "type": "player", "size": 2},
        {"distance": 30, "type": "tree", "size": 15},
        {"distance": 80, "type": "enemy", "size": 2},
        {"distance": 150, "type": "rock", "size": 5}
    ]
    
    camera_data = {"position": [0, 0, 0]}
    settings = {"quality": "high"}
    
    for obj in objects:
        lod = determine_lod_level(obj, camera_data, settings)
        print(f"   {obj['type']} (distance: {obj['distance']}): {lod} LOD")
    
    # 2. Culling optimization
    print("\n2. Culling Optimization:")
    
    def should_cull_object(object_data, camera_data, culling_settings):
        """Determine if an object should be culled"""
        # Distance culling
        if object_data["distance"] > culling_settings["max_distance"]:
            return True, "distance"
        
        # Frustum culling (simplified)
        if not object_data.get("in_frustum", True):
            return True, "frustum"
        
        # Occlusion culling (simplified)
        if object_data.get("occluded", False):
            return True, "occlusion"
        
        # Size-based culling for very small objects
        if (object_data["size"] < culling_settings["min_size"] and 
            object_data["distance"] > culling_settings["small_object_distance"]):
            return True, "size"
        
        # Type-based culling
        if object_data["type"] in culling_settings["cull_types"]:
            return True, "type"
        
        return False, "none"
    
    # Test culling
    test_objects = [
        {"distance": 150, "size": 5, "type": "tree", "in_frustum": True, "occluded": False},
        {"distance": 50, "size": 0.1, "type": "particle", "in_frustum": True, "occluded": False},
        {"distance": 30, "size": 10, "type": "building", "in_frustum": False, "occluded": False},
        {"distance": 20, "size": 2, "type": "player", "in_frustum": True, "occluded": True}
    ]
    
    culling_settings = {
        "max_distance": 100,
        "min_size": 0.5,
        "small_object_distance": 30,
        "cull_types": ["debug", "invisible"]
    }
    
    for obj in test_objects:
        should_cull, reason = should_cull_object(obj, {}, culling_settings)
        print(f"   {obj['type']}: {'Cull' if should_cull else 'Render'} ({reason})")

def main():
    """Main function to run all complex condition demonstrations"""
    print("=== Python Complex Conditions for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_logical_operators()
    demonstrate_complex_conditions()
    demonstrate_3d_complex_conditions()
    demonstrate_game_logic_conditions()
    demonstrate_optimization_conditions()
    
    print("\n=== Summary ===")
    print("This chapter covered complex conditional logic:")
    print("✓ Logical operators (and, or, not)")
    print("✓ Complex conditions with parentheses")
    print("✓ Nested and compound conditions")
    print("✓ 3D-specific complex conditions (frustum culling, lighting)")
    print("✓ Game logic with multiple conditions")
    print("✓ Performance optimization conditions")
    
    print("\nComplex conditions are essential for:")
    print("- Creating sophisticated game logic and AI")
    print("- Implementing advanced rendering techniques")
    print("- Optimizing performance with smart culling")
    print("- Building realistic physics and collision systems")
    print("- Creating dynamic and responsive applications")

if __name__ == "__main__":
    main()
