#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
Nested Control Example

This example demonstrates combining conditionals and loops in complex scenarios,
focusing on applications in 3D graphics programming and game development.
"""

def demonstrate_nested_conditionals():
    """Demonstrate nested conditional statements"""
    print("=== Nested Conditionals ===\n")
    
    # 1. Complex object state checking
    print("1. Complex Object State Checking:")
    
    def check_object_state(object_data, camera_data):
        """Check complex object state for rendering"""
        # Check if object exists and is valid
        if object_data is None:
            return "invalid_object"
        
        # Check object type and visibility
        if object_data.get("visible", True):
            if object_data["type"] == "player":
                # Player-specific checks
                if object_data.get("health", 100) > 0:
                    if object_data.get("invisible", False):
                        return "player_invisible"
                    else:
                        return "player_visible"
                else:
                    return "player_dead"
            
            elif object_data["type"] == "enemy":
                # Enemy-specific checks
                distance = object_data.get("distance", 0)
                if distance < 50:
                    if object_data.get("aggressive", False):
                        return "enemy_aggressive"
                    else:
                        return "enemy_patrol"
                else:
                    return "enemy_too_far"
            
            elif object_data["type"] == "item":
                # Item-specific checks
                if object_data.get("collected", False):
                    return "item_collected"
                elif object_data.get("glowing", False):
                    return "item_glowing"
                else:
                    return "item_normal"
            
            else:
                return "unknown_type"
        else:
            return "object_hidden"
    
    # Test object state checking
    test_objects = [
        {"type": "player", "visible": True, "health": 75, "invisible": False},
        {"type": "enemy", "visible": True, "distance": 30, "aggressive": True},
        {"type": "item", "visible": True, "collected": False, "glowing": True},
        {"type": "tree", "visible": False},
        None
    ]
    
    camera_data = {"position": [0, 0, 0]}
    
    for i, obj in enumerate(test_objects):
        state = check_object_state(obj, camera_data)
        print(f"   Object {i+1}: {state}")
    
    # 2. Multi-level menu system
    print("\n2. Multi-Level Menu System:")
    
    def process_menu_selection(menu_path, menu_data):
        """Process nested menu selections"""
        current_level = menu_data
        
        for level, selection in menu_path:
            if current_level is None:
                return "invalid_menu"
            
            if level not in current_level:
                return "invalid_level"
            
            if selection not in current_level[level]:
                return "invalid_selection"
            
            current_level = current_level[level][selection]
        
        return current_level
    
    # Test menu system
    menu_structure = {
        "graphics": {
            "quality": {
                "low": "Graphics quality set to low",
                "medium": "Graphics quality set to medium",
                "high": "Graphics quality set to high"
            },
            "resolution": {
                "720p": "Resolution set to 720p",
                "1080p": "Resolution set to 1080p",
                "4k": "Resolution set to 4k"
            }
        },
        "audio": {
            "volume": {
                "mute": "Audio muted",
                "low": "Audio volume low",
                "high": "Audio volume high"
            }
        }
    }
    
    test_selections = [
        [("graphics", "quality"), ("quality", "high")],
        [("graphics", "resolution"), ("resolution", "1080p")],
        [("audio", "volume"), ("volume", "mute")],
        [("invalid", "test")],
        [("graphics", "quality"), ("quality", "invalid")]
    ]
    
    for selection in test_selections:
        result = process_menu_selection(selection, menu_structure)
        print(f"   Selection {selection}: {result}")

def demonstrate_loops_with_nested_conditionals():
    """Demonstrate loops with nested conditional logic"""
    print("\n=== Loops with Nested Conditionals ===\n")
    
    # 1. Complex object filtering and processing
    print("1. Complex Object Filtering:")
    
    def process_scene_objects(objects, camera_pos, game_state):
        """Process objects with complex filtering logic"""
        processed_objects = []
        skipped_objects = []
        
        for obj in objects:
            # Skip if object is invalid
            if obj is None or "type" not in obj:
                skipped_objects.append({"reason": "invalid_object", "object": obj})
                continue
            
            # Calculate distance to camera
            if "position" in obj and camera_pos is not None:
                dx = obj["position"][0] - camera_pos[0]
                dy = obj["position"][1] - camera_pos[1]
                dz = obj["position"][2] - camera_pos[2]
                distance = (dx**2 + dy**2 + dz**2)**0.5
                obj["distance"] = distance
            else:
                obj["distance"] = 0
            
            # Apply type-specific processing
            if obj["type"] == "player":
                if obj.get("health", 100) > 0:
                    if obj.get("invisible", False):
                        skipped_objects.append({"reason": "player_invisible", "object": obj})
                    else:
                        processed_objects.append({"action": "render_player", "object": obj})
                else:
                    skipped_objects.append({"reason": "player_dead", "object": obj})
            
            elif obj["type"] == "enemy":
                if obj["distance"] < 100:  # Render distance
                    if obj.get("aggressive", False):
                        processed_objects.append({"action": "render_aggressive_enemy", "object": obj})
                    else:
                        processed_objects.append({"action": "render_patrol_enemy", "object": obj})
                else:
                    skipped_objects.append({"reason": "enemy_too_far", "object": obj})
            
            elif obj["type"] == "item":
                if not obj.get("collected", False):
                    if obj["distance"] < 50:
                        if obj.get("glowing", False):
                            processed_objects.append({"action": "render_glowing_item", "object": obj})
                        else:
                            processed_objects.append({"action": "render_normal_item", "object": obj})
                    else:
                        skipped_objects.append({"reason": "item_too_far", "object": obj})
                else:
                    skipped_objects.append({"reason": "item_collected", "object": obj})
            
            else:
                # Default processing for unknown types
                if obj.get("visible", True):
                    processed_objects.append({"action": "render_generic", "object": obj})
                else:
                    skipped_objects.append({"reason": "object_hidden", "object": obj})
        
        return processed_objects, skipped_objects
    
    # Test object processing
    test_objects = [
        {"type": "player", "position": [0, 0, 0], "health": 75, "invisible": False},
        {"type": "enemy", "position": [10, 0, 0], "aggressive": True},
        {"type": "item", "position": [5, 0, 0], "collected": False, "glowing": True},
        {"type": "tree", "position": [20, 0, 0], "visible": False},
        {"type": "enemy", "position": [150, 0, 0], "aggressive": False},
        None
    ]
    
    camera_position = [0, 0, 0]
    game_state = {"level": 1}
    
    processed, skipped = process_scene_objects(test_objects, camera_position, game_state)
    
    print(f"   Processed {len(processed)} objects:")
    for item in processed:
        print(f"     {item['action']}: {item['object']['type']}")
    
    print(f"   Skipped {len(skipped)} objects:")
    for item in skipped:
        print(f"     {item['reason']}: {item['object']}")
    
    # 2. Animation state machine
    print("\n2. Animation State Machine:")
    
    def update_animation_states(entities, delta_time):
        """Update animation states for multiple entities"""
        for entity in entities:
            if entity is None or "animation" not in entity:
                continue
            
            current_state = entity["animation"]["current_state"]
            state_time = entity["animation"].get("state_time", 0)
            
            # Update state time
            entity["animation"]["state_time"] = state_time + delta_time
            
            # State-specific logic
            if current_state == "idle":
                # Check for transitions from idle
                if entity.get("moving", False):
                    entity["animation"]["current_state"] = "walk"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: idle -> walk")
                
                elif entity.get("attacking", False):
                    entity["animation"]["current_state"] = "attack"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: idle -> attack")
            
            elif current_state == "walk":
                # Check for transitions from walk
                if not entity.get("moving", False):
                    entity["animation"]["current_state"] = "idle"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: walk -> idle")
                
                elif entity.get("running", False):
                    entity["animation"]["current_state"] = "run"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: walk -> run")
            
            elif current_state == "attack":
                # Attack animations have fixed duration
                attack_duration = entity["animation"].get("attack_duration", 1.0)
                if state_time >= attack_duration:
                    entity["animation"]["current_state"] = "idle"
                    entity["animation"]["state_time"] = 0
                    entity["attacking"] = False
                    print(f"     {entity['name']}: attack -> idle")
            
            elif current_state == "run":
                # Check for transitions from run
                if not entity.get("moving", False):
                    entity["animation"]["current_state"] = "idle"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: run -> idle")
                
                elif not entity.get("running", False):
                    entity["animation"]["current_state"] = "walk"
                    entity["animation"]["state_time"] = 0
                    print(f"     {entity['name']}: run -> walk")
    
    # Test animation state machine
    entities = [
        {
            "name": "player",
            "animation": {"current_state": "idle", "state_time": 0},
            "moving": True,
            "running": False,
            "attacking": False
        },
        {
            "name": "enemy",
            "animation": {"current_state": "walk", "state_time": 0.5},
            "moving": True,
            "running": True,
            "attacking": False
        },
        {
            "name": "boss",
            "animation": {"current_state": "attack", "state_time": 0.8, "attack_duration": 1.0},
            "moving": False,
            "running": False,
            "attacking": True
        }
    ]
    
    update_animation_states(entities, 0.1)

def demonstrate_nested_loops():
    """Demonstrate nested loops with complex logic"""
    print("\n=== Nested Loops ===\n")
    
    # 1. 3D grid processing with conditions
    print("1. 3D Grid Processing:")
    
    def process_3d_grid(grid_size, processing_rules):
        """Process a 3D grid with nested loops and conditions"""
        processed_cells = []
        skipped_cells = []
        
        for x in range(grid_size):
            for y in range(grid_size):
                for z in range(grid_size):
                    cell_data = {
                        "position": [x, y, z],
                        "value": x + y + z,
                        "distance_from_center": ((x - grid_size//2)**2 + 
                                               (y - grid_size//2)**2 + 
                                               (z - grid_size//2)**2)**0.5
                    }
                    
                    # Apply processing rules
                    should_process = True
                    process_reason = "normal"
                    
                    # Check distance rule
                    if "max_distance" in processing_rules:
                        if cell_data["distance_from_center"] > processing_rules["max_distance"]:
                            should_process = False
                            process_reason = "too_far"
                    
                    # Check value rule
                    if "min_value" in processing_rules:
                        if cell_data["value"] < processing_rules["min_value"]:
                            should_process = False
                            process_reason = "value_too_low"
                    
                    # Check position rules
                    if "skip_edges" in processing_rules and processing_rules["skip_edges"]:
                        if (x == 0 or x == grid_size-1 or 
                            y == 0 or y == grid_size-1 or 
                            z == 0 or z == grid_size-1):
                            should_process = False
                            process_reason = "edge_cell"
                    
                    # Process or skip based on rules
                    if should_process:
                        processed_cells.append(cell_data)
                    else:
                        skipped_cells.append({"cell": cell_data, "reason": process_reason})
        
        return processed_cells, skipped_cells
    
    # Test 3D grid processing
    grid_size = 3
    rules = {
        "max_distance": 2.0,
        "min_value": 2,
        "skip_edges": True
    }
    
    processed, skipped = process_3d_grid(grid_size, rules)
    
    print(f"   Processed {len(processed)} cells:")
    for cell in processed[:5]:  # Show first 5
        print(f"     Position {cell['position']}: value {cell['value']}")
    
    print(f"   Skipped {len(skipped)} cells:")
    for item in skipped[:5]:  # Show first 5
        print(f"     Position {item['cell']['position']}: {item['reason']}")
    
    # 2. Matrix operations with nested loops
    print("\n2. Matrix Operations:")
    
    def matrix_operations(matrix_a, matrix_b, operations):
        """Perform matrix operations with nested loops"""
        results = {}
        
        for operation in operations:
            if operation == "addition":
                if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
                    results["addition"] = "incompatible_dimensions"
                else:
                    result_matrix = []
                    for i in range(len(matrix_a)):
                        row = []
                        for j in range(len(matrix_a[0])):
                            row.append(matrix_a[i][j] + matrix_b[i][j])
                        result_matrix.append(row)
                    results["addition"] = result_matrix
            
            elif operation == "multiplication":
                if len(matrix_a[0]) != len(matrix_b):
                    results["multiplication"] = "incompatible_dimensions"
                else:
                    result_matrix = []
                    for i in range(len(matrix_a)):
                        row = []
                        for j in range(len(matrix_b[0])):
                            element = 0
                            for k in range(len(matrix_a[0])):
                                element += matrix_a[i][k] * matrix_b[k][j]
                            row.append(element)
                        result_matrix.append(row)
                    results["multiplication"] = result_matrix
            
            elif operation == "transpose":
                result_matrix = []
                for j in range(len(matrix_a[0])):
                    row = []
                    for i in range(len(matrix_a)):
                        row.append(matrix_a[i][j])
                    result_matrix.append(row)
                results["transpose"] = result_matrix
        
        return results
    
    # Test matrix operations
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    operations = ["addition", "multiplication", "transpose"]
    
    matrix_results = matrix_operations(matrix_a, matrix_b, operations)
    
    for operation, result in matrix_results.items():
        if isinstance(result, list):
            print(f"   {operation.capitalize()}:")
            for row in result:
                print(f"     {row}")
        else:
            print(f"   {operation.capitalize()}: {result}")

def demonstrate_complex_control_structures():
    """Demonstrate complex control structures"""
    print("\n=== Complex Control Structures ===\n")
    
    # 1. Game AI decision tree
    print("1. Game AI Decision Tree:")
    
    def ai_decision_making(ai_state, game_state, player_state):
        """Complex AI decision making with nested control"""
        decisions = []
        
        # Check if AI is alive
        if ai_state["health"] <= 0:
            return ["dead"]
        
        # Check for immediate threats
        if ai_state.get("under_attack", False):
            if ai_state["health"] < 30:
                decisions.append("retreat")
            else:
                decisions.append("defend")
        
        # Check for opportunities
        if player_state.get("health", 100) < 50:
            if ai_state["health"] > 70:
                decisions.append("aggressive_attack")
            else:
                decisions.append("cautious_attack")
        
        # Check for resource management
        if ai_state.get("ammo", 0) < 10:
            if ai_state.get("has_ammo_pickup", False):
                decisions.append("collect_ammo")
            else:
                decisions.append("conserve_ammo")
        
        # Check for tactical positioning
        distance_to_player = ai_state.get("distance_to_player", 100)
        if distance_to_player > 50:
            if ai_state.get("has_cover", False):
                decisions.append("take_cover")
            else:
                decisions.append("advance")
        
        # Check for special abilities
        if ai_state.get("special_ability_ready", False):
            if ai_state["health"] < 50:
                decisions.append("use_healing_ability")
            elif player_state.get("health", 100) > 80:
                decisions.append("use_damage_ability")
        
        # Default behavior if no specific decisions
        if not decisions:
            if ai_state.get("patrol_route", []):
                decisions.append("patrol")
            else:
                decisions.append("idle")
        
        return decisions
    
    # Test AI decision making
    ai_state = {
        "health": 60,
        "ammo": 5,
        "distance_to_player": 30,
        "under_attack": True,
        "has_cover": True,
        "special_ability_ready": True
    }
    
    game_state = {"level": 2}
    player_state = {"health": 40}
    
    ai_decisions = ai_decision_making(ai_state, game_state, player_state)
    print(f"   AI decisions: {ai_decisions}")
    
    # 2. Rendering pipeline simulation
    print("\n2. Rendering Pipeline:")
    
    def simulate_rendering_pipeline(objects, camera, render_settings):
        """Simulate a rendering pipeline with complex control"""
        rendered_objects = []
        culled_objects = []
        
        for obj in objects:
            # Frustum culling
            if not is_in_frustum(obj, camera):
                culled_objects.append({"object": obj, "reason": "frustum_cull"})
                continue
            
            # Distance culling
            distance = calculate_distance(obj["position"], camera["position"])
            if distance > render_settings["max_distance"]:
                culled_objects.append({"object": obj, "reason": "distance_cull"})
                continue
            
            # LOD selection
            lod_level = select_lod_level(distance, obj["type"], render_settings)
            
            # Material processing
            if obj.get("material") == "transparent":
                if render_settings.get("transparency_enabled", True):
                    rendered_objects.append({
                        "object": obj,
                        "lod": lod_level,
                        "render_order": "transparent"
                    })
                else:
                    culled_objects.append({"object": obj, "reason": "transparency_disabled"})
            else:
                rendered_objects.append({
                    "object": obj,
                    "lod": lod_level,
                    "render_order": "opaque"
                })
        
        # Sort by render order
        opaque_objects = [obj for obj in rendered_objects if obj["render_order"] == "opaque"]
        transparent_objects = [obj for obj in rendered_objects if obj["render_order"] == "transparent"]
        
        return opaque_objects, transparent_objects, culled_objects
    
    def is_in_frustum(obj, camera):
        """Simplified frustum culling check"""
        return True  # Simplified for example
    
    def calculate_distance(pos1, pos2):
        """Calculate distance between two positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    def select_lod_level(distance, obj_type, settings):
        """Select level of detail based on distance and type"""
        if obj_type == "player":
            return "high"  # Always high detail for player
        elif distance < 10:
            return "high"
        elif distance < 50:
            return "medium"
        else:
            return "low"
    
    # Test rendering pipeline
    test_objects = [
        {"position": [0, 0, 0], "type": "player", "material": "opaque"},
        {"position": [10, 0, 0], "type": "enemy", "material": "opaque"},
        {"position": [20, 0, 0], "type": "item", "material": "transparent"},
        {"position": [100, 0, 0], "type": "tree", "material": "opaque"}
    ]
    
    camera = {"position": [0, 0, 0]}
    settings = {"max_distance": 80, "transparency_enabled": True}
    
    opaque, transparent, culled = simulate_rendering_pipeline(test_objects, camera, settings)
    
    print(f"   Opaque objects: {len(opaque)}")
    for obj in opaque:
        print(f"     {obj['object']['type']} (LOD: {obj['lod']})")
    
    print(f"   Transparent objects: {len(transparent)}")
    for obj in transparent:
        print(f"     {obj['object']['type']} (LOD: {obj['lod']})")
    
    print(f"   Culled objects: {len(culled)}")
    for item in culled:
        print(f"     {item['object']['type']}: {item['reason']}")

def main():
    """Main function to run all nested control demonstrations"""
    print("=== Python Nested Control for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_nested_conditionals()
    demonstrate_loops_with_nested_conditionals()
    demonstrate_nested_loops()
    demonstrate_complex_control_structures()
    
    print("\n=== Summary ===")
    print("This chapter covered nested control structures:")
    print("✓ Nested conditional statements")
    print("✓ Loops with complex conditional logic")
    print("✓ Nested loops for multi-dimensional processing")
    print("✓ Complex control structures for AI and rendering")
    print("✓ Combining conditionals and loops effectively")
    
    print("\nNested control structures are essential for:")
    print("- Complex game logic and AI decision making")
    print("- Multi-dimensional data processing")
    print("- Rendering pipelines and graphics processing")
    print("- State machines and animation systems")
    print("- Advanced algorithms and simulations")

if __name__ == "__main__":
    main()
