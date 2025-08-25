#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
While Loops Example

This example demonstrates while loops in Python, focusing on applications
in 3D graphics programming and game development.
"""

def demonstrate_basic_while_loops():
    """Demonstrate basic while loop concepts"""
    print("=== Basic While Loops ===\n")
    
    # 1. Simple counter
    print("1. Simple Counter:")
    
    count = 0
    while count < 5:
        print(f"   Count: {count}")
        count += 1
    
    # 2. User input validation
    print("\n2. Input Validation:")
    
    def get_valid_number(min_val, max_val):
        """Get a valid number from user input"""
        while True:
            try:
                # Simulate user input (in real app, this would be input())
                user_input = "10"  # Simulated input
                number = int(user_input)
                
                if min_val <= number <= max_val:
                    return number
                else:
                    print(f"   Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("   Please enter a valid number")
    
    # Simulate getting a valid number
    valid_number = get_valid_number(1, 100)
    print(f"   Valid number entered: {valid_number}")
    
    # 3. Processing until condition is met
    print("\n3. Processing Until Condition:")
    
    target_value = 100
    current_value = 0
    step = 15
    
    while current_value < target_value:
        current_value += step
        print(f"   Current value: {current_value}")
    
    print(f"   Final value: {current_value}")

def demonstrate_3d_while_loops():
    """Demonstrate while loops in 3D context"""
    print("\n=== 3D Graphics While Loops ===\n")
    
    # 1. Object spawning until limit reached
    print("1. Object Spawning:")
    
    max_objects = 5
    spawned_count = 0
    objects = []
    
    while spawned_count < max_objects:
        # Simulate spawning an object
        object_id = f"object_{spawned_count}"
        position = [spawned_count * 2, 0, 0]  # Space them out
        
        objects.append({
            "id": object_id,
            "position": position,
            "active": True
        })
        
        spawned_count += 1
        print(f"   Spawned {object_id} at {position}")
    
    print(f"   Total objects spawned: {len(objects)}")
    
    # 2. Collision detection loop
    print("\n2. Collision Detection Loop:")
    
    def check_collisions(objects, max_iterations=100):
        """Check for collisions between objects"""
        iterations = 0
        collisions_found = 0
        
        while iterations < max_iterations:
            collisions_in_iteration = 0
            
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    if not objects[i]["active"] or not objects[j]["active"]:
                        continue
                    
                    # Simple distance-based collision
                    pos1 = objects[i]["position"]
                    pos2 = objects[j]["position"]
                    distance = ((pos1[0] - pos2[0])**2 + 
                              (pos1[1] - pos2[1])**2 + 
                              (pos1[2] - pos2[2])**2)**0.5
                    
                    if distance < 1.0:  # Collision threshold
                        print(f"     Collision between {objects[i]['id']} and {objects[j]['id']}")
                        collisions_in_iteration += 1
                        collisions_found += 1
            
            # If no collisions found in this iteration, we can stop
            if collisions_in_iteration == 0:
                break
            
            iterations += 1
        
        return collisions_found
    
    # Test collision detection
    test_objects = [
        {"id": "obj1", "position": [0, 0, 0], "active": True},
        {"id": "obj2", "position": [0.5, 0, 0], "active": True},
        {"id": "obj3", "position": [10, 0, 0], "active": True},
        {"id": "obj4", "position": [10.5, 0, 0], "active": True}
    ]
    
    total_collisions = check_collisions(test_objects)
    print(f"   Total collisions detected: {total_collisions}")
    
    # 3. Animation loop simulation
    print("\n3. Animation Loop:")
    
    def animate_object(object_data, duration, fps=60):
        """Animate an object over time"""
        frame_count = 0
        total_frames = int(duration * fps)
        
        while frame_count < total_frames:
            # Calculate animation progress (0 to 1)
            progress = frame_count / total_frames
            
            # Simple linear animation
            start_pos = object_data["start_position"]
            end_pos = object_data["end_position"]
            
            current_pos = [
                start_pos[0] + progress * (end_pos[0] - start_pos[0]),
                start_pos[1] + progress * (end_pos[1] - start_pos[1]),
                start_pos[2] + progress * (end_pos[2] - start_pos[2])
            ]
            
            if frame_count % 10 == 0:  # Print every 10th frame
                print(f"     Frame {frame_count}: position {[f'{p:.2f}' for p in current_pos]}")
            
            frame_count += 1
        
        print(f"   Animation completed in {frame_count} frames")
    
    # Test animation
    animation_data = {
        "start_position": [0, 0, 0],
        "end_position": [10, 5, 0]
    }
    
    animate_object(animation_data, 1.0, 30)  # 1 second at 30 FPS

def demonstrate_game_loops():
    """Demonstrate game loop patterns with while loops"""
    print("\n=== Game Loops ===\n")
    
    # 1. Basic game loop
    print("1. Basic Game Loop:")
    
    def basic_game_loop(max_frames=10):
        """Simulate a basic game loop"""
        frame = 0
        game_running = True
        
        while game_running and frame < max_frames:
            # Update game state
            frame += 1
            
            # Simulate game logic
            if frame == 5:
                print("     Game event: Player collected item!")
            elif frame == 8:
                print("     Game event: Enemy spawned!")
            
            print(f"     Frame {frame}: Game running...")
            
            # Check for game end condition
            if frame >= max_frames:
                game_running = False
        
        print("   Game loop ended")
    
    basic_game_loop()
    
    # 2. Physics simulation loop
    print("\n2. Physics Simulation Loop:")
    
    def physics_simulation(initial_velocity, gravity, max_time=5.0, time_step=0.1):
        """Simulate physics with while loop"""
        time = 0.0
        position = 0.0
        velocity = initial_velocity
        
        while time < max_time and position >= 0:
            # Update physics
            position += velocity * time_step
            velocity -= gravity * time_step
            time += time_step
            
            if time % 1.0 < time_step:  # Print every second
                print(f"     Time {time:.1f}s: Position = {position:.2f}, Velocity = {velocity:.2f}")
            
            # Check for ground collision
            if position <= 0:
                position = 0
                print(f"     Object hit ground at time {time:.1f}s")
                break
        
        return time, position
    
    # Test physics simulation
    final_time, final_position = physics_simulation(initial_velocity=20, gravity=9.8)
    print(f"   Simulation ended: Time = {final_time:.1f}s, Position = {final_position:.2f}")
    
    # 3. AI behavior loop
    print("\n3. AI Behavior Loop:")
    
    def ai_behavior_loop(ai_state, max_actions=10):
        """Simulate AI behavior with while loop"""
        action_count = 0
        
        while action_count < max_actions and ai_state["health"] > 0:
            # AI decision making
            if ai_state["health"] < 30:
                action = "heal"
                ai_state["health"] = min(100, ai_state["health"] + 20)
            elif ai_state["enemy_nearby"]:
                action = "attack"
                ai_state["enemy_nearby"] = False
            else:
                action = "patrol"
                ai_state["patrol_progress"] += 1
            
            action_count += 1
            print(f"     Action {action_count}: {action} (Health: {ai_state['health']})")
            
            # Simulate taking damage occasionally
            if action_count % 3 == 0:
                ai_state["health"] -= 10
                print(f"     AI took damage! Health: {ai_state['health']}")
        
        return action_count
    
    # Test AI behavior
    ai_state = {
        "health": 100,
        "enemy_nearby": False,
        "patrol_progress": 0
    }
    
    actions_taken = ai_behavior_loop(ai_state)
    print(f"   AI completed {actions_taken} actions")

def demonstrate_while_loops_with_break_continue():
    """Demonstrate while loops with break and continue"""
    print("\n=== While Loops with Break/Continue ===\n")
    
    # 1. Break example - finding target
    print("1. Break Example - Finding Target:")
    
    def find_target_in_area(search_area, target_value):
        """Search for a target value in a 2D area"""
        x = 0
        y = 0
        
        while y < len(search_area):
            while x < len(search_area[0]):
                if search_area[y][x] == target_value:
                    print(f"     Found target {target_value} at position ({x}, {y})")
                    return x, y
                x += 1
            x = 0
            y += 1
        
        print(f"     Target {target_value} not found")
        return None
    
    # Test search
    search_grid = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    
    target_position = find_target_in_area(search_grid, 7)
    
    # 2. Continue example - processing valid objects
    print("\n2. Continue Example - Processing Valid Objects:")
    
    def process_valid_objects(objects):
        """Process only valid objects"""
        index = 0
        processed_count = 0
        
        while index < len(objects):
            obj = objects[index]
            
            # Skip invalid objects
            if not obj.get("valid", True):
                index += 1
                continue
            
            # Skip objects that are too far
            if obj.get("distance", 0) > 100:
                index += 1
                continue
            
            # Process valid object
            print(f"     Processing {obj['name']} at distance {obj.get('distance', 0)}")
            processed_count += 1
            index += 1
        
        return processed_count
    
    # Test object processing
    test_objects = [
        {"name": "obj1", "valid": True, "distance": 50},
        {"name": "obj2", "valid": False, "distance": 30},
        {"name": "obj3", "valid": True, "distance": 150},
        {"name": "obj4", "valid": True, "distance": 25}
    ]
    
    processed = process_valid_objects(test_objects)
    print(f"   Processed {processed} valid objects")
    
    # 3. Nested while loops with break
    print("\n3. Nested While Loops with Break:")
    
    def collision_resolution(objects, max_iterations=5):
        """Resolve collisions between objects"""
        iteration = 0
        
        while iteration < max_iterations:
            collision_found = False
            i = 0
            
            while i < len(objects):
                j = i + 1
                
                while j < len(objects):
                    # Check for collision
                    pos1 = objects[i]["position"]
                    pos2 = objects[j]["position"]
                    distance = ((pos1[0] - pos2[0])**2 + 
                              (pos1[1] - pos2[1])**2)**0.5
                    
                    if distance < 1.0:
                        print(f"     Resolving collision between objects {i} and {j}")
                        # Move objects apart
                        objects[i]["position"][0] += 0.5
                        objects[j]["position"][0] -= 0.5
                        collision_found = True
                        break  # Break inner loop
                    
                    j += 1
                
                if collision_found:
                    break  # Break middle loop
                
                i += 1
            
            if not collision_found:
                print(f"     No collisions found in iteration {iteration + 1}")
                break  # Break outer loop
            
            iteration += 1
        
        return iteration
    
    # Test collision resolution
    collision_objects = [
        {"position": [0, 0]},
        {"position": [0.5, 0]},
        {"position": [10, 0]},
        {"position": [10.5, 0]}
    ]
    
    iterations_needed = collision_resolution(collision_objects)
    print(f"   Collision resolution completed in {iterations_needed} iterations")

def demonstrate_practical_examples():
    """Demonstrate practical while loop examples"""
    print("\n=== Practical While Loop Examples ===\n")
    
    # 1. Loading progress simulation
    print("1. Loading Progress:")
    
    def simulate_loading(total_items, load_speed=0.1):
        """Simulate loading progress"""
        loaded_items = 0
        start_time = 0  # Simulate time
        
        while loaded_items < total_items:
            # Simulate loading time
            start_time += load_speed
            
            # Load an item
            loaded_items += 1
            progress = (loaded_items / total_items) * 100
            
            if loaded_items % 5 == 0:  # Print every 5 items
                print(f"     Loaded {loaded_items}/{total_items} items ({progress:.1f}%)")
        
        print(f"   Loading completed in {start_time:.1f} seconds")
        return start_time
    
    loading_time = simulate_loading(20)
    
    # 2. Resource management
    print("\n2. Resource Management:")
    
    def manage_resources(initial_resources, consumption_rate, regeneration_rate):
        """Manage resource consumption and regeneration"""
        resources = initial_resources
        time_steps = 0
        max_time = 20
        
        while time_steps < max_time and resources > 0:
            # Consume resources
            resources -= consumption_rate
            
            # Regenerate resources (but not above initial)
            if resources < initial_resources:
                resources = min(initial_resources, resources + regeneration_rate)
            
            time_steps += 1
            
            if time_steps % 5 == 0:  # Print every 5 time steps
                print(f"     Time {time_steps}: Resources = {resources:.2f}")
            
            # Check for resource depletion
            if resources <= 0:
                print(f"     Resources depleted at time {time_steps}")
                break
        
        return time_steps, resources
    
    # Test resource management
    final_time, final_resources = manage_resources(
        initial_resources=100,
        consumption_rate=8,
        regeneration_rate=2
    )
    print(f"   Final state: Time = {final_time}, Resources = {final_resources:.2f}")
    
    # 3. Pathfinding simulation
    print("\n3. Pathfinding Simulation:")
    
    def simple_pathfinding(start_pos, target_pos, max_steps=50):
        """Simple pathfinding with while loop"""
        current_pos = start_pos.copy()
        steps = 0
        path = [current_pos.copy()]
        
        while steps < max_steps:
            # Calculate distance to target
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = (dx**2 + dy**2)**0.5
            
            # Check if we've reached the target
            if distance < 0.5:
                print(f"     Reached target in {steps} steps")
                break
            
            # Move towards target
            if abs(dx) > abs(dy):
                # Move horizontally
                current_pos[0] += 1 if dx > 0 else -1
            else:
                # Move vertically
                current_pos[1] += 1 if dy > 0 else -1
            
            path.append(current_pos.copy())
            steps += 1
            
            if steps % 5 == 0:  # Print every 5 steps
                print(f"     Step {steps}: Position {current_pos}, Distance {distance:.2f}")
        
        if steps >= max_steps:
            print(f"     Pathfinding failed after {max_steps} steps")
        
        return path, steps
    
    # Test pathfinding
    start = [0, 0]
    target = [8, 6]
    
    path, steps_taken = simple_pathfinding(start, target)
    print(f"   Path length: {len(path)} positions")

def main():
    """Main function to run all while loop demonstrations"""
    print("=== Python While Loops for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_while_loops()
    demonstrate_3d_while_loops()
    demonstrate_game_loops()
    demonstrate_while_loops_with_break_continue()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered while loops:")
    print("✓ Basic while loop syntax and control")
    print("✓ While loops for conditional repetition")
    print("✓ Game loops and simulation patterns")
    print("✓ Using break and continue in while loops")
    print("✓ Nested while loops for complex logic")
    print("✓ Practical applications in 3D graphics and games")
    
    print("\nWhile loops are essential for:")
    print("- Game loops and real-time applications")
    print("- Physics simulations and animations")
    print("- AI behavior and decision making")
    print("- Resource management and loading")
    print("- Iterative algorithms and pathfinding")

if __name__ == "__main__":
    main()
