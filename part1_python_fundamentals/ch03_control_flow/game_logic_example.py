#!/usr/bin/env python3
"""
Chapter 3: Control Flow (Conditionals and Loops)
Game Logic Example

This example demonstrates practical game logic implementation using
control flow concepts, focusing on 3D graphics programming applications.
"""

import math
import random

class GameState:
    """Simple game state management"""
    def __init__(self):
        self.running = True
        self.score = 0
        self.level = 1
        self.time = 0.0

class Player:
    """Player class with basic properties and methods"""
    def __init__(self, name="Player"):
        self.name = name
        self.position = [0, 0, 0]
        self.health = 100
        self.max_health = 100
        self.energy = 100
        self.max_energy = 100
        self.inventory = []
        self.weapon = "fists"
        self.armor = "none"
        self.experience = 0
        self.level = 1
        
    def take_damage(self, damage):
        """Player takes damage"""
        if self.armor == "heavy":
            damage = damage * 0.5
        elif self.armor == "light":
            damage = damage * 0.8
        
        self.health = max(0, self.health - damage)
        return self.health <= 0
    
    def heal(self, amount):
        """Player heals"""
        self.health = min(self.max_health, self.health + amount)
    
    def use_energy(self, amount):
        """Player uses energy"""
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False
    
    def gain_experience(self, exp):
        """Player gains experience and levels up"""
        self.experience += exp
        new_level = self.experience // 100 + 1
        
        if new_level > self.level:
            self.level = new_level
            self.max_health += 10
            self.max_energy += 5
            self.health = self.max_health
            self.energy = self.max_energy
            return True
        return False

class Enemy:
    """Enemy class with AI behavior"""
    def __init__(self, enemy_type, position):
        self.type = enemy_type
        self.position = position
        self.health = self.get_max_health()
        self.max_health = self.health
        self.damage = self.get_damage()
        self.speed = self.get_speed()
        self.aggressive = False
        self.patrol_route = []
        self.current_patrol_index = 0
        self.last_attack_time = 0
        self.attack_cooldown = 2.0
        
    def get_max_health(self):
        """Get max health based on enemy type"""
        if self.type == "goblin":
            return 30
        elif self.type == "orc":
            return 60
        elif self.type == "dragon":
            return 200
        else:
            return 50
    
    def get_damage(self):
        """Get damage based on enemy type"""
        if self.type == "goblin":
            return 10
        elif self.type == "orc":
            return 20
        elif self.type == "dragon":
            return 50
        else:
            return 15
    
    def get_speed(self):
        """Get speed based on enemy type"""
        if self.type == "goblin":
            return 2.0
        elif self.type == "orc":
            return 1.5
        elif self.type == "dragon":
            return 3.0
        else:
            return 1.0
    
    def take_damage(self, damage):
        """Enemy takes damage"""
        self.health = max(0, self.health - damage)
        return self.health <= 0
    
    def calculate_distance_to_player(self, player_position):
        """Calculate distance to player"""
        dx = self.position[0] - player_position[0]
        dy = self.position[1] - player_position[1]
        dz = self.position[2] - player_position[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def update_ai(self, player_position, current_time):
        """Update enemy AI behavior"""
        distance_to_player = self.calculate_distance_to_player(player_position)
        
        # Check if player is in detection range
        detection_range = 10 if self.type == "goblin" else 15 if self.type == "orc" else 25
        
        if distance_to_player <= detection_range:
            self.aggressive = True
            # Move towards player
            self.move_towards_player(player_position)
            
            # Attack if close enough and cooldown is ready
            attack_range = 2 if self.type == "goblin" else 3 if self.type == "orc" else 5
            if distance_to_player <= attack_range and current_time - self.last_attack_time >= self.attack_cooldown:
                self.last_attack_time = current_time
                return "attack"
        else:
            self.aggressive = False
            # Patrol behavior
            self.patrol()
        
        return "patrol"
    
    def move_towards_player(self, player_position):
        """Move towards player"""
        dx = player_position[0] - self.position[0]
        dy = player_position[1] - self.position[1]
        dz = player_position[2] - self.position[2]
        
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance > 0:
            dx /= distance
            dy /= distance
            dz /= distance
            
            # Move towards player
            self.position[0] += dx * self.speed * 0.1  # Simplified movement
            self.position[1] += dy * self.speed * 0.1
            self.position[2] += dz * self.speed * 0.1
    
    def patrol(self):
        """Patrol behavior"""
        if not self.patrol_route:
            # Generate simple patrol route
            self.patrol_route = [
                [self.position[0] - 5, self.position[1], self.position[2]],
                [self.position[0] + 5, self.position[1], self.position[2]],
                [self.position[0], self.position[1] - 5, self.position[2]],
                [self.position[0], self.position[1] + 5, self.position[2]]
            ]
        
        # Move towards current patrol point
        target = self.patrol_route[self.current_patrol_index]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dz = target[2] - self.position[2]
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < 1.0:
            # Move to next patrol point
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_route)
        else:
            # Move towards current patrol point
            if distance > 0:
                dx /= distance
                dy /= distance
                dz /= distance
                
                self.position[0] += dx * self.speed * 0.05
                self.position[1] += dy * self.speed * 0.05
                self.position[2] += dz * self.speed * 0.05

class Item:
    """Item class for collectibles and equipment"""
    def __init__(self, item_type, position, value=0):
        self.type = item_type
        self.position = position
        self.value = value
        self.collected = False
        
    def get_effect(self):
        """Get item effect based on type"""
        if self.type == "health_potion":
            return {"health": 25}
        elif self.type == "energy_potion":
            return {"energy": 30}
        elif self.type == "sword":
            return {"weapon": "sword"}
        elif self.type == "armor":
            return {"armor": "light"}
        elif self.type == "treasure":
            return {"score": self.value}
        else:
            return {}

class GameLogic:
    """Main game logic controller"""
    def __init__(self):
        self.game_state = GameState()
        self.player = Player()
        self.enemies = []
        self.items = []
        self.spawn_timer = 0
        self.spawn_interval = 5.0
        
    def initialize_level(self, level):
        """Initialize a new level"""
        self.game_state.level = level
        self.enemies.clear()
        self.items.clear()
        
        # Spawn enemies based on level
        num_enemies = min(level + 2, 10)
        for i in range(num_enemies):
            enemy_type = self.select_enemy_type(level)
            position = self.get_random_position()
            enemy = Enemy(enemy_type, position)
            self.enemies.append(enemy)
        
        # Spawn items
        num_items = level + 1
        for i in range(num_items):
            item_type = self.select_item_type()
            position = self.get_random_position()
            value = random.randint(10, 50) if item_type == "treasure" else 0
            item = Item(item_type, position, value)
            self.items.append(item)
    
    def select_enemy_type(self, level):
        """Select enemy type based on level"""
        if level <= 2:
            return "goblin"
        elif level <= 5:
            return random.choice(["goblin", "orc"])
        else:
            return random.choice(["goblin", "orc", "dragon"])
    
    def select_item_type(self):
        """Select random item type"""
        return random.choice(["health_potion", "energy_potion", "sword", "armor", "treasure"])
    
    def get_random_position(self):
        """Get random position in game world"""
        return [
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-20, 20)
        ]
    
    def update(self, delta_time):
        """Update game logic"""
        self.game_state.time += delta_time
        
        # Update enemies
        self.update_enemies(delta_time)
        
        # Check item collection
        self.check_item_collection()
        
        # Check level completion
        if self.check_level_completion():
            self.complete_level()
        
        # Spawn new enemies periodically
        self.spawn_timer += delta_time
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_new_enemy()
            self.spawn_timer = 0
    
    def update_enemies(self, delta_time):
        """Update all enemies"""
        for enemy in self.enemies[:]:  # Copy list to allow removal
            if enemy.health <= 0:
                # Enemy defeated
                self.enemy_defeated(enemy)
                self.enemies.remove(enemy)
                continue
            
            # Update enemy AI
            action = enemy.update_ai(self.player.position, self.game_state.time)
            
            if action == "attack":
                # Enemy attacks player
                self.enemy_attack(enemy)
    
    def enemy_defeated(self, enemy):
        """Handle enemy defeat"""
        exp_gain = 20 if enemy.type == "goblin" else 40 if enemy.type == "orc" else 100
        self.game_state.score += exp_gain
        
        if self.player.gain_experience(exp_gain):
            print(f"   {self.player.name} leveled up! Level {self.player.level}")
        
        print(f"   Defeated {enemy.type}! +{exp_gain} exp, +{exp_gain} score")
    
    def enemy_attack(self, enemy):
        """Handle enemy attack on player"""
        if self.player.take_damage(enemy.damage):
            print(f"   {enemy.type} defeated {self.player.name}!")
            self.game_state.running = False
        else:
            print(f"   {enemy.type} attacks! Player health: {self.player.health}")
    
    def check_item_collection(self):
        """Check if player collected any items"""
        for item in self.items[:]:  # Copy list to allow removal
            if item.collected:
                continue
            
            distance = self.calculate_distance(self.player.position, item.position)
            if distance < 2.0:  # Collection range
                self.collect_item(item)
                self.items.remove(item)
    
    def collect_item(self, item):
        """Handle item collection"""
        effect = item.get_effect()
        
        if "health" in effect:
            self.player.heal(effect["health"])
            print(f"   Collected {item.type}! +{effect['health']} health")
        
        elif "energy" in effect:
            self.player.energy = min(self.player.max_energy, self.player.energy + effect["energy"])
            print(f"   Collected {item.type}! +{effect['energy']} energy")
        
        elif "weapon" in effect:
            self.player.weapon = effect["weapon"]
            print(f"   Collected {item.type}! Equipped {effect['weapon']}")
        
        elif "armor" in effect:
            self.player.armor = effect["armor"]
            print(f"   Collected {item.type}! Equipped {effect['armor']} armor")
        
        elif "score" in effect:
            self.game_state.score += effect["score"]
            print(f"   Collected treasure! +{effect['score']} score")
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def check_level_completion(self):
        """Check if level is completed"""
        return len(self.enemies) == 0 and len(self.items) == 0
    
    def complete_level(self):
        """Complete current level"""
        level_bonus = self.game_state.level * 100
        self.game_state.score += level_bonus
        print(f"   Level {self.game_state.level} completed! +{level_bonus} bonus")
        
        self.game_state.level += 1
        self.initialize_level(self.game_state.level)
    
    def spawn_new_enemy(self):
        """Spawn a new enemy"""
        if len(self.enemies) < 5:  # Max enemies limit
            enemy_type = self.select_enemy_type(self.game_state.level)
            position = self.get_random_position()
            enemy = Enemy(enemy_type, position)
            self.enemies.append(enemy)
            print(f"   New {enemy_type} spawned!")
    
    def player_action(self, action, target_position=None):
        """Handle player actions"""
        if action == "move" and target_position:
            self.player.position = target_position
            print(f"   {self.player.name} moved to {target_position}")
        
        elif action == "attack":
            if self.player.use_energy(10):
                # Find closest enemy
                closest_enemy = None
                min_distance = float('inf')
                
                for enemy in self.enemies:
                    distance = self.calculate_distance(self.player.position, enemy.position)
                    if distance < min_distance and distance <= 5:  # Attack range
                        min_distance = distance
                        closest_enemy = enemy
                
                if closest_enemy:
                    damage = 15 if self.player.weapon == "fists" else 25
                    if closest_enemy.take_damage(damage):
                        print(f"   {self.player.name} defeated {closest_enemy.type}!")
                        self.enemy_defeated(closest_enemy)
                        self.enemies.remove(closest_enemy)
                    else:
                        print(f"   {self.player.name} attacks {closest_enemy.type}! Enemy health: {closest_enemy.health}")
                else:
                    print(f"   No enemies in range!")
            else:
                print(f"   Not enough energy to attack!")
        
        elif action == "heal":
            if self.player.use_energy(20):
                self.player.heal(30)
                print(f"   {self.player.name} healed! Health: {self.player.health}")
            else:
                print(f"   Not enough energy to heal!")
    
    def get_game_status(self):
        """Get current game status"""
        return {
            "player_health": self.player.health,
            "player_energy": self.player.energy,
            "player_level": self.player.level,
            "player_experience": self.player.experience,
            "score": self.game_state.score,
            "level": self.game_state.level,
            "enemies_remaining": len(self.enemies),
            "items_remaining": len(self.items),
            "time": self.game_state.time
        }

def demonstrate_game_logic():
    """Demonstrate the game logic system"""
    print("=== Game Logic Demonstration ===\n")
    
    # Create game instance
    game = GameLogic()
    game.initialize_level(1)
    
    print("Game initialized! Starting simulation...\n")
    
    # Simulate game loop
    simulation_time = 0
    max_simulation_time = 30  # 30 seconds simulation
    
    while game.game_state.running and simulation_time < max_simulation_time:
        delta_time = 1.0  # 1 second per update
        
        # Player actions (simulated)
        if simulation_time % 3 == 0:  # Every 3 seconds
            game.player_action("attack")
        
        if simulation_time % 5 == 0:  # Every 5 seconds
            # Move player to random position
            new_position = game.get_random_position()
            game.player_action("move", new_position)
        
        if simulation_time % 7 == 0:  # Every 7 seconds
            game.player_action("heal")
        
        # Update game
        game.update(delta_time)
        
        # Print status every 5 seconds
        if simulation_time % 5 == 0:
            status = game.get_game_status()
            print(f"\n--- Time: {simulation_time}s ---")
            print(f"Player: Level {status['player_level']}, Health {status['player_health']}, Energy {status['player_energy']}")
            print(f"Score: {status['score']}, Level: {status['level']}")
            print(f"Enemies: {status['enemies_remaining']}, Items: {status['items_remaining']}")
        
        simulation_time += delta_time
    
    # Final status
    print(f"\n=== Final Game Status ===")
    status = game.get_game_status()
    print(f"Final Score: {status['score']}")
    print(f"Level Reached: {status['level']}")
    print(f"Player Level: {status['player_level']}")
    print(f"Simulation Time: {simulation_time}s")
    
    if not game.game_state.running:
        print("Game Over - Player was defeated!")
    else:
        print("Simulation completed!")

def demonstrate_control_flow_patterns():
    """Demonstrate control flow patterns used in the game"""
    print("\n=== Control Flow Patterns ===\n")
    
    # 1. State machine pattern
    print("1. State Machine Pattern:")
    
    class GameStateMachine:
        def __init__(self):
            self.state = "menu"
            self.states = {
                "menu": self.menu_state,
                "playing": self.playing_state,
                "paused": self.paused_state,
                "game_over": self.game_over_state
            }
        
        def update(self, input_action):
            if self.state in self.states:
                self.states[self.state](input_action)
        
        def menu_state(self, action):
            if action == "start_game":
                self.state = "playing"
                print("     Transition: menu -> playing")
            elif action == "quit":
                self.state = "game_over"
                print("     Transition: menu -> game_over")
        
        def playing_state(self, action):
            if action == "pause":
                self.state = "paused"
                print("     Transition: playing -> paused")
            elif action == "player_died":
                self.state = "game_over"
                print("     Transition: playing -> game_over")
        
        def paused_state(self, action):
            if action == "resume":
                self.state = "playing"
                print("     Transition: paused -> playing")
            elif action == "quit":
                self.state = "game_over"
                print("     Transition: paused -> game_over")
        
        def game_over_state(self, action):
            if action == "restart":
                self.state = "menu"
                print("     Transition: game_over -> menu")
    
    # Test state machine
    state_machine = GameStateMachine()
    actions = ["start_game", "pause", "resume", "player_died", "restart"]
    
    for action in actions:
        state_machine.update(action)
    
    # 2. Event-driven pattern
    print("\n2. Event-Driven Pattern:")
    
    class EventSystem:
        def __init__(self):
            self.event_handlers = {}
        
        def register_handler(self, event_type, handler):
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []
            self.event_handlers[event_type].append(handler)
        
        def trigger_event(self, event_type, data=None):
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    handler(data)
    
    # Test event system
    event_system = EventSystem()
    
    def on_enemy_defeated(data):
        print(f"     Event: Enemy defeated - {data}")
    
    def on_item_collected(data):
        print(f"     Event: Item collected - {data}")
    
    def on_level_completed(data):
        print(f"     Event: Level completed - {data}")
    
    event_system.register_handler("enemy_defeated", on_enemy_defeated)
    event_system.register_handler("item_collected", on_item_collected)
    event_system.register_handler("level_completed", on_level_completed)
    
    # Trigger events
    event_system.trigger_event("enemy_defeated", "goblin")
    event_system.trigger_event("item_collected", "health_potion")
    event_system.trigger_event("level_completed", "Level 1")
    
    # 3. Command pattern
    print("\n3. Command Pattern:")
    
    class Command:
        def execute(self):
            pass
    
    class MoveCommand(Command):
        def __init__(self, player, target_position):
            self.player = player
            self.target_position = target_position
        
        def execute(self):
            self.player.position = self.target_position
            print(f"     Command: Move to {self.target_position}")
    
    class AttackCommand(Command):
        def __init__(self, player, target_enemy):
            self.player = player
            self.target_enemy = target_enemy
        
        def execute(self):
            if self.player.use_energy(10):
                damage = 15
                self.target_enemy.take_damage(damage)
                print(f"     Command: Attack {self.target_enemy.type} for {damage} damage")
            else:
                print(f"     Command: Attack failed - not enough energy")
    
    # Test command pattern
    player = Player()
    enemy = Enemy("goblin", [5, 0, 0])
    
    commands = [
        MoveCommand(player, [10, 0, 0]),
        AttackCommand(player, enemy),
        MoveCommand(player, [0, 0, 0])
    ]
    
    for command in commands:
        command.execute()

def main():
    """Main function to run game logic demonstrations"""
    print("=== Python Game Logic for 3D Graphics ===\n")
    
    # Run demonstrations
    demonstrate_game_logic()
    demonstrate_control_flow_patterns()
    
    print("\n=== Summary ===")
    print("This chapter covered practical game logic implementation:")
    print("✓ Object-oriented game design")
    print("✓ State machines and AI behavior")
    print("✓ Event-driven programming")
    print("✓ Command pattern for actions")
    print("✓ Complex control flow in game loops")
    
    print("\nGame logic patterns are essential for:")
    print("- Creating interactive 3D applications")
    print("- Implementing AI and behavior systems")
    print("- Managing game state and progression")
    print("- Building responsive user interfaces")
    print("- Developing complex simulation systems")

if __name__ == "__main__":
    main()
