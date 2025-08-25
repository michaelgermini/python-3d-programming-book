#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Design Patterns Example

Demonstrates common design patterns with 3D graphics applications.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# ============================================================================
# SINGLETON PATTERN
# ============================================================================

class GameManager:
    """Singleton pattern - ensures only one instance exists"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not GameManager._initialized:
            self.game_state = "menu"
            self.current_level = 1
            self.player_score = 0
            self.settings = {
                "graphics_quality": "high",
                "sound_enabled": True,
                "fullscreen": False
            }
            GameManager._initialized = True
    
    def update_game_state(self, new_state):
        """Update the game state"""
        self.game_state = new_state
        print(f"   Game state changed to: {new_state}")
    
    def get_setting(self, key):
        """Get a setting value"""
        return self.settings.get(key)
    
    def set_setting(self, key, value):
        """Set a setting value"""
        self.settings[key] = value
        print(f"   Setting '{key}' changed to: {value}")

# ============================================================================
# FACTORY PATTERN
# ============================================================================

class GameObject(ABC):
    """Abstract base class for game objects"""
    
    @abstractmethod
    def render(self):
        """Render the object"""
        pass
    
    @abstractmethod
    def update(self, delta_time):
        """Update the object"""
        pass

class Cube(GameObject):
    """Cube game object"""
    
    def __init__(self, position=(0, 0, 0), size=1.0):
        self.position = position
        self.size = size
        self.material = "default"
    
    def render(self):
        print(f"   Rendering cube at {self.position} with size {self.size}")
    
    def update(self, delta_time):
        print(f"   Updating cube at {self.position}")

class Sphere(GameObject):
    """Sphere game object"""
    
    def __init__(self, position=(0, 0, 0), radius=1.0):
        self.position = position
        self.radius = radius
        self.material = "default"
    
    def render(self):
        print(f"   Rendering sphere at {self.position} with radius {self.radius}")
    
    def update(self, delta_time):
        print(f"   Updating sphere at {self.position}")

class GameObjectFactory:
    """Factory for creating game objects"""
    
    @staticmethod
    def create_object(object_type: str, **kwargs) -> GameObject:
        """Create a game object based on type"""
        if object_type.lower() == "cube":
            return Cube(**kwargs)
        elif object_type.lower() == "sphere":
            return Sphere(**kwargs)
        else:
            raise ValueError(f"Unknown object type: {object_type}")

# ============================================================================
# OBSERVER PATTERN
# ============================================================================

class Event:
    """Event class for observer pattern"""
    
    def __init__(self, event_type: str, data: Any = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

class Observer(ABC):
    """Abstract observer interface"""
    
    @abstractmethod
    def update(self, event: Event):
        """Handle event update"""
        pass

class Subject(ABC):
    """Abstract subject interface"""
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: Event):
        """Notify all observers"""
        for observer in self._observers:
            observer.update(event)

class Player(Subject):
    """Player class that notifies observers of events"""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.health = 100
        self.position = (0, 0, 0)
    
    def take_damage(self, damage: int):
        """Take damage and notify observers"""
        old_health = self.health
        self.health = max(0, self.health - damage)
        
        event = Event("damage_taken", {
            "player": self.name,
            "damage": damage,
            "old_health": old_health,
            "new_health": self.health
        })
        self.notify(event)
    
    def move(self, new_position):
        """Move player and notify observers"""
        old_position = self.position
        self.position = new_position
        
        event = Event("player_moved", {
            "player": self.name,
            "old_position": old_position,
            "new_position": new_position
        })
        self.notify(event)

class HealthUI(Observer):
    """UI component that observes player health"""
    
    def __init__(self, player_name: str):
        self.player_name = player_name
    
    def update(self, event: Event):
        if event.event_type == "damage_taken" and event.data["player"] == self.player_name:
            print(f"   [HealthUI] {self.player_name} took {event.data['damage']} damage!")
            print(f"   [HealthUI] Health: {event.data['old_health']} -> {event.data['new_health']}")

class SoundManager(Observer):
    """Sound manager that observes player events"""
    
    def update(self, event: Event):
        if event.event_type == "damage_taken":
            print(f"   [SoundManager] Playing damage sound for {event.data['player']}")
        elif event.event_type == "player_moved":
            print(f"   [SoundManager] Playing footstep sound for {event.data['player']}")

# ============================================================================
# COMPONENT PATTERN
# ============================================================================

class Component(ABC):
    """Abstract component base class"""
    
    def __init__(self, entity):
        self.entity = entity
    
    @abstractmethod
    def update(self, delta_time):
        """Update the component"""
        pass

class TransformComponent(Component):
    """Transform component for position, rotation, scale"""
    
    def __init__(self, entity, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        super().__init__(entity)
        self.position = list(position)
        self.rotation = list(rotation)
        self.scale = list(scale)
    
    def update(self, delta_time):
        # Transform updates would go here
        pass
    
    def translate(self, dx, dy, dz):
        """Translate the transform"""
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz

class RenderComponent(Component):
    """Render component for visual representation"""
    
    def __init__(self, entity, mesh="cube", material="default"):
        super().__init__(entity)
        self.mesh = mesh
        self.material = material
        self.visible = True
    
    def update(self, delta_time):
        if self.visible:
            transform = self.entity.get_component(TransformComponent)
            if transform:
                print(f"   Rendering {self.mesh} at {transform.position}")

class Entity:
    """Entity class that can have multiple components"""
    
    def __init__(self, name: str):
        self.name = name
        self.components: Dict[type, Component] = {}
        self.active = True
    
    def add_component(self, component: Component):
        """Add a component to the entity"""
        self.components[type(component)] = component
    
    def get_component(self, component_type: type) -> Optional[Component]:
        """Get a component by type"""
        return self.components.get(component_type)
    
    def has_component(self, component_type: type) -> bool:
        """Check if entity has a component"""
        return component_type in self.components
    
    def update(self, delta_time):
        """Update all components"""
        if not self.active:
            return
        
        for component in self.components.values():
            component.update(delta_time)

# ============================================================================
# COMMAND PATTERN
# ============================================================================

class Command(ABC):
    """Abstract command interface"""
    
    @abstractmethod
    def execute(self):
        """Execute the command"""
        pass
    
    @abstractmethod
    def undo(self):
        """Undo the command"""
        pass

class MoveCommand(Command):
    """Command for moving an entity"""
    
    def __init__(self, entity: Entity, old_position, new_position):
        self.entity = entity
        self.old_position = old_position
        self.new_position = new_position
    
    def execute(self):
        transform = self.entity.get_component(TransformComponent)
        if transform:
            transform.position = list(self.new_position)
            print(f"   [MoveCommand] Moved {self.entity.name} to {self.new_position}")
    
    def undo(self):
        transform = self.entity.get_component(TransformComponent)
        if transform:
            transform.position = list(self.old_position)
            print(f"   [MoveCommand] Undid move for {self.entity.name} back to {self.old_position}")

class CommandManager:
    """Manager for command history and undo/redo"""
    
    def __init__(self):
        self.command_history: List[Command] = []
        self.current_index = -1
    
    def execute_command(self, command: Command):
        """Execute a command and add to history"""
        command.execute()
        
        # Remove any commands after current position (for redo)
        self.command_history = self.command_history[:self.current_index + 1]
        self.command_history.append(command)
        self.current_index += 1
    
    def undo(self):
        """Undo the last command"""
        if self.current_index >= 0:
            command = self.command_history[self.current_index]
            command.undo()
            self.current_index -= 1
        else:
            print("   [CommandManager] Nothing to undo")
    
    def redo(self):
        """Redo the next command"""
        if self.current_index + 1 < len(self.command_history):
            self.current_index += 1
            command = self.command_history[self.current_index]
            command.execute()
        else:
            print("   [CommandManager] Nothing to redo")

# ============================================================================
# STATE PATTERN
# ============================================================================

class GameState(ABC):
    """Abstract game state"""
    
    @abstractmethod
    def enter(self):
        """Enter the state"""
        pass
    
    @abstractmethod
    def update(self, delta_time):
        """Update the state"""
        pass
    
    @abstractmethod
    def exit(self):
        """Exit the state"""
        pass

class MenuState(GameState):
    """Menu state"""
    
    def enter(self):
        print("   [MenuState] Entering menu state")
    
    def update(self, delta_time):
        print("   [MenuState] Updating menu")
    
    def exit(self):
        print("   [MenuState] Exiting menu state")

class PlayingState(GameState):
    """Playing state"""
    
    def enter(self):
        print("   [PlayingState] Entering playing state")
    
    def update(self, delta_time):
        print("   [PlayingState] Updating game")
    
    def exit(self):
        print("   [PlayingState] Exiting playing state")

class GameStateMachine:
    """State machine for managing game states"""
    
    def __init__(self):
        self.current_state: Optional[GameState] = None
        self.states = {
            "menu": MenuState(),
            "playing": PlayingState()
        }
    
    def change_state(self, state_name: str):
        """Change to a new state"""
        if state_name not in self.states:
            print(f"   [StateMachine] Unknown state: {state_name}")
            return
        
        if self.current_state:
            self.current_state.exit()
        
        self.current_state = self.states[state_name]
        self.current_state.enter()
    
    def update(self, delta_time):
        """Update the current state"""
        if self.current_state:
            self.current_state.update(delta_time)

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_singleton():
    """Demonstrate singleton pattern"""
    print("=== Singleton Pattern ===\n")
    
    # Create multiple instances (should be the same)
    manager1 = GameManager()
    manager2 = GameManager()
    
    print(f"manager1 is manager2: {manager1 is manager2}")
    print(f"manager1.game_state: {manager1.game_state}")
    
    # Update through one instance
    manager1.update_game_state("playing")
    print(f"manager2.game_state: {manager2.game_state}")
    
    # Settings are shared
    manager1.set_setting("graphics_quality", "medium")
    print(f"manager2.get_setting('graphics_quality'): {manager2.get_setting('graphics_quality')}")

def demonstrate_factory():
    """Demonstrate factory pattern"""
    print("\n=== Factory Pattern ===\n")
    
    factory = GameObjectFactory()
    
    # Create different types of objects
    objects = [
        factory.create_object("cube", position=(0, 0, 0), size=2.0),
        factory.create_object("sphere", position=(5, 0, 0), radius=1.5)
    ]
    
    print("Created objects:")
    for obj in objects:
        print(f"   {type(obj).__name__}: {obj}")
        obj.render()
        obj.update(0.016)

def demonstrate_observer():
    """Demonstrate observer pattern"""
    print("\n=== Observer Pattern ===\n")
    
    # Create player and observers
    player = Player("Hero")
    health_ui = HealthUI("Hero")
    sound_manager = SoundManager()
    
    # Attach observers
    player.attach(health_ui)
    player.attach(sound_manager)
    
    # Trigger events
    print("Player taking damage:")
    player.take_damage(25)
    
    print("\nPlayer moving:")
    player.move((10, 0, 5))
    
    # Detach an observer
    player.detach(sound_manager)
    print("\nAfter detaching sound manager:")
    player.take_damage(10)

def demonstrate_component():
    """Demonstrate component pattern"""
    print("\n=== Component Pattern ===\n")
    
    # Create entity with components
    entity = Entity("Player")
    
    transform = TransformComponent(entity, position=(0, 0, 0))
    render = RenderComponent(entity, mesh="character", material="player")
    
    entity.add_component(transform)
    entity.add_component(render)
    
    print(f"Entity: {entity.name}")
    print(f"Has transform: {entity.has_component(TransformComponent)}")
    print(f"Has render: {entity.has_component(RenderComponent)}")
    
    # Update entity
    print("\nUpdating entity:")
    entity.update(0.016)

def demonstrate_command():
    """Demonstrate command pattern"""
    print("\n=== Command Pattern ===\n")
    
    # Create entity and command manager
    entity = Entity("Cube")
    transform = TransformComponent(entity, position=(0, 0, 0))
    entity.add_component(transform)
    
    command_manager = CommandManager()
    
    # Execute commands
    print("Executing commands:")
    move_cmd = MoveCommand(entity, (0, 0, 0), (5, 0, 0))
    command_manager.execute_command(move_cmd)
    
    # Undo commands
    print("\nUndoing commands:")
    command_manager.undo()
    
    # Redo commands
    print("\nRedoing commands:")
    command_manager.redo()

def demonstrate_state():
    """Demonstrate state pattern"""
    print("\n=== State Pattern ===\n")
    
    # Create state machine
    state_machine = GameStateMachine()
    
    # Change states
    print("Changing game states:")
    state_machine.change_state("menu")
    state_machine.update(0.016)
    
    state_machine.change_state("playing")
    state_machine.update(0.016)

def main():
    """Main function to run all demonstrations"""
    print("=== Python Design Patterns for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_singleton()
    demonstrate_factory()
    demonstrate_observer()
    demonstrate_component()
    demonstrate_command()
    demonstrate_state()
    
    print("\n=== Summary ===")
    print("This chapter covered design patterns:")
    print("✓ Singleton: Ensures single instance of a class")
    print("✓ Factory: Creates objects without specifying exact class")
    print("✓ Observer: Notifies objects of state changes")
    print("✓ Component: Composes objects from reusable components")
    print("✓ Command: Encapsulates requests as objects")
    print("✓ State: Manages object behavior based on state")
    
    print("\nKey benefits of design patterns:")
    print("- Code reusability: Proven solutions to common problems")
    print("- Maintainability: Well-structured, organized code")
    print("- Flexibility: Easy to modify and extend")
    print("- Scalability: Patterns scale with project size")
    print("- Team collaboration: Common vocabulary and structure")
    print("- Best practices: Industry-standard approaches")

if __name__ == "__main__":
    main()
