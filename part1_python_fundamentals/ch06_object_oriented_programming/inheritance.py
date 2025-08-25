#!/usr/bin/env python3
"""
Chapter 6: Object-Oriented Programming (OOP)
Inheritance Example

Demonstrates inheritance concepts including single and multiple inheritance,
method overriding, and the super() function with 3D graphics applications.
"""

import math

class GameObject:
    """Base class for all game objects"""
    
    def __init__(self, name, position=(0, 0, 0)):
        self.name = name
        self.position = list(position)
        self.active = True
        self.visible = True
    
    def update(self, delta_time):
        if not self.active:
            return
        print(f"   Updating {self.name}")
    
    def render(self):
        if not self.active or not self.visible:
            return
        print(f"   Rendering {self.name}")
    
    def move(self, dx, dy, dz):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz

class Renderable:
    """Mixin class for renderable objects"""
    
    def __init__(self):
        self.mesh_data = {}
        self.material = "default"
    
    def set_material(self, material):
        self.material = material
    
    def render(self):
        print(f"   Rendering mesh with material '{self.material}' for '{self.name}'")

class Collidable:
    """Mixin class for collidable objects"""
    
    def __init__(self):
        self.collider_type = "box"
        self.enabled = True
    
    def check_collision(self, other):
        if not self.enabled or not hasattr(other, 'enabled') or not other.enabled:
            return False
        return True

# Single Inheritance Examples

class Character(GameObject):
    """Character class - inherits from GameObject"""
    
    def __init__(self, name, position=(0, 0, 0), health=100):
        super().__init__(name, position)
        self.health = health
        self.max_health = health
        self.speed = 5.0
    
    def take_damage(self, damage):
        self.health = max(0, self.health - damage)
        print(f"   {self.name} took {damage} damage. Health: {self.health}")
    
    def heal(self, amount):
        self.health = min(self.max_health, self.health + amount)
        print(f"   {self.name} healed {amount}. Health: {self.health}")
    
    def move(self, dx, dy, dz):
        super().move(dx * self.speed, dy * self.speed, dz * self.speed)
        print(f"   {self.name} moved with speed {self.speed}")

class Enemy(Character):
    """Enemy class - inherits from Character"""
    
    def __init__(self, name, position=(0, 0, 0), health=50, damage=10):
        super().__init__(name, position, health)
        self.damage = damage
        self.target = None
    
    def attack(self, target):
        if target and hasattr(target, 'take_damage'):
            target.take_damage(self.damage)
            print(f"   {self.name} attacked {target.name} for {self.damage} damage")
    
    def set_target(self, target):
        self.target = target
        if target:
            print(f"   {self.name} is targeting {target.name}")

# Multiple Inheritance Examples

class Player(Character, Renderable, Collidable):
    """Player class - inherits from multiple classes"""
    
    def __init__(self, name, position=(0, 0, 0), health=100):
        Character.__init__(self, name, position, health)
        Renderable.__init__(self)
        Collidable.__init__(self)
        self.experience = 0
        self.level = 1
    
    def gain_experience(self, amount):
        self.experience += amount
        print(f"   {self.name} gained {amount} experience. Total: {self.experience}")
        if self.experience >= self.level * 100:
            self.level_up()
    
    def level_up(self):
        self.level += 1
        self.max_health += 20
        self.health = self.max_health
        print(f"   {self.name} leveled up to level {self.level}!")

class Item(GameObject, Renderable, Collidable):
    """Item class - inherits from multiple classes"""
    
    def __init__(self, name, position=(0, 0, 0), item_type="generic", value=0):
        GameObject.__init__(self, name, position)
        Renderable.__init__(self)
        Collidable.__init__(self)
        self.item_type = item_type
        self.value = value
        self.pickupable = True
    
    def pickup(self, player):
        if not self.pickupable:
            return False
        print(f"   {player.name} picked up {self.name}")
        self.visible = False
        return True

def demonstrate_single_inheritance():
    """Demonstrate single inheritance"""
    print("=== Single Inheritance ===\n")
    
    character = Character("Hero", (0, 0, 0), 100)
    print(f"Created: {character.name}")
    
    character.take_damage(20)
    character.heal(10)
    character.move(1, 0, 0)
    
    enemy = Enemy("Goblin", (5, 0, 0), 50, 15)
    enemy.set_target(character)
    enemy.attack(character)

def demonstrate_multiple_inheritance():
    """Demonstrate multiple inheritance"""
    print("\n=== Multiple Inheritance ===\n")
    
    player = Player("Adventurer", (0, 0, 0), 100)
    player.set_material("player_material")
    player.gain_experience(150)
    
    sword = Item("Iron Sword", (2, 0, 0), "weapon", 100)
    sword.set_material("metal")
    sword.pickup(player)

def demonstrate_method_resolution():
    """Demonstrate method resolution order (MRO)"""
    print("\n=== Method Resolution Order ===\n")
    
    print("Player MRO:")
    for i, cls in enumerate(Player.__mro__):
        print(f"   {i}: {cls.__name__}")

def demonstrate_super_function():
    """Demonstrate the super() function"""
    print("\n=== Super() Function ===\n")
    
    player = Player("Hero", (0, 0, 0), 100)
    player.take_damage(30)
    player.move(1, 0, 0)

def demonstrate_practical_examples():
    """Demonstrate practical inheritance examples"""
    print("\n=== Practical Examples ===\n")
    
    player = Player("Hero", (0, 0, 0), 100)
    enemy = Enemy("Orc", (5, 0, 0), 80, 20)
    sword = Item("Magic Sword", (2, 0, 0), "weapon", 200)
    
    enemy.set_target(player)
    
    for i in range(2):
        print(f"Frame {i+1}:")
        player.update(0.016)
        enemy.update(0.016)
        player.render()
        enemy.render()
        sword.render()
    
    enemy.attack(player)
    sword.pickup(player)

def main():
    """Main function"""
    print("=== Python Inheritance for 3D Graphics ===\n")
    
    demonstrate_single_inheritance()
    demonstrate_multiple_inheritance()
    demonstrate_method_resolution()
    demonstrate_super_function()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("Inheritance concepts covered:")
    print("✓ Single inheritance: Character → Enemy")
    print("✓ Multiple inheritance: Player inherits from Character, Renderable, Collidable")
    print("✓ Method overriding: Customizing parent behavior")
    print("✓ Super() function: Calling parent methods")
    print("✓ Method Resolution Order (MRO)")
    print("✓ Mixin classes: Reusable behavior components")

if __name__ == "__main__":
    main()
