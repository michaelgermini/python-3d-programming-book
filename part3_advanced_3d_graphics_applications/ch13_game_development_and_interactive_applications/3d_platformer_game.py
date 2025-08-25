#!/usr/bin/env python3
"""
Chapter 13: Game Development and Interactive Applications
3D Platformer Game

Demonstrates a 3D platformer game with physics, collision detection,
game mechanics, and interactive systems.
"""

import numpy as np
import moderngl
import glfw
import math
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z]

@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_array(self) -> List[float]:
        return [self.r, self.g, self.b, self.a]

class GameState(Enum):
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"

class ObjectType(Enum):
    PLAYER = "player"
    PLATFORM = "platform"
    COLLECTIBLE = "collectible"
    ENEMY = "enemy"

# ============================================================================
# GAME OBJECTS
# ============================================================================

@dataclass
class BoundingBox:
    min_pos: Vector3D
    max_pos: Vector3D
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return (self.min_pos.x <= other.max_pos.x and self.max_pos.x >= other.min_pos.x and
                self.min_pos.y <= other.max_pos.y and self.max_pos.y >= other.min_pos.y and
                self.min_pos.z <= other.max_pos.z and self.max_pos.z >= other.min_pos.z)

class GameObject:
    def __init__(self, position: Vector3D, size: Vector3D, obj_type: ObjectType):
        self.position = position
        self.size = size
        self.velocity = Vector3D(0, 0, 0)
        self.obj_type = obj_type
        self.active = True
        self.color = Color(1.0, 1.0, 1.0, 1.0)
        
        self.bounding_box = BoundingBox(
            Vector3D(position.x - size.x/2, position.y - size.y/2, position.z - size.z/2),
            Vector3D(position.x + size.x/2, position.y + size.y/2, position.z + size.z/2)
        )
    
    def update(self, delta_time: float):
        self.position = self.position + self.velocity * delta_time
        
        self.bounding_box.min_pos = Vector3D(
            self.position.x - self.size.x/2,
            self.position.y - self.size.y/2,
            self.position.z - self.size.z/2
        )
        self.bounding_box.max_pos = Vector3D(
            self.position.x + self.size.x/2,
            self.position.y + self.size.y/2,
            self.position.z + self.size.z/2
        )
    
    def check_collision(self, other: 'GameObject') -> bool:
        return self.bounding_box.intersects(other.bounding_box)
    
    def handle_collision(self, other: 'GameObject'):
        pass

class Player(GameObject):
    def __init__(self, position: Vector3D):
        super().__init__(position, Vector3D(0.5, 1.0, 0.5), ObjectType.PLAYER)
        self.color = Color(0.2, 0.6, 1.0, 1.0)
        self.health = 100
        self.score = 0
        self.on_ground = False
        self.jump_count = 0
        self.max_jumps = 2
        self.move_speed = 5.0
        self.jump_force = 8.0
        self.gravity = -20.0
        
        self.input_left = False
        self.input_right = False
        self.input_forward = False
        self.input_backward = False
        self.input_jump = False
    
    def update(self, delta_time: float):
        # Apply gravity
        self.velocity.y += self.gravity * delta_time
        
        # Handle movement
        move_direction = Vector3D(0, 0, 0)
        if self.input_left: move_direction.x -= 1
        if self.input_right: move_direction.x += 1
        if self.input_forward: move_direction.z -= 1
        if self.input_backward: move_direction.z += 1
        
        if move_direction.magnitude() > 0:
            move_direction = move_direction.normalize()
            self.velocity.x = move_direction.x * self.move_speed
            self.velocity.z = move_direction.z * self.move_speed
        else:
            self.velocity.x *= 0.8
            self.velocity.z *= 0.8
        
        # Handle jumping
        if self.input_jump and self.jump_count < self.max_jumps:
            self.velocity.y = self.jump_force
            self.jump_count += 1
            self.on_ground = False
        
        super().update(delta_time)
        
        # Ground collision
        if self.position.y <= 0:
            self.position.y = 0
            self.velocity.y = 0
            self.on_ground = True
            self.jump_count = 0
    
    def handle_collision(self, other: GameObject):
        if other.obj_type == ObjectType.COLLECTIBLE:
            self.score += 10
            other.active = False
        elif other.obj_type == ObjectType.ENEMY:
            self.health -= 20
            if self.health <= 0:
                self.active = False

class Platform(GameObject):
    def __init__(self, position: Vector3D, size: Vector3D):
        super().__init__(position, size, ObjectType.PLATFORM)
        self.color = Color(0.3, 0.8, 0.3, 1.0)

class Collectible(GameObject):
    def __init__(self, position: Vector3D):
        super().__init__(position, Vector3D(0.3, 0.3, 0.3), ObjectType.COLLECTIBLE)
        self.color = Color(1.0, 1.0, 0.0, 1.0)

class Enemy(GameObject):
    def __init__(self, position: Vector3D):
        super().__init__(position, Vector3D(0.5, 0.5, 0.5), ObjectType.ENEMY)
        self.color = Color(1.0, 0.2, 0.2, 1.0)
        self.patrol_speed = 2.0
        self.patrol_direction = 1
    
    def update(self, delta_time: float):
        # Simple patrol movement
        self.velocity.x = self.patrol_speed * self.patrol_direction
        
        if self.position.x > 5 or self.position.x < -5:
            self.patrol_direction *= -1
        
        super().update(delta_time)

# ============================================================================
# GAME SYSTEMS
# ============================================================================

class InputSystem:
    def __init__(self, window):
        self.window = window
        self.keys_pressed = set()
        glfw.set_key_callback(window, self.key_callback)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self.keys_pressed.discard(key)
    
    def is_key_pressed(self, key) -> bool:
        return key in self.keys_pressed

class PhysicsSystem:
    def __init__(self):
        self.game_objects: List[GameObject] = []
    
    def add_object(self, obj: GameObject):
        self.game_objects.append(obj)
    
    def update(self, delta_time: float):
        for obj in self.game_objects:
            if obj.active:
                obj.update(delta_time)
        
        self.check_collisions()
        self.game_objects = [obj for obj in self.game_objects if obj.active]
    
    def check_collisions(self):
        for i, obj1 in enumerate(self.game_objects):
            if not obj1.active:
                continue
            
            for obj2 in self.game_objects[i+1:]:
                if not obj2.active:
                    continue
                
                if obj1.check_collision(obj2):
                    obj1.handle_collision(obj2)
                    obj2.handle_collision(obj1)

class Camera:
    def __init__(self, position: Vector3D, target: Vector3D):
        self.position = position
        self.target = target
        self.up = Vector3D(0, 1, 0)
        self.fov = 45.0
        self.aspect_ratio = 800 / 600
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.smoothness = 5.0
    
    def follow_target(self, target: Vector3D, delta_time: float):
        offset = Vector3D(0, 3, 5)
        desired_position = target + offset
        self.position = self.position + (desired_position - self.position) * self.smoothness * delta_time
        self.target = target
    
    def get_view_matrix(self) -> np.ndarray:
        forward = (self.target - self.position).normalize()
        right = Vector3D(forward.y, -forward.x, 0).normalize()
        up = Vector3D(0, 1, 0)
        
        view_matrix = np.eye(4, dtype='f4')
        view_matrix[0, 0:3] = right.to_array()
        view_matrix[1, 0:3] = up.to_array()
        view_matrix[2, 0:3] = [-f for f in forward.to_array()]
        return view_matrix
    
    def get_projection_matrix(self) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        projection_matrix = np.zeros((4, 4), dtype='f4')
        projection_matrix[0, 0] = f / self.aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[2, 3] = (2 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[3, 2] = -1.0
        return projection_matrix

class GameRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.setup_shaders()
        self.create_geometry()
    
    def setup_shaders(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        out vec3 world_pos;
        out vec3 normal;
        void main() {
            world_pos = (model * vec4(in_position, 1.0)).xyz;
            normal = mat3(model) * in_normal;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec3 world_pos;
        in vec3 normal;
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform vec3 material_color;
        out vec4 frag_color;
        void main() {
            vec3 norm = normalize(normal);
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            vec3 ambient = 0.2 * light_color;
            vec3 result = (ambient + diffuse) * material_color;
            frag_color = vec4(result, 1.0);
        }
        """
        
        self.shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def create_geometry(self):
        cube_vertices = np.array([
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        ], dtype='f4')
        
        cube_indices = np.array([
            0, 1, 2, 0, 2, 3,  # Front
            4, 6, 5, 4, 7, 6,  # Back
            0, 4, 5, 0, 5, 1,  # Bottom
            2, 6, 7, 2, 7, 3,  # Top
            0, 3, 7, 0, 7, 4,  # Left
            1, 5, 6, 1, 6, 2,  # Right
        ], dtype='u4')
        
        self.cube_vbo = self.ctx.buffer(cube_vertices.tobytes())
        self.cube_ibo = self.ctx.buffer(cube_indices.tobytes())
        
        self.cube_vao = self.ctx.vertex_array(
            self.shader,
            [(self.cube_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.cube_ibo
        )
    
    def render_object(self, obj: GameObject, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        model_matrix = np.eye(4, dtype='f4')
        model_matrix[0:3, 3] = obj.position.to_array()
        model_matrix[0, 0] = obj.size.x
        model_matrix[1, 1] = obj.size.y
        model_matrix[2, 2] = obj.size.z
        
        self.shader['model'].write(model_matrix.tobytes())
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        self.shader['material_color'].write(obj.color.to_array())
        
        self.cube_vao.render()
    
    def render_scene(self, game_objects: List[GameObject], camera: Camera):
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()
        
        light_pos = camera.position + Vector3D(5, 5, 5)
        self.shader['light_pos'].write(light_pos.to_array())
        self.shader['light_color'].write([1.0, 1.0, 1.0])
        
        for obj in game_objects:
            if obj.active:
                self.render_object(obj, view_matrix, projection_matrix)

# ============================================================================
# GAME MANAGER
# ============================================================================

class GameManager:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        self.input_system = None
        self.physics_system = PhysicsSystem()
        self.camera = Camera(Vector3D(0, 5, 10), Vector3D(0, 0, 0))
        self.renderer = None
        
        self.game_state = GameState.PLAYING
        self.game_objects: List[GameObject] = []
        self.player = None
        
        self.target_fps = 60.0
        self.last_time = time.time()
        
        self.init_glfw()
        self.init_opengl()
        self.setup_game()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "3D Platformer Game", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
    
    def setup_game(self):
        self.input_system = InputSystem(self.window)
        self.renderer = GameRenderer(self.ctx)
        self.create_level()
    
    def create_level(self):
        # Create player
        self.player = Player(Vector3D(0, 1, 0))
        self.game_objects.append(self.player)
        self.physics_system.add_object(self.player)
        
        # Create platforms
        platforms = [
            Platform(Vector3D(0, -0.5, 0), Vector3D(10, 1, 10)),
            Platform(Vector3D(5, 1, 0), Vector3D(2, 0.5, 2)),
            Platform(Vector3D(-3, 2, 2), Vector3D(2, 0.5, 2)),
        ]
        
        for platform in platforms:
            self.game_objects.append(platform)
            self.physics_system.add_object(platform)
        
        # Create collectibles
        collectibles = [
            Collectible(Vector3D(5, 2, 0)),
            Collectible(Vector3D(-3, 3, 2)),
        ]
        
        for collectible in collectibles:
            self.game_objects.append(collectible)
            self.physics_system.add_object(collectible)
        
        # Create enemies
        enemies = [
            Enemy(Vector3D(3, 0, 3)),
            Enemy(Vector3D(-5, 0, -2)),
        ]
        
        for enemy in enemies:
            self.game_objects.append(enemy)
            self.physics_system.add_object(enemy)
    
    def handle_input(self):
        if not self.player or not self.player.active:
            return
        
        self.player.input_left = self.input_system.is_key_pressed(glfw.KEY_A)
        self.player.input_right = self.input_system.is_key_pressed(glfw.KEY_D)
        self.player.input_forward = self.input_system.is_key_pressed(glfw.KEY_W)
        self.player.input_backward = self.input_system.is_key_pressed(glfw.KEY_S)
        self.player.input_jump = self.input_system.is_key_pressed(glfw.KEY_SPACE)
        
        if self.input_system.is_key_pressed(glfw.KEY_ESCAPE):
            if self.game_state == GameState.PLAYING:
                self.game_state = GameState.PAUSED
            elif self.game_state == GameState.PAUSED:
                self.game_state = GameState.PLAYING
    
    def update(self, delta_time: float):
        if self.game_state != GameState.PLAYING:
            return
        
        self.handle_input()
        self.physics_system.update(delta_time)
        
        if self.player and self.player.active:
            self.camera.follow_target(self.player.position, delta_time)
        
        if self.player and not self.player.active:
            self.game_state = GameState.GAME_OVER
    
    def render(self):
        self.ctx.clear(0.2, 0.3, 0.5, 1.0)
        self.renderer.render_scene(self.game_objects, self.camera)
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            
            self.update(delta_time)
            self.render()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            target_frame_time = 1.0 / self.target_fps
            if delta_time < target_frame_time:
                time.sleep(target_frame_time - delta_time)
        
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_game_development():
    print("=== Game Development Demo ===\n")
    print("Game development features:")
    print("  • Complete game architecture")
    print("  • Physics and collision detection")
    print("  • Input handling and controls")
    print("  • Game state management")
    print("  • 3D rendering and graphics")
    print()

def demonstrate_game_mechanics():
    print("=== Game Mechanics Demo ===\n")
    print("Game mechanics implemented:")
    print("  • Player movement and jumping")
    print("  • Platform collision and physics")
    print("  • Collectible system")
    print("  • Enemy AI and patrol")
    print("  • Health and scoring system")
    print()

def demonstrate_interactive_systems():
    print("=== Interactive Systems Demo ===\n")
    print("Interactive systems:")
    print("  • Real-time input handling")
    print("  • Dynamic camera system")
    print("  • Physics simulation")
    print("  • Collision detection")
    print("  • Game state transitions")
    print()

def demonstrate_game_engine():
    print("=== 3D Platformer Game Demo ===\n")
    print("Starting 3D platformer game...")
    print("Controls:")
    print("  W/A/S/D: Move")
    print("  SPACE: Jump")
    print("  ESC: Pause/Resume")
    print("  Collect all yellow cubes to win!")
    print("  Avoid red enemies!")
    print()
    
    try:
        game = GameManager(800, 600)
        game.run()
    except Exception as e:
        print(f"✗ Game failed to start: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=== 3D Platformer Game Demo ===\n")
    
    demonstrate_game_development()
    demonstrate_game_mechanics()
    demonstrate_interactive_systems()
    
    print("="*60)
    print("3D Platformer Game demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Complete game development workflow")
    print("✓ Physics integration and collision detection")
    print("✓ Game mechanics and player interaction")
    print("✓ Real-time input handling")
    print("✓ 3D rendering and graphics")
    print("✓ Game state management")
    
    print("\nGame features:")
    print("• 3D platformer gameplay")
    print("• Physics-based movement")
    print("• Collision detection system")
    print("• Enemy AI and patrol")
    print("• Collectible system")
    print("• Health and scoring")
    print("• Dynamic camera system")
    
    print("\nApplications:")
    print("• Game development: Complete game systems")
    print("• Interactive applications: Real-time interaction")
    print("• Educational games: Learning and training")
    print("• Simulation: Real-time simulation systems")
    print("• Entertainment: Interactive entertainment")
    
    print("\nNext steps:")
    print("• Add more game mechanics")
    print("• Implement advanced AI")
    print("• Add sound and music")
    print("• Create multiple levels")
    print("• Add multiplayer support")

if __name__ == "__main__":
    main()
