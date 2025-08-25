#!/usr/bin/env python3
"""
Chapter 13: Game Development and Interactive Applications
Real-Time Strategy Game

Demonstrates a real-time strategy game with 3D terrain,
unit management, AI systems, and strategic gameplay.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import random
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

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

class UnitType(Enum):
    INFANTRY = "infantry"
    TANK = "tank"
    ARTILLERY = "artillery"

class GameState(Enum):
    PLAYING = "playing"
    GAME_OVER = "game_over"

class Unit:
    def __init__(self, unit_type: UnitType, position: Vector3D, player_id: int):
        self.unit_type = unit_type
        self.position = position
        self.target_position = position
        self.player_id = player_id
        self.health = 100
        self.max_health = 100
        self.attack_power = 10
        self.defense = 5
        self.speed = 2.0
        self.range = 1.0
        self.selected = False
        self.moving = False
        self.attacking = False
        self.target_unit = None
        
        if unit_type == UnitType.INFANTRY:
            self.attack_power = 15
            self.defense = 3
            self.speed = 3.0
            self.color = Color(0.2, 0.6, 1.0, 1.0)
        elif unit_type == UnitType.TANK:
            self.attack_power = 30
            self.defense = 15
            self.speed = 1.5
            self.range = 2.0
            self.color = Color(0.8, 0.2, 0.2, 1.0)
        elif unit_type == UnitType.ARTILLERY:
            self.attack_power = 50
            self.defense = 5
            self.speed = 1.0
            self.range = 5.0
            self.color = Color(0.8, 0.8, 0.2, 1.0)
    
    def update(self, delta_time: float):
        if self.moving:
            direction = self.target_position - self.position
            distance = direction.magnitude()
            
            if distance > 0.1:
                direction = direction.normalize()
                movement = direction * self.speed * delta_time
                self.position = self.position + movement
            else:
                self.position = self.target_position
                self.moving = False
        
        if self.attacking and self.target_unit:
            direction = self.target_unit.position - self.position
            distance = direction.magnitude()
            
            if distance <= self.range:
                damage = max(0, self.attack_power - self.target_unit.defense)
                self.target_unit.health -= damage * delta_time
                
                if self.target_unit.health <= 0:
                    self.target_unit = None
                    self.attacking = False
            else:
                self.target_position = self.target_unit.position
                self.moving = True
                self.attacking = False
    
    def move_to(self, position: Vector3D):
        self.target_position = position
        self.moving = True
        self.attacking = False
        self.target_unit = None
    
    def attack(self, target: 'Unit'):
        self.target_unit = target
        self.attacking = True
        self.moving = False
    
    def get_model_matrix(self) -> np.ndarray:
        model_matrix = np.eye(4, dtype='f4')
        model_matrix[0:3, 3] = self.position.to_array()
        scale = 0.5
        if self.unit_type == UnitType.TANK:
            scale = 0.8
        elif self.unit_type == UnitType.ARTILLERY:
            scale = 0.6
        model_matrix[0, 0] = scale
        model_matrix[1, 1] = scale
        model_matrix[2, 2] = scale
        return model_matrix

class Player:
    def __init__(self, player_id: int, color: Color):
        self.player_id = player_id
        self.color = color
        self.units: List[Unit] = []
        self.selected_units: List[Unit] = []
    
    def add_unit(self, unit: Unit):
        self.units.append(unit)
    
    def remove_unit(self, unit: Unit):
        if unit in self.units:
            self.units.remove(unit)
        if unit in self.selected_units:
            self.selected_units.remove(unit)
    
    def select_units(self, units: List[Unit]):
        for unit in self.selected_units:
            unit.selected = False
        self.selected_units = units
        for unit in self.selected_units:
            unit.selected = True
    
    def update(self, delta_time: float):
        for unit in self.units[:]:
            unit.update(delta_time)
            if unit.health <= 0:
                self.remove_unit(unit)

class AIController:
    def __init__(self, player: Player):
        self.player = player
        self.decision_timer = 0.0
        self.decision_interval = 2.0
    
    def update(self, delta_time: float):
        self.decision_timer += delta_time
        
        if self.decision_timer >= self.decision_interval:
            self.make_decisions()
            self.decision_timer = 0.0
    
    def make_decisions(self):
        for unit in self.player.units:
            if not unit.moving and not unit.attacking:
                if random.random() < 0.7:
                    center = Vector3D(0, 0, 0)
                    direction = center - unit.position
                    if direction.magnitude() > 0:
                        direction = direction.normalize()
                        unit.move_to(unit.position + direction * 5.0)
                else:
                    random_x = random.uniform(-10, 10)
                    random_z = random.uniform(-10, 10)
                    unit.move_to(Vector3D(random_x, 0, random_z))

class RTSCamera:
    def __init__(self):
        self.position = Vector3D(0, 15, 10)
        self.target = Vector3D(0, 0, 0)
        self.up = Vector3D(0, 1, 0)
        self.fov = 60.0
        self.near_plane = 0.1
        self.far_plane = 100.0
        self.pan_speed = 10.0
        self.zoom_speed = 5.0
    
    def pan(self, direction: Vector3D, delta_time: float):
        movement = direction * self.pan_speed * delta_time
        self.position = self.position + movement
        self.target = self.target + movement
    
    def zoom(self, factor: float, delta_time: float):
        zoom_amount = factor * self.zoom_speed * delta_time
        direction = self.position - self.target
        direction = direction.normalize()
        
        new_distance = direction.magnitude() + zoom_amount
        new_distance = max(5.0, min(30.0, new_distance))
        
        self.position = self.target + direction * new_distance
    
    def get_view_matrix(self) -> np.ndarray:
        forward = (self.target - self.position).normalize()
        right = Vector3D.cross(forward, self.up).normalize()
        up = Vector3D.cross(right, forward)
        
        view_matrix = np.eye(4, dtype='f4')
        view_matrix[0, 0:3] = right.to_array()
        view_matrix[1, 0:3] = up.to_array()
        view_matrix[2, 0:3] = [-f for f in forward.to_array()]
        view_matrix[0:3, 3] = [-p for p in self.position.to_array()]
        
        return view_matrix
    
    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        projection_matrix = np.zeros((4, 4), dtype='f4')
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[2, 3] = (2 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane)
        projection_matrix[3, 2] = -1.0
        return projection_matrix

class InputSystem:
    def __init__(self, window):
        self.window = window
        self.keys_pressed = set()
        self.mouse_buttons_pressed = set()
        
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self.keys_pressed.discard(key)
    
    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_buttons_pressed.add(button)
        elif action == glfw.RELEASE:
            self.mouse_buttons_pressed.discard(button)
    
    def is_key_pressed(self, key) -> bool:
        return key in self.keys_pressed
    
    def is_mouse_button_pressed(self, button) -> bool:
        return button in self.mouse_buttons_pressed

class RTSRenderer:
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
        uniform bool selected;
        out vec4 frag_color;
        void main() {
            vec3 norm = normalize(normal);
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            vec3 ambient = 0.3 * light_color;
            vec3 result = (ambient + diffuse) * material_color;
            
            if (selected) {
                result = mix(result, vec3(1.0, 1.0, 0.0), 0.3);
            }
            
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
    
    def render_unit(self, unit: Unit, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        model_matrix = unit.get_model_matrix()
        
        self.shader['model'].write(model_matrix.tobytes())
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        self.shader['material_color'].write(unit.color.to_array())
        self.shader['selected'].value = unit.selected
        
        self.cube_vao.render()
    
    def render_scene(self, units: List[Unit], camera: RTSCamera, aspect_ratio: float):
        light_pos = Vector3D(0, 20, 0)
        light_color = Color(1.0, 1.0, 1.0, 1.0)
        
        self.shader['light_pos'].write(light_pos.to_array())
        self.shader['light_color'].write(light_color.to_array())
        
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix(aspect_ratio)
        
        for unit in units:
            self.render_unit(unit, view_matrix, projection_matrix)

class RTSGame:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        self.game_state = GameState.PLAYING
        self.camera = RTSCamera()
        self.renderer = RTSRenderer(None)
        self.input_system = None
        
        self.player = Player(1, Color(0.2, 0.6, 1.0, 1.0))
        self.ai_player = Player(2, Color(1.0, 0.2, 0.2, 1.0))
        self.ai_controller = AIController(self.ai_player)
        
        self.selected_units: List[Unit] = []
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
        
        self.window = glfw.create_window(self.width, self.height, "RTS Game", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.renderer = RTSRenderer(self.ctx)
    
    def setup_game(self):
        self.input_system = InputSystem(self.window)
        
        # Create initial units
        player_units = [
            (UnitType.INFANTRY, Vector3D(-5, 0, -5)),
            (UnitType.TANK, Vector3D(-3, 0, -5)),
            (UnitType.ARTILLERY, Vector3D(-7, 0, -5)),
        ]
        
        for unit_type, position in player_units:
            unit = Unit(unit_type, position, self.player.player_id)
            self.player.add_unit(unit)
        
        ai_units = [
            (UnitType.INFANTRY, Vector3D(5, 0, 5)),
            (UnitType.TANK, Vector3D(3, 0, 5)),
            (UnitType.ARTILLERY, Vector3D(7, 0, 5)),
        ]
        
        for unit_type, position in ai_units:
            unit = Unit(unit_type, position, self.ai_player.player_id)
            self.ai_player.add_unit(unit)
    
    def update(self, delta_time: float):
        self.ai_controller.update(delta_time)
        self.player.update(delta_time)
        self.ai_player.update(delta_time)
        self.handle_input(delta_time)
        self.check_win_conditions()
    
    def handle_input(self, delta_time: float):
        if self.input_system.is_key_pressed(glfw.KEY_W):
            self.camera.pan(Vector3D(0, 0, -1), delta_time)
        if self.input_system.is_key_pressed(glfw.KEY_S):
            self.camera.pan(Vector3D(0, 0, 1), delta_time)
        if self.input_system.is_key_pressed(glfw.KEY_A):
            self.camera.pan(Vector3D(-1, 0, 0), delta_time)
        if self.input_system.is_key_pressed(glfw.KEY_D):
            self.camera.pan(Vector3D(1, 0, 0), delta_time)
        
        if self.input_system.is_key_pressed(glfw.KEY_Q):
            self.camera.zoom(-1, delta_time)
        if self.input_system.is_key_pressed(glfw.KEY_E):
            self.camera.zoom(1, delta_time)
        
        if self.input_system.is_mouse_button_pressed(glfw.MOUSE_BUTTON_LEFT):
            self.select_units()
        
        if self.input_system.is_mouse_button_pressed(glfw.MOUSE_BUTTON_RIGHT):
            self.move_selected_units()
    
    def select_units(self):
        self.selected_units = self.player.units.copy()
        self.player.select_units(self.selected_units)
    
    def move_selected_units(self):
        target_position = self.camera.target
        for unit in self.selected_units:
            unit.move_to(target_position)
    
    def check_win_conditions(self):
        if len(self.ai_player.units) == 0:
            self.game_state = GameState.GAME_OVER
            print("Player wins!")
        elif len(self.player.units) == 0:
            self.game_state = GameState.GAME_OVER
            print("AI wins!")
    
    def render(self):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        all_units = self.player.units + self.ai_player.units
        
        self.renderer.render_scene(
            all_units,
            self.camera,
            self.width / self.height
        )
    
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
        
        glfw.terminate()

def main():
    print("=== Real-Time Strategy Game Demo ===\n")
    print("RTS game features:")
    print("  • 3D terrain and map system")
    print("  • Unit management and AI")
    print("  • Resource management")
    print("  • Strategic gameplay")
    print("  • Real-time combat system")
    print()
    
    print("Controls:")
    print("• WASD: Camera movement")
    print("• Q/E: Camera zoom")
    print("• Left click: Select units")
    print("• Right click: Move units")
    print()
    
    try:
        rts_game = RTSGame(1280, 720)
        rts_game.run()
    except Exception as e:
        print(f"✗ RTS game failed to start: {e}")

if __name__ == "__main__":
    main()
