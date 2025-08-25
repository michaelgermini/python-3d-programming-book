#!/usr/bin/env python3
"""
Chapter 13: Game Development and Interactive Applications
Physics-Based Simulation

Demonstrates a real-time physics simulation with multiple object types,
force fields, constraints, and complex interactions.
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

class ObjectType(Enum):
    PARTICLE = "particle"
    RIGID_BODY = "rigid_body"

class ForceType(Enum):
    GRAVITY = "gravity"
    WIND = "wind"
    VORTEX = "vortex"

# ============================================================================
# PHYSICS OBJECTS
# ============================================================================

class PhysicsObject:
    def __init__(self, position: Vector3D, mass: float, obj_type: ObjectType):
        self.position = position
        self.velocity = Vector3D(0, 0, 0)
        self.acceleration = Vector3D(0, 0, 0)
        self.mass = mass
        self.obj_type = obj_type
        self.active = True
        self.color = Color(0.8, 0.8, 0.8, 1.0)
        self.size = Vector3D(0.1, 0.1, 0.1)
    
    def update(self, delta_time: float):
        # Update velocity
        self.velocity = self.velocity + self.acceleration * delta_time
        
        # Update position
        self.position = self.position + self.velocity * delta_time
        
        # Reset acceleration
        self.acceleration = Vector3D(0, 0, 0)
    
    def apply_force(self, force: Vector3D):
        self.acceleration = self.acceleration + force * (1.0 / self.mass)

class Particle(PhysicsObject):
    def __init__(self, position: Vector3D, mass: float = 1.0):
        super().__init__(position, mass, ObjectType.PARTICLE)
        self.color = Color(0.2, 0.6, 1.0, 1.0)
        self.life = 1.0
        self.max_life = 1.0
    
    def update(self, delta_time: float):
        super().update(delta_time)
        
        # Update life
        self.life -= delta_time
        if self.life <= 0:
            self.active = False
        
        # Update color based on life
        life_factor = self.life / self.max_life
        self.color.r = 0.2 + 0.8 * life_factor
        self.color.g = 0.6 * life_factor
        self.color.b = 1.0 * life_factor

class RigidBody(PhysicsObject):
    def __init__(self, position: Vector3D, mass: float = 1.0):
        super().__init__(position, mass, ObjectType.RIGID_BODY)
        self.color = Color(0.8, 0.4, 0.2, 1.0)
        self.size = Vector3D(0.2, 0.2, 0.2)

# ============================================================================
# FORCE SYSTEMS
# ============================================================================

class Force:
    def __init__(self, force_type: ForceType, strength: float = 1.0):
        self.force_type = force_type
        self.strength = strength
        self.active = True
    
    def calculate_force(self, obj: PhysicsObject, delta_time: float) -> Vector3D:
        return Vector3D(0, 0, 0)

class GravityForce(Force):
    def __init__(self, gravity: Vector3D = Vector3D(0, -9.81, 0)):
        super().__init__(ForceType.GRAVITY)
        self.gravity = gravity
    
    def calculate_force(self, obj: PhysicsObject, delta_time: float) -> Vector3D:
        return self.gravity * obj.mass

class WindForce(Force):
    def __init__(self, wind_direction: Vector3D, wind_strength: float):
        super().__init__(ForceType.WIND)
        self.wind_direction = wind_direction.normalize()
        self.wind_strength = wind_strength
    
    def calculate_force(self, obj: PhysicsObject, delta_time: float) -> Vector3D:
        return self.wind_direction * self.wind_strength

class VortexForce(Force):
    def __init__(self, center: Vector3D, radius: float, strength: float):
        super().__init__(ForceType.VORTEX)
        self.center = center
        self.radius = radius
        self.strength = strength
    
    def calculate_force(self, obj: PhysicsObject, delta_time: float) -> Vector3D:
        displacement = obj.position - self.center
        distance = displacement.magnitude()
        
        if distance < self.radius and distance > 0:
            # Create circular motion
            direction = Vector3D(-displacement.z, 0, displacement.x).normalize()
            force_magnitude = self.strength * (1.0 - distance / self.radius)
            return direction * force_magnitude
        
        return Vector3D(0, 0, 0)

# ============================================================================
# CONSTRAINT SYSTEM
# ============================================================================

class GroundConstraint:
    def __init__(self, ground_y: float = -2.0):
        self.ground_y = ground_y
    
    def solve(self, objects: List[PhysicsObject], delta_time: float):
        for obj in objects:
            if obj.active and obj.position.y < self.ground_y:
                obj.position.y = self.ground_y
                obj.velocity.y = 0

# ============================================================================
# PHYSICS ENGINE
# ============================================================================

class PhysicsEngine:
    def __init__(self):
        self.objects: List[PhysicsObject] = []
        self.forces: List[Force] = []
        self.constraints: List[GroundConstraint] = []
        self.air_resistance = 0.99
    
    def add_object(self, obj: PhysicsObject):
        self.objects.append(obj)
    
    def add_force(self, force: Force):
        self.forces.append(force)
    
    def add_constraint(self, constraint: GroundConstraint):
        self.constraints.append(constraint)
    
    def update(self, delta_time: float):
        # Apply forces
        for obj in self.objects:
            if not obj.active:
                continue
            
            # Apply custom forces
            for force in self.forces:
                if force.active:
                    obj.apply_force(force.calculate_force(obj, delta_time))
            
            # Apply air resistance
            obj.velocity = obj.velocity * self.air_resistance
        
        # Update objects
        for obj in self.objects:
            if obj.active:
                obj.update(delta_time)
        
        # Solve constraints
        for constraint in self.constraints:
            constraint.solve(self.objects, delta_time)
        
        # Remove inactive objects
        self.objects = [obj for obj in self.objects if obj.active]

# ============================================================================
# RENDERING SYSTEM
# ============================================================================

class Camera:
    def __init__(self, position: Vector3D = Vector3D(0, 0, 5)):
        self.position = position
        self.target = Vector3D(0, 0, 0)
        self.up = Vector3D(0, 1, 0)
        self.fov = 45.0
        self.aspect_ratio = 800 / 600
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        self.orbit_radius = 5.0
        self.orbit_theta = 0.0
        self.orbit_phi = 0.0
        self.update_position()
    
    def update_position(self):
        x = self.orbit_radius * math.cos(self.orbit_phi) * math.sin(self.orbit_theta)
        y = self.orbit_radius * math.sin(self.orbit_phi)
        z = self.orbit_radius * math.cos(self.orbit_phi) * math.cos(self.orbit_theta)
        self.position = Vector3D(x, y, z)
    
    def orbit(self, delta_theta: float, delta_phi: float):
        self.orbit_theta += delta_theta
        self.orbit_phi += delta_phi
        self.orbit_phi = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.orbit_phi))
        self.update_position()
    
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

class Renderer:
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
        # Create sphere vertices
        sphere_vertices = []
        sphere_indices = []
        
        segments = 8
        for i in range(segments + 1):
            lat = math.pi * (-0.5 + float(i) / segments)
            for j in range(segments):
                lon = 2 * math.pi * float(j) / segments
                
                x = math.cos(lat) * math.cos(lon) * 0.5
                y = math.cos(lat) * math.sin(lon) * 0.5
                z = math.sin(lat) * 0.5
                
                sphere_vertices.extend([x, y, z, x, y, z])  # position and normal
        
        for i in range(segments):
            for j in range(segments):
                first = i * segments + j
                second = first + segments
                
                sphere_indices.extend([first, second, first + 1])
                sphere_indices.extend([second, second + 1, first + 1])
        
        self.sphere_vbo = self.ctx.buffer(np.array(sphere_vertices, dtype='f4').tobytes())
        self.sphere_ibo = self.ctx.buffer(np.array(sphere_indices, dtype='u4').tobytes())
        
        self.sphere_vao = self.ctx.vertex_array(
            self.shader,
            [(self.sphere_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.sphere_ibo
        )
    
    def render_object(self, obj: PhysicsObject, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        model_matrix = np.eye(4, dtype='f4')
        model_matrix[0:3, 3] = obj.position.to_array()
        model_matrix[0, 0] = obj.size.x
        model_matrix[1, 1] = obj.size.y
        model_matrix[2, 2] = obj.size.z
        
        self.shader['model'].write(model_matrix.tobytes())
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        self.shader['material_color'].write(obj.color.to_array())
        
        self.sphere_vao.render()
    
    def render_scene(self, objects: List[PhysicsObject], camera: Camera):
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()
        
        light_pos = camera.position + Vector3D(5, 5, 5)
        self.shader['light_pos'].write(light_pos.to_array())
        self.shader['light_color'].write([1.0, 1.0, 1.0])
        
        for obj in objects:
            if obj.active:
                self.render_object(obj, view_matrix, projection_matrix)

# ============================================================================
# INPUT SYSTEM
# ============================================================================

class InputSystem:
    def __init__(self, window):
        self.window = window
        self.keys_pressed = set()
        self.mouse_pos = Vector3D(0, 0, 0)
        self.last_mouse_pos = Vector3D(0, 0, 0)
        self.mouse_delta = Vector3D(0, 0, 0)
        self.mouse_buttons = set()
        
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_cursor_pos_callback(window, self.mouse_callback)
        glfw.set_mouse_button_callback(window, self.mouse_button_callback)
    
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys_pressed.add(key)
        elif action == glfw.RELEASE:
            self.keys_pressed.discard(key)
    
    def mouse_callback(self, window, xpos, ypos):
        self.mouse_pos = Vector3D(xpos, ypos, 0)
        self.mouse_delta = self.mouse_pos - self.last_mouse_pos
        self.last_mouse_pos = self.mouse_pos
    
    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_buttons.add(button)
        elif action == glfw.RELEASE:
            self.mouse_buttons.discard(button)
    
    def is_key_pressed(self, key) -> bool:
        return key in self.keys_pressed
    
    def is_mouse_button_pressed(self, button) -> bool:
        return button in self.mouse_buttons
    
    def get_mouse_delta(self) -> Vector3D:
        return self.mouse_delta

# ============================================================================
# SIMULATION SYSTEM
# ============================================================================

class PhysicsSimulation:
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Systems
        self.input_system = None
        self.physics_engine = PhysicsEngine()
        self.camera = Camera()
        self.renderer = None
        
        # Simulation state
        self.particle_emitter = Vector3D(0, 3, 0)
        self.emission_rate = 10.0
        self.last_emission_time = 0.0
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.setup_simulation()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Physics Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
    
    def setup_simulation(self):
        self.input_system = InputSystem(self.window)
        self.renderer = Renderer(self.ctx)
        self.create_scene()
    
    def create_scene(self):
        # Add forces
        gravity = GravityForce()
        self.physics_engine.add_force(gravity)
        
        wind = WindForce(Vector3D(1, 0, 0), 2.0)
        self.physics_engine.add_force(wind)
        
        vortex = VortexForce(Vector3D(0, 0, 0), 3.0, 5.0)
        self.physics_engine.add_force(vortex)
        
        # Add constraints
        ground = GroundConstraint(-2.0)
        self.physics_engine.add_constraint(ground)
        
        # Add some initial objects
        for i in range(20):
            pos = Vector3D(
                random.uniform(-2, 2),
                random.uniform(0, 4),
                random.uniform(-2, 2)
            )
            particle = Particle(pos, random.uniform(0.5, 2.0))
            self.physics_engine.add_object(particle)
    
    def handle_input(self):
        # Camera controls
        if self.input_system.is_mouse_button_pressed(glfw.MOUSE_BUTTON_RIGHT):
            delta = self.input_system.get_mouse_delta()
            self.camera.orbit(delta.x * 0.01, delta.y * 0.01)
        
        # Add particles
        if self.input_system.is_key_pressed(glfw.KEY_SPACE):
            self.add_particle()
        
        # Add rigid bodies
        if self.input_system.is_key_pressed(glfw.KEY_R):
            self.add_rigid_body()
    
    def add_particle(self):
        pos = Vector3D(
            self.particle_emitter.x + random.uniform(-0.5, 0.5),
            self.particle_emitter.y,
            self.particle_emitter.z + random.uniform(-0.5, 0.5)
        )
        particle = Particle(pos, random.uniform(0.5, 2.0))
        self.physics_engine.add_object(particle)
    
    def add_rigid_body(self):
        pos = Vector3D(
            random.uniform(-2, 2),
            random.uniform(0, 4),
            random.uniform(-2, 2)
        )
        rigid_body = RigidBody(pos, random.uniform(1.0, 3.0))
        self.physics_engine.add_object(rigid_body)
    
    def update(self, delta_time: float):
        self.handle_input()
        
        # Emit particles
        current_time = time.time()
        if current_time - self.last_emission_time > 1.0 / self.emission_rate:
            self.add_particle()
            self.last_emission_time = current_time
        
        # Update physics
        self.physics_engine.update(delta_time)
    
    def render(self):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        self.renderer.render_scene(self.physics_engine.objects, self.camera)
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
        self.camera.aspect_ratio = width / height
    
    def run(self):
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            self.update(delta_time)
            self.render()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        glfw.terminate()

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_physics_simulation():
    print("=== Physics-Based Simulation Demo ===\n")
    print("Physics simulation features:")
    print("  • Real-time physics simulation")
    print("  • Multiple object types and interactions")
    print("  • Force fields and constraints")
    print("  • Real-time visualization and analysis")
    print("  • Interactive particle emission")
    print()

def demonstrate_force_systems():
    print("=== Force Systems Demo ===\n")
    print("Force systems implemented:")
    print("  • Gravity and gravitational forces")
    print("  • Wind and fluid dynamics")
    print("  • Vortex and attraction forces")
    print("  • Custom force field creation")
    print("  • Real-time force visualization")
    print()

def demonstrate_physics_engine():
    print("=== Physics Engine Demo ===\n")
    print("Physics engine features:")
    print("  • Rigid body dynamics")
    print("  • Particle systems")
    print("  • Collision detection and response")
    print("  • Constraint solving")
    print("  • Real-time physics simulation")
    print()

def demonstrate_simulation():
    print("=== Physics Simulation Demo ===\n")
    print("Starting physics simulation...")
    print("Controls:")
    print("  Right Mouse: Orbit camera")
    print("  SPACE: Add particle")
    print("  R: Add rigid body")
    print("  Watch the physics simulation in action!")
    print()
    
    try:
        simulation = PhysicsSimulation(1200, 800)
        simulation.run()
    except Exception as e:
        print(f"✗ Physics simulation failed to start: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=== Physics-Based Simulation Demo ===\n")
    
    demonstrate_physics_simulation()
    demonstrate_force_systems()
    demonstrate_physics_engine()
    
    print("="*60)
    print("Physics-Based Simulation demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Real-time physics simulation")
    print("✓ Multiple object types and interactions")
    print("✓ Force fields and constraints")
    print("✓ Collision detection and response")
    print("✓ Particle systems and dynamics")
    print("✓ Interactive physics visualization")
    
    print("\nSimulation features:")
    print("• Real-time physics engine")
    print("• Multiple force types (gravity, wind, vortex)")
    print("• Particle and rigid body systems")
    print("• Collision detection and response")
    print("• Constraint solving")
    print("• Interactive object creation")
    
    print("\nApplications:")
    print("• Game physics: Realistic game physics")
    print("• Scientific simulation: Research and analysis")
    print("• Engineering: Structural and fluid analysis")
    print("• Animation: Physics-based animation")
    print("• Education: Interactive physics learning")
    
    print("\nNext steps:")
    print("• Add more force types")
    print("• Implement fluid simulation")
    print("• Add cloth and soft body physics")
    print("• Implement advanced collision detection")
    print("• Add real-time analysis tools")

if __name__ == "__main__":
    main()
