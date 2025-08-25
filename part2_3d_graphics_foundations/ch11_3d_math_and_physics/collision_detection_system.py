#!/usr/bin/env python3
"""
Chapter 11: 3D Math and Physics
Collision Detection System

Demonstrates broad phase and narrow phase collision detection algorithms,
collision response, and performance optimization techniques.
"""

import numpy as np
import moderngl
import glfw
import math
import time
import random
from typing import Dict, List, Tuple, Any, Optional
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
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
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

class CollisionShape(Enum):
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    MESH = "mesh"

@dataclass
class BoundingBox:
    min_point: Vector3D
    max_point: Vector3D
    
    def contains_point(self, point: Vector3D) -> bool:
        return (self.min_point.x <= point.x <= self.max_point.x and
                self.min_point.y <= point.y <= self.max_point.y and
                self.min_point.z <= point.z <= self.max_point.z)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.max_point.x < other.min_point.x or
                   self.min_point.x > other.max_point.x or
                   self.max_point.y < other.min_point.y or
                   self.min_point.y > other.max_point.y or
                   self.max_point.z < other.min_point.z or
                   self.min_point.z > other.max_point.z)
    
    def get_center(self) -> Vector3D:
        return Vector3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )
    
    def get_size(self) -> Vector3D:
        return Vector3D(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )

class CollisionObject:
    def __init__(self, position: Vector3D, shape: CollisionShape, size: Vector3D):
        self.position = position
        self.shape = shape
        self.size = size
        self.velocity = Vector3D(0, 0, 0)
        self.bounding_box = self.calculate_bounding_box()
        self.active = True
        self.collision_count = 0
    
    def calculate_bounding_box(self) -> BoundingBox:
        half_size = self.size * 0.5
        return BoundingBox(
            self.position - half_size,
            self.position + half_size
        )
    
    def update_position(self, delta_time: float):
        self.position = self.position + self.velocity * delta_time
        self.bounding_box = self.calculate_bounding_box()
    
    def check_collision_sphere_sphere(self, other: 'CollisionObject') -> bool:
        if self.shape != CollisionShape.SPHERE or other.shape != CollisionShape.SPHERE:
            return False
        
        distance = (self.position - other.position).magnitude()
        combined_radius = (self.size.x + other.size.x) / 2
        return distance < combined_radius
    
    def check_collision_sphere_box(self, other: 'CollisionObject') -> bool:
        if self.shape != CollisionShape.SPHERE or other.shape != CollisionShape.BOX:
            return False
        
        # Find closest point on box to sphere center
        closest_x = max(other.bounding_box.min_point.x, 
                       min(self.position.x, other.bounding_box.max_point.x))
        closest_y = max(other.bounding_box.min_point.y, 
                       min(self.position.y, other.bounding_box.max_point.y))
        closest_z = max(other.bounding_box.min_point.z, 
                       min(self.position.z, other.bounding_box.max_point.z))
        
        closest_point = Vector3D(closest_x, closest_y, closest_z)
        distance = (self.position - closest_point).magnitude()
        sphere_radius = self.size.x / 2
        
        return distance < sphere_radius
    
    def check_collision_box_box(self, other: 'CollisionObject') -> bool:
        if self.shape != CollisionShape.BOX or other.shape != CollisionShape.BOX:
            return False
        
        return self.bounding_box.intersects(other.bounding_box)
    
    def check_collision(self, other: 'CollisionObject') -> bool:
        if not self.active or not other.active:
            return False
        
        if self.shape == CollisionShape.SPHERE and other.shape == CollisionShape.SPHERE:
            return self.check_collision_sphere_sphere(other)
        elif self.shape == CollisionShape.SPHERE and other.shape == CollisionShape.BOX:
            return self.check_collision_sphere_box(other)
        elif self.shape == CollisionShape.BOX and other.shape == CollisionShape.SPHERE:
            return other.check_collision_sphere_box(self)
        elif self.shape == CollisionShape.BOX and other.shape == CollisionShape.BOX:
            return self.check_collision_box_box(other)
        
        return False

class SpatialHashGrid:
    def __init__(self, cell_size: float = 2.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[CollisionObject]] = {}
    
    def get_cell_key(self, position: Vector3D) -> Tuple[int, int, int]:
        x = int(position.x / self.cell_size)
        y = int(position.y / self.cell_size)
        z = int(position.z / self.cell_size)
        return (x, y, z)
    
    def add_object(self, obj: CollisionObject):
        cell_key = self.get_cell_key(obj.position)
        if cell_key not in self.grid:
            self.grid[cell_key] = []
        self.grid[cell_key].append(obj)
    
    def get_nearby_objects(self, obj: CollisionObject) -> List[CollisionObject]:
        nearby = []
        cell_key = self.get_cell_key(obj.position)
        
        # Check current cell and neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (cell_key[0] + dx, cell_key[1] + dy, cell_key[2] + dz)
                    if neighbor_key in self.grid:
                        nearby.extend(self.grid[neighbor_key])
        
        return nearby
    
    def clear(self):
        self.grid.clear()

class CollisionDetector:
    def __init__(self):
        self.spatial_grid = SpatialHashGrid()
        self.collision_pairs = []
        self.broad_phase_calls = 0
        self.narrow_phase_calls = 0
    
    def broad_phase(self, objects: List[CollisionObject]) -> List[Tuple[CollisionObject, CollisionObject]]:
        self.broad_phase_calls += 1
        potential_pairs = []
        
        # Clear and rebuild spatial grid
        self.spatial_grid.clear()
        for obj in objects:
            if obj.active:
                self.spatial_grid.add_object(obj)
        
        # Find potential collision pairs
        checked_pairs = set()
        for obj in objects:
            if not obj.active:
                continue
            
            nearby_objects = self.spatial_grid.get_nearby_objects(obj)
            for other in nearby_objects:
                if obj == other:
                    continue
                
                pair_key = tuple(sorted([id(obj), id(other)]))
                if pair_key not in checked_pairs:
                    checked_pairs.add(pair_key)
                    potential_pairs.append((obj, other))
        
        return potential_pairs
    
    def narrow_phase(self, potential_pairs: List[Tuple[CollisionObject, CollisionObject]]) -> List[Tuple[CollisionObject, CollisionObject]]:
        self.narrow_phase_calls += 1
        actual_collisions = []
        
        for obj1, obj2 in potential_pairs:
            if obj1.check_collision(obj2):
                actual_collisions.append((obj1, obj2))
                obj1.collision_count += 1
                obj2.collision_count += 1
        
        return actual_collisions
    
    def detect_collisions(self, objects: List[CollisionObject]) -> List[Tuple[CollisionObject, CollisionObject]]:
        # Broad phase: find potential collision pairs
        potential_pairs = self.broad_phase(objects)
        
        # Narrow phase: precise collision detection
        actual_collisions = self.narrow_phase(potential_pairs)
        
        self.collision_pairs = actual_collisions
        return actual_collisions
    
    def resolve_collisions(self, collisions: List[Tuple[CollisionObject, CollisionObject]]):
        for obj1, obj2 in collisions:
            self.resolve_collision(obj1, obj2)
    
    def resolve_collision(self, obj1: CollisionObject, obj2: CollisionObject):
        # Simple collision response: separate objects
        if obj1.shape == CollisionShape.SPHERE and obj2.shape == CollisionShape.SPHERE:
            self.resolve_sphere_sphere_collision(obj1, obj2)
        elif obj1.shape == CollisionShape.SPHERE and obj2.shape == CollisionShape.BOX:
            self.resolve_sphere_box_collision(obj1, obj2)
        elif obj1.shape == CollisionShape.BOX and obj2.shape == CollisionShape.BOX:
            self.resolve_box_box_collision(obj1, obj2)
    
    def resolve_sphere_sphere_collision(self, sphere1: CollisionObject, sphere2: CollisionObject):
        # Calculate separation vector
        separation = sphere1.position - sphere2.position
        distance = separation.magnitude()
        
        if distance == 0:
            separation = Vector3D(1, 0, 0)
            distance = 1
        
        combined_radius = (sphere1.size.x + sphere2.size.x) / 2
        overlap = combined_radius - distance
        
        if overlap > 0:
            # Separate spheres
            separation_normal = separation.normalize()
            separation_vector = separation_normal * (overlap * 0.5)
            
            sphere1.position = sphere1.position + separation_vector
            sphere2.position = sphere2.position - separation_vector
            
            # Update bounding boxes
            sphere1.bounding_box = sphere1.calculate_bounding_box()
            sphere2.bounding_box = sphere2.calculate_bounding_box()
    
    def resolve_sphere_box_collision(self, sphere: CollisionObject, box: CollisionObject):
        # Find closest point on box to sphere center
        closest_x = max(box.bounding_box.min_point.x, 
                       min(sphere.position.x, box.bounding_box.max_point.x))
        closest_y = max(box.bounding_box.min_point.y, 
                       min(sphere.position.y, box.bounding_box.max_point.y))
        closest_z = max(box.bounding_box.min_point.z, 
                       min(sphere.position.z, box.bounding_box.max_point.z))
        
        closest_point = Vector3D(closest_x, closest_y, closest_z)
        separation = sphere.position - closest_point
        distance = separation.magnitude()
        sphere_radius = sphere.size.x / 2
        
        if distance < sphere_radius:
            # Separate sphere from box
            separation_normal = separation.normalize()
            overlap = sphere_radius - distance
            separation_vector = separation_normal * overlap
            
            sphere.position = sphere.position + separation_vector
            sphere.bounding_box = sphere.calculate_bounding_box()
    
    def resolve_box_box_collision(self, box1: CollisionObject, box2: CollisionObject):
        # Simple box-box separation: move boxes apart along shortest axis
        center1 = box1.bounding_box.get_center()
        center2 = box2.bounding_box.get_center()
        separation = center1 - center2
        
        size1 = box1.bounding_box.get_size()
        size2 = box2.bounding_box.get_size()
        combined_size = (size1 + size2) * 0.5
        
        # Find shortest separation axis
        overlap_x = combined_size.x - abs(separation.x)
        overlap_y = combined_size.y - abs(separation.y)
        overlap_z = combined_size.z - abs(separation.z)
        
        if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
            min_overlap = min(overlap_x, overlap_y, overlap_z)
            
            if min_overlap == overlap_x:
                separation_vector = Vector3D(separation.x / abs(separation.x) * min_overlap * 0.5, 0, 0)
            elif min_overlap == overlap_y:
                separation_vector = Vector3D(0, separation.y / abs(separation.y) * min_overlap * 0.5, 0)
            else:
                separation_vector = Vector3D(0, 0, separation.z / abs(separation.z) * min_overlap * 0.5)
            
            box1.position = box1.position + separation_vector
            box2.position = box2.position - separation_vector
            
            box1.bounding_box = box1.calculate_bounding_box()
            box2.bounding_box = box2.calculate_bounding_box()

class CollisionDetectionSystem:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Collision detection components
        self.collision_detector = CollisionDetector()
        self.objects = []
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.detection_times = []
        
        # Statistics
        self.stats = {
            'total_objects': 0,
            'active_collisions': 0,
            'broad_phase_calls': 0,
            'narrow_phase_calls': 0,
            'avg_detection_time': 0.0
        }
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.create_scene()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Collision Detection System", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
    
    def init_opengl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.setup_shaders()
        self.create_geometry()
    
    def setup_shaders(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 color;
        out vec3 frag_color;
        void main() {
            frag_color = color;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec3 frag_color;
        out vec4 out_color;
        void main() {
            out_color = vec4(frag_color, 1.0);
        }
        """
        
        self.shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def create_geometry(self):
        # Create sphere geometry
        sphere_vertices = []
        sphere_indices = []
        
        segments = 16
        rings = 8
        
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for segment in range(segments + 1):
                theta = 2 * math.pi * segment / segments
                
                x = math.sin(phi) * math.cos(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.sin(theta)
                
                sphere_vertices.extend([x * 0.5, y * 0.5, z * 0.5])
        
        for ring in range(rings):
            for segment in range(segments):
                first = ring * (segments + 1) + segment
                second = first + segments + 1
                
                sphere_indices.extend([first, second, first + 1])
                sphere_indices.extend([second, second + 1, first + 1])
        
        sphere_vertices = np.array(sphere_vertices, dtype='f4')
        sphere_indices = np.array(sphere_indices, dtype='u4')
        
        self.sphere_vbo = self.ctx.buffer(sphere_vertices.tobytes())
        self.sphere_ibo = self.ctx.buffer(sphere_indices.tobytes())
        
        self.sphere_vao = self.ctx.vertex_array(
            self.shader,
            [(self.sphere_vbo, '3f', 'in_position')],
            self.sphere_ibo
        )
        
        # Create cube geometry
        cube_vertices = [
            # Front face
            -0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,
             0.5,  0.5,  0.5,
            -0.5,  0.5,  0.5,
            # Back face
            -0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,
             0.5,  0.5, -0.5,
            -0.5,  0.5, -0.5,
        ]
        
        cube_indices = [
            0, 1, 2, 2, 3, 0,  # Front
            1, 5, 6, 6, 2, 1,  # Right
            5, 4, 7, 7, 6, 5,  # Back
            4, 0, 3, 3, 7, 4,  # Left
            3, 2, 6, 6, 7, 3,  # Top
            4, 5, 1, 1, 0, 4   # Bottom
        ]
        
        cube_vertices = np.array(cube_vertices, dtype='f4')
        cube_indices = np.array(cube_indices, dtype='u4')
        
        self.cube_vbo = self.ctx.buffer(cube_vertices.tobytes())
        self.cube_ibo = self.ctx.buffer(cube_indices.tobytes())
        
        self.cube_vao = self.ctx.vertex_array(
            self.shader,
            [(self.cube_vbo, '3f', 'in_position')],
            self.cube_ibo
        )
    
    def create_scene(self):
        # Create various collision objects
        for i in range(20):
            x = random.uniform(-8, 8)
            y = random.uniform(-8, 8)
            z = random.uniform(-8, 8)
            position = Vector3D(x, y, z)
            
            # Random shape and size
            shape = random.choice([CollisionShape.SPHERE, CollisionShape.BOX])
            size = Vector3D(
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0)
            )
            
            obj = CollisionObject(position, shape, size)
            
            # Add some velocity for movement
            obj.velocity = Vector3D(
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-2, 2)
            )
            
            self.objects.append(obj)
        
        self.stats['total_objects'] = len(self.objects)
    
    def update_objects(self, delta_time: float):
        for obj in self.objects:
            if obj.active:
                obj.update_position(delta_time)
                
                # Bounce off boundaries
                for axis in ['x', 'y', 'z']:
                    pos_val = getattr(obj.position, axis)
                    vel_val = getattr(obj.velocity, axis)
                    size_val = getattr(obj.size, axis)
                    
                    if abs(pos_val) + size_val/2 > 10:
                        setattr(obj.velocity, axis, -vel_val * 0.8)
    
    def detect_and_resolve_collisions(self):
        start_time = time.time()
        
        # Detect collisions
        collisions = self.collision_detector.detect_collisions(self.objects)
        
        # Resolve collisions
        self.collision_detector.resolve_collisions(collisions)
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        # Update statistics
        self.stats['active_collisions'] = len(collisions)
        self.stats['broad_phase_calls'] = self.collision_detector.broad_phase_calls
        self.stats['narrow_phase_calls'] = self.collision_detector.narrow_phase_calls
        
        if self.detection_times:
            self.stats['avg_detection_time'] = np.mean(self.detection_times[-30:])
    
    def render_object(self, obj: CollisionObject):
        # Choose color based on collision state
        if obj.collision_count > 0:
            color = [1.0, 0.2, 0.2]  # Red for colliding objects
        else:
            color = [0.2, 0.8, 0.2]  # Green for non-colliding objects
        
        self.shader['color'].write(color)
        
        # Create model matrix
        model_matrix = np.eye(4, dtype='f4')
        model_matrix[0:3, 3] = obj.position.to_array()
        model_matrix[0, 0] = obj.size.x
        model_matrix[1, 1] = obj.size.y
        model_matrix[2, 2] = obj.size.z
        
        self.shader['model'].write(model_matrix.tobytes())
        
        # Render based on shape
        if obj.shape == CollisionShape.SPHERE:
            self.sphere_vao.render()
        else:
            self.cube_vao.render()
    
    def render_scene(self):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        view_matrix = np.eye(4, dtype='f4')
        view_matrix[2, 3] = -15
        projection_matrix = np.eye(4, dtype='f4')
        
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        
        # Render all objects
        for obj in self.objects:
            if obj.active:
                self.render_object(obj)
    
    def handle_input(self):
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            # Add new object
            position = Vector3D(
                random.uniform(-8, 8),
                random.uniform(-8, 8),
                random.uniform(-8, 8)
            )
            shape = random.choice([CollisionShape.SPHERE, CollisionShape.BOX])
            size = Vector3D(
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0),
                random.uniform(0.5, 2.0)
            )
            
            obj = CollisionObject(position, shape, size)
            obj.velocity = Vector3D(
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-2, 2)
            )
            
            self.objects.append(obj)
            self.stats['total_objects'] = len(self.objects)
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        last_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            self.handle_input()
            self.update_objects(delta_time)
            self.detect_and_resolve_collisions()
            self.render_scene()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Print statistics
            self.frame_count += 1
            if current_time - self.last_fps_time >= 2.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                print(f"FPS: {fps:.1f}, Objects: {self.stats['total_objects']}, "
                      f"Collisions: {self.stats['active_collisions']}, "
                      f"Avg Detection Time: {self.stats['avg_detection_time']*1000:.2f}ms")
                self.frame_count = 0
                self.last_fps_time = current_time
        
        glfw.terminate()

def main():
    print("=== Collision Detection System Demo ===\n")
    print("Collision detection features:")
    print("  • Broad phase collision detection")
    print("  • Narrow phase collision detection")
    print("  • Spatial hash grid optimization")
    print("  • Multiple collision shapes (sphere, box)")
    print("  • Collision response and resolution")
    print("  • Performance tracking and statistics")
    
    print("\nControls:")
    print("• SPACE: Add new collision object")
    
    print("\nApplications:")
    print("• Game physics and collision detection")
    print("• Simulation and modeling")
    print("• Performance optimization research")
    print("• Algorithm development and testing")
    
    try:
        collision_system = CollisionDetectionSystem(800, 600)
        collision_system.run()
    except Exception as e:
        print(f"✗ Collision detection system failed to start: {e}")

if __name__ == "__main__":
    main()
