#!/usr/bin/env python3
"""
Chapter 14: Computer Graphics Research and Development
Procedural Content Generation

Demonstrates AI-assisted 3D content creation with procedural generation,
automated asset generation, and research methodologies.
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

class ContentType(Enum):
    TERRAIN = "terrain"
    VEGETATION = "vegetation"
    BUILDINGS = "buildings"
    ROCKS = "rocks"
    WATER = "water"

class GenerationMethod(Enum):
    NOISE_BASED = "noise_based"
    FRACTAL = "fractal"
    CELLULAR = "cellular"
    AI_GENERATED = "ai_generated"

class NoiseGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.permutation_table = list(range(256))
        random.shuffle(self.permutation_table)
        self.permutation_table *= 2
    
    def noise_2d(self, x: float, y: float) -> float:
        # Simplified Perlin noise implementation
        xi = int(x) & 255
        yi = int(y) & 255
        xf = x - int(x)
        yf = y - int(y)
        
        # Hash the coordinates
        aa = self.permutation_table[self.permutation_table[xi] + yi]
        ab = self.permutation_table[self.permutation_table[xi] + yi + 1]
        ba = self.permutation_table[self.permutation_table[xi + 1] + yi]
        bb = self.permutation_table[self.permutation_table[xi + 1] + yi + 1]
        
        # Interpolate
        u = self.fade(xf)
        v = self.fade(yf)
        
        x1 = self.lerp(self.grad(aa, xf, yf), self.grad(ba, xf - 1, yf), u)
        x2 = self.lerp(self.grad(ab, xf, yf - 1), self.grad(bb, xf - 1, yf - 1), u)
        
        return self.lerp(x1, x2, v)
    
    def fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)
    
    def grad(self, hash_val: int, x: float, y: float) -> float:
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h != 12 and h != 14 else 0)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def fractal_noise(self, x: float, y: float, octaves: int = 4, persistence: float = 0.5) -> float:
        total = 0
        frequency = 1
        amplitude = 1
        max_value = 0
        
        for _ in range(octaves):
            total += self.noise_2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        
        return total / max_value

class TerrainGenerator:
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.noise_generator = NoiseGenerator()
        self.height_map = np.zeros((width, height), dtype=np.float32)
        self.moisture_map = np.zeros((width, height), dtype=np.float32)
        self.temperature_map = np.zeros((width, height), dtype=np.float32)
        
    def generate_terrain(self, scale: float = 0.1, octaves: int = 4) -> np.ndarray:
        for x in range(self.width):
            for y in range(self.height):
                nx = x * scale
                ny = y * scale
                
                # Generate height using fractal noise
                height = self.noise_generator.fractal_noise(nx, ny, octaves)
                
                # Add some variation
                height += 0.1 * self.noise_generator.noise_2d(nx * 2, ny * 2)
                
                self.height_map[x, y] = height
        
        return self.height_map
    
    def generate_moisture(self, scale: float = 0.05) -> np.ndarray:
        for x in range(self.width):
            for y in range(self.height):
                nx = x * scale
                ny = y * scale
                self.moisture_map[x, y] = self.noise_generator.fractal_noise(nx, ny, 3)
        return self.moisture_map
    
    def generate_temperature(self, scale: float = 0.03) -> np.ndarray:
        for x in range(self.width):
            for y in range(self.height):
                nx = x * scale
                ny = y * scale
                self.temperature_map[x, y] = self.noise_generator.fractal_noise(nx, ny, 2)
        return self.temperature_map

class VegetationGenerator:
    def __init__(self):
        self.tree_templates = {
            'pine': {'height': 8, 'width': 2, 'density': 0.3},
            'oak': {'height': 6, 'width': 4, 'density': 0.4},
            'bush': {'height': 2, 'width': 1.5, 'density': 0.6}
        }
    
    def generate_vegetation(self, terrain: TerrainGenerator, density: float = 0.1) -> List[Dict]:
        vegetation = []
        
        for x in range(terrain.width):
            for y in range(terrain.height):
                if random.random() < density:
                    height = terrain.height_map[x, y]
                    moisture = terrain.moisture_map[x, y]
                    temperature = terrain.temperature_map[x, y]
                    
                    # Determine vegetation type based on conditions
                    if height > 0.6 and moisture > 0.5:
                        tree_type = 'pine'
                    elif height > 0.3 and moisture > 0.3:
                        tree_type = 'oak'
                    else:
                        tree_type = 'bush'
                    
                    template = self.tree_templates[tree_type]
                    
                    # Add variation
                    scale = random.uniform(0.8, 1.2)
                    rotation = random.uniform(0, 2 * math.pi)
                    
                    vegetation.append({
                        'type': tree_type,
                        'position': Vector3D(x - terrain.width/2, height, y - terrain.height/2),
                        'scale': scale,
                        'rotation': rotation,
                        'template': template
                    })
        
        return vegetation

class BuildingGenerator:
    def __init__(self):
        self.building_templates = {
            'house': {'width': 4, 'height': 3, 'depth': 4},
            'tower': {'width': 2, 'height': 8, 'depth': 2},
            'barn': {'width': 6, 'height': 4, 'depth': 8}
        }
    
    def generate_buildings(self, terrain: TerrainGenerator, count: int = 5) -> List[Dict]:
        buildings = []
        
        for _ in range(count):
            x = random.randint(0, terrain.width - 1)
            y = random.randint(0, terrain.height - 1)
            height = terrain.height_map[x, y]
            
            # Only place buildings on flat areas
            if height < 0.4:
                building_type = random.choice(list(self.building_templates.keys()))
                template = self.building_templates[building_type]
                
                scale = random.uniform(0.8, 1.2)
                rotation = random.uniform(0, 2 * math.pi)
                
                buildings.append({
                    'type': building_type,
                    'position': Vector3D(x - terrain.width/2, height, y - terrain.height/2),
                    'scale': scale,
                    'rotation': rotation,
                    'template': template
                })
        
        return buildings

class ProceduralContentGenerator:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Content generation components
        self.terrain_generator = TerrainGenerator(32, 32)
        self.vegetation_generator = VegetationGenerator()
        self.building_generator = BuildingGenerator()
        
        # Generation settings
        self.content_type = ContentType.TERRAIN
        self.generation_method = GenerationMethod.NOISE_BASED
        self.auto_generate = True
        self.generation_interval = 5.0
        
        # Generated content
        self.terrain = None
        self.vegetation = []
        self.buildings = []
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.generation_times = []
        
        # Research metrics
        self.quality_metrics = {
            'diversity_score': 0.0,
            'coherence_score': 0.0,
            'performance_score': 0.0,
            'generation_speed': 0.0
        }
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.generate_content()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Procedural Content Generation", None, None)
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
        in vec3 in_normal;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 color;
        out vec3 world_pos;
        out vec3 normal;
        out vec3 frag_color;
        void main() {
            world_pos = (model * vec4(in_position, 1.0)).xyz;
            normal = mat3(model) * in_normal;
            frag_color = color;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        in vec3 world_pos;
        in vec3 normal;
        in vec3 frag_color;
        uniform vec3 light_pos;
        uniform vec3 light_color;
        out vec4 out_color;
        void main() {
            vec3 norm = normalize(normal);
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            vec3 ambient = 0.3 * light_color;
            vec3 result = (ambient + diffuse) * frag_color;
            out_color = vec4(result, 1.0);
        }
        """
        
        self.shader = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def create_geometry(self):
        # Create cube geometry for buildings and vegetation
        cube_vertices = [
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
            # Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
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
            [(self.cube_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.cube_ibo
        )
    
    def generate_content(self):
        start_time = time.time()
        
        if self.content_type == ContentType.TERRAIN:
            self.terrain = self.terrain_generator.generate_terrain()
            self.terrain_generator.generate_moisture()
            self.terrain_generator.generate_temperature()
        
        elif self.content_type == ContentType.VEGETATION:
            if self.terrain is None:
                self.terrain = self.terrain_generator.generate_terrain()
            self.vegetation = self.vegetation_generator.generate_vegetation(self.terrain_generator)
        
        elif self.content_type == ContentType.BUILDINGS:
            if self.terrain is None:
                self.terrain = self.terrain_generator.generate_terrain()
            self.buildings = self.building_generator.generate_buildings(self.terrain_generator)
        
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        self.update_quality_metrics()
        print(f"Generated {self.content_type.value} in {generation_time*1000:.2f}ms")
    
    def update_quality_metrics(self):
        # Calculate diversity score based on content variety
        if self.content_type == ContentType.VEGETATION:
            tree_types = set(item['type'] for item in self.vegetation)
            self.quality_metrics['diversity_score'] = len(tree_types) / len(self.vegetation_generator.tree_templates)
        elif self.content_type == ContentType.BUILDINGS:
            building_types = set(item['type'] for item in self.buildings)
            self.quality_metrics['diversity_score'] = len(building_types) / len(self.building_generator.building_templates)
        else:
            self.quality_metrics['diversity_score'] = 0.8
        
        # Calculate coherence score based on terrain consistency
        if self.terrain is not None:
            height_variance = np.var(self.terrain)
            self.quality_metrics['coherence_score'] = 1.0 / (1.0 + height_variance)
        
        # Calculate generation speed
        if self.generation_times:
            avg_generation_time = np.mean(self.generation_times[-5:])
            self.quality_metrics['generation_speed'] = 1.0 / (avg_generation_time + 0.001)
        
        # Calculate performance score
        self.quality_metrics['performance_score'] = (
            self.quality_metrics['diversity_score'] * 0.3 +
            self.quality_metrics['coherence_score'] * 0.3 +
            self.quality_metrics['generation_speed'] * 0.4
        )
    
    def render_terrain(self):
        if self.terrain is None:
            return
        
        # Render terrain as a simple grid
        for x in range(self.terrain_generator.width - 1):
            for y in range(self.terrain_generator.height - 1):
                height = self.terrain[x, y]
                color = [0.3 + height * 0.4, 0.2 + height * 0.3, 0.1 + height * 0.2]
                
                self.shader['color'].write(color)
                
                model_matrix = np.eye(4, dtype='f4')
                model_matrix[0:3, 3] = [x - self.terrain_generator.width/2, height, y - self.terrain_generator.height/2]
                model_matrix[0, 0] = 0.5
                model_matrix[2, 2] = 0.5
                
                self.shader['model'].write(model_matrix.tobytes())
                self.cube_vao.render()
    
    def render_vegetation(self):
        for item in self.vegetation:
            template = item['template']
            color = [0.1, 0.6, 0.1] if item['type'] in ['pine', 'oak'] else [0.3, 0.5, 0.1]
            
            self.shader['color'].write(color)
            
            model_matrix = np.eye(4, dtype='f4')
            model_matrix[0:3, 3] = item['position'].to_array()
            model_matrix[0, 0] = template['width'] * item['scale']
            model_matrix[1, 1] = template['height'] * item['scale']
            model_matrix[2, 2] = template['width'] * item['scale']
            
            # Apply rotation
            cos_r = math.cos(item['rotation'])
            sin_r = math.sin(item['rotation'])
            model_matrix[0, 0] *= cos_r
            model_matrix[0, 2] *= sin_r
            model_matrix[2, 0] *= -sin_r
            model_matrix[2, 2] *= cos_r
            
            self.shader['model'].write(model_matrix.tobytes())
            self.cube_vao.render()
    
    def render_buildings(self):
        for item in self.buildings:
            template = item['template']
            color = [0.6, 0.4, 0.2] if item['type'] == 'house' else [0.4, 0.4, 0.4]
            
            self.shader['color'].write(color)
            
            model_matrix = np.eye(4, dtype='f4')
            model_matrix[0:3, 3] = item['position'].to_array()
            model_matrix[0, 0] = template['width'] * item['scale']
            model_matrix[1, 1] = template['height'] * item['scale']
            model_matrix[2, 2] = template['depth'] * item['scale']
            
            # Apply rotation
            cos_r = math.cos(item['rotation'])
            sin_r = math.sin(item['rotation'])
            model_matrix[0, 0] *= cos_r
            model_matrix[0, 2] *= sin_r
            model_matrix[2, 0] *= -sin_r
            model_matrix[2, 2] *= cos_r
            
            self.shader['model'].write(model_matrix.tobytes())
            self.cube_vao.render()
    
    def render_scene(self):
        start_time = time.time()
        
        self.ctx.clear(0.2, 0.3, 0.4, 1.0)
        
        view_matrix = np.eye(4, dtype='f4')
        view_matrix[2, 3] = -10
        projection_matrix = np.eye(4, dtype='f4')
        
        light_pos = Vector3D(5, 5, 5)
        light_color = Color(1.0, 1.0, 1.0, 1.0)
        
        self.shader['light_pos'].write(light_pos.to_array())
        self.shader['light_color'].write(light_color.to_array())
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        
        # Render terrain
        self.render_terrain()
        
        # Render vegetation
        if self.content_type == ContentType.VEGETATION:
            self.render_vegetation()
        
        # Render buildings
        if self.content_type == ContentType.BUILDINGS:
            self.render_buildings()
        
        render_time = time.time() - start_time
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            print(f"FPS: {fps:.1f}, Quality Metrics: {self.quality_metrics}")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def handle_input(self):
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.content_type = ContentType.TERRAIN
            print("Switched to Terrain Generation")
            self.generate_content()
        elif glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.content_type = ContentType.VEGETATION
            print("Switched to Vegetation Generation")
            self.generate_content()
        elif glfw.get_key(self.window, glfw.KEY_3) == glfw.PRESS:
            self.content_type = ContentType.BUILDINGS
            print("Switched to Building Generation")
            self.generate_content()
        
        if glfw.get_key(self.window, glfw.KEY_G) == glfw.PRESS:
            self.generate_content()
        
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.auto_generate = not self.auto_generate
            print(f"Auto-generation: {self.auto_generate}")
    
    def framebuffer_size_callback(self, window, width, height):
        self.width = width
        self.height = height
        self.ctx.viewport = (0, 0, width, height)
    
    def run(self):
        last_time = time.time()
        last_generation_time = time.time()
        
        while not glfw.window_should_close(self.window):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Auto-generate content
            if self.auto_generate and current_time - last_generation_time > self.generation_interval:
                self.generate_content()
                last_generation_time = current_time
            
            self.handle_input()
            self.render_scene()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        glfw.terminate()

def main():
    print("=== Procedural Content Generation Demo ===\n")
    print("Procedural generation features:")
    print("  • AI-assisted 3D content creation")
    print("  • Automated asset generation")
    print("  • Terrain, vegetation, and building generation")
    print("  • Noise-based and fractal algorithms")
    print("  • Research methodology and quality metrics")
    
    print("\nControls:")
    print("• 1: Terrain generation")
    print("• 2: Vegetation generation")
    print("• 3: Building generation")
    print("• G: Generate new content")
    print("• A: Toggle auto-generation")
    
    print("\nApplications:")
    print("• Game development and level design")
    print("• Film production and visual effects")
    print("• Architectural visualization")
    print("• Research and development")
    
    try:
        generator = ProceduralContentGenerator(800, 600)
        generator.run()
    except Exception as e:
        print(f"✗ Procedural content generator failed to start: {e}")

if __name__ == "__main__":
    main()
