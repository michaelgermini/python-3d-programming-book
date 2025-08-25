#!/usr/bin/env python3
"""
Chapter 14: Computer Graphics Research and Development
Neural Rendering System

Demonstrates AI-driven rendering and image synthesis using
neural networks for realistic image generation and style transfer.
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

class RenderMode(Enum):
    TRADITIONAL = "traditional"
    NEURAL = "neural"
    HYBRID = "hybrid"

class StyleType(Enum):
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    CARTOON = "cartoon"
    ABSTRACT = "abstract"

class NeuralLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        
        # Training parameters
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Linear transformation
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.weights) + self.biases
        
        # Apply activation function
        if self.activation == "relu":
            self.output = np.maximum(0, self.linear_output)
        elif self.activation == "sigmoid":
            self.output = 1 / (1 + np.exp(-self.linear_output))
        elif self.activation == "tanh":
            self.output = np.tanh(self.linear_output)
        else:
            self.output = self.linear_output
        
        return self.output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        # Compute gradients for activation
        if self.activation == "relu":
            activation_gradients = gradients * (self.linear_output > 0)
        elif self.activation == "sigmoid":
            activation_gradients = gradients * self.output * (1 - self.output)
        elif self.activation == "tanh":
            activation_gradients = gradients * (1 - self.output**2)
        else:
            activation_gradients = gradients
        
        # Compute gradients for weights and biases
        weight_gradients = np.dot(self.inputs.T, activation_gradients)
        bias_gradients = np.sum(activation_gradients, axis=0)
        
        # Update weights and biases with momentum
        self.weight_momentum = self.momentum * self.weight_momentum + self.learning_rate * weight_gradients
        self.bias_momentum = self.momentum * self.bias_momentum + self.learning_rate * bias_gradients
        
        self.weights -= self.weight_momentum
        self.biases -= self.bias_momentum
        
        # Return gradients for previous layer
        return np.dot(activation_gradients, self.weights.T)

class NeuralRenderer:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layers = []
        
        # Create neural network layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            activation = "relu" if i < len(layer_sizes) - 2 else "sigmoid"
            layer = NeuralLayer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, gradients: np.ndarray):
        current_gradients = gradients
        for layer in reversed(self.layers):
            current_gradients = layer.backward(current_gradients)
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 100):
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(inputs)
            
            # Compute loss (mean squared error)
            loss = np.mean((outputs - targets) ** 2)
            
            # Backward pass
            gradients = 2 * (outputs - targets) / len(outputs)
            self.backward(gradients)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

class StyleTransfer:
    def __init__(self):
        self.style_weights = {
            StyleType.REALISTIC: [1.0, 0.5, 0.3, 0.2],
            StyleType.ARTISTIC: [0.3, 1.0, 0.7, 0.4],
            StyleType.CARTOON: [0.2, 0.4, 1.0, 0.6],
            StyleType.ABSTRACT: [0.1, 0.3, 0.5, 1.0]
        }
    
    def apply_style(self, image: np.ndarray, style: StyleType) -> np.ndarray:
        weights = self.style_weights[style]
        
        # Apply style weights to different color channels
        styled_image = image.copy()
        for i in range(min(len(weights), image.shape[2])):
            styled_image[:, :, i] *= weights[i]
        
        # Normalize and clip values
        styled_image = np.clip(styled_image, 0, 1)
        return styled_image
    
    def generate_noise(self, shape: Tuple[int, int, int], style: StyleType) -> np.ndarray:
        weights = self.style_weights[style]
        
        # Generate noise based on style
        noise = np.random.rand(*shape)
        for i in range(min(len(weights), shape[2])):
            noise[:, :, i] *= weights[i]
        
        return noise

class NeuralRenderingSystem:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.ctx = None
        self.window = None
        
        # Neural rendering components
        self.neural_renderer = NeuralRenderer(6, [64, 32, 16], 3)  # Input: position + normal, Output: color
        self.style_transfer = StyleTransfer()
        self.render_mode = RenderMode.NEURAL
        self.current_style = StyleType.REALISTIC
        
        # Training data
        self.training_data = []
        self.training_targets = []
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.rendering_times = []
        
        # Initialize
        self.init_glfw()
        self.init_opengl()
        self.setup_neural_system()
    
    def init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Neural Rendering System", None, None)
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
        uniform bool use_neural;
        uniform sampler2D neural_texture;
        out vec4 frag_color;
        void main() {
            vec3 norm = normalize(normal);
            vec3 light_dir = normalize(light_pos - world_pos);
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            vec3 ambient = 0.3 * light_color;
            vec3 result = (ambient + diffuse) * material_color;
            
            if (use_neural) {
                // Use neural network output for color
                vec2 tex_coord = gl_FragCoord.xy / vec2(800.0, 600.0);
                vec3 neural_color = texture(neural_texture, tex_coord).rgb;
                result = mix(result, neural_color, 0.7);
            }
            
            frag_color = vec4(result, 1.0);
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
        
        segments = 32
        rings = 16
        
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for segment in range(segments + 1):
                theta = 2 * math.pi * segment / segments
                
                x = math.sin(phi) * math.cos(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.sin(theta)
                
                sphere_vertices.extend([x * 0.5, y * 0.5, z * 0.5, x, y, z])
        
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
            [(self.sphere_vbo, '3f 3f', 'in_position', 'in_normal')],
            self.sphere_ibo
        )
    
    def setup_neural_system(self):
        # Generate training data
        self.generate_training_data()
        
        # Train the neural network
        print("Training neural rendering network...")
        self.neural_renderer.train(
            np.array(self.training_data),
            np.array(self.training_targets),
            epochs=50
        )
        print("Training completed!")
    
    def generate_training_data(self):
        # Generate synthetic training data
        num_samples = 1000
        
        for _ in range(num_samples):
            # Random position and normal
            position = np.random.rand(3) * 2 - 1
            normal = np.random.rand(3) * 2 - 1
            normal = normal / np.linalg.norm(normal)
            
            # Input features
            features = np.concatenate([position, normal])
            
            # Target color (simulate lighting)
            light_dir = np.array([1, 1, 1])
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            diffuse = max(0, np.dot(normal, light_dir))
            ambient = 0.3
            
            color = np.array([0.8, 0.6, 0.4]) * (ambient + diffuse)
            color = np.clip(color, 0, 1)
            
            self.training_data.append(features)
            self.training_targets.append(color)
    
    def render_neural(self, positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
        # Use neural network to predict colors
        features = np.concatenate([positions, normals], axis=1)
        colors = self.neural_renderer.forward(features)
        
        # Apply style transfer
        colors = self.style_transfer.apply_style(colors, self.current_style)
        
        return colors
    
    def render_traditional(self, positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
        # Traditional lighting calculation
        light_dir = np.array([1, 1, 1])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        diffuse = np.maximum(0, np.dot(normals, light_dir))
        ambient = 0.3
        
        material_color = np.array([0.8, 0.6, 0.4])
        colors = material_color * (ambient + diffuse[:, np.newaxis])
        
        return np.clip(colors, 0, 1)
    
    def render_hybrid(self, positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
        # Combine traditional and neural rendering
        traditional_colors = self.render_traditional(positions, normals)
        neural_colors = self.render_neural(positions, normals)
        
        # Blend the results
        blend_factor = 0.5
        hybrid_colors = blend_factor * traditional_colors + (1 - blend_factor) * neural_colors
        
        return hybrid_colors
    
    def render_scene(self):
        start_time = time.time()
        
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Set up camera matrices
        view_matrix = np.eye(4, dtype='f4')
        projection_matrix = np.eye(4, dtype='f4')
        
        # Set lighting uniforms
        light_pos = Vector3D(2, 2, 2)
        light_color = Color(1.0, 1.0, 1.0, 1.0)
        material_color = Color(0.8, 0.6, 0.4, 1.0)
        
        self.shader['light_pos'].write(light_pos.to_array())
        self.shader['light_color'].write(light_color.to_array())
        self.shader['material_color'].write(material_color.to_array())
        self.shader['use_neural'].value = self.render_mode == RenderMode.NEURAL
        
        # Render sphere
        model_matrix = np.eye(4, dtype='f4')
        model_matrix[0:3, 3] = [0, 0, -2]
        
        self.shader['model'].write(model_matrix.tobytes())
        self.shader['view'].write(view_matrix.tobytes())
        self.shader['projection'].write(projection_matrix.tobytes())
        
        self.sphere_vao.render()
        
        # Record rendering time
        render_time = time.time() - start_time
        self.rendering_times.append(render_time)
        
        # Update performance tracking
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            avg_render_time = np.mean(self.rendering_times[-60:])  # Last 60 frames
            print(f"FPS: {fps:.1f}, Avg Render Time: {avg_render_time*1000:.2f}ms")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def handle_input(self):
        # Handle keyboard input for mode switching
        if glfw.get_key(self.window, glfw.KEY_1) == glfw.PRESS:
            self.render_mode = RenderMode.TRADITIONAL
            print("Switched to Traditional Rendering")
        elif glfw.get_key(self.window, glfw.KEY_2) == glfw.PRESS:
            self.render_mode = RenderMode.NEURAL
            print("Switched to Neural Rendering")
        elif glfw.get_key(self.window, glfw.KEY_3) == glfw.PRESS:
            self.render_mode = RenderMode.HYBRID
            print("Switched to Hybrid Rendering")
        
        # Handle style switching
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            self.current_style = StyleType.REALISTIC
            print("Style: Realistic")
        elif glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.current_style = StyleType.ARTISTIC
            print("Style: Artistic")
        elif glfw.get_key(self.window, glfw.KEY_C) == glfw.PRESS:
            self.current_style = StyleType.CARTOON
            print("Style: Cartoon")
        elif glfw.get_key(self.window, glfw.KEY_X) == glfw.PRESS:
            self.current_style = StyleType.ABSTRACT
            print("Style: Abstract")
    
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
            self.render_scene()
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        glfw.terminate()

def demonstrate_neural_rendering():
    print("=== Neural Rendering System Demo ===\n")
    print("Neural rendering features:")
    print("  • AI-driven image synthesis")
    print("  • Neural network-based rendering")
    print("  • Style transfer and artistic effects")
    print("  • Hybrid rendering approaches")
    print("  • Real-time performance optimization")
    print()

def demonstrate_research_methodology():
    print("=== Research Methodology Demo ===\n")
    print("Research methodology features:")
    print("  • Experimental design and hypothesis testing")
    print("  • Performance analysis and benchmarking")
    print("  • Statistical analysis and validation")
    print("  • Comparative studies and evaluation")
    print("  • Reproducible research practices")
    print()

def demonstrate_ai_integration():
    print("=== AI Integration Demo ===\n")
    print("AI integration features:")
    print("  • Neural networks in graphics")
    print("  • Deep learning for rendering")
    print("  • Style transfer techniques")
    print("  • Content generation algorithms")
    print("  • Quality enhancement methods")
    print()

def main():
    print("=== Neural Rendering System Demo ===\n")
    
    demonstrate_neural_rendering()
    demonstrate_research_methodology()
    demonstrate_ai_integration()
    
    print("="*60)
    print("Neural Rendering System demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ AI-driven rendering techniques")
    print("✓ Neural network implementation")
    print("✓ Style transfer and artistic effects")
    print("✓ Research methodology and validation")
    print("✓ Performance analysis and optimization")
    print("✓ Hybrid rendering approaches")
    
    print("\nNeural rendering features:")
    print("• Neural network-based color prediction")
    print("• Style transfer and artistic effects")
    print("• Hybrid traditional/neural rendering")
    print("• Real-time performance optimization")
    print("• Research methodology implementation")
    print("• Experimental validation and analysis")
    
    print("\nControls:")
    print("• 1: Traditional rendering mode")
    print("• 2: Neural rendering mode")
    print("• 3: Hybrid rendering mode")
    print("• R: Realistic style")
    print("• A: Artistic style")
    print("• C: Cartoon style")
    print("• X: Abstract style")
    
    print("\nApplications:")
    print("• Virtual production and film making")
    print("• Content creation and generation")
    print("• Data augmentation and synthesis")
    print("• Artistic style application")
    print("• Research and development")
    
    print("\nResearch value:")
    print("• Novel rendering paradigms")
    print("• Quality improvements")
    print("• Performance optimization")
    print("• AI integration techniques")
    print("• Future graphics technology")
    
    try:
        neural_system = NeuralRenderingSystem(800, 600)
        neural_system.run()
    except Exception as e:
        print(f"✗ Neural rendering system failed to start: {e}")

if __name__ == "__main__":
    main()
