"""
Chapter 27: Particle Systems and Visual Effects - Visual Effects
=============================================================

This module demonstrates visual effects and particle rendering techniques.

Key Concepts:
- Particle rendering and visualization
- Visual effects systems and management
- Particle shaders and rendering pipelines
- Effect blending and composition
- Performance optimization for visual effects
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class EffectType(Enum):
    """Visual effect type enumeration."""
    FIRE = "fire"
    SMOKE = "smoke"
    EXPLOSION = "explosion"
    SPARKLE = "sparkle"
    TRAIL = "trail"
    BURST = "burst"


class BlendMode(Enum):
    """Blending mode enumeration."""
    ADDITIVE = "additive"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ALPHA = "alpha"


@dataclass
class VisualEffect:
    """Represents a visual effect configuration."""
    effect_type: EffectType
    duration: float
    intensity: float
    color: np.ndarray
    size: float
    blend_mode: BlendMode = BlendMode.ADDITIVE
    fade_in: float = 0.1
    fade_out: float = 0.3
    
    def __post_init__(self):
        if self.color is None:
            self.color = np.array([1.0, 1.0, 1.0, 1.0])


class ParticleRenderer:
    """Handles particle rendering and visualization."""
    
    def __init__(self):
        self.vao = 0
        self.vbo_positions = 0
        self.vbo_colors = 0
        self.vbo_sizes = 0
        self.shader_program = 0
        self.setup_rendering()
    
    def setup_rendering(self):
        """Setup OpenGL rendering resources."""
        # Create VAO and VBOs
        self.vao = gl.glGenVertexArrays(1)
        self.vbo_positions = gl.glGenBuffers(1)
        self.vbo_colors = gl.glGenBuffers(1)
        self.vbo_sizes = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.vao)
        
        # Setup position buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_positions)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        
        # Setup color buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_colors)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)
        
        # Setup size buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_sizes)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)
        
        # Create shader program (simplified)
        self.shader_program = self._create_particle_shader()
    
    def _create_particle_shader(self) -> int:
        """Create particle rendering shader program."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec4 color;
        layout (location = 2) in float size;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec4 fragColor;
        out vec2 texCoord;
        
        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
            gl_PointSize = size;
            fragColor = color;
            texCoord = gl_PointCoord;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec4 fragColor;
        in vec2 texCoord;
        
        out vec4 outColor;
        
        void main() {
            float dist = length(texCoord - vec2(0.5));
            float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
            outColor = fragColor * alpha;
        }
        """
        
        # Compile shaders (simplified - in practice you'd use proper shader compilation)
        return 1  # Placeholder
    
    def render_particles(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray):
        """Render particles using OpenGL."""
        if len(positions) == 0:
            return
        
        gl.glUseProgram(self.shader_program)
        gl.glBindVertexArray(self.vao)
        
        # Update position buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_positions)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, positions, gl.GL_DYNAMIC_DRAW)
        
        # Update color buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_colors)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_DYNAMIC_DRAW)
        
        # Update size buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_sizes)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sizes.nbytes, sizes, gl.GL_DYNAMIC_DRAW)
        
        # Enable blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Render particles
        gl.glDrawArrays(gl.GL_POINTS, 0, len(positions))
        
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)
    
    def cleanup(self):
        """Clean up rendering resources."""
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.vbo_positions:
            gl.glDeleteBuffers(1, [self.vbo_positions])
        if self.vbo_colors:
            gl.glDeleteBuffers(1, [self.vbo_colors])
        if self.vbo_sizes:
            gl.glDeleteBuffers(1, [self.vbo_sizes])


class VisualEffectsManager:
    """Manages visual effects and their rendering."""
    
    def __init__(self):
        self.effects: List[VisualEffect] = []
        self.renderer = ParticleRenderer()
        self.effect_templates = self._create_effect_templates()
    
    def _create_effect_templates(self) -> Dict[EffectType, VisualEffect]:
        """Create predefined effect templates."""
        templates = {}
        
        # Fire effect
        templates[EffectType.FIRE] = VisualEffect(
            effect_type=EffectType.FIRE,
            duration=3.0,
            intensity=1.0,
            color=np.array([1.0, 0.3, 0.0, 1.0]),
            size=2.0,
            blend_mode=BlendMode.ADDITIVE,
            fade_in=0.1,
            fade_out=0.5
        )
        
        # Smoke effect
        templates[EffectType.SMOKE] = VisualEffect(
            effect_type=EffectType.SMOKE,
            duration=5.0,
            intensity=0.7,
            color=np.array([0.3, 0.3, 0.3, 0.8]),
            size=3.0,
            blend_mode=BlendMode.ALPHA,
            fade_in=0.2,
            fade_out=1.0
        )
        
        # Explosion effect
        templates[EffectType.EXPLOSION] = VisualEffect(
            effect_type=EffectType.EXPLOSION,
            duration=2.0,
            intensity=1.5,
            color=np.array([1.0, 0.8, 0.0, 1.0]),
            size=4.0,
            blend_mode=BlendMode.ADDITIVE,
            fade_in=0.0,
            fade_out=0.3
        )
        
        # Sparkle effect
        templates[EffectType.SPARKLE] = VisualEffect(
            effect_type=EffectType.SPARKLE,
            duration=1.5,
            intensity=0.8,
            color=np.array([1.0, 1.0, 1.0, 1.0]),
            size=1.0,
            blend_mode=BlendMode.ADDITIVE,
            fade_in=0.0,
            fade_out=0.2
        )
        
        return templates
    
    def create_effect(self, effect_type: EffectType, position: np.ndarray, 
                     custom_params: Optional[Dict] = None) -> VisualEffect:
        """Create a visual effect based on template."""
        template = self.effect_templates[effect_type]
        
        # Create effect with template parameters
        effect = VisualEffect(
            effect_type=template.effect_type,
            duration=template.duration,
            intensity=template.intensity,
            color=template.color.copy(),
            size=template.size,
            blend_mode=template.blend_mode,
            fade_in=template.fade_in,
            fade_out=template.fade_out
        )
        
        # Apply custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if hasattr(effect, key):
                    setattr(effect, key, value)
        
        self.effects.append(effect)
        return effect
    
    def update_effects(self, delta_time: float):
        """Update all visual effects."""
        for effect in self.effects[:]:  # Copy list to avoid modification during iteration
            effect.duration -= delta_time
            
            if effect.duration <= 0:
                self.effects.remove(effect)
    
    def render_effects(self, camera_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render all visual effects."""
        for effect in self.effects:
            self._render_effect(effect, camera_matrix, projection_matrix)
    
    def _render_effect(self, effect: VisualEffect, camera_matrix: np.ndarray, 
                      projection_matrix: np.ndarray):
        """Render a single visual effect."""
        # Calculate effect parameters based on time
        time_ratio = 1.0 - (effect.duration / effect.effect_templates[effect.effect_type].duration)
        
        # Calculate fade
        fade = self._calculate_fade(effect, time_ratio)
        
        # Generate particle data for effect
        positions, colors, sizes = self._generate_effect_particles(effect, time_ratio, fade)
        
        # Set blending mode
        self._set_blend_mode(effect.blend_mode)
        
        # Render particles
        self.renderer.render_particles(positions, colors, sizes)
    
    def _calculate_fade(self, effect: VisualEffect, time_ratio: float) -> float:
        """Calculate fade value based on effect timing."""
        if time_ratio < effect.fade_in:
            return time_ratio / effect.fade_in
        elif time_ratio > (1.0 - effect.fade_out):
            return (1.0 - time_ratio) / effect.fade_out
        else:
            return 1.0
    
    def _generate_effect_particles(self, effect: VisualEffect, time_ratio: float, 
                                 fade: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate particle data for visual effect."""
        # This is a simplified version - in practice you'd generate particles based on effect type
        
        # Generate some sample particles
        num_particles = 50
        
        # Generate positions in a sphere around origin
        positions = []
        colors = []
        sizes = []
        
        for i in range(num_particles):
            # Random position in sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            radius = np.random.uniform(0, 2.0)
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            positions.append([x, y, z])
            
            # Color with fade
            color = effect.color.copy()
            color[3] *= fade
            colors.append(color)
            
            # Size with variation
            size = effect.size * np.random.uniform(0.5, 1.5)
            sizes.append(size)
        
        return (np.array(positions, dtype=np.float32),
                np.array(colors, dtype=np.float32),
                np.array(sizes, dtype=np.float32))
    
    def _set_blend_mode(self, blend_mode: BlendMode):
        """Set OpenGL blending mode."""
        if blend_mode == BlendMode.ADDITIVE:
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        elif blend_mode == BlendMode.MULTIPLY:
            gl.glBlendFunc(gl.GL_DST_COLOR, gl.GL_ZERO)
        elif blend_mode == BlendMode.SCREEN:
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_COLOR)
        elif blend_mode == BlendMode.OVERLAY:
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        else:  # ALPHA
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


class EffectCompositor:
    """Composes multiple visual effects together."""
    
    def __init__(self):
        self.effects_layers: List[List[VisualEffect]] = []
        self.compositing_shader = 0
        self.framebuffer = 0
        self.setup_compositing()
    
    def setup_compositing(self):
        """Setup compositing resources."""
        # Create framebuffer for effect compositing
        self.framebuffer = gl.glGenFramebuffers(1)
        
        # Create compositing shader (simplified)
        self.compositing_shader = self._create_compositing_shader()
    
    def _create_compositing_shader(self) -> int:
        """Create effect compositing shader."""
        # Simplified shader creation
        return 1  # Placeholder
    
    def add_effect_layer(self, effects: List[VisualEffect]):
        """Add a layer of effects for compositing."""
        self.effects_layers.append(effects)
    
    def composite_effects(self, output_texture: int):
        """Composite all effect layers."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                 gl.GL_TEXTURE_2D, output_texture, 0)
        
        gl.glUseProgram(self.compositing_shader)
        
        # Render each layer
        for layer in self.effects_layers:
            self._render_effect_layer(layer)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    
    def _render_effect_layer(self, effects: List[VisualEffect]):
        """Render a single effect layer."""
        # Simplified layer rendering
        pass
    
    def cleanup(self):
        """Clean up compositing resources."""
        if self.framebuffer:
            gl.glDeleteFramebuffers(1, [self.framebuffer])


def demonstrate_visual_effects():
    """Demonstrate visual effects functionality."""
    print("=== Particle Systems and Visual Effects - Visual Effects ===\n")

    # Create visual effects manager
    effects_manager = VisualEffectsManager()
    
    print("1. Available effect templates:")
    for effect_type in EffectType:
        template = effects_manager.effect_templates[effect_type]
        print(f"   - {effect_type.value}: duration={template.duration}s, intensity={template.intensity}")

    print("\n2. Creating visual effects...")
    
    # Create different effects
    fire_effect = effects_manager.create_effect(
        EffectType.FIRE,
        position=np.array([0.0, 0.0, 0.0]),
        custom_params={"intensity": 1.2, "size": 2.5}
    )
    print(f"   Created fire effect: intensity={fire_effect.intensity}, size={fire_effect.size}")
    
    explosion_effect = effects_manager.create_effect(
        EffectType.EXPLOSION,
        position=np.array([5.0, 0.0, 0.0]),
        custom_params={"duration": 1.5, "color": np.array([1.0, 0.5, 0.0, 1.0])}
    )
    print(f"   Created explosion effect: duration={explosion_effect.duration}s")
    
    sparkle_effect = effects_manager.create_effect(
        EffectType.SPARKLE,
        position=np.array([-5.0, 0.0, 0.0])
    )
    print(f"   Created sparkle effect: blend_mode={sparkle_effect.blend_mode.value}")

    print("\n3. Effect management...")
    
    print(f"   Active effects: {len(effects_manager.effects)}")
    
    # Simulate effect updates
    delta_time = 0.016
    simulation_time = 1.0
    steps = int(simulation_time / delta_time)
    
    for i in range(steps):
        effects_manager.update_effects(delta_time)
        
        if i % 30 == 0:  # Print every 0.5 seconds
            active_count = len(effects_manager.effects)
            print(f"   Time {i * delta_time:.1f}s: {active_count} active effects")

    print("\n4. Rendering setup...")
    
    # Create renderer
    renderer = ParticleRenderer()
    print("   Created particle renderer")
    
    # Create compositor
    compositor = EffectCompositor()
    print("   Created effect compositor")

    print("\n5. Blending modes:")
    for blend_mode in BlendMode:
        print(f"   - {blend_mode.value}")

    print("\n6. Features demonstrated:")
    print("   - Visual effect templates")
    print("   - Effect lifecycle management")
    print("   - Particle rendering pipeline")
    print("   - Multiple blending modes")
    print("   - Effect compositing")
    print("   - Custom effect parameters")

    # Cleanup
    renderer.cleanup()
    compositor.cleanup()
    print("\n7. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_visual_effects()
