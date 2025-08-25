"""
Chapter 24: Framebuffers and Render-to-Texture - Render to Texture
================================================================

This module demonstrates render-to-texture techniques and post-processing effects.

Key Concepts:
- Render-to-texture for off-screen rendering
- Post-processing effects and filters
- Multi-pass rendering techniques
- Screen-space effects and shaders
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import math


class PostProcessEffect(Enum):
    """Post-processing effect enumeration."""
    NONE = "none"
    BLUR = "blur"
    SHARPEN = "sharpen"
    EDGE_DETECTION = "edge_detection"
    BLOOM = "bloom"
    TONE_MAPPING = "tone_mapping"
    GAMMA_CORRECTION = "gamma_correction"


@dataclass
class RenderTarget:
    """Represents a render target with texture and framebuffer."""
    name: str
    texture_id: int
    framebuffer_id: int
    width: int
    height: int
    format: int
    data_type: int


class RenderToTexture:
    """Manages render-to-texture operations."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.render_targets: Dict[str, RenderTarget] = {}
        self.current_target: Optional[RenderTarget] = None

    def create_render_target(self, name: str, format: int = gl.GL_RGBA, 
                           data_type: int = gl.GL_UNSIGNED_BYTE) -> RenderTarget:
        """Create a render target with texture and framebuffer."""
        # Create texture
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        # Allocate texture storage
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format, self.width, self.height, 
                       0, format, data_type, None)
        
        # Create framebuffer
        framebuffer_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer_id)
        
        # Attach texture to framebuffer
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                 gl.GL_TEXTURE_2D, texture_id, 0)
        
        # Check completeness
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer incomplete: {status}")
        
        # Create render target
        render_target = RenderTarget(
            name=name,
            texture_id=texture_id,
            framebuffer_id=framebuffer_id,
            width=self.width,
            height=self.height,
            format=format,
            data_type=data_type
        )
        
        self.render_targets[name] = render_target
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        
        return render_target

    def bind_render_target(self, name: str):
        """Bind a render target for rendering."""
        if name in self.render_targets:
            target = self.render_targets[name]
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target.framebuffer_id)
            self.current_target = target
        else:
            print(f"Render target '{name}' not found")

    def unbind_render_target(self):
        """Unbind current render target."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.current_target = None

    def bind_texture(self, name: str, texture_unit: int = 0):
        """Bind a render target texture to a texture unit."""
        if name in self.render_targets:
            target = self.render_targets[name]
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, target.texture_id)

    def get_texture_id(self, name: str) -> Optional[int]:
        """Get texture ID for a render target."""
        if name in self.render_targets:
            return self.render_targets[name].texture_id
        return None

    def cleanup(self):
        """Clean up render target resources."""
        for target in self.render_targets.values():
            gl.glDeleteTextures(1, [target.texture_id])
            gl.glDeleteFramebuffers(1, [target.framebuffer_id])
        self.render_targets.clear()
        self.current_target = None


class PostProcessor:
    """Handles post-processing effects and filters."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.render_to_texture = RenderToTexture(width, height)
        self.effects: Dict[PostProcessEffect, Callable] = {}
        self.setup_effects()

    def setup_effects(self):
        """Setup post-processing effects."""
        self.effects[PostProcessEffect.BLUR] = self.apply_blur
        self.effects[PostProcessEffect.SHARPEN] = self.apply_sharpen
        self.effects[PostProcessEffect.EDGE_DETECTION] = self.apply_edge_detection
        self.effects[PostProcessEffect.BLOOM] = self.apply_bloom
        self.effects[PostProcessEffect.TONE_MAPPING] = self.apply_tone_mapping
        self.effects[PostProcessEffect.GAMMA_CORRECTION] = self.apply_gamma_correction

    def create_effect_targets(self):
        """Create render targets for post-processing effects."""
        # Main render target
        self.render_to_texture.create_render_target("main", gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        
        # Effect render targets
        self.render_to_texture.create_render_target("effect1", gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        self.render_to_texture.create_render_target("effect2", gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        
        # High dynamic range target
        self.render_to_texture.create_render_target("hdr", gl.GL_RGBA16F, gl.GL_FLOAT)

    def apply_blur(self, source_texture: int, target_name: str, radius: float = 5.0):
        """Apply Gaussian blur effect."""
        # This would use a blur shader in practice
        # For demonstration, we'll simulate the effect
        print(f"Applying blur effect with radius {radius}")
        
        # Bind target for blur result
        self.render_to_texture.bind_render_target(target_name)
        
        # In a real implementation, this would:
        # 1. Bind the source texture
        # 2. Use a blur shader
        # 3. Render a full-screen quad
        # 4. Apply Gaussian blur kernel
        
        self.render_to_texture.unbind_render_target()

    def apply_sharpen(self, source_texture: int, target_name: str, strength: float = 1.0):
        """Apply sharpening effect."""
        print(f"Applying sharpen effect with strength {strength}")
        
        self.render_to_texture.bind_render_target(target_name)
        
        # Sharpening would use a convolution kernel like:
        # [ 0, -1,  0]
        # [-1,  5, -1]
        # [ 0, -1,  0]
        
        self.render_to_texture.unbind_render_target()

    def apply_edge_detection(self, source_texture: int, target_name: str):
        """Apply edge detection effect."""
        print("Applying edge detection effect")
        
        self.render_to_texture.bind_render_target(target_name)
        
        # Edge detection would use Sobel or similar operators
        
        self.render_to_texture.unbind_render_target()

    def apply_bloom(self, source_texture: int, target_name: str, threshold: float = 1.0):
        """Apply bloom effect."""
        print(f"Applying bloom effect with threshold {threshold}")
        
        # Bloom effect steps:
        # 1. Extract bright areas (above threshold)
        # 2. Blur the bright areas
        # 3. Combine with original image
        
        self.render_to_texture.bind_render_target("effect1")
        # Extract bright areas
        
        self.render_to_texture.bind_render_target("effect2")
        # Apply blur to bright areas
        
        self.render_to_texture.bind_render_target(target_name)
        # Combine with original
        
        self.render_to_texture.unbind_render_target()

    def apply_tone_mapping(self, source_texture: int, target_name: str, exposure: float = 1.0):
        """Apply tone mapping for HDR to LDR conversion."""
        print(f"Applying tone mapping with exposure {exposure}")
        
        self.render_to_texture.bind_render_target(target_name)
        
        # Tone mapping would convert HDR values to LDR
        # Common methods: Reinhard, ACES, Uncharted 2
        
        self.render_to_texture.unbind_render_target()

    def apply_gamma_correction(self, source_texture: int, target_name: str, gamma: float = 2.2):
        """Apply gamma correction."""
        print(f"Applying gamma correction with gamma {gamma}")
        
        self.render_to_texture.bind_render_target(target_name)
        
        # Gamma correction: output = pow(input, 1.0 / gamma)
        
        self.render_to_texture.unbind_render_target()

    def process_effect(self, effect: PostProcessEffect, source_name: str, target_name: str, **params):
        """Process a post-processing effect."""
        if effect not in self.effects:
            print(f"Effect '{effect.value}' not supported")
            return
        
        source_texture = self.render_to_texture.get_texture_id(source_name)
        if source_texture is None:
            print(f"Source texture '{source_name}' not found")
            return
        
        # Apply the effect
        self.effects[effect](source_texture, target_name, **params)

    def render_fullscreen_quad(self):
        """Render a full-screen quad for post-processing."""
        # This would render a simple quad covering the entire screen
        # Used for applying post-processing effects
        print("Rendering full-screen quad for post-processing")


class MultiPassRenderer:
    """Handles multi-pass rendering with render-to-texture."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.post_processor = PostProcessor(width, height)
        self.passes: List[Dict[str, Any]] = []

    def add_pass(self, name: str, render_function: Callable, 
                source_target: str = None, target_name: str = None):
        """Add a render pass to the pipeline."""
        pass_info = {
            "name": name,
            "render_function": render_function,
            "source_target": source_target,
            "target_name": target_name
        }
        self.passes.append(pass_info)

    def setup_pipeline(self):
        """Setup the rendering pipeline."""
        self.post_processor.create_effect_targets()

    def execute_pipeline(self):
        """Execute the complete rendering pipeline."""
        print("Executing multi-pass rendering pipeline...")
        
        for i, pass_info in enumerate(self.passes):
            print(f"Pass {i + 1}: {pass_info['name']}")
            
            # Bind target if specified
            if pass_info['target_name']:
                self.post_processor.render_to_texture.bind_render_target(pass_info['target_name'])
            
            # Execute render function
            pass_info['render_function']()
            
            # Unbind target
            if pass_info['target_name']:
                self.post_processor.render_to_texture.unbind_render_target()

    def apply_post_processing(self, effects: List[PostProcessEffect]):
        """Apply post-processing effects to the final result."""
        print("Applying post-processing effects...")
        
        current_source = "main"
        
        for i, effect in enumerate(effects):
            target_name = f"effect{i + 1}"
            self.post_processor.process_effect(effect, current_source, target_name)
            current_source = target_name

    def cleanup(self):
        """Clean up rendering resources."""
        self.post_processor.render_to_texture.cleanup()


def demonstrate_render_to_texture():
    """Demonstrate render-to-texture techniques and post-processing."""
    print("=== Framebuffers and Render-to-Texture - Render to Texture ===\n")

    # Create multi-pass renderer
    renderer = MultiPassRenderer(1024, 768)
    renderer.setup_pipeline()

    # Define render passes
    def geometry_pass():
        """Render geometry to main target."""
        print("  Rendering geometry...")
        # This would render 3D geometry

    def lighting_pass():
        """Apply lighting calculations."""
        print("  Applying lighting...")
        # This would apply lighting to the geometry

    def post_process_pass():
        """Apply post-processing effects."""
        print("  Applying post-processing...")
        renderer.post_processor.render_fullscreen_quad()

    # Add render passes
    renderer.add_pass("Geometry", geometry_pass, target_name="main")
    renderer.add_pass("Lighting", lighting_pass, source_target="main", target_name="hdr")
    renderer.add_pass("PostProcess", post_process_pass, source_target="hdr", target_name="effect1")

    # Execute pipeline
    print("1. Executing rendering pipeline...")
    renderer.execute_pipeline()

    # Apply post-processing effects
    print("\n2. Applying post-processing effects...")
    effects = [
        PostProcessEffect.TONE_MAPPING,
        PostProcessEffect.BLOOM,
        PostProcessEffect.GAMMA_CORRECTION
    ]
    renderer.apply_post_processing(effects)

    # Demonstrate individual effects
    print("\n3. Demonstrating individual effects...")
    
    # Blur effect
    renderer.post_processor.process_effect(
        PostProcessEffect.BLUR, "main", "effect1", radius=3.0
    )
    
    # Sharpen effect
    renderer.post_processor.process_effect(
        PostProcessEffect.SHARPEN, "main", "effect2", strength=1.5
    )
    
    # Edge detection
    renderer.post_processor.process_effect(
        PostProcessEffect.EDGE_DETECTION, "main", "effect1"
    )

    # Display render target information
    print(f"\n4. Render Target Information:")
    for name, target in renderer.post_processor.render_to_texture.render_targets.items():
        print(f"  {name}:")
        print(f"    Size: {target.width}x{target.height}")
        print(f"    Format: {target.format}")
        print(f"    Data Type: {target.data_type}")

    # Cleanup
    renderer.cleanup()
    print("\n5. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_render_to_texture()
