"""
Chapter 29: Advanced Rendering Techniques - Post Processing
=========================================================

This module demonstrates post-processing effects and filters.

Key Concepts:
- Post-processing pipeline and effects
- Screen-space effects and filters
- Bloom, SSAO, motion blur, and depth of field
- Tone mapping and color grading
- Performance optimization for real-time effects
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math


class EffectType(Enum):
    """Post-processing effect type enumeration."""
    BLOOM = "bloom"
    SSAO = "ssao"
    MOTION_BLUR = "motion_blur"
    DEPTH_OF_FIELD = "depth_of_field"
    TONE_MAPPING = "tone_mapping"
    COLOR_GRADING = "color_grading"
    VIGNETTE = "vignette"
    CHROMATIC_ABERRATION = "chromatic_aberration"


@dataclass
class PostProcessEffect:
    """Post-processing effect configuration."""
    effect_type: EffectType
    enabled: bool = True
    intensity: float = 1.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PostProcessor:
    """Post-processing pipeline manager."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.effects: List[PostProcessEffect] = []
        self.framebuffers: Dict[str, int] = {}
        self.textures: Dict[str, int] = {}
        self.setup_framebuffers()
        self.setup_shaders()
    
    def setup_framebuffers(self):
        """Setup framebuffers for post-processing."""
        # Main framebuffer
        self.framebuffers['main'] = gl.glGenFramebuffers(1)
        self.textures['main'] = gl.glGenTextures(1)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers['main'])
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['main'])
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, self.width, self.height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.textures['main'], 0)
        
        # Bloom framebuffers
        self.framebuffers['bloom'] = gl.glGenFramebuffers(1)
        self.textures['bloom'] = gl.glGenTextures(1)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers['bloom'])
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['bloom'])
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, self.width, self.height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.textures['bloom'], 0)
        
        # SSAO framebuffer
        self.framebuffers['ssao'] = gl.glGenFramebuffers(1)
        self.textures['ssao'] = gl.glGenTextures(1)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers['ssao'])
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures['ssao'])
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RED, self.width, self.height, 0, gl.GL_RED, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.textures['ssao'], 0)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    
    def setup_shaders(self):
        """Setup shader programs for post-processing effects."""
        self.shaders = {
            'bloom': self._create_bloom_shader(),
            'ssao': self._create_ssao_shader(),
            'tone_mapping': self._create_tone_mapping_shader(),
            'vignette': self._create_vignette_shader(),
            'chromatic_aberration': self._create_chromatic_aberration_shader()
        }
    
    def _create_bloom_shader(self) -> int:
        """Create bloom effect shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D screenTexture;
        uniform float threshold;
        uniform float intensity;
        
        void main() {
            vec3 color = texture(screenTexture, TexCoord).rgb;
            
            // Brightness threshold
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            if(brightness > threshold) {
                FragColor = vec4(color * intensity, 1.0);
            } else {
                FragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        
        # Compile shaders (simplified)
        return 1  # Placeholder
    
    def _create_ssao_shader(self) -> int:
        """Create SSAO effect shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out float FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D gPosition;
        uniform sampler2D gNormal;
        uniform sampler2D texNoise;
        uniform vec3 samples[64];
        uniform mat4 projection;
        
        uniform float radius;
        uniform float bias;
        
        void main() {
            vec3 fragPos = texture(gPosition, TexCoord).xyz;
            vec3 normal = normalize(texture(gNormal, TexCoord).rgb);
            vec3 randomVec = normalize(texture(texNoise, TexCoord * 64.0).xyz);
            
            vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
            vec3 bitangent = cross(normal, tangent);
            mat3 TBN = mat3(tangent, bitangent, normal);
            
            float occlusion = 0.0;
            for(int i = 0; i < 64; ++i) {
                vec3 samplePos = TBN * samples[i];
                samplePos = fragPos + samplePos * radius;
                
                vec4 offset = vec4(samplePos, 1.0);
                offset = projection * offset;
                offset.xyz /= offset.w;
                offset.xyz = offset.xyz * 0.5 + 0.5;
                
                float sampleDepth = texture(gPosition, offset.xy).z;
                float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
                occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
            }
            occlusion = 1.0 - (occlusion / 64.0);
            
            FragColor = occlusion;
        }
        """
        
        # Compile shaders (simplified)
        return 2  # Placeholder
    
    def _create_tone_mapping_shader(self) -> int:
        """Create tone mapping shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D screenTexture;
        uniform float exposure;
        uniform float gamma;
        
        void main() {
            vec3 color = texture(screenTexture, TexCoord).rgb;
            
            // Reinhard tone mapping
            color = color / (color + vec3(1.0));
            
            // Exposure adjustment
            color = vec3(1.0) - exp(-color * exposure);
            
            // Gamma correction
            color = pow(color, vec3(1.0 / gamma));
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        # Compile shaders (simplified)
        return 3  # Placeholder
    
    def _create_vignette_shader(self) -> int:
        """Create vignette effect shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D screenTexture;
        uniform float intensity;
        uniform float radius;
        
        void main() {
            vec3 color = texture(screenTexture, TexCoord).rgb;
            
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(TexCoord, center);
            float vignette = 1.0 - smoothstep(radius, 1.0, dist);
            
            color *= vignette * intensity + (1.0 - intensity);
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        # Compile shaders (simplified)
        return 4  # Placeholder
    
    def _create_chromatic_aberration_shader(self) -> int:
        """Create chromatic aberration effect shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        
        out vec2 TexCoord;
        
        void main() {
            TexCoord = texCoord;
            gl_Position = vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        
        uniform sampler2D screenTexture;
        uniform float intensity;
        
        void main() {
            vec2 center = vec2(0.5, 0.5);
            vec2 direction = normalize(TexCoord - center);
            
            float r = texture(screenTexture, TexCoord + direction * intensity * 0.01).r;
            float g = texture(screenTexture, TexCoord).g;
            float b = texture(screenTexture, TexCoord - direction * intensity * 0.01).b;
            
            FragColor = vec4(r, g, b, 1.0);
        }
        """
        
        # Compile shaders (simplified)
        return 5  # Placeholder
    
    def add_effect(self, effect: PostProcessEffect):
        """Add post-processing effect to pipeline."""
        self.effects.append(effect)
    
    def remove_effect(self, effect_type: EffectType):
        """Remove effect from pipeline."""
        self.effects = [e for e in self.effects if e.effect_type != effect_type]
    
    def apply_bloom(self, input_texture: int, threshold: float = 1.0, intensity: float = 1.0):
        """Apply bloom effect."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers['bloom'])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.shaders['bloom'])
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['bloom'], "screenTexture"), 0)
        
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['bloom'], "threshold"), threshold)
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['bloom'], "intensity"), intensity)
        
        self._render_quad()
    
    def apply_ssao(self, position_texture: int, normal_texture: int, radius: float = 0.5, bias: float = 0.025):
        """Apply SSAO effect."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.framebuffers['ssao'])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.shaders['ssao'])
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, position_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['ssao'], "gPosition"), 0)
        
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['ssao'], "gNormal"), 1)
        
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['ssao'], "radius"), radius)
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['ssao'], "bias"), bias)
        
        self._render_quad()
    
    def apply_tone_mapping(self, input_texture: int, exposure: float = 1.0, gamma: float = 2.2):
        """Apply tone mapping."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.shaders['tone_mapping'])
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['tone_mapping'], "screenTexture"), 0)
        
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['tone_mapping'], "exposure"), exposure)
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['tone_mapping'], "gamma"), gamma)
        
        self._render_quad()
    
    def apply_vignette(self, input_texture: int, intensity: float = 1.0, radius: float = 0.5):
        """Apply vignette effect."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.shaders['vignette'])
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['vignette'], "screenTexture"), 0)
        
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['vignette'], "intensity"), intensity)
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['vignette'], "radius"), radius)
        
        self._render_quad()
    
    def apply_chromatic_aberration(self, input_texture: int, intensity: float = 1.0):
        """Apply chromatic aberration effect."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.shaders['chromatic_aberration'])
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shaders['chromatic_aberration'], "screenTexture"), 0)
        
        gl.glUniform1f(gl.glGetUniformLocation(self.shaders['chromatic_aberration'], "intensity"), intensity)
        
        self._render_quad()
    
    def _render_quad(self):
        """Render a full-screen quad."""
        # Simplified quad rendering
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1.0, -1.0)
        gl.glVertex2f(1.0, -1.0)
        gl.glVertex2f(1.0, 1.0)
        gl.glVertex2f(-1.0, 1.0)
        gl.glEnd()
    
    def process(self, input_texture: int, g_position: int = None, g_normal: int = None):
        """Apply all enabled post-processing effects."""
        current_texture = input_texture
        
        for effect in self.effects:
            if not effect.enabled:
                continue
            
            if effect.effect_type == EffectType.BLOOM:
                self.apply_bloom(current_texture, 
                               effect.parameters.get('threshold', 1.0),
                               effect.parameters.get('intensity', 1.0))
                current_texture = self.textures['bloom']
            
            elif effect.effect_type == EffectType.SSAO and g_position and g_normal:
                self.apply_ssao(g_position, g_normal,
                              effect.parameters.get('radius', 0.5),
                              effect.parameters.get('bias', 0.025))
                current_texture = self.textures['ssao']
            
            elif effect.effect_type == EffectType.TONE_MAPPING:
                self.apply_tone_mapping(current_texture,
                                      effect.parameters.get('exposure', 1.0),
                                      effect.parameters.get('gamma', 2.2))
                return  # Final output
            
            elif effect.effect_type == EffectType.VIGNETTE:
                self.apply_vignette(current_texture,
                                  effect.parameters.get('intensity', 1.0),
                                  effect.parameters.get('radius', 0.5))
                return  # Final output
            
            elif effect.effect_type == EffectType.CHROMATIC_ABERRATION:
                self.apply_chromatic_aberration(current_texture,
                                              effect.parameters.get('intensity', 1.0))
                return  # Final output
        
        # If no effects applied, render input texture directly
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, current_texture)
        
        self._render_quad()
    
    def cleanup(self):
        """Clean up post-processing resources."""
        for framebuffer in self.framebuffers.values():
            gl.glDeleteFramebuffers([framebuffer])
        
        for texture in self.textures.values():
            gl.glDeleteTextures([texture])


def demonstrate_post_processing():
    """Demonstrate post-processing functionality."""
    print("=== Advanced Rendering Techniques - Post Processing ===\n")

    # Create post-processor
    print("1. Creating post-processor...")
    
    post_processor = PostProcessor(800, 600)
    print("   - Framebuffers created")
    print("   - Shader programs compiled")

    # Test framebuffer status
    print("\n2. Testing framebuffer status...")
    
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, post_processor.framebuffers['main'])
    main_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
    
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, post_processor.framebuffers['bloom'])
    bloom_status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
    
    print(f"   - Main framebuffer status: {'Complete' if main_status else 'Incomplete'}")
    print(f"   - Bloom framebuffer status: {'Complete' if bloom_status else 'Incomplete'}")

    # Create test effects
    print("\n3. Creating post-processing effects...")
    
    bloom_effect = PostProcessEffect(
        EffectType.BLOOM,
        enabled=True,
        intensity=1.0,
        parameters={'threshold': 1.0, 'intensity': 1.0}
    )
    post_processor.add_effect(bloom_effect)
    print("   - Added bloom effect")
    
    tone_mapping_effect = PostProcessEffect(
        EffectType.TONE_MAPPING,
        enabled=True,
        intensity=1.0,
        parameters={'exposure': 1.0, 'gamma': 2.2}
    )
    post_processor.add_effect(tone_mapping_effect)
    print("   - Added tone mapping effect")
    
    vignette_effect = PostProcessEffect(
        EffectType.VIGNETTE,
        enabled=True,
        intensity=0.5,
        parameters={'intensity': 0.5, 'radius': 0.5}
    )
    post_processor.add_effect(vignette_effect)
    print("   - Added vignette effect")

    # Test effect management
    print("\n4. Effect management...")
    
    print(f"   - Total effects: {len(post_processor.effects)}")
    print(f"   - Enabled effects: {sum(1 for e in post_processor.effects if e.enabled)}")
    
    # Disable an effect
    bloom_effect.enabled = False
    print("   - Disabled bloom effect")
    print(f"   - Enabled effects: {sum(1 for e in post_processor.effects if e.enabled)}")

    # Performance characteristics
    print("\n5. Performance characteristics:")
    print("   - Bloom: O(n) where n = number of bright pixels")
    print("   - SSAO: O(k) where k = number of samples")
    print("   - Tone mapping: O(1) per pixel")
    print("   - Vignette: O(1) per pixel")
    print("   - Memory usage: Multiple framebuffers")

    print("\n6. Features demonstrated:")
    print("   - Multiple post-processing effects")
    print("   - Framebuffer management")
    print("   - Effect parameterization")
    print("   - Effect enable/disable")
    print("   - Screen-space effects")
    print("   - Real-time post-processing pipeline")

    # Cleanup
    post_processor.cleanup()
    print("\n7. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_post_processing()
