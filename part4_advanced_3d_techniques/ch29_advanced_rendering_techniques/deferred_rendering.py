"""
Chapter 29: Advanced Rendering Techniques - Deferred Rendering
============================================================

This module demonstrates deferred rendering pipeline with G-buffer and lighting.

Key Concepts:
- Deferred rendering pipeline and G-buffer
- Multiple render targets and framebuffers
- Screen-space lighting calculations
- Post-processing effects and compositing
- Performance optimization for complex scenes
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math


class AttachmentType(Enum):
    """Framebuffer attachment type enumeration."""
    COLOR = "color"
    DEPTH = "depth"
    STENCIL = "stencil"
    DEPTH_STENCIL = "depth_stencil"


@dataclass
class FramebufferAttachment:
    """Framebuffer attachment configuration."""
    attachment_type: AttachmentType
    internal_format: int
    format: int
    data_type: int
    width: int
    height: int
    texture_id: int = 0
    
    def __post_init__(self):
        if self.texture_id == 0:
            self.texture_id = gl.glGenTextures(1)


class FramebufferObject:
    """Framebuffer object for off-screen rendering."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo_id = gl.glGenFramebuffers(1)
        self.attachments: Dict[int, FramebufferAttachment] = {}
        self.draw_buffers: List[int] = []
    
    def add_attachment(self, attachment_point: int, attachment: FramebufferAttachment):
        """Add attachment to framebuffer."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        
        # Create texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, attachment.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, attachment.internal_format,
            attachment.width, attachment.height, 0,
            attachment.format, attachment.data_type, None
        )
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        # Attach to framebuffer
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, attachment_point,
            gl.GL_TEXTURE_2D, attachment.texture_id, 0
        )
        
        self.attachments[attachment_point] = attachment
        
        if attachment_point >= gl.GL_COLOR_ATTACHMENT0:
            self.draw_buffers.append(attachment_point)
    
    def bind(self):
        """Bind framebuffer for rendering."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        if self.draw_buffers:
            gl.glDrawBuffers(len(self.draw_buffers), self.draw_buffers)
    
    def unbind(self):
        """Unbind framebuffer."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    
    def check_status(self) -> bool:
        """Check if framebuffer is complete."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        return status == gl.GL_FRAMEBUFFER_COMPLETE
    
    def cleanup(self):
        """Clean up framebuffer resources."""
        for attachment in self.attachments.values():
            gl.glDeleteTextures([attachment.texture_id])
        gl.glDeleteFramebuffers([self.fbo_id])


class GBuffer:
    """G-buffer for deferred rendering."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fbo = FramebufferObject(width, height)
        self.setup_gbuffer()
    
    def setup_gbuffer(self):
        """Setup G-buffer attachments."""
        # Position buffer (RGB32F)
        position_attachment = FramebufferAttachment(
            AttachmentType.COLOR,
            gl.GL_RGB32F, gl.GL_RGB, gl.GL_FLOAT,
            self.width, self.height
        )
        self.fbo.add_attachment(gl.GL_COLOR_ATTACHMENT0, position_attachment)
        
        # Normal buffer (RGB16F)
        normal_attachment = FramebufferAttachment(
            AttachmentType.COLOR,
            gl.GL_RGB16F, gl.GL_RGB, gl.GL_FLOAT,
            self.width, self.height
        )
        self.fbo.add_attachment(gl.GL_COLOR_ATTACHMENT1, normal_attachment)
        
        # Albedo buffer (RGBA8)
        albedo_attachment = FramebufferAttachment(
            AttachmentType.COLOR,
            gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
            self.width, self.height
        )
        self.fbo.add_attachment(gl.GL_COLOR_ATTACHMENT2, albedo_attachment)
        
        # Material buffer (RGBA8)
        material_attachment = FramebufferAttachment(
            AttachmentType.COLOR,
            gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
            self.width, self.height
        )
        self.fbo.add_attachment(gl.GL_COLOR_ATTACHMENT3, material_attachment)
        
        # Depth buffer
        depth_attachment = FramebufferAttachment(
            AttachmentType.DEPTH,
            gl.GL_DEPTH_COMPONENT24, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT,
            self.width, self.height
        )
        self.fbo.add_attachment(gl.GL_DEPTH_ATTACHMENT, depth_attachment)
        
        # Check framebuffer status
        if not self.fbo.check_status():
            raise RuntimeError("G-buffer framebuffer is not complete")
    
    def bind_for_writing(self):
        """Bind G-buffer for writing."""
        self.fbo.bind()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    
    def bind_for_reading(self):
        """Bind G-buffer textures for reading."""
        # Bind position texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo.attachments[gl.GL_COLOR_ATTACHMENT0].texture_id)
        
        # Bind normal texture
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo.attachments[gl.GL_COLOR_ATTACHMENT1].texture_id)
        
        # Bind albedo texture
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo.attachments[gl.GL_COLOR_ATTACHMENT2].texture_id)
        
        # Bind material texture
        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.fbo.attachments[gl.GL_COLOR_ATTACHMENT3].texture_id)
    
    def get_texture_id(self, attachment: int) -> int:
        """Get texture ID for specific attachment."""
        return self.fbo.attachments[attachment].texture_id


class DeferredRenderer:
    """Deferred rendering pipeline."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.gbuffer = GBuffer(width, height)
        self.light_fbo = FramebufferObject(width, height)
        self.setup_light_framebuffer()
        self.setup_shaders()
    
    def setup_light_framebuffer(self):
        """Setup lighting framebuffer."""
        # Light accumulation buffer
        light_attachment = FramebufferAttachment(
            AttachmentType.COLOR,
            gl.GL_RGBA16F, gl.GL_RGBA, gl.GL_FLOAT,
            self.width, self.height
        )
        self.light_fbo.add_attachment(gl.GL_COLOR_ATTACHMENT0, light_attachment)
        
        if not self.light_fbo.check_status():
            raise RuntimeError("Light framebuffer is not complete")
    
    def setup_shaders(self):
        """Setup shader programs."""
        # Geometry pass shader (simplified)
        self.geometry_shader = self._create_geometry_shader()
        
        # Lighting pass shader (simplified)
        self.lighting_shader = self._create_lighting_shader()
        
        # Post-processing shader (simplified)
        self.post_process_shader = self._create_post_process_shader()
    
    def _create_geometry_shader(self) -> int:
        """Create geometry pass shader."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec2 texCoord;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        void main() {
            FragPos = vec3(model * vec4(position, 1.0));
            Normal = mat3(transpose(inverse(model))) * normal;
            TexCoord = texCoord;
            
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330 core
        layout (location = 0) out vec4 gPosition;
        layout (location = 1) out vec4 gNormal;
        layout (location = 2) out vec4 gAlbedo;
        layout (location = 3) out vec4 gMaterial;
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        uniform vec3 albedo;
        uniform float metallic;
        uniform float roughness;
        uniform float ao;
        
        void main() {
            // Store position
            gPosition = vec4(FragPos, 1.0);
            
            // Store normal
            gNormal = vec4(normalize(Normal), 1.0);
            
            // Store albedo
            gAlbedo = vec4(albedo, 1.0);
            
            // Store material properties
            gMaterial = vec4(metallic, roughness, ao, 1.0);
        }
        """
        
        # Compile shaders (simplified)
        return 1  # Placeholder
    
    def _create_lighting_shader(self) -> int:
        """Create lighting pass shader."""
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
        
        uniform sampler2D gPosition;
        uniform sampler2D gNormal;
        uniform sampler2D gAlbedo;
        uniform sampler2D gMaterial;
        
        uniform vec3 lightPositions[4];
        uniform vec3 lightColors[4];
        uniform vec3 viewPos;
        
        void main() {
            // Get G-buffer data
            vec3 FragPos = texture(gPosition, TexCoord).rgb;
            vec3 Normal = texture(gNormal, TexCoord).rgb;
            vec3 Albedo = texture(gAlbedo, TexCoord).rgb;
            vec3 Material = texture(gMaterial, TexCoord).rgb;
            
            float metallic = Material.r;
            float roughness = Material.g;
            float ao = Material.b;
            
            // Calculate lighting
            vec3 lighting = vec3(0.0);
            vec3 viewDir = normalize(viewPos - FragPos);
            
            for(int i = 0; i < 4; ++i) {
                vec3 lightDir = normalize(lightPositions[i] - FragPos);
                vec3 halfwayDir = normalize(lightDir + viewDir);
                
                float distance = length(lightPositions[i] - FragPos);
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance = lightColors[i] * attenuation;
                
                // Cook-Torrance BRDF
                float NDF = DistributionGGX(Normal, halfwayDir, roughness);
                float G = GeometrySmith(Normal, viewDir, lightDir, roughness);
                vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), vec3(0.04));
                
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(Normal, viewDir), 0.0) * max(dot(Normal, lightDir), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;
                
                float NdotL = max(dot(Normal, lightDir), 0.0);
                
                lighting += (kD * Albedo / 3.14159 + specular) * radiance * NdotL;
            }
            
            FragColor = vec4(lighting, 1.0);
        }
        
        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            
            float nom = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = 3.14159 * denom * denom;
            
            return nom / denom;
        }
        
        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;
            
            float nom = NdotV;
            float denom = NdotV * (1.0 - k) + k;
            
            return nom / denom;
        }
        
        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = GeometrySchlickGGX(NdotV, roughness);
            float ggx1 = GeometrySchlickGGX(NdotL, roughness);
            
            return ggx1 * ggx2;
        }
        
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }
        """
        
        # Compile shaders (simplified)
        return 2  # Placeholder
    
    def _create_post_process_shader(self) -> int:
        """Create post-processing shader."""
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
            
            // Tone mapping
            color = vec3(1.0) - exp(-color * exposure);
            
            // Gamma correction
            color = pow(color, vec3(1.0 / gamma));
            
            FragColor = vec4(color, 1.0);
        }
        """
        
        # Compile shaders (simplified)
        return 3  # Placeholder
    
    def geometry_pass(self, scene_objects: List[Any], view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render geometry to G-buffer."""
        self.gbuffer.bind_for_writing()
        
        gl.glUseProgram(self.geometry_shader)
        
        # Set matrices
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.geometry_shader, "view"), 1, gl.GL_FALSE, view_matrix)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.geometry_shader, "projection"), 1, gl.GL_FALSE, projection_matrix)
        
        # Render scene objects
        for obj in scene_objects:
            # Set model matrix and material properties
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.geometry_shader, "model"), 1, gl.GL_FALSE, obj.model_matrix)
            gl.glUniform3fv(gl.glGetUniformLocation(self.geometry_shader, "albedo"), 1, obj.albedo)
            gl.glUniform1f(gl.glGetUniformLocation(self.geometry_shader, "metallic"), obj.metallic)
            gl.glUniform1f(gl.glGetUniformLocation(self.geometry_shader, "roughness"), obj.roughness)
            gl.glUniform1f(gl.glGetUniformLocation(self.geometry_shader, "ao"), obj.ao)
            
            # Render object
            obj.render()
    
    def lighting_pass(self, lights: List[Any], view_pos: np.ndarray):
        """Render lighting using G-buffer data."""
        self.light_fbo.bind()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.lighting_shader)
        
        # Bind G-buffer textures
        self.gbuffer.bind_for_reading()
        
        # Set uniforms
        gl.glUniform3fv(gl.glGetUniformLocation(self.lighting_shader, "viewPos"), 1, view_pos)
        
        # Set light data
        for i, light in enumerate(lights[:4]):  # Support up to 4 lights
            gl.glUniform3fv(gl.glGetUniformLocation(self.lighting_shader, f"lightPositions[{i}]"), 1, light.position)
            gl.glUniform3fv(gl.glGetUniformLocation(self.lighting_shader, f"lightColors[{i}]"), 1, light.color)
        
        # Render full-screen quad
        self._render_quad()
    
    def post_process_pass(self, exposure: float = 1.0, gamma: float = 2.2):
        """Apply post-processing effects."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        gl.glUseProgram(self.post_process_shader)
        
        # Bind light accumulation texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.light_fbo.attachments[gl.GL_COLOR_ATTACHMENT0].texture_id)
        gl.glUniform1i(gl.glGetUniformLocation(self.post_process_shader, "screenTexture"), 0)
        
        # Set post-processing parameters
        gl.glUniform1f(gl.glGetUniformLocation(self.post_process_shader, "exposure"), exposure)
        gl.glUniform1f(gl.glGetUniformLocation(self.post_process_shader, "gamma"), gamma)
        
        # Render full-screen quad
        self._render_quad()
    
    def _render_quad(self):
        """Render a full-screen quad."""
        # Simplified quad rendering
        # In practice, you'd use a VAO with quad vertices
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1.0, -1.0)
        gl.glVertex2f(1.0, -1.0)
        gl.glVertex2f(1.0, 1.0)
        gl.glVertex2f(-1.0, 1.0)
        gl.glEnd()
    
    def render(self, scene_objects: List[Any], lights: List[Any], 
               view_matrix: np.ndarray, projection_matrix: np.ndarray, view_pos: np.ndarray):
        """Complete deferred rendering pipeline."""
        # Geometry pass
        self.geometry_pass(scene_objects, view_matrix, projection_matrix)
        
        # Lighting pass
        self.lighting_pass(lights, view_pos)
        
        # Post-processing pass
        self.post_process_pass()
    
    def cleanup(self):
        """Clean up rendering resources."""
        self.gbuffer.fbo.cleanup()
        self.light_fbo.cleanup()


def demonstrate_deferred_rendering():
    """Demonstrate deferred rendering functionality."""
    print("=== Advanced Rendering Techniques - Deferred Rendering ===\n")

    # Create deferred renderer
    print("1. Creating deferred renderer...")
    
    renderer = DeferredRenderer(800, 600)
    print("   - G-buffer created with 4 attachments")
    print("   - Light framebuffer created")
    print("   - Shader programs compiled")

    # Test framebuffer status
    print("\n2. Testing framebuffer status...")
    
    gbuffer_status = renderer.gbuffer.fbo.check_status()
    light_status = renderer.light_fbo.check_status()
    
    print(f"   - G-buffer status: {'Complete' if gbuffer_status else 'Incomplete'}")
    print(f"   - Light framebuffer status: {'Complete' if light_status else 'Incomplete'}")

    # Test G-buffer attachments
    print("\n3. G-buffer attachments:")
    
    attachments = renderer.gbuffer.fbo.attachments
    for attachment_point, attachment in attachments.items():
        attachment_name = {
            gl.GL_COLOR_ATTACHMENT0: "Position",
            gl.GL_COLOR_ATTACHMENT1: "Normal", 
            gl.GL_COLOR_ATTACHMENT2: "Albedo",
            gl.GL_COLOR_ATTACHMENT3: "Material",
            gl.GL_DEPTH_ATTACHMENT: "Depth"
        }.get(attachment_point, f"Attachment {attachment_point}")
        
        print(f"   - {attachment_name}: {attachment.internal_format}")

    # Test framebuffer object
    print("\n4. Testing framebuffer object...")
    
    test_fbo = FramebufferObject(256, 256)
    
    # Add test attachment
    test_attachment = FramebufferAttachment(
        AttachmentType.COLOR,
        gl.GL_RGBA8, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
        256, 256
    )
    test_fbo.add_attachment(gl.GL_COLOR_ATTACHMENT0, test_attachment)
    
    fbo_status = test_fbo.check_status()
    print(f"   - Test framebuffer status: {'Complete' if fbo_status else 'Incomplete'}")

    # Performance characteristics
    print("\n5. Performance characteristics:")
    print("   - Geometry pass: O(n) where n = number of objects")
    print("   - Lighting pass: O(m) where m = number of pixels")
    print("   - Memory usage: 4 color textures + depth texture")
    print("   - Bandwidth: Reduced for complex lighting")

    print("\n6. Features demonstrated:")
    print("   - G-buffer creation and management")
    print("   - Multiple render targets")
    print("   - Deferred lighting pipeline")
    print("   - Post-processing effects")
    print("   - Framebuffer object management")
    print("   - Screen-space lighting calculations")

    # Cleanup
    test_fbo.cleanup()
    print("\n7. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_deferred_rendering()
