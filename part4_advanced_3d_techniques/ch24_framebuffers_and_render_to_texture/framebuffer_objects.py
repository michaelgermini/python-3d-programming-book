"""
Chapter 24: Framebuffers and Render-to-Texture - Framebuffer Objects
==================================================================

This module demonstrates framebuffer objects and off-screen rendering techniques.

Key Concepts:
- Framebuffer Objects (FBOs) for off-screen rendering
- Multiple render targets and attachments
- Color, depth, and stencil buffer management
- Render-to-texture techniques for advanced effects
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class AttachmentType(Enum):
    """Framebuffer attachment type enumeration."""
    COLOR = "color"
    DEPTH = "depth"
    STENCIL = "stencil"
    DEPTH_STENCIL = "depth_stencil"


class TextureFormat(Enum):
    """Texture format enumeration."""
    RGB = gl.GL_RGB
    RGBA = gl.GL_RGBA
    DEPTH = gl.GL_DEPTH_COMPONENT
    DEPTH_STENCIL = gl.GL_DEPTH_STENCIL


@dataclass
class FramebufferAttachment:
    """Represents a framebuffer attachment."""
    attachment_type: AttachmentType
    texture_id: int
    format: TextureFormat
    width: int
    height: int
    internal_format: int
    data_type: int


class FramebufferObject:
    """Represents a Framebuffer Object (FBO)."""

    def __init__(self, width: int, height: int):
        self.fbo_id = gl.glGenFramebuffers(1)
        self.width = width
        self.height = height
        self.attachments: Dict[int, FramebufferAttachment] = {}
        self.bound = False
        self.complete = False

    def bind(self):
        """Bind the framebuffer."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        self.bound = True

    def unbind(self):
        """Unbind the framebuffer (bind to default framebuffer)."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.bound = False

    def add_color_attachment(self, slot: int, format: TextureFormat = TextureFormat.RGBA) -> bool:
        """Add a color attachment to the framebuffer."""
        # Create texture
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        # Allocate texture storage
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format.value, self.width, self.height, 
                       0, format.value, gl.GL_UNSIGNED_BYTE, None)
        
        # Attach to framebuffer
        attachment_point = gl.GL_COLOR_ATTACHMENT0 + slot
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, attachment_point, gl.GL_TEXTURE_2D, texture_id, 0)
        
        # Store attachment info
        self.attachments[attachment_point] = FramebufferAttachment(
            attachment_type=AttachmentType.COLOR,
            texture_id=texture_id,
            format=format,
            width=self.width,
            height=self.height,
            internal_format=format.value,
            data_type=gl.GL_UNSIGNED_BYTE
        )
        
        return True

    def add_depth_attachment(self, format: TextureFormat = TextureFormat.DEPTH) -> bool:
        """Add a depth attachment to the framebuffer."""
        # Create texture
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        # Allocate texture storage
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, format.value, self.width, self.height, 
                       0, format.value, gl.GL_FLOAT, None)
        
        # Attach to framebuffer
        attachment_point = gl.GL_DEPTH_ATTACHMENT
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, attachment_point, gl.GL_TEXTURE_2D, texture_id, 0)
        
        # Store attachment info
        self.attachments[attachment_point] = FramebufferAttachment(
            attachment_type=AttachmentType.DEPTH,
            texture_id=texture_id,
            format=format,
            width=self.width,
            height=self.height,
            internal_format=format.value,
            data_type=gl.GL_FLOAT
        )
        
        return True

    def add_depth_stencil_attachment(self) -> bool:
        """Add a depth-stencil attachment to the framebuffer."""
        # Create renderbuffer for depth-stencil
        rbo_id = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo_id)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH24_STENCIL8, self.width, self.height)
        
        # Attach to framebuffer
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_STENCIL_ATTACHMENT, 
                                    gl.GL_RENDERBUFFER, rbo_id)
        
        # Store attachment info (using texture_id field for renderbuffer ID)
        self.attachments[gl.GL_DEPTH_STENCIL_ATTACHMENT] = FramebufferAttachment(
            attachment_type=AttachmentType.DEPTH_STENCIL,
            texture_id=rbo_id,  # Actually a renderbuffer ID
            format=TextureFormat.DEPTH_STENCIL,
            width=self.width,
            height=self.height,
            internal_format=gl.GL_DEPTH24_STENCIL8,
            data_type=gl.GL_UNSIGNED_INT_24_8
        )
        
        return True

    def check_completeness(self) -> bool:
        """Check if the framebuffer is complete."""
        if not self.bound:
            self.bind()
        
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        self.complete = (status == gl.GL_FRAMEBUFFER_COMPLETE)
        
        if not self.complete:
            print(f"Framebuffer incomplete: {status}")
        
        return self.complete

    def set_draw_buffers(self, color_attachments: List[int]):
        """Set which color attachments to draw to."""
        if not self.bound:
            self.bind()
        
        draw_buffers = [gl.GL_COLOR_ATTACHMENT0 + slot for slot in color_attachments]
        gl.glDrawBuffers(len(draw_buffers), draw_buffers)

    def set_read_buffer(self, color_attachment: int):
        """Set which color attachment to read from."""
        if not self.bound:
            self.bind()
        
        read_buffer = gl.GL_COLOR_ATTACHMENT0 + color_attachment
        gl.glReadBuffer(read_buffer)

    def get_texture_id(self, attachment_type: AttachmentType, slot: int = 0) -> Optional[int]:
        """Get the texture ID for a specific attachment."""
        if attachment_type == AttachmentType.COLOR:
            attachment_point = gl.GL_COLOR_ATTACHMENT0 + slot
        elif attachment_type == AttachmentType.DEPTH:
            attachment_point = gl.GL_DEPTH_ATTACHMENT
        elif attachment_type == AttachmentType.DEPTH_STENCIL:
            attachment_point = gl.GL_DEPTH_STENCIL_ATTACHMENT
        else:
            return None
        
        if attachment_point in self.attachments:
            return self.attachments[attachment_point].texture_id
        return None

    def bind_texture(self, attachment_type: AttachmentType, slot: int = 0, texture_unit: int = 0):
        """Bind a framebuffer texture to a texture unit."""
        texture_id = self.get_texture_id(attachment_type, slot)
        if texture_id:
            gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    def cleanup(self):
        """Clean up framebuffer resources."""
        # Clean up textures
        for attachment in self.attachments.values():
            if attachment.attachment_type == AttachmentType.DEPTH_STENCIL:
                gl.glDeleteRenderbuffers(1, [attachment.texture_id])
            else:
                gl.glDeleteTextures(1, [attachment.texture_id])
        
        # Clean up framebuffer
        if self.fbo_id:
            gl.glDeleteFramebuffers(1, [self.fbo_id])
            self.fbo_id = 0


class FramebufferManager:
    """Manages multiple framebuffers and their resources."""

    def __init__(self):
        self.framebuffers: Dict[str, FramebufferObject] = {}
        self.active_framebuffer: Optional[FramebufferObject] = None

    def create_framebuffer(self, name: str, width: int, height: int) -> FramebufferObject:
        """Create a new framebuffer."""
        fbo = FramebufferObject(width, height)
        self.framebuffers[name] = fbo
        return fbo

    def get_framebuffer(self, name: str) -> Optional[FramebufferObject]:
        """Get a framebuffer by name."""
        return self.framebuffers.get(name)

    def bind_framebuffer(self, name: str):
        """Bind a framebuffer by name."""
        if name in self.framebuffers:
            self.framebuffers[name].bind()
            self.active_framebuffer = self.framebuffers[name]
        else:
            print(f"Framebuffer '{name}' not found")

    def unbind_framebuffer(self):
        """Unbind the current framebuffer."""
        if self.active_framebuffer:
            self.active_framebuffer.unbind()
            self.active_framebuffer = None

    def create_g_buffer(self, name: str, width: int, height: int) -> FramebufferObject:
        """Create a G-Buffer for deferred rendering."""
        fbo = self.create_framebuffer(name, width, height)
        fbo.bind()
        
        # Position buffer
        fbo.add_color_attachment(0, TextureFormat.RGBA)  # Position (RGB) + unused (A)
        
        # Normal buffer
        fbo.add_color_attachment(1, TextureFormat.RGBA)  # Normal (RGB) + unused (A)
        
        # Albedo + Specular buffer
        fbo.add_color_attachment(2, TextureFormat.RGBA)  # Albedo (RGB) + Specular (A)
        
        # Depth buffer
        fbo.add_depth_attachment()
        
        # Set draw buffers
        fbo.set_draw_buffers([0, 1, 2])
        
        # Check completeness
        if not fbo.check_completeness():
            print(f"G-Buffer '{name}' creation failed")
        
        fbo.unbind()
        return fbo

    def create_shadow_map(self, name: str, width: int, height: int) -> FramebufferObject:
        """Create a shadow map framebuffer."""
        fbo = self.create_framebuffer(name, width, height)
        fbo.bind()
        
        # Only depth attachment for shadow mapping
        fbo.add_depth_attachment()
        
        # No color attachments for shadow mapping
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        
        # Check completeness
        if not fbo.check_completeness():
            print(f"Shadow map '{name}' creation failed")
        
        fbo.unbind()
        return fbo

    def create_post_process_buffer(self, name: str, width: int, height: int) -> FramebufferObject:
        """Create a post-processing framebuffer."""
        fbo = self.create_framebuffer(name, width, height)
        fbo.bind()
        
        # Color attachment for post-processing
        fbo.add_color_attachment(0, TextureFormat.RGBA)
        
        # Check completeness
        if not fbo.check_completeness():
            print(f"Post-process buffer '{name}' creation failed")
        
        fbo.unbind()
        return fbo

    def cleanup(self):
        """Clean up all framebuffer resources."""
        for fbo in self.framebuffers.values():
            fbo.cleanup()
        self.framebuffers.clear()
        self.active_framebuffer = None


def demonstrate_framebuffer_objects():
    """Demonstrate framebuffer objects and off-screen rendering."""
    print("=== Framebuffers and Render-to-Texture - Framebuffer Objects ===\n")

    # Create framebuffer manager
    manager = FramebufferManager()

    # Create different types of framebuffers
    print("1. Creating G-Buffer for deferred rendering...")
    g_buffer = manager.create_g_buffer("g_buffer", 1024, 768)
    print(f"G-Buffer created: {g_buffer.complete}")

    print("\n2. Creating shadow map...")
    shadow_map = manager.create_shadow_map("shadow_map", 2048, 2048)
    print(f"Shadow map created: {shadow_map.complete}")

    print("\n3. Creating post-process buffer...")
    post_process = manager.create_post_process_buffer("post_process", 1024, 768)
    print(f"Post-process buffer created: {post_process.complete}")

    # Display framebuffer information
    print(f"\n4. Framebuffer Information:")
    for name, fbo in manager.framebuffers.items():
        print(f"  {name}:")
        print(f"    Size: {fbo.width}x{fbo.height}")
        print(f"    Complete: {fbo.complete}")
        print(f"    Attachments: {len(fbo.attachments)}")
        for attachment_point, attachment in fbo.attachments.items():
            print(f"      {attachment.attachment_type.value}: {attachment_point}")

    # Demonstrate usage
    print(f"\n5. Usage Examples:")
    print("  To render to G-Buffer:")
    print("    manager.bind_framebuffer('g_buffer')")
    print("    # Render geometry...")
    print("    manager.unbind_framebuffer()")
    
    print("\n  To use G-Buffer textures:")
    print("    g_buffer.bind_texture(AttachmentType.COLOR, 0, 0)  # Position")
    print("    g_buffer.bind_texture(AttachmentType.COLOR, 1, 1)  # Normal")
    print("    g_buffer.bind_texture(AttachmentType.COLOR, 2, 2)  # Albedo")

    # Cleanup
    manager.cleanup()
    print("\n6. Resources cleaned up successfully")


if __name__ == "__main__":
    demonstrate_framebuffer_objects()
