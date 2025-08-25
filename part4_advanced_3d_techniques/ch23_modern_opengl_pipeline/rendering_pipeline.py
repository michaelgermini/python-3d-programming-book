"""
Chapter 23: Modern OpenGL Pipeline - Rendering Pipeline
=====================================================

This module demonstrates a complete modern OpenGL rendering pipeline.

Key Concepts:
- Complete rendering pipeline setup and management
- State management and optimization
- Batch rendering and draw call optimization
- Performance monitoring and profiling
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time


class RenderState(Enum):
    """OpenGL render state enumeration."""
    DEPTH_TEST = "depth_test"
    BLENDING = "blending"
    CULLING = "culling"
    WIREFRAME = "wireframe"


@dataclass
class RenderStats:
    """Rendering statistics and performance metrics."""
    draw_calls: int = 0
    triangles_rendered: int = 0
    vertices_processed: int = 0
    frame_time: float = 0.0
    fps: float = 0.0
    gpu_memory_used: int = 0


class RenderStateManager:
    """Manages OpenGL render states and optimizations."""

    def __init__(self):
        self.current_states: Dict[RenderState, bool] = {}
        self.state_stack: List[Dict[RenderState, bool]] = []

    def enable_state(self, state: RenderState):
        """Enable a render state."""
        if state == RenderState.DEPTH_TEST:
            gl.glEnable(gl.GL_DEPTH_TEST)
        elif state == RenderState.BLENDING:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        elif state == RenderState.CULLING:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
        elif state == RenderState.WIREFRAME:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        
        self.current_states[state] = True

    def disable_state(self, state: RenderState):
        """Disable a render state."""
        if state == RenderState.DEPTH_TEST:
            gl.glDisable(gl.GL_DEPTH_TEST)
        elif state == RenderState.BLENDING:
            gl.glDisable(gl.GL_BLEND)
        elif state == RenderState.CULLING:
            gl.glDisable(gl.GL_CULL_FACE)
        elif state == RenderState.WIREFRAME:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        self.current_states[state] = False

    def push_states(self):
        """Push current states onto stack."""
        self.state_stack.append(self.current_states.copy())

    def pop_states(self):
        """Pop states from stack and restore."""
        if self.state_stack:
            previous_states = self.state_stack.pop()
            for state, enabled in previous_states.items():
                if enabled:
                    self.enable_state(state)
                else:
                    self.disable_state(state)
            self.current_states = previous_states

    def set_depth_function(self, func: int):
        """Set depth test function."""
        gl.glDepthFunc(func)

    def set_blend_function(self, src: int, dst: int):
        """Set blending function."""
        gl.glBlendFunc(src, dst)

    def set_cull_face(self, face: int):
        """Set culling face."""
        gl.glCullFace(face)


class RenderPass:
    """Represents a render pass with specific settings."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.clear_flags = gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        self.render_callback: Optional[Callable] = None

    def set_clear_flags(self, flags: int):
        """Set clear flags for this pass."""
        self.clear_flags = flags

    def set_clear_color(self, r: float, g: float, b: float, a: float = 1.0):
        """Set clear color for this pass."""
        self.clear_color = (r, g, b, a)

    def set_render_callback(self, callback: Callable):
        """Set the render callback for this pass."""
        self.render_callback = callback

    def execute(self):
        """Execute this render pass."""
        if not self.enabled or not self.render_callback:
            return

        # Set clear color
        gl.glClearColor(*self.clear_color)
        
        # Clear buffers
        gl.glClear(self.clear_flags)
        
        # Execute render callback
        self.render_callback()


class RenderPipeline:
    """Complete modern OpenGL rendering pipeline."""

    def __init__(self):
        self.render_passes: List[RenderPass] = []
        self.state_manager = RenderStateManager()
        self.stats = RenderStats()
        self.frame_start_time = 0.0
        self.frame_count = 0

    def add_render_pass(self, render_pass: RenderPass):
        """Add a render pass to the pipeline."""
        self.render_passes.append(render_pass)

    def remove_render_pass(self, name: str):
        """Remove a render pass by name."""
        self.render_passes = [pass_obj for pass_obj in self.render_passes if pass_obj.name != name]

    def get_render_pass(self, name: str) -> Optional[RenderPass]:
        """Get a render pass by name."""
        for render_pass in self.render_passes:
            if render_pass.name == name:
                return render_pass
        return None

    def begin_frame(self):
        """Begin a new frame."""
        self.frame_start_time = time.time()
        self.stats.draw_calls = 0
        self.stats.triangles_rendered = 0
        self.stats.vertices_processed = 0

    def end_frame(self):
        """End the current frame."""
        frame_time = time.time() - self.frame_start_time
        self.stats.frame_time = frame_time
        self.stats.fps = 1.0 / frame_time if frame_time > 0 else 0
        self.frame_count += 1

    def render_frame(self):
        """Render a complete frame."""
        self.begin_frame()

        # Execute all render passes
        for render_pass in self.render_passes:
            render_pass.execute()

        self.end_frame()

    def get_stats(self) -> RenderStats:
        """Get current rendering statistics."""
        return self.stats

    def record_draw_call(self, triangle_count: int = 0, vertex_count: int = 0):
        """Record a draw call for statistics."""
        self.stats.draw_calls += 1
        self.stats.triangles_rendered += triangle_count
        self.stats.vertices_processed += vertex_count


class BatchRenderer:
    """Efficient batch rendering system."""

    def __init__(self, pipeline: RenderPipeline):
        self.pipeline = pipeline
        self.batches: Dict[str, List[Any]] = {}
        self.batch_size = 1000  # Maximum objects per batch

    def add_to_batch(self, batch_name: str, render_data: Any):
        """Add render data to a batch."""
        if batch_name not in self.batches:
            self.batches[batch_name] = []
        
        self.batches[batch_name].append(render_data)

    def render_batch(self, batch_name: str, render_function: Callable):
        """Render a batch of objects."""
        if batch_name not in self.batches:
            return

        batch = self.batches[batch_name]
        if not batch:
            return

        # Render in chunks to avoid excessive draw calls
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i:i + self.batch_size]
            render_function(chunk)
            self.pipeline.record_draw_call(len(chunk))

    def clear_batch(self, batch_name: str):
        """Clear a batch."""
        if batch_name in self.batches:
            self.batches[batch_name].clear()

    def clear_all_batches(self):
        """Clear all batches."""
        for batch_name in self.batches:
            self.batches[batch_name].clear()


class PerformanceProfiler:
    """Performance profiling and monitoring."""

    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.timer_stack: List[str] = []
        self.frame_times: List[float] = []
        self.max_frame_history = 60

    def start_timer(self, name: str):
        """Start a timer."""
        self.timers[name] = time.time()
        self.timer_stack.append(name)

    def end_timer(self, name: str) -> float:
        """End a timer and return elapsed time."""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            
            # Remove from stack
            if name in self.timer_stack:
                self.timer_stack.remove(name)
            
            return elapsed
        return 0.0

    def end_current_timer(self) -> float:
        """End the most recent timer."""
        if self.timer_stack:
            return self.end_timer(self.timer_stack[-1])
        return 0.0

    def record_frame_time(self, frame_time: float):
        """Record frame time for averaging."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)

    def get_average_fps(self) -> float:
        """Get average FPS over recorded frames."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_min_fps(self) -> float:
        """Get minimum FPS over recorded frames."""
        if not self.frame_times:
            return 0.0
        
        max_frame_time = max(self.frame_times)
        return 1.0 / max_frame_time if max_frame_time > 0 else 0.0

    def get_max_fps(self) -> float:
        """Get maximum FPS over recorded frames."""
        if not self.frame_times:
            return 0.0
        
        min_frame_time = min(self.frame_times)
        return 1.0 / min_frame_time if min_frame_time > 0 else 0.0


class ModernRenderEngine:
    """Complete modern OpenGL rendering engine."""

    def __init__(self):
        self.pipeline = RenderPipeline()
        self.batch_renderer = BatchRenderer(self.pipeline)
        self.profiler = PerformanceProfiler()
        self.initialized = False

    def initialize(self):
        """Initialize the rendering engine."""
        # Enable common OpenGL features
        self.pipeline.state_manager.enable_state(RenderState.DEPTH_TEST)
        self.pipeline.state_manager.enable_state(RenderState.CULLING)
        self.pipeline.state_manager.set_depth_function(gl.GL_LESS)
        
        # Set viewport (would be set by window system)
        # gl.glViewport(0, 0, width, height)
        
        self.initialized = True
        print("Modern OpenGL rendering engine initialized")

    def create_render_pass(self, name: str, callback: Callable) -> RenderPass:
        """Create and add a render pass."""
        render_pass = RenderPass(name)
        render_pass.set_render_callback(callback)
        self.pipeline.add_render_pass(render_pass)
        return render_pass

    def render_frame(self):
        """Render a complete frame with profiling."""
        if not self.initialized:
            print("Rendering engine not initialized")
            return

        self.profiler.start_timer("frame")
        
        # Render the frame
        self.pipeline.render_frame()
        
        # Record frame time
        frame_time = self.profiler.end_timer("frame")
        self.profiler.record_frame_time(frame_time)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        render_stats = self.pipeline.get_stats()
        
        return {
            "frame_time": render_stats.frame_time,
            "fps": render_stats.fps,
            "avg_fps": self.profiler.get_average_fps(),
            "min_fps": self.profiler.get_min_fps(),
            "max_fps": self.profiler.get_max_fps(),
            "draw_calls": render_stats.draw_calls,
            "triangles": render_stats.triangles_rendered,
            "vertices": render_stats.vertices_processed
        }

    def cleanup(self):
        """Clean up rendering engine resources."""
        # Cleanup would be handled by individual components
        print("Rendering engine cleanup complete")


def demonstrate_rendering_pipeline():
    """Demonstrate the complete modern OpenGL rendering pipeline."""
    print("=== Modern OpenGL Pipeline - Rendering Pipeline ===\n")

    # Create rendering engine
    engine = ModernRenderEngine()
    engine.initialize()

    # Create render passes
    print("1. Creating render passes...")
    
    def geometry_pass():
        """Geometry rendering pass."""
        print("  Executing geometry pass")
        # Simulate rendering geometry
        engine.pipeline.record_draw_call(100, 300)
    
    def lighting_pass():
        """Lighting calculation pass."""
        print("  Executing lighting pass")
        # Simulate lighting calculations
        engine.pipeline.record_draw_call(50, 150)
    
    def post_process_pass():
        """Post-processing pass."""
        print("  Executing post-processing pass")
        # Simulate post-processing effects
        engine.pipeline.record_draw_call(10, 30)

    # Add render passes
    geometry_pass_obj = engine.create_render_pass("Geometry", geometry_pass)
    geometry_pass_obj.set_clear_color(0.2, 0.3, 0.4, 1.0)
    
    lighting_pass_obj = engine.create_render_pass("Lighting", lighting_pass)
    lighting_pass_obj.set_clear_flags(gl.GL_DEPTH_BUFFER_BIT)
    
    post_process_pass_obj = engine.create_render_pass("PostProcess", post_process_pass)
    post_process_pass_obj.set_clear_flags(0)  # Don't clear for post-processing

    print("Render passes created successfully")

    # Demonstrate batch rendering
    print("\n2. Setting up batch rendering...")
    
    # Add some objects to batches
    for i in range(100):
        engine.batch_renderer.add_to_batch("static_objects", f"object_{i}")
        engine.batch_renderer.add_to_batch("dynamic_objects", f"dynamic_{i}")

    def render_static_batch(objects):
        """Render static objects."""
        print(f"    Rendering {len(objects)} static objects")
        engine.pipeline.record_draw_call(len(objects) * 10, len(objects) * 30)

    def render_dynamic_batch(objects):
        """Render dynamic objects."""
        print(f"    Rendering {len(objects)} dynamic objects")
        engine.pipeline.record_draw_call(len(objects) * 5, len(objects) * 15)

    # Render batches
    engine.batch_renderer.render_batch("static_objects", render_static_batch)
    engine.batch_renderer.render_batch("dynamic_objects", render_dynamic_batch)

    # Demonstrate frame rendering
    print("\n3. Rendering frames...")
    for frame in range(3):
        print(f"  Frame {frame + 1}:")
        engine.render_frame()
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"    FPS: {stats['fps']:.1f}, Draw Calls: {stats['draw_calls']}")

    # Display final statistics
    print(f"\n4. Final Performance Statistics:")
    final_stats = engine.get_performance_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Cleanup
    engine.cleanup()
    print("\n5. Rendering pipeline demonstration complete")


if __name__ == "__main__":
    demonstrate_rendering_pipeline()
