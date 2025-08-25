"""
Chapter 27: Particle Systems and Visual Effects - Performance Optimization
======================================================================

This module demonstrates performance optimization techniques for particle systems.

Key Concepts:
- Particle system performance profiling and optimization
- GPU-based particle processing and rendering
- Memory management and data structures
- LOD (Level of Detail) systems for particles
- Batch processing and instanced rendering
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import cProfile
import pstats
import threading
from concurrent.futures import ThreadPoolExecutor


class OptimizationLevel(Enum):
    """Optimization level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class PerformanceMetrics:
    """Performance metrics for particle systems."""
    frame_time: float
    particle_count: int
    draw_calls: int
    memory_usage: float
    cpu_usage: float
    gpu_usage: float
    
    def __post_init__(self):
        if self.frame_time is None:
            self.frame_time = 0.0
        if self.particle_count is None:
            self.particle_count = 0
        if self.draw_calls is None:
            self.draw_calls = 0
        if self.memory_usage is None:
            self.memory_usage = 0.0
        if self.cpu_usage is None:
            self.cpu_usage = 0.0
        if self.gpu_usage is None:
            self.gpu_usage = 0.0


class ParticleLOD:
    """Level of Detail system for particles."""
    
    def __init__(self, distance_thresholds: List[float], detail_levels: List[int]):
        self.distance_thresholds = distance_thresholds
        self.detail_levels = detail_levels
        self.camera_position = np.array([0.0, 0.0, 0.0])
    
    def get_lod_level(self, particle_position: np.ndarray) -> int:
        """Get LOD level based on distance from camera."""
        distance = np.linalg.norm(particle_position - self.camera_position)
        
        for i, threshold in enumerate(self.distance_thresholds):
            if distance <= threshold:
                return self.detail_levels[i]
        
        return self.detail_levels[-1]  # Lowest detail level
    
    def update_camera_position(self, position: np.ndarray):
        """Update camera position for LOD calculations."""
        self.camera_position = position.copy()
    
    def get_particle_count_for_lod(self, base_count: int, lod_level: int) -> int:
        """Get particle count for specific LOD level."""
        # Reduce particle count based on LOD level
        reduction_factor = 1.0 / (2 ** lod_level)
        return max(1, int(base_count * reduction_factor))


class GPUParticleProcessor:
    """GPU-based particle processing using compute shaders."""
    
    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        self.compute_shader = 0
        self.particle_buffer = 0
        self.velocity_buffer = 0
        self.setup_gpu_processing()
    
    def setup_gpu_processing(self):
        """Setup GPU resources for particle processing."""
        # Create compute shader (simplified)
        self.compute_shader = self._create_compute_shader()
        
        # Create buffers
        self.particle_buffer = gl.glGenBuffers(1)
        self.velocity_buffer = gl.glGenBuffers(1)
        
        # Initialize buffers
        self._initialize_buffers()
    
    def _create_compute_shader(self) -> int:
        """Create compute shader for particle processing."""
        compute_shader_source = """
        #version 430
        
        layout(local_size_x = 256) in;
        
        layout(std430, binding = 0) buffer ParticleBuffer {
            vec4 particles[];
        };
        
        layout(std430, binding = 1) buffer VelocityBuffer {
            vec4 velocities[];
        };
        
        uniform float deltaTime;
        uniform vec3 gravity;
        
        void main() {
            uint index = gl_GlobalInvocationID.x;
            if (index >= particles.length()) return;
            
            // Update velocity
            velocities[index].xyz += gravity * deltaTime;
            
            // Update position
            particles[index].xyz += velocities[index].xyz * deltaTime;
            
            // Update life
            particles[index].w -= deltaTime;
        }
        """
        
        # Compile shader (simplified)
        return 1  # Placeholder
    
    def _initialize_buffers(self):
        """Initialize GPU buffers with particle data."""
        # Create particle data
        particle_data = np.zeros((self.max_particles, 4), dtype=np.float32)
        velocity_data = np.zeros((self.max_particles, 4), dtype=np.float32)
        
        # Bind and upload particle buffer
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.particle_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, particle_data.nbytes, particle_data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.particle_buffer)
        
        # Bind and upload velocity buffer
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.velocity_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, velocity_data.nbytes, velocity_data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.velocity_buffer)
    
    def process_particles(self, delta_time: float, active_particles: int):
        """Process particles on GPU."""
        gl.glUseProgram(self.compute_shader)
        
        # Set uniforms
        gl.glUniform1f(gl.glGetUniformLocation(self.compute_shader, "deltaTime"), delta_time)
        gl.glUniform3f(gl.glGetUniformLocation(self.compute_shader, "gravity"), 0.0, -9.81, 0.0)
        
        # Dispatch compute shader
        work_groups = (active_particles + 255) // 256
        gl.glDispatchCompute(work_groups, 1, 1)
        
        # Memory barrier
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
    
    def get_particle_data(self) -> np.ndarray:
        """Get processed particle data from GPU."""
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.particle_buffer)
        data = gl.glMapBuffer(gl.GL_SHADER_STORAGE_BUFFER, gl.GL_READ_ONLY)
        
        # Convert to numpy array (simplified)
        particle_data = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
        
        gl.glUnmapBuffer(gl.GL_SHADER_STORAGE_BUFFER)
        return particle_data
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.particle_buffer:
            gl.glDeleteBuffers(1, [self.particle_buffer])
        if self.velocity_buffer:
            gl.glDeleteBuffers(1, [self.velocity_buffer])


class InstancedParticleRenderer:
    """Instanced rendering for particles."""
    
    def __init__(self):
        self.vao = 0
        self.instance_buffer = 0
        self.shader_program = 0
        self.setup_instanced_rendering()
    
    def setup_instanced_rendering(self):
        """Setup instanced rendering resources."""
        # Create VAO
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        
        # Create instance buffer
        self.instance_buffer = gl.glGenBuffers(1)
        
        # Create shader program
        self.shader_program = self._create_instanced_shader()
        
        # Setup vertex attributes
        self._setup_vertex_attributes()
    
    def _create_instanced_shader(self) -> int:
        """Create shader for instanced particle rendering."""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec4 color;
        layout (location = 2) in float size;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec4 fragColor;
        
        void main() {
            gl_Position = projection * view * model * vec4(position * size, 1.0);
            fragColor = color;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec4 fragColor;
        out vec4 outColor;
        
        void main() {
            outColor = fragColor;
        }
        """
        
        # Compile shaders (simplified)
        return 1  # Placeholder
    
    def _setup_vertex_attributes(self):
        """Setup vertex attributes for instanced rendering."""
        # Position attribute
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        
        # Instance attributes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_buffer)
        
        # Color attribute (instanced)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 20, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribDivisor(1, 1)
        
        # Size attribute (instanced)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, 20, ctypes.c_void_p(16))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribDivisor(2, 1)
    
    def render_instanced(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray):
        """Render particles using instanced rendering."""
        if len(positions) == 0:
            return
        
        gl.glUseProgram(self.shader_program)
        gl.glBindVertexArray(self.vao)
        
        # Create instance data
        instance_data = np.column_stack([positions, colors, sizes])
        
        # Update instance buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, gl.GL_DYNAMIC_DRAW)
        
        # Enable blending
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Render instanced
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, len(positions))  # 6 vertices for quad
        
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)
    
    def cleanup(self):
        """Clean up instanced rendering resources."""
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.instance_buffer:
            gl.glDeleteBuffers(1, [self.instance_buffer])


class PerformanceProfiler:
    """Profiles particle system performance."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.profiler = cProfile.Profile()
        self.is_profiling = False
    
    def start_profiling(self):
        """Start performance profiling."""
        self.profiler.enable()
        self.is_profiling = True
    
    def stop_profiling(self):
        """Stop performance profiling."""
        self.profiler.disable()
        self.is_profiling = False
    
    def get_profile_stats(self) -> pstats.Stats:
        """Get profiling statistics."""
        return pstats.Stats(self.profiler)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
    
    def get_average_metrics(self) -> PerformanceMetrics:
        """Get average performance metrics."""
        if not self.metrics_history:
            return PerformanceMetrics(0.0, 0, 0, 0.0, 0.0, 0.0)
        
        avg_frame_time = np.mean([m.frame_time for m in self.metrics_history])
        avg_particle_count = np.mean([m.particle_count for m in self.metrics_history])
        avg_draw_calls = np.mean([m.draw_calls for m in self.metrics_history])
        avg_memory_usage = np.mean([m.memory_usage for m in self.metrics_history])
        avg_cpu_usage = np.mean([m.cpu_usage for m in self.metrics_history])
        avg_gpu_usage = np.mean([m.gpu_usage for m in self.metrics_history])
        
        return PerformanceMetrics(
            avg_frame_time, int(avg_particle_count), int(avg_draw_calls),
            avg_memory_usage, avg_cpu_usage, avg_gpu_usage
        )


class OptimizedParticleSystem:
    """Optimized particle system with performance features."""
    
    def __init__(self, max_particles: int, optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM):
        self.max_particles = max_particles
        self.optimization_level = optimization_level
        self.gpu_processor = GPUParticleProcessor(max_particles)
        self.instanced_renderer = InstancedParticleRenderer()
        self.lod_system = ParticleLOD([10.0, 50.0, 100.0], [3, 2, 1, 0])
        self.profiler = PerformanceProfiler()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance settings based on optimization level
        self._setup_performance_settings()
    
    def _setup_performance_settings(self):
        """Setup performance settings based on optimization level."""
        if self.optimization_level == OptimizationLevel.LOW:
            self.batch_size = 100
            self.update_frequency = 30  # Hz
            self.use_gpu_processing = False
            self.use_instanced_rendering = False
        
        elif self.optimization_level == OptimizationLevel.MEDIUM:
            self.batch_size = 500
            self.update_frequency = 60  # Hz
            self.use_gpu_processing = True
            self.use_instanced_rendering = True
        
        elif self.optimization_level == OptimizationLevel.HIGH:
            self.batch_size = 1000
            self.update_frequency = 120  # Hz
            self.use_gpu_processing = True
            self.use_instanced_rendering = True
        
        else:  # ULTRA
            self.batch_size = 2000
            self.update_frequency = 240  # Hz
            self.use_gpu_processing = True
            self.use_instanced_rendering = True
    
    def update_system(self, delta_time: float, particle_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Update particle system with optimization."""
        start_time = time.time()
        
        positions, colors, sizes = particle_data
        active_particles = len(positions)
        
        # Apply LOD
        if self.optimization_level != OptimizationLevel.LOW:
            positions, colors, sizes = self._apply_lod(positions, colors, sizes)
        
        # Process particles
        if self.use_gpu_processing:
            self.gpu_processor.process_particles(delta_time, active_particles)
            processed_positions = self.gpu_processor.get_particle_data()
        else:
            processed_positions = positions  # CPU processing would happen here
        
        # Render particles
        if self.use_instanced_rendering:
            self.instanced_renderer.render_instanced(processed_positions, colors, sizes)
        else:
            # Fallback to regular rendering
            pass
        
        # Record metrics
        frame_time = time.time() - start_time
        metrics = PerformanceMetrics(
            frame_time=frame_time,
            particle_count=active_particles,
            draw_calls=1 if self.use_instanced_rendering else active_particles,
            memory_usage=self._estimate_memory_usage(active_particles),
            cpu_usage=self._estimate_cpu_usage(frame_time),
            gpu_usage=self._estimate_gpu_usage(frame_time)
        )
        self.profiler.record_metrics(metrics)
    
    def _apply_lod(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Level of Detail to particles."""
        filtered_positions = []
        filtered_colors = []
        filtered_sizes = []
        
        for i, position in enumerate(positions):
            lod_level = self.lod_system.get_lod_level(position)
            
            # Skip particles based on LOD
            if np.random.random() < (1.0 / (2 ** lod_level)):
                filtered_positions.append(position)
                filtered_colors.append(colors[i])
                filtered_sizes.append(sizes[i] * (1.0 / (2 ** lod_level)))
        
        return (np.array(filtered_positions),
                np.array(filtered_colors),
                np.array(filtered_sizes))
    
    def _estimate_memory_usage(self, particle_count: int) -> float:
        """Estimate memory usage in MB."""
        # 4 floats per position + 4 floats per color + 1 float per size
        bytes_per_particle = (4 + 4 + 1) * 4
        total_bytes = particle_count * bytes_per_particle
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _estimate_cpu_usage(self, frame_time: float) -> float:
        """Estimate CPU usage percentage."""
        target_frame_time = 1.0 / 60.0  # 60 FPS target
        return min(100.0, (frame_time / target_frame_time) * 100.0)
    
    def _estimate_gpu_usage(self, frame_time: float) -> float:
        """Estimate GPU usage percentage."""
        # Simplified GPU usage estimation
        return min(100.0, (frame_time / 0.016) * 50.0)  # Assume 50% GPU usage at 60 FPS
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        avg_metrics = self.profiler.get_average_metrics()
        
        return {
            "optimization_level": self.optimization_level.value,
            "max_particles": self.max_particles,
            "average_frame_time": avg_metrics.frame_time,
            "average_particle_count": avg_metrics.particle_count,
            "average_draw_calls": avg_metrics.draw_calls,
            "average_memory_usage_mb": avg_metrics.memory_usage,
            "average_cpu_usage_percent": avg_metrics.cpu_usage,
            "average_gpu_usage_percent": avg_metrics.gpu_usage,
            "fps": 1.0 / avg_metrics.frame_time if avg_metrics.frame_time > 0 else 0,
            "gpu_processing": self.use_gpu_processing,
            "instanced_rendering": self.use_instanced_rendering
        }
    
    def cleanup(self):
        """Clean up optimized particle system."""
        self.gpu_processor.cleanup()
        self.instanced_renderer.cleanup()
        self.thread_pool.shutdown()


def demonstrate_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print("=== Particle Systems and Visual Effects - Performance Optimization ===\n")

    # Test different optimization levels
    optimization_levels = [OptimizationLevel.LOW, OptimizationLevel.MEDIUM, 
                          OptimizationLevel.HIGH, OptimizationLevel.ULTRA]
    
    for level in optimization_levels:
        print(f"1. Testing {level.value.upper()} optimization level...")
        
        # Create optimized particle system
        system = OptimizedParticleSystem(max_particles=5000, optimization_level=level)
        
        # Generate test particle data
        positions = np.random.uniform(-10, 10, (1000, 3)).astype(np.float32)
        colors = np.random.uniform(0, 1, (1000, 4)).astype(np.float32)
        sizes = np.random.uniform(0.1, 2.0, 1000).astype(np.float32)
        
        # Simulate for a few frames
        delta_time = 0.016
        simulation_frames = 60
        
        system.profiler.start_profiling()
        
        for i in range(simulation_frames):
            system.update_system(delta_time, (positions, colors, sizes))
        
        system.profiler.stop_profiling()
        
        # Get performance report
        report = system.get_performance_report()
        
        print(f"   Average FPS: {report['fps']:.1f}")
        print(f"   Average frame time: {report['average_frame_time']*1000:.2f}ms")
        print(f"   Average particle count: {report['average_particle_count']}")
        print(f"   Average draw calls: {report['average_draw_calls']}")
        print(f"   Memory usage: {report['average_memory_usage_mb']:.2f}MB")
        print(f"   CPU usage: {report['average_cpu_usage_percent']:.1f}%")
        print(f"   GPU usage: {report['average_gpu_usage_percent']:.1f}%")
        print(f"   GPU processing: {report['gpu_processing']}")
        print(f"   Instanced rendering: {report['instanced_rendering']}")
        
        system.cleanup()
        print()

    print("2. Performance optimization features demonstrated:")
    print("   - GPU-based particle processing")
    print("   - Instanced rendering")
    print("   - Level of Detail (LOD) systems")
    print("   - Performance profiling and metrics")
    print("   - Multi-threading support")
    print("   - Memory usage optimization")
    print("   - Configurable optimization levels")
    print("   - Batch processing")
    print("   - Draw call reduction")


if __name__ == "__main__":
    demonstrate_performance_optimization()
