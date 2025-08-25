"""
Chapter 27: Particle Systems and Visual Effects - Particle System
==============================================================

This module demonstrates particle system implementation for dynamic visual effects.

Key Concepts:
- Particle system architecture and management
- Particle lifecycle and behavior
- Emitter systems and particle generation
- Performance optimization for large particle counts
- Integration with rendering systems
"""

import numpy as np
import OpenGL.GL as gl
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
import time


class ParticleState(Enum):
    """Particle state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEAD = "dead"


class EmitterType(Enum):
    """Emitter type enumeration."""
    POINT = "point"
    SPHERE = "sphere"
    BOX = "box"
    LINE = "line"
    PLANE = "plane"


@dataclass
class Particle:
    """Represents a single particle in the system."""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    color: np.ndarray
    size: float
    life: float
    max_life: float
    state: ParticleState = ParticleState.ACTIVE
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0])
        if self.velocity is None:
            self.velocity = np.array([0.0, 0.0, 0.0])
        if self.acceleration is None:
            self.acceleration = np.array([0.0, 0.0, 0.0])
        if self.color is None:
            self.color = np.array([1.0, 1.0, 1.0, 1.0])


@dataclass
class ParticleEmitter:
    """Configures particle emission behavior."""
    emitter_type: EmitterType = EmitterType.POINT
    position: np.ndarray = None
    direction: np.ndarray = None
    spread_angle: float = 45.0  # Degrees
    emission_rate: float = 10.0  # Particles per second
    max_particles: int = 1000
    particle_life: float = 5.0
    particle_size: float = 1.0
    particle_color: np.ndarray = None
    velocity_range: Tuple[float, float] = (1.0, 5.0)
    size_range: Tuple[float, float] = (0.5, 2.0)
    color_variation: float = 0.2
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0])
        if self.direction is None:
            self.direction = np.array([0.0, 1.0, 0.0])
        if self.particle_color is None:
            self.particle_color = np.array([1.0, 1.0, 1.0, 1.0])


class ParticlePool:
    """Manages a pool of reusable particles for performance optimization."""
    
    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.active_particles: List[Particle] = []
        self.inactive_particles: List[Particle] = []
        self.initialize_pool()
    
    def initialize_pool(self):
        """Initialize the particle pool with inactive particles."""
        for _ in range(self.max_particles):
            particle = Particle(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                acceleration=np.array([0.0, 0.0, 0.0]),
                color=np.array([1.0, 1.0, 1.0, 1.0]),
                size=1.0,
                life=0.0,
                max_life=1.0,
                state=ParticleState.INACTIVE
            )
            self.particles.append(particle)
            self.inactive_particles.append(particle)
    
    def get_particle(self) -> Optional[Particle]:
        """Get an inactive particle from the pool."""
        if self.inactive_particles:
            particle = self.inactive_particles.pop()
            particle.state = ParticleState.ACTIVE
            self.active_particles.append(particle)
            return particle
        return None
    
    def return_particle(self, particle: Particle):
        """Return a particle to the inactive pool."""
        if particle in self.active_particles:
            self.active_particles.remove(particle)
            particle.state = ParticleState.INACTIVE
            particle.life = 0.0
            self.inactive_particles.append(particle)
    
    def get_active_count(self) -> int:
        """Get the number of active particles."""
        return len(self.active_particles)
    
    def get_inactive_count(self) -> int:
        """Get the number of inactive particles."""
        return len(self.inactive_particles)


class ParticleEmitter:
    """Handles particle emission and generation."""
    
    def __init__(self, config: ParticleEmitter):
        self.config = config
        self.last_emission_time = 0.0
        self.particles_to_emit = 0.0
    
    def should_emit(self, current_time: float) -> bool:
        """Check if particles should be emitted based on emission rate."""
        time_since_last = current_time - self.last_emission_time
        self.particles_to_emit += self.config.emission_rate * time_since_last
        
        if self.particles_to_emit >= 1.0:
            self.particles_to_emit -= 1.0
            self.last_emission_time = current_time
            return True
        return False
    
    def generate_particle(self, pool: ParticlePool) -> Optional[Particle]:
        """Generate a new particle from the emitter."""
        particle = pool.get_particle()
        if not particle:
            return None
        
        # Set initial position based on emitter type
        particle.position = self._get_emission_position()
        
        # Set initial velocity based on direction and spread
        particle.velocity = self._get_emission_velocity()
        
        # Set particle properties
        particle.acceleration = np.array([0.0, -9.81, 0.0])  # Gravity
        particle.size = random.uniform(*self.config.size_range)
        particle.max_life = self.config.particle_life
        particle.life = particle.max_life
        
        # Set color with variation
        color_variation = random.uniform(-self.config.color_variation, self.config.color_variation)
        particle.color = np.clip(self.config.particle_color + color_variation, 0.0, 1.0)
        
        return particle
    
    def _get_emission_position(self) -> np.ndarray:
        """Get emission position based on emitter type."""
        if self.config.emitter_type == EmitterType.POINT:
            return self.config.position.copy()
        
        elif self.config.emitter_type == EmitterType.SPHERE:
            # Random position on sphere surface
            radius = 1.0
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            return self.config.position + np.array([x, y, z])
        
        elif self.config.emitter_type == EmitterType.BOX:
            # Random position inside box
            size = 2.0
            x = random.uniform(-size/2, size/2)
            y = random.uniform(-size/2, size/2)
            z = random.uniform(-size/2, size/2)
            return self.config.position + np.array([x, y, z])
        
        elif self.config.emitter_type == EmitterType.LINE:
            # Random position along line
            length = 5.0
            t = random.uniform(-length/2, length/2)
            return self.config.position + self.config.direction * t
        
        else:  # PLANE
            # Random position on plane
            size = 5.0
            x = random.uniform(-size/2, size/2)
            z = random.uniform(-size/2, size/2)
            return self.config.position + np.array([x, 0.0, z])
    
    def _get_emission_velocity(self) -> np.ndarray:
        """Get emission velocity based on direction and spread."""
        # Base direction
        base_direction = self.config.direction / np.linalg.norm(self.config.direction)
        
        # Add random spread
        spread_rad = np.radians(self.config.spread_angle)
        
        # Generate random direction within cone
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, spread_rad)
        
        # Create random direction
        random_direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.cos(phi),
            np.sin(phi) * np.sin(theta)
        ])
        
        # Combine with base direction
        final_direction = base_direction + random_direction
        final_direction = final_direction / np.linalg.norm(final_direction)
        
        # Apply velocity magnitude
        velocity_magnitude = random.uniform(*self.config.velocity_range)
        return final_direction * velocity_magnitude


class ParticleSystem:
    """Main particle system that manages emitters and particles."""
    
    def __init__(self, max_particles: int = 10000):
        self.pool = ParticlePool(max_particles)
        self.emitters: List[ParticleEmitter] = []
        self.forces: List[Callable[[Particle, float], np.ndarray]] = []
        self.current_time = 0.0
        
        # Add default forces
        self.add_force(self._gravity_force)
        self.add_force(self._drag_force)
    
    def add_emitter(self, emitter: ParticleEmitter):
        """Add a particle emitter to the system."""
        self.emitters.append(emitter)
    
    def add_force(self, force_function: Callable[[Particle, float], np.ndarray]):
        """Add a force function to the system."""
        self.forces.append(force_function)
    
    def update(self, delta_time: float):
        """Update the particle system."""
        self.current_time += delta_time
        
        # Emit new particles
        self._emit_particles()
        
        # Update existing particles
        self._update_particles(delta_time)
        
        # Clean up dead particles
        self._cleanup_particles()
    
    def _emit_particles(self):
        """Emit new particles from all emitters."""
        for emitter in self.emitters:
            if emitter.should_emit(self.current_time):
                particle = emitter.generate_particle(self.pool)
                if particle:
                    # Check if we've reached max particles
                    if self.pool.get_active_count() >= emitter.config.max_particles:
                        break
    
    def _update_particles(self, delta_time: float):
        """Update all active particles."""
        for particle in self.pool.active_particles[:]:  # Copy list to avoid modification during iteration
            if particle.state == ParticleState.ACTIVE:
                # Update life
                particle.life -= delta_time
                
                if particle.life <= 0:
                    particle.state = ParticleState.DEAD
                    continue
                
                # Apply forces
                total_force = np.zeros(3)
                for force_func in self.forces:
                    total_force += force_func(particle, delta_time)
                
                # Update acceleration
                particle.acceleration = total_force
                
                # Update velocity
                particle.velocity += particle.acceleration * delta_time
                
                # Update position
                particle.position += particle.velocity * delta_time
                
                # Update color based on life
                life_ratio = particle.life / particle.max_life
                particle.color[3] = life_ratio  # Alpha fade
    
    def _cleanup_particles(self):
        """Remove dead particles from the system."""
        for particle in self.pool.active_particles[:]:
            if particle.state == ParticleState.DEAD:
                self.pool.return_particle(particle)
    
    def _gravity_force(self, particle: Particle, delta_time: float) -> np.ndarray:
        """Apply gravity force to particle."""
        return np.array([0.0, -9.81, 0.0])
    
    def _drag_force(self, particle: Particle, delta_time: float) -> np.ndarray:
        """Apply drag force to particle."""
        drag_coefficient = 0.1
        velocity_magnitude = np.linalg.norm(particle.velocity)
        if velocity_magnitude > 0:
            drag_direction = -particle.velocity / velocity_magnitude
            return drag_direction * drag_coefficient * velocity_magnitude * velocity_magnitude
        return np.zeros(3)
    
    def get_particle_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get particle data for rendering."""
        positions = []
        colors = []
        sizes = []
        
        for particle in self.pool.active_particles:
            if particle.state == ParticleState.ACTIVE:
                positions.append(particle.position)
                colors.append(particle.color)
                sizes.append(particle.size)
        
        if positions:
            return (np.array(positions, dtype=np.float32),
                   np.array(colors, dtype=np.float32),
                   np.array(sizes, dtype=np.float32))
        else:
            return (np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32),
                   np.array([], dtype=np.float32))


def demonstrate_particle_system():
    """Demonstrate particle system functionality."""
    print("=== Particle Systems and Visual Effects - Particle System ===\n")

    # Create particle system
    system = ParticleSystem(max_particles=1000)
    
    print("1. Creating particle emitters...")
    
    # Create different types of emitters
    point_emitter = ParticleEmitter(
        emitter_type=EmitterType.POINT,
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        emission_rate=20.0,
        particle_life=3.0,
        particle_color=np.array([1.0, 0.5, 0.0, 1.0])  # Orange
    )
    system.add_emitter(point_emitter)
    print("   Added point emitter")
    
    sphere_emitter = ParticleEmitter(
        emitter_type=EmitterType.SPHERE,
        position=np.array([5.0, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        emission_rate=15.0,
        particle_life=4.0,
        particle_color=np.array([0.0, 0.5, 1.0, 1.0])  # Blue
    )
    system.add_emitter(sphere_emitter)
    print("   Added sphere emitter")
    
    box_emitter = ParticleEmitter(
        emitter_type=EmitterType.BOX,
        position=np.array([-5.0, 0.0, 0.0]),
        direction=np.array([0.0, 1.0, 0.0]),
        emission_rate=10.0,
        particle_life=5.0,
        particle_color=np.array([0.5, 1.0, 0.0, 1.0])  # Green
    )
    system.add_emitter(box_emitter)
    print("   Added box emitter")

    print("\n2. Simulating particle system...")
    
    # Simulate for a few seconds
    delta_time = 0.016  # 60 FPS
    simulation_time = 2.0
    steps = int(simulation_time / delta_time)
    
    for i in range(steps):
        system.update(delta_time)
        
        if i % 60 == 0:  # Print every second
            active_count = system.pool.get_active_count()
            inactive_count = system.pool.get_inactive_count()
            print(f"   Time {i * delta_time:.1f}s: {active_count} active, {inactive_count} inactive particles")

    print("\n3. Particle data for rendering...")
    
    positions, colors, sizes = system.get_particle_data()
    print(f"   Particle positions shape: {positions.shape}")
    print(f"   Particle colors shape: {colors.shape}")
    print(f"   Particle sizes shape: {sizes.shape}")
    
    if len(positions) > 0:
        print(f"   Sample particle position: {positions[0]}")
        print(f"   Sample particle color: {colors[0]}")
        print(f"   Sample particle size: {sizes[0]}")

    print("\n4. Performance metrics...")
    
    print(f"   Total particles in pool: {len(system.pool.particles)}")
    print(f"   Active particles: {system.pool.get_active_count()}")
    print(f"   Inactive particles: {system.pool.get_inactive_count()}")
    print(f"   Emitters: {len(system.emitters)}")
    print(f"   Forces: {len(system.forces)}")

    print("\n5. Features demonstrated:")
    print("   - Particle pool management")
    print("   - Multiple emitter types (point, sphere, box)")
    print("   - Particle lifecycle management")
    print("   - Force system (gravity, drag)")
    print("   - Performance optimization")
    print("   - Particle data extraction for rendering")


if __name__ == "__main__":
    demonstrate_particle_system()
