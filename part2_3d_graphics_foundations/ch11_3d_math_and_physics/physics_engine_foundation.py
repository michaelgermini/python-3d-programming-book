#!/usr/bin/env python3
"""
Chapter 11: 3D Math and Physics
Physics Engine Foundation

Demonstrates force accumulation, motion integration, constraint systems,
and rigid body dynamics for 3D physics simulation.
"""

import math
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Physics Engine Foundation"
__description__ = "Basic physics engine with forces, motion, and constraints"

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Vector3D:
    """3D vector class for representing positions and directions"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self * scalar
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product of two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product of two vectors"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def magnitude_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

@dataclass
class Quaternion:
    """Quaternion class for representing 3D rotations"""
    w: float  # Real part
    x: float  # i component
    y: float  # j component
    z: float  # k component
    
    def __post_init__(self):
        """Normalize the quaternion after initialization"""
        self.normalize()
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (Hamilton product)"""
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )
    
    def conjugate(self) -> 'Quaternion':
        """Quaternion conjugate"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def inverse(self) -> 'Quaternion':
        """Quaternion inverse"""
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq == 0:
            raise ValueError("Cannot invert zero quaternion")
        return self.conjugate() * (1.0 / norm_sq)
    
    def magnitude(self) -> float:
        """Quaternion magnitude"""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Normalize the quaternion"""
        mag = self.magnitude()
        if mag > 0:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    def normalized(self) -> 'Quaternion':
        """Return a normalized copy of the quaternion"""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(
            self.w / mag,
            self.x / mag,
            self.y / mag,
            self.z / mag
        )
    
    def __str__(self):
        return f"Quaternion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

# ============================================================================
# PHYSICS COMPONENTS
# ============================================================================

class ForceType(Enum):
    """Types of forces"""
    GRAVITY = "gravity"
    SPRING = "spring"
    DAMPING = "damping"
    DRAG = "drag"
    BUOYANCY = "buoyancy"
    CUSTOM = "custom"

class Force:
    """Base force class"""
    
    def __init__(self, force_type: ForceType, name: str = ""):
        self.force_type = force_type
        self.name = name or force_type.value
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate force vector (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.force_type.value.capitalize()}Force('{self.name}')"

class GravityForce(Force):
    """Gravitational force"""
    
    def __init__(self, gravity: Vector3D = Vector3D(0, -9.81, 0)):
        super().__init__(ForceType.GRAVITY, "gravity")
        self.gravity = gravity
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate gravitational force"""
        return self.gravity * body.mass

class SpringForce(Force):
    """Spring force between two bodies"""
    
    def __init__(self, body1: 'RigidBody', body2: 'RigidBody', 
                 rest_length: float, spring_constant: float, damping: float = 0.0):
        super().__init__(ForceType.SPRING, "spring")
        self.body1 = body1
        self.body2 = body2
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        self.damping = damping
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate spring force"""
        # Calculate displacement vector
        displacement = body.position - self.body2.position
        distance = displacement.magnitude()
        
        if distance == 0:
            return Vector3D(0, 0, 0)
        
        # Calculate spring force
        spring_force = displacement.normalize() * (distance - self.rest_length) * self.spring_constant
        
        # Calculate damping force
        relative_velocity = body.velocity - self.body2.velocity
        damping_force = relative_velocity * self.damping
        
        return spring_force + damping_force

class DragForce(Force):
    """Air resistance/drag force"""
    
    def __init__(self, drag_coefficient: float = 0.1):
        super().__init__(ForceType.DRAG, "drag")
        self.drag_coefficient = drag_coefficient
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate drag force"""
        velocity_squared = body.velocity.magnitude_squared()
        if velocity_squared == 0:
            return Vector3D(0, 0, 0)
        
        # Drag force is opposite to velocity direction
        drag_force = body.velocity.normalize() * (-velocity_squared * self.drag_coefficient)
        return drag_force

class BuoyancyForce(Force):
    """Buoyancy force in fluid"""
    
    def __init__(self, fluid_density: float = 1000.0, fluid_level: float = 0.0):
        super().__init__(ForceType.BUOYANCY, "buoyancy")
        self.fluid_density = fluid_density
        self.fluid_level = fluid_level
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate buoyancy force"""
        # Simple buoyancy: if body is below fluid level, apply upward force
        if body.position.y < self.fluid_level:
            # Calculate submerged volume (simplified)
            submerged_volume = body.mass / self.fluid_density
            buoyancy_force = Vector3D(0, 9.81 * self.fluid_density * submerged_volume, 0)
            return buoyancy_force
        return Vector3D(0, 0, 0)

class CustomForce(Force):
    """Custom force defined by a function"""
    
    def __init__(self, force_function: Callable[['RigidBody', float], Vector3D], name: str = "custom"):
        super().__init__(ForceType.CUSTOM, name)
        self.force_function = force_function
    
    def calculate(self, body: 'RigidBody', dt: float) -> Vector3D:
        """Calculate custom force"""
        return self.force_function(body, dt)

# ============================================================================
# RIGID BODY
# ============================================================================

class RigidBody:
    """Rigid body for physics simulation"""
    
    def __init__(self, name: str = "body"):
        self.name = name
        
        # Physical properties
        self.mass = 1.0
        self.inverse_mass = 1.0 / self.mass
        
        # Position and orientation
        self.position = Vector3D(0, 0, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        
        # Linear and angular motion
        self.velocity = Vector3D(0, 0, 0)
        self.angular_velocity = Vector3D(0, 0, 0)
        
        # Forces and torques
        self.force = Vector3D(0, 0, 0)
        self.torque = Vector3D(0, 0, 0)
        
        # Inertia (simplified as scalar)
        self.inertia = 1.0
        self.inverse_inertia = 1.0 / self.inertia
        
        # State flags
        self.is_static = False
        self.is_awake = True
        
        # Collision properties
        self.bounding_radius = 1.0
        
    def set_mass(self, mass: float):
        """Set the mass of the body"""
        self.mass = mass
        self.inverse_mass = 1.0 / mass if mass > 0 else 0.0
    
    def set_static(self, is_static: bool):
        """Set whether the body is static"""
        self.is_static = is_static
        if is_static:
            self.inverse_mass = 0.0
            self.inverse_inertia = 0.0
    
    def apply_force(self, force: Vector3D, point: Vector3D = None):
        """Apply a force to the body"""
        if self.is_static:
            return
        
        self.force = self.force + force
        
        # Apply torque if force is applied at a specific point
        if point is not None:
            r = point - self.position
            torque = r.cross(force)
            self.torque = self.torque + torque
    
    def apply_impulse(self, impulse: Vector3D, point: Vector3D = None):
        """Apply an impulse to the body"""
        if self.is_static:
            return
        
        self.velocity = self.velocity + impulse * self.inverse_mass
        
        # Apply angular impulse if impulse is applied at a specific point
        if point is not None:
            r = point - self.position
            angular_impulse = r.cross(impulse)
            self.angular_velocity = self.angular_velocity + angular_impulse * self.inverse_inertia
    
    def clear_forces(self):
        """Clear accumulated forces and torques"""
        self.force = Vector3D(0, 0, 0)
        self.torque = Vector3D(0, 0, 0)
    
    def integrate(self, dt: float):
        """Integrate motion over time step"""
        if self.is_static:
            return
        
        # Linear motion integration (Euler method)
        acceleration = self.force * self.inverse_mass
        self.velocity = self.velocity + acceleration * dt
        self.position = self.position + self.velocity * dt
        
        # Angular motion integration (simplified)
        angular_acceleration = self.torque * self.inverse_inertia
        self.angular_velocity = self.angular_velocity + angular_acceleration * dt
        
        # Update orientation (simplified quaternion integration)
        angular_quat = Quaternion(0, self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z)
        orientation_dot = angular_quat * self.orientation * 0.5 * dt
        self.orientation.w += orientation_dot.w
        self.orientation.x += orientation_dot.x
        self.orientation.y += orientation_dot.y
        self.orientation.z += orientation_dot.z
        self.orientation.normalize()
    
    def get_kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        linear_ke = 0.5 * self.mass * self.velocity.magnitude_squared()
        angular_ke = 0.5 * self.inertia * self.angular_velocity.magnitude_squared()
        return linear_ke + angular_ke
    
    def get_momentum(self) -> Vector3D:
        """Calculate linear momentum"""
        return self.velocity * self.mass
    
    def __str__(self):
        return f"RigidBody('{self.name}', pos={self.position}, vel={self.velocity}, mass={self.mass})"

# ============================================================================
# CONSTRAINT SYSTEM
# ============================================================================

class ConstraintType(Enum):
    """Types of constraints"""
    DISTANCE = "distance"
    HINGE = "hinge"
    SLIDER = "slider"
    FIXED = "fixed"

class Constraint:
    """Base constraint class"""
    
    def __init__(self, constraint_type: ConstraintType, body1: RigidBody, body2: RigidBody):
        self.constraint_type = constraint_type
        self.body1 = body1
        self.body2 = body2
        self.enabled = True
    
    def solve(self, dt: float):
        """Solve the constraint (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def __str__(self):
        return f"{self.constraint_type.value.capitalize()}Constraint({self.body1.name}, {self.body2.name})"

class DistanceConstraint(Constraint):
    """Distance constraint between two bodies"""
    
    def __init__(self, body1: RigidBody, body2: RigidBody, distance: float):
        super().__init__(ConstraintType.DISTANCE, body1, body2)
        self.distance = distance
    
    def solve(self, dt: float):
        """Solve distance constraint using position correction"""
        if not self.enabled or (self.body1.is_static and self.body2.is_static):
            return
        
        # Calculate current distance
        displacement = self.body2.position - self.body1.position
        current_distance = displacement.magnitude()
        
        if current_distance == 0:
            return
        
        # Calculate correction
        correction = displacement.normalize() * (current_distance - self.distance)
        
        # Apply correction based on inverse masses
        total_inverse_mass = self.body1.inverse_mass + self.body2.inverse_mass
        if total_inverse_mass == 0:
            return
        
        correction1 = correction * (self.body1.inverse_mass / total_inverse_mass)
        correction2 = correction * (self.body2.inverse_mass / total_inverse_mass)
        
        # Apply corrections
        if not self.body1.is_static:
            self.body1.position = self.body1.position - correction1
        if not self.body2.is_static:
            self.body2.position = self.body2.position + correction2

class FixedConstraint(Constraint):
    """Fixed constraint (bodies maintain relative position)"""
    
    def __init__(self, body1: RigidBody, body2: RigidBody, offset: Vector3D):
        super().__init__(ConstraintType.FIXED, body1, body2)
        self.offset = offset
    
    def solve(self, dt: float):
        """Solve fixed constraint"""
        if not self.enabled or (self.body1.is_static and self.body2.is_static):
            return
        
        # Calculate desired position for body2
        desired_position = self.body1.position + self.offset
        
        # Apply correction
        if not self.body2.is_static:
            self.body2.position = desired_position

# ============================================================================
# PHYSICS WORLD
# ============================================================================

class PhysicsWorld:
    """Physics world for managing bodies, forces, and constraints"""
    
    def __init__(self):
        self.bodies: List[RigidBody] = []
        self.forces: List[Force] = []
        self.constraints: List[Constraint] = []
        self.gravity = Vector3D(0, -9.81, 0)
        self.time_step = 1.0 / 60.0
        self.iterations = 10
        
        # Add default gravity force
        self.add_force(GravityForce(self.gravity))
    
    def add_body(self, body: RigidBody):
        """Add a body to the physics world"""
        self.bodies.append(body)
    
    def remove_body(self, body: RigidBody):
        """Remove a body from the physics world"""
        if body in self.bodies:
            self.bodies.remove(body)
    
    def add_force(self, force: Force):
        """Add a force to the physics world"""
        self.forces.append(force)
    
    def remove_force(self, force: Force):
        """Remove a force from the physics world"""
        if force in self.forces:
            self.forces.remove(force)
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the physics world"""
        self.constraints.append(constraint)
    
    def remove_constraint(self, constraint: Constraint):
        """Remove a constraint from the physics world"""
        if constraint in self.constraints:
            self.constraints.remove(constraint)
    
    def step(self, dt: float = None):
        """Step the physics simulation"""
        if dt is None:
            dt = self.time_step
        
        # Apply forces
        self.apply_forces(dt)
        
        # Integrate motion
        self.integrate_motion(dt)
        
        # Solve constraints
        self.solve_constraints(dt)
        
        # Clear forces
        self.clear_forces()
    
    def apply_forces(self, dt: float):
        """Apply all forces to bodies"""
        for body in self.bodies:
            if body.is_static:
                continue
            
            # Apply all forces
            for force in self.forces:
                force_vector = force.calculate(body, dt)
                body.apply_force(force_vector)
    
    def integrate_motion(self, dt: float):
        """Integrate motion for all bodies"""
        for body in self.bodies:
            body.integrate(dt)
    
    def solve_constraints(self, dt: float):
        """Solve all constraints"""
        for _ in range(self.iterations):
            for constraint in self.constraints:
                constraint.solve(dt)
    
    def clear_forces(self):
        """Clear forces from all bodies"""
        for body in self.bodies:
            body.clear_forces()
    
    def get_total_energy(self) -> float:
        """Calculate total energy in the system"""
        total_energy = 0.0
        for body in self.bodies:
            total_energy += body.get_kinetic_energy()
        return total_energy
    
    def get_total_momentum(self) -> Vector3D:
        """Calculate total momentum in the system"""
        total_momentum = Vector3D(0, 0, 0)
        for body in self.bodies:
            total_momentum = total_momentum + body.get_momentum()
        return total_momentum

# ============================================================================
# NUMERICAL INTEGRATION
# ============================================================================

class Integrator:
    """Numerical integration methods"""
    
    @staticmethod
    def euler_integrate(body: RigidBody, dt: float):
        """Euler integration method"""
        if body.is_static:
            return
        
        # Linear motion
        acceleration = body.force * body.inverse_mass
        body.velocity = body.velocity + acceleration * dt
        body.position = body.position + body.velocity * dt
        
        # Angular motion (simplified)
        angular_acceleration = body.torque * body.inverse_inertia
        body.angular_velocity = body.angular_velocity + angular_acceleration * dt
    
    @staticmethod
    def verlet_integrate(body: RigidBody, dt: float, prev_position: Vector3D):
        """Verlet integration method"""
        if body.is_static:
            return prev_position
        
        # Verlet integration for position
        acceleration = body.force * body.inverse_mass
        new_position = body.position * 2 - prev_position + acceleration * dt * dt
        
        # Update velocity
        body.velocity = (new_position - body.position) / dt
        
        # Update position
        body.position = new_position
        
        return body.position
    
    @staticmethod
    def rk4_integrate(body: RigidBody, dt: float):
        """Fourth-order Runge-Kutta integration method"""
        if body.is_static:
            return
        
        # RK4 integration for position
        k1 = body.velocity
        k2 = body.velocity + (body.force * body.inverse_mass) * dt * 0.5
        k3 = body.velocity + (body.force * body.inverse_mass) * dt * 0.5
        k4 = body.velocity + (body.force * body.inverse_mass) * dt
        
        body.position = body.position + (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6.0
        body.velocity = body.velocity + (body.force * body.inverse_mass) * dt

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_physics():
    """Demonstrate basic physics concepts"""
    print("=== Basic Physics Concepts ===\n")
    
    # Create a physics world
    world = PhysicsWorld()
    
    # Create bodies
    body1 = RigidBody("ball1")
    body1.set_mass(1.0)
    body1.position = Vector3D(0, 10, 0)
    
    body2 = RigidBody("ball2")
    body2.set_mass(2.0)
    body2.position = Vector3D(5, 10, 0)
    
    # Add bodies to world
    world.add_body(body1)
    world.add_body(body2)
    
    print("Initial state:")
    print(f"  {body1}")
    print(f"  {body2}")
    
    # Simulate for a few steps
    print("\nSimulation steps:")
    for i in range(5):
        world.step(0.1)
        print(f"  Step {i+1}:")
        print(f"    {body1.name}: pos={body1.position}, vel={body1.velocity}")
        print(f"    {body2.name}: pos={body2.position}, vel={body2.velocity}")
    
    print()

def demonstrate_forces():
    """Demonstrate different types of forces"""
    print("=== Force Demonstration ===\n")
    
    # Create physics world
    world = PhysicsWorld()
    
    # Create a body
    body = RigidBody("test_body")
    body.set_mass(1.0)
    body.position = Vector3D(0, 5, 0)
    
    world.add_body(body)
    
    # Test different forces
    forces = [
        GravityForce(Vector3D(0, -9.81, 0)),
        DragForce(0.1),
        BuoyancyForce(1000.0, 0.0)
    ]
    
    print("Testing different forces:")
    for force in forces:
        world.remove_force(world.forces[0])  # Remove gravity
        world.add_force(force)
        
        print(f"\n{force}:")
        for i in range(3):
            world.step(0.1)
            print(f"  Step {i+1}: pos={body.position}, vel={body.velocity}")
        
        # Reset body
        body.position = Vector3D(0, 5, 0)
        body.velocity = Vector3D(0, 0, 0)
    
    # Restore gravity
    world.add_force(GravityForce(Vector3D(0, -9.81, 0)))
    
    print()

def demonstrate_spring_system():
    """Demonstrate spring force system"""
    print("=== Spring System ===\n")
    
    # Create physics world
    world = PhysicsWorld()
    
    # Create two bodies connected by a spring
    body1 = RigidBody("mass1")
    body1.set_mass(1.0)
    body1.position = Vector3D(-2, 0, 0)
    
    body2 = RigidBody("mass2")
    body2.set_mass(1.0)
    body2.position = Vector3D(2, 0, 0)
    
    # Create spring force
    spring = SpringForce(body1, body2, rest_length=4.0, spring_constant=10.0, damping=0.5)
    
    # Add to world
    world.add_body(body1)
    world.add_body(body2)
    world.add_force(spring)
    
    print("Spring system simulation:")
    for i in range(20):
        world.step(0.05)
        if i % 5 == 0:
            distance = (body2.position - body1.position).magnitude()
            print(f"  Step {i}: distance={distance:.3f}, pos1={body1.position}, pos2={body2.position}")
    
    print()

def demonstrate_constraints():
    """Demonstrate constraint system"""
    print("=== Constraint System ===\n")
    
    # Create physics world
    world = PhysicsWorld()
    
    # Create bodies
    body1 = RigidBody("anchor")
    body1.set_static(True)
    body1.position = Vector3D(0, 5, 0)
    
    body2 = RigidBody("pendulum")
    body2.set_mass(1.0)
    body2.position = Vector3D(0, 0, 0)
    
    # Create distance constraint
    constraint = DistanceConstraint(body1, body2, distance=5.0)
    
    # Add to world
    world.add_body(body1)
    world.add_body(body2)
    world.add_constraint(constraint)
    
    print("Pendulum simulation:")
    for i in range(15):
        world.step(0.1)
        if i % 3 == 0:
            distance = (body2.position - body1.position).magnitude()
            print(f"  Step {i}: distance={distance:.3f}, pos={body2.position}")
    
    print()

def demonstrate_integration_methods():
    """Demonstrate different integration methods"""
    print("=== Integration Methods ===\n")
    
    # Create bodies for different integration methods
    bodies = {
        "Euler": RigidBody("euler"),
        "Verlet": RigidBody("verlet"),
        "RK4": RigidBody("rk4")
    }
    
    # Initialize bodies
    for body in bodies.values():
        body.set_mass(1.0)
        body.position = Vector3D(0, 10, 0)
        body.velocity = Vector3D(5, 0, 0)
    
    # Apply gravity force
    gravity = Vector3D(0, -9.81, 0)
    
    print("Integration method comparison:")
    dt = 0.1
    
    for step in range(10):
        # Euler integration
        bodies["Euler"].apply_force(gravity * bodies["Euler"].mass)
        Integrator.euler_integrate(bodies["Euler"], dt)
        bodies["Euler"].clear_forces()
        
        # Verlet integration (simplified)
        bodies["Verlet"].apply_force(gravity * bodies["Verlet"].mass)
        if step == 0:
            prev_pos = bodies["Verlet"].position
        prev_pos = Integrator.verlet_integrate(bodies["Verlet"], dt, prev_pos)
        bodies["Verlet"].clear_forces()
        
        # RK4 integration
        bodies["RK4"].apply_force(gravity * bodies["RK4"].mass)
        Integrator.rk4_integrate(bodies["RK4"], dt)
        bodies["RK4"].clear_forces()
        
        if step % 3 == 0:
            print(f"  Step {step}:")
            for name, body in bodies.items():
                print(f"    {name}: pos={body.position}, vel={body.velocity}")
    
    print()

def demonstrate_energy_conservation():
    """Demonstrate energy conservation"""
    print("=== Energy Conservation ===\n")
    
    # Create physics world
    world = PhysicsWorld()
    
    # Create a bouncing ball
    ball = RigidBody("bouncing_ball")
    ball.set_mass(1.0)
    ball.position = Vector3D(0, 10, 0)
    ball.velocity = Vector3D(5, 0, 0)
    
    world.add_body(ball)
    
    print("Energy conservation test:")
    initial_energy = ball.get_kinetic_energy()
    print(f"  Initial kinetic energy: {initial_energy:.3f}")
    
    for i in range(20):
        world.step(0.1)
        if i % 5 == 0:
            kinetic_energy = ball.get_kinetic_energy()
            energy_loss = (initial_energy - kinetic_energy) / initial_energy * 100
            print(f"  Step {i}: KE={kinetic_energy:.3f}, Loss={energy_loss:.1f}%")
    
    print()

def demonstrate_performance_test():
    """Demonstrate performance testing"""
    print("=== Performance Test ===\n")
    
    import time
    
    # Create physics world with many bodies
    world = PhysicsWorld()
    
    # Create multiple bodies
    num_bodies = 100
    for i in range(num_bodies):
        body = RigidBody(f"body_{i}")
        body.set_mass(1.0)
        body.position = Vector3D(
            random.uniform(-10, 10),
            random.uniform(0, 20),
            random.uniform(-10, 10)
        )
        world.add_body(body)
    
    print(f"Testing performance with {num_bodies} bodies:")
    
    # Test simulation performance
    start_time = time.time()
    num_steps = 100
    
    for i in range(num_steps):
        world.step(0.016)  # 60 FPS
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    steps_per_second = num_steps / total_time
    
    print(f"  Total simulation time: {total_time:.3f} seconds")
    print(f"  Average time per step: {avg_time_per_step*1000:.2f} ms")
    print(f"  Steps per second: {steps_per_second:.1f}")
    print(f"  Bodies per second: {steps_per_second * num_bodies:.0f}")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate physics engine foundation"""
    print("=== Physics Engine Foundation Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate basic physics
    demonstrate_basic_physics()
    
    # Demonstrate forces
    demonstrate_forces()
    
    # Demonstrate spring system
    demonstrate_spring_system()
    
    # Demonstrate constraints
    demonstrate_constraints()
    
    # Demonstrate integration methods
    demonstrate_integration_methods()
    
    # Demonstrate energy conservation
    demonstrate_energy_conservation()
    
    # Demonstrate performance test
    demonstrate_performance_test()
    
    print("="*60)
    print("Physics Engine Foundation demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Force accumulation and application")
    print("✓ Motion integration (Euler, Verlet, RK4)")
    print("✓ Constraint systems and solving")
    print("✓ Rigid body dynamics")
    print("✓ Energy conservation and performance")
    print("✓ Physics world management")
    
    print("\nPhysics components covered:")
    print("• Forces: Gravity, spring, drag, buoyancy, custom forces")
    print("• Bodies: Rigid bodies with mass, position, velocity, orientation")
    print("• Constraints: Distance, fixed constraints with position correction")
    print("• Integration: Multiple numerical integration methods")
    print("• World: Physics world with body, force, and constraint management")
    
    print("\nApplications:")
    print("• Game physics: Realistic object movement and interaction")
    print("• Simulation: Scientific and engineering simulations")
    print("• Animation: Physics-based character and object animation")
    print("• Virtual reality: Interactive physics in VR environments")
    print("• Robotics: Motion planning and collision avoidance")
    
    print("\nNext steps:")
    print("• Implement collision detection and response")
    print("• Add advanced constraint types (hinge, slider, etc.)")
    print("• Optimize performance with spatial partitioning")
    print("• Add soft body physics and fluid simulation")
    print("• Integrate with rendering systems for visualization")

if __name__ == "__main__":
    main()
