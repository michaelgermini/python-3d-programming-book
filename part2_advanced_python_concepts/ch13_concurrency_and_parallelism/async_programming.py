"""
Chapter 13: Concurrency and Parallelism - Asynchronous Programming
=================================================================

This module demonstrates asynchronous programming patterns for 3D graphics
applications, including async/await, event loops, and non-blocking operations.

Key Concepts:
- Async/await syntax and patterns
- Event loops and coroutines
- Non-blocking I/O operations
- Async rendering and computation
- Performance optimization with async programming
"""

import asyncio
import time
import random
import aiofiles
import json
from typing import List, Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
import math


class AsyncTaskStatus(Enum):
    """Status of an async task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AsyncTaskInfo:
    """Information about an async task."""
    name: str
    task: asyncio.Task
    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    
    def get_duration(self) -> float:
        """Get task execution duration."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class AsyncRenderer:
    """Asynchronous 3D renderer."""
    
    def __init__(self):
        self.rendering_tasks: Dict[str, AsyncTaskInfo] = {}
        self.render_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.results_queue: asyncio.Queue = asyncio.Queue()
        self.rendering = False
        self.worker_tasks: List[asyncio.Task] = []
    
    async def start_rendering(self, num_workers: int = 3):
        """Start the async rendering system."""
        if self.rendering:
            return
        
        self.rendering = True
        print(f"  🚀 Starting async renderer with {num_workers} workers")
        
        # Start worker tasks
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._render_worker(f"Worker-{i+1}"))
            self.worker_tasks.append(worker_task)
    
    async def stop_rendering(self):
        """Stop the async rendering system."""
        if not self.rendering:
            return
        
        print("  🛑 Stopping async renderer...")
        self.rendering = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        print("  ✅ Async renderer stopped")
    
    async def render_scene_async(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render a scene asynchronously."""
        if not self.rendering:
            await self.start_rendering()
        
        print(f"  🎨 Starting async rendering of scene with {len(scene_data.get('objects', []))} objects")
        
        # Create rendering tasks for each object
        objects = scene_data.get('objects', [])
        render_tasks = []
        
        for obj in objects:
            task_name = f"render_{obj.get('name', 'object')}"
            task = asyncio.create_task(self._render_object_async(obj))
            
            task_info = AsyncTaskInfo(name=task_name, task=task)
            self.rendering_tasks[task_name] = task_info
            
            render_tasks.append(task)
        
        # Wait for all rendering tasks to complete
        start_time = time.time()
        results = await asyncio.gather(*render_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ Rendering task failed: {result}")
            else:
                valid_results.append(result)
        
        print(f"  ✅ Async rendering completed in {end_time - start_time:.3f}s")
        print(f"  📊 Rendered {len(valid_results)} objects successfully")
        
        return valid_results
    
    async def _render_object_async(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single object asynchronously."""
        obj_name = obj.get('name', 'unknown')
        
        # Simulate async rendering time
        render_time = random.uniform(0.01, 0.1)
        await asyncio.sleep(render_time)
        
        # Simulate rendering result
        result = {
            "object_name": obj_name,
            "render_time": render_time,
            "vertices_rendered": obj.get('vertex_count', 100),
            "textures_loaded": obj.get('texture_count', 1),
            "pixels_processed": obj.get('pixel_count', 1000)
        }
        
        print(f"  🎨 Rendered {obj_name} in {render_time:.3f}s")
        return result
    
    async def _render_worker(self, worker_name: str):
        """Worker coroutine for processing render tasks."""
        print(f"  👷 {worker_name} started")
        
        while self.rendering:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.render_queue.get(), timeout=0.1)
                
                # Process task
                result = await self._process_render_task(task)
                
                # Put result in results queue
                await self.results_queue.put(result)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  ❌ {worker_name} encountered error: {e}")
        
        print(f"  👷 {worker_name} stopped")
    
    async def _process_render_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a render task."""
        # Simulate task processing
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return {"task": task, "processed": True}


class AsyncResourceLoader:
    """Asynchronous resource loader for 3D graphics."""
    
    def __init__(self):
        self.loading_tasks: Dict[str, AsyncTaskInfo] = {}
        self.cache: Dict[str, Any] = {}
    
    async def load_texture_async(self, texture_path: str) -> Dict[str, Any]:
        """Load a texture asynchronously."""
        if texture_path in self.cache:
            print(f"  📦 Texture {texture_path} loaded from cache")
            return self.cache[texture_path]
        
        task_name = f"load_texture_{texture_path}"
        task = asyncio.create_task(self._load_texture_file(texture_path))
        
        task_info = AsyncTaskInfo(name=task_name, task=task)
        self.loading_tasks[task_name] = task_info
        
        try:
            result = await task
            self.cache[texture_path] = result
            return result
        except Exception as e:
            print(f"  ❌ Failed to load texture {texture_path}: {e}")
            raise
    
    async def load_mesh_async(self, mesh_path: str) -> Dict[str, Any]:
        """Load a mesh asynchronously."""
        if mesh_path in self.cache:
            print(f"  📦 Mesh {mesh_path} loaded from cache")
            return self.cache[mesh_path]
        
        task_name = f"load_mesh_{mesh_path}"
        task = asyncio.create_task(self._load_mesh_file(mesh_path))
        
        task_info = AsyncTaskInfo(name=task_name, task=task)
        self.loading_tasks[task_name] = task_info
        
        try:
            result = await task
            self.cache[mesh_path] = result
            return result
        except Exception as e:
            print(f"  ❌ Failed to load mesh {mesh_path}: {e}")
            raise
    
    async def load_shader_async(self, shader_path: str) -> Dict[str, Any]:
        """Load a shader asynchronously."""
        if shader_path in self.cache:
            print(f"  📦 Shader {shader_path} loaded from cache")
            return self.cache[shader_path]
        
        task_name = f"load_shader_{shader_path}"
        task = asyncio.create_task(self._load_shader_file(shader_path))
        
        task_info = AsyncTaskInfo(name=task_name, task=task)
        self.loading_tasks[task_name] = task_info
        
        try:
            result = await task
            self.cache[shader_path] = result
            return result
        except Exception as e:
            print(f"  ❌ Failed to load shader {shader_path}: {e}")
            raise
    
    async def _load_texture_file(self, texture_path: str) -> Dict[str, Any]:
        """Load texture file asynchronously (simulated)."""
        # Simulate file loading time
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate texture data
        result = {
            "type": "texture",
            "path": texture_path,
            "width": random.randint(256, 2048),
            "height": random.randint(256, 2048),
            "channels": 4,
            "data_size": random.randint(1024, 10240)
        }
        
        print(f"  📦 Loaded texture: {texture_path}")
        return result
    
    async def _load_mesh_file(self, mesh_path: str) -> Dict[str, Any]:
        """Load mesh file asynchronously (simulated)."""
        # Simulate file loading time
        await asyncio.sleep(random.uniform(0.2, 0.8))
        
        # Simulate mesh data
        result = {
            "type": "mesh",
            "path": mesh_path,
            "vertex_count": random.randint(100, 10000),
            "face_count": random.randint(50, 5000),
            "data_size": random.randint(2048, 20480)
        }
        
        print(f"  📦 Loaded mesh: {mesh_path}")
        return result
    
    async def _load_shader_file(self, shader_path: str) -> Dict[str, Any]:
        """Load shader file asynchronously (simulated)."""
        # Simulate file loading time
        await asyncio.sleep(random.uniform(0.05, 0.2))
        
        # Simulate shader data
        result = {
            "type": "shader",
            "path": shader_path,
            "shader_type": random.choice(["vertex", "fragment", "geometry"]),
            "source_size": random.randint(100, 2000)
        }
        
        print(f"  📦 Loaded shader: {shader_path}")
        return result
    
    async def load_resources_batch_async(self, resources: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Load multiple resources asynchronously."""
        print(f"  📦 Starting batch load of {len(resources)} resources")
        
        # Create loading tasks
        loading_tasks = []
        for resource in resources:
            resource_type = resource.get('type')
            resource_path = resource.get('path')
            
            if resource_type == 'texture':
                task = self.load_texture_async(resource_path)
            elif resource_type == 'mesh':
                task = self.load_mesh_async(resource_path)
            elif resource_type == 'shader':
                task = self.load_shader_async(resource_path)
            else:
                print(f"  ⚠️  Unknown resource type: {resource_type}")
                continue
            
            loading_tasks.append(task)
        
        # Load all resources concurrently
        start_time = time.time()
        results = await asyncio.gather(*loading_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ Resource loading failed: {result}")
            else:
                valid_results.append(result)
        
        print(f"  ✅ Batch loading completed in {end_time - start_time:.3f}s")
        print(f"  📊 Loaded {len(valid_results)} resources successfully")
        
        return valid_results


class AsyncPhysicsEngine:
    """Asynchronous physics engine for 3D graphics."""
    
    def __init__(self):
        self.physics_tasks: Dict[str, AsyncTaskInfo] = {}
        self.simulation_running = False
        self.physics_objects: List[Dict[str, Any]] = []
    
    async def start_physics_simulation(self, objects: List[Dict[str, Any]], 
                                     time_steps: int = 100, fps: int = 60):
        """Start asynchronous physics simulation."""
        if self.simulation_running:
            print("  ⚠️  Physics simulation already running")
            return
        
        self.simulation_running = True
        self.physics_objects = objects.copy()
        
        print(f"  🌍 Starting async physics simulation for {len(objects)} objects")
        print(f"  ⏱️  Simulating {time_steps} time steps at {fps} FPS")
        
        # Initialize object states
        for obj in self.physics_objects:
            obj["position"] = list(obj.get("position", [0, 0, 0]))
            obj["velocity"] = list(obj.get("velocity", [0, 0, 0]))
            obj["mass"] = obj.get("mass", 1.0)
        
        # Run physics simulation
        simulation_task = asyncio.create_task(
            self._run_physics_simulation(time_steps, fps)
        )
        
        task_info = AsyncTaskInfo(name="physics_simulation", task=simulation_task)
        self.physics_tasks["physics_simulation"] = task_info
        
        try:
            await simulation_task
        except Exception as e:
            print(f"  ❌ Physics simulation failed: {e}")
            raise
    
    async def _run_physics_simulation(self, time_steps: int, fps: int):
        """Run the physics simulation asynchronously."""
        dt = 1.0 / fps
        gravity = (0, -9.81, 0)
        
        for step in range(time_steps):
            # Create physics update tasks for each object
            update_tasks = []
            for obj in self.physics_objects:
                task = asyncio.create_task(self._update_physics_object(obj, gravity, dt))
                update_tasks.append(task)
            
            # Update all objects concurrently
            await asyncio.gather(*update_tasks)
            
            # Simulate frame time
            await asyncio.sleep(dt)
            
            if step % 10 == 0:
                print(f"  🌍 Physics step {step}/{time_steps}")
        
        self.simulation_running = False
        print("  ✅ Physics simulation completed")
    
    async def _update_physics_object(self, obj: Dict[str, Any], gravity: tuple, dt: float):
        """Update physics for a single object asynchronously."""
        # Apply gravity
        obj["velocity"][1] += gravity[1] * dt
        
        # Update position
        obj["position"][0] += obj["velocity"][0] * dt
        obj["position"][1] += obj["velocity"][1] * dt
        obj["position"][2] += obj["velocity"][2] * dt
        
        # Simple ground collision
        if obj["position"][1] < 0:
            obj["position"][1] = 0
            obj["velocity"][1] *= -0.8  # Bounce with energy loss
        
        # Small delay to simulate computation
        await asyncio.sleep(0.001)
    
    def get_physics_objects(self) -> List[Dict[str, Any]]:
        """Get current physics object states."""
        return self.physics_objects.copy()


class AsyncEventSystem:
    """Asynchronous event system for 3D graphics applications."""
    
    def __init__(self):
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.processor_task: Optional[asyncio.Task] = None
    
    async def start_event_processing(self):
        """Start the event processing system."""
        if self.processing:
            return
        
        self.processing = True
        self.processor_task = asyncio.create_task(self._event_processor())
        print("  📡 Started async event processing")
    
    async def stop_event_processing(self):
        """Stop the event processing system."""
        if not self.processing:
            return
        
        self.processing = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        print("  📡 Stopped async event processing")
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        print(f"  📡 Registered handler for event: {event_type}")
    
    async def emit_event(self, event_type: str, event_data: Any = None):
        """Emit an event asynchronously."""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": time.time()
        }
        
        await self.event_queue.put(event)
    
    async def _event_processor(self):
        """Process events asynchronously."""
        while self.processing:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
                
                # Process event
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  ❌ Event processing error: {e}")
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process a single event."""
        event_type = event["type"]
        event_data = event["data"]
        
        if event_type in self.event_handlers:
            # Create handler tasks
            handler_tasks = []
            for handler in self.event_handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(event_data))
                else:
                    # Wrap synchronous handler
                    task = asyncio.create_task(self._run_sync_handler(handler, event_data))
                handler_tasks.append(task)
            
            # Execute all handlers concurrently
            await asyncio.gather(*handler_tasks, return_exceptions=True)
    
    async def _run_sync_handler(self, handler: Callable, event_data: Any):
        """Run a synchronous handler in the event loop."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, handler, event_data)


# Example Usage and Demonstration
async def demonstrate_async_programming():
    """Demonstrates asynchronous programming for 3D graphics."""
    print("=== Asynchronous Programming for 3D Graphics ===\n")
    
    # Create async renderer
    renderer = AsyncRenderer()
    
    # Create sample scene data
    scene_data = {
        "objects": [
            {"name": f"object_{i}", "vertex_count": random.randint(50, 200), 
             "texture_count": random.randint(1, 3), "pixel_count": random.randint(500, 2000)} 
            for i in range(10)
        ]
    }
    
    # Render scene asynchronously
    print("=== Async Scene Rendering ===")
    render_results = await renderer.render_scene_async(scene_data)
    
    # Create async resource loader
    resource_loader = AsyncResourceLoader()
    
    # Define resources to load
    resources = [
        {"type": "texture", "path": "textures/diffuse.png"},
        {"type": "texture", "path": "textures/normal.png"},
        {"type": "mesh", "path": "meshes/character.obj"},
        {"type": "mesh", "path": "meshes/environment.obj"},
        {"type": "shader", "path": "shaders/phong.vert"},
        {"type": "shader", "path": "shaders/phong.frag"}
    ]
    
    # Load resources asynchronously
    print("\n=== Async Resource Loading ===")
    loaded_resources = await resource_loader.load_resources_batch_async(resources)
    
    # Create async physics engine
    physics_engine = AsyncPhysicsEngine()
    
    # Create physics objects
    physics_objects = [
        {"name": f"physics_obj_{i}", "position": [random.uniform(-2, 2), 5, random.uniform(-2, 2)], 
         "velocity": [0, 0, 0], "mass": random.uniform(0.5, 2.0)}
        for i in range(5)
    ]
    
    # Run physics simulation asynchronously
    print("\n=== Async Physics Simulation ===")
    physics_task = asyncio.create_task(
        physics_engine.start_physics_simulation(physics_objects, time_steps=30, fps=30)
    )
    
    # Create async event system
    event_system = AsyncEventSystem()
    
    # Register event handlers
    async def on_object_rendered(data):
        print(f"  📡 Event: Object rendered - {data}")
    
    async def on_resource_loaded(data):
        print(f"  📡 Event: Resource loaded - {data}")
    
    event_system.register_handler("object_rendered", on_object_rendered)
    event_system.register_handler("resource_loaded", on_resource_loaded)
    
    # Start event processing
    await event_system.start_event_processing()
    
    # Emit some events
    await event_system.emit_event("object_rendered", {"object": "cube", "time": 0.1})
    await event_system.emit_event("resource_loaded", {"resource": "texture.png", "size": 1024})
    
    # Wait for physics simulation to complete
    await physics_task
    
    # Stop event processing
    await event_system.stop_event_processing()
    
    # Stop renderer
    await renderer.stop_rendering()
    
    # Display results
    print("\n=== Results Summary ===")
    print(f"Rendering: {len(render_results)} objects rendered")
    print(f"Resources: {len(loaded_resources)} resources loaded")
    print(f"Physics: {len(physics_objects)} objects simulated")


async def demonstrate_async_patterns():
    """Demonstrates various async programming patterns."""
    print("\n=== Async Programming Patterns ===\n")
    
    # Pattern 1: Concurrent execution
    async def task_1():
        await asyncio.sleep(1)
        return "Task 1 completed"
    
    async def task_2():
        await asyncio.sleep(0.5)
        return "Task 2 completed"
    
    async def task_3():
        await asyncio.sleep(0.8)
        return "Task 3 completed"
    
    print("Running tasks concurrently:")
    start_time = time.time()
    results = await asyncio.gather(task_1(), task_2(), task_3())
    end_time = time.time()
    
    print(f"  ✅ All tasks completed in {end_time - start_time:.3f}s")
    for result in results:
        print(f"  📋 {result}")
    
    # Pattern 2: Timeout handling
    async def long_running_task():
        await asyncio.sleep(5)
        return "Long task completed"
    
    print("\nRunning task with timeout:")
    try:
        result = await asyncio.wait_for(long_running_task(), timeout=2.0)
        print(f"  ✅ {result}")
    except asyncio.TimeoutError:
        print("  ⏰ Task timed out")
    
    # Pattern 3: Cancellation
    async def cancellable_task():
        try:
            await asyncio.sleep(10)
            return "Task completed"
        except asyncio.CancelledError:
            print("  🚫 Task was cancelled")
            raise
    
    print("\nCancelling a task:")
    task = asyncio.create_task(cancellable_task())
    await asyncio.sleep(0.1)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("  ✅ Task cancelled successfully")


if __name__ == "__main__":
    # Run async demonstrations
    asyncio.run(demonstrate_async_programming())
    asyncio.run(demonstrate_async_patterns())
