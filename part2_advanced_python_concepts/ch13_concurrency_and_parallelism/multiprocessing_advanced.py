"""
Chapter 13: Concurrency and Parallelism - Multiprocessing Advanced
==================================================================

This module demonstrates advanced multiprocessing concepts for 3D graphics
applications, including process pools, shared memory, and parallel computation.

Key Concepts:
- Process creation and management
- Process pools for parallel computation
- Shared memory and inter-process communication
- Parallel rendering and computation
- Performance optimization with multiprocessing
"""

import multiprocessing as mp
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math
import os


class ProcessStatus(Enum):
    """Status of a process."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProcessInfo:
    """Information about a process."""
    name: str
    pid: int
    status: ProcessStatus = ProcessStatus.IDLE
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None


class SharedMemoryManager:
    """Manages shared memory for 3D graphics data."""
    
    def __init__(self):
        self.shared_arrays: Dict[str, mp.Array] = {}
        self.shared_values: Dict[str, mp.Value] = {}
        self.locks: Dict[str, mp.Lock] = {}
    
    def create_shared_array(self, name: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        """Create a shared NumPy array."""
        size = int(np.prod(shape))
        shared_array = mp.Array(dtype, size)
        numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
        
        self.shared_arrays[name] = shared_array
        self.locks[name] = mp.Lock()
        
        print(f"  📦 Created shared array '{name}' with shape {shape}")
        return numpy_array
    
    def get_shared_array(self, name: str) -> Optional[np.ndarray]:
        """Get a shared NumPy array."""
        if name in self.shared_arrays:
            shared_array = self.shared_arrays[name]
            return np.frombuffer(shared_array.get_obj(), dtype=np.float32)
        return None
    
    def create_shared_value(self, name: str, value_type, initial_value=0) -> mp.Value:
        """Create a shared value."""
        shared_value = mp.Value(value_type, initial_value)
        self.shared_values[name] = shared_value
        self.locks[name] = mp.Lock()
        
        print(f"  📦 Created shared value '{name}' with initial value {initial_value}")
        return shared_value
    
    def get_shared_value(self, name: str) -> Optional[mp.Value]:
        """Get a shared value."""
        return self.shared_values.get(name)
    
    def acquire_lock(self, name: str):
        """Acquire lock for a shared resource."""
        if name in self.locks:
            self.locks[name].acquire()
    
    def release_lock(self, name: str):
        """Release lock for a shared resource."""
        if name in self.locks:
            self.locks[name].release()
    
    def cleanup(self):
        """Clean up shared memory resources."""
        self.shared_arrays.clear()
        self.shared_values.clear()
        self.locks.clear()
        print("  🧹 Cleaned up shared memory resources")


class ParallelRenderer:
    """Parallel 3D renderer using multiprocessing."""
    
    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or mp.cpu_count()
        self.process_pool = mp.Pool(processes=self.num_processes)
        self.shared_memory = SharedMemoryManager()
        self.rendering = False
        
        print(f"  🚀 Initialized parallel renderer with {self.num_processes} processes")
    
    def render_scene_parallel(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render a scene using parallel processing."""
        if self.rendering:
            print("  ⚠️  Already rendering, please wait...")
            return []
        
        self.rendering = True
        print(f"  🎨 Starting parallel rendering of scene with {len(scene_data.get('objects', []))} objects")
        
        # Split scene into chunks for parallel processing
        objects = scene_data.get('objects', [])
        chunk_size = max(1, len(objects) // self.num_processes)
        chunks = [objects[i:i + chunk_size] for i in range(0, len(objects), chunk_size)]
        
        # Create tasks for each chunk
        tasks = []
        for i, chunk in enumerate(chunks):
            task_data = {
                "chunk_id": i,
                "objects": chunk,
                "camera": scene_data.get('camera', {}),
                "lights": scene_data.get('lights', []),
                "settings": scene_data.get('settings', {})
            }
            tasks.append(task_data)
        
        # Submit tasks to process pool
        start_time = time.time()
        results = self.process_pool.map(self._render_chunk, tasks)
        end_time = time.time()
        
        # Flatten results
        all_results = []
        for chunk_results in results:
            all_results.extend(chunk_results)
        
        self.rendering = False
        
        print(f"  ✅ Parallel rendering completed in {end_time - start_time:.3f}s")
        print(f"  📊 Rendered {len(all_results)} objects using {self.num_processes} processes")
        
        return all_results
    
    def _render_chunk(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Render a chunk of objects (runs in separate process)."""
        chunk_id = task_data["chunk_id"]
        objects = task_data["objects"]
        process_id = os.getpid()
        
        print(f"  🎨 Process {process_id} rendering chunk {chunk_id} with {len(objects)} objects")
        
        results = []
        for obj in objects:
            # Simulate rendering time
            render_time = random.uniform(0.01, 0.05)
            time.sleep(render_time)
            
            result = {
                "object_name": obj.get('name', 'unknown'),
                "chunk_id": chunk_id,
                "process_id": process_id,
                "render_time": render_time,
                "vertices_rendered": obj.get('vertex_count', 100),
                "pixels_processed": obj.get('pixel_count', 1000)
            }
            results.append(result)
        
        return results
    
    def close(self):
        """Close the parallel renderer."""
        self.process_pool.close()
        self.process_pool.join()
        self.shared_memory.cleanup()
        print("  🛑 Parallel renderer closed")


class ParallelComputingEngine:
    """Engine for parallel computation in 3D graphics."""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.process_pool = mp.Pool(processes=self.num_workers)
        self.tasks: List[Dict[str, Any]] = []
        self.results: List[Any] = []
    
    def add_computation_task(self, task_func: Callable, *args, **kwargs):
        """Add a computation task."""
        task = {
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": time.time()
        }
        self.tasks.append(task)
    
    def execute_tasks_parallel(self) -> List[Any]:
        """Execute all tasks in parallel."""
        if not self.tasks:
            print("  ⚠️  No tasks to execute")
            return []
        
        print(f"  🚀 Executing {len(self.tasks)} tasks using {self.num_workers} workers")
        
        # Prepare tasks for process pool
        pool_tasks = []
        for i, task in enumerate(self.tasks):
            pool_task = (task["func"], task["args"], task["kwargs"], i)
            pool_tasks.append(pool_task)
        
        # Execute tasks
        start_time = time.time()
        results = self.process_pool.map(self._execute_task, pool_tasks)
        end_time = time.time()
        
        # Store results
        self.results = results
        
        print(f"  ✅ Parallel execution completed in {end_time - start_time:.3f}s")
        print(f"  📊 Processed {len(results)} tasks")
        
        return results
    
    def _execute_task(self, task_data: Tuple) -> Dict[str, Any]:
        """Execute a single task (runs in separate process)."""
        func, args, kwargs, task_id = task_data
        process_id = os.getpid()
        
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            return {
                "task_id": task_id,
                "process_id": process_id,
                "result": result,
                "execution_time": end_time - start_time,
                "status": "success"
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "process_id": process_id,
                "error": str(e),
                "status": "error"
            }
    
    def clear_tasks(self):
        """Clear all tasks."""
        self.tasks.clear()
        self.results.clear()
    
    def close(self):
        """Close the computing engine."""
        self.process_pool.close()
        self.process_pool.join()


class PhysicsSimulator:
    """Parallel physics simulator for 3D graphics."""
    
    def __init__(self, num_physics_processes: int = None):
        self.num_processes = num_physics_processes or max(1, mp.cpu_count() - 1)
        self.process_pool = mp.Pool(processes=self.num_processes)
        self.shared_memory = SharedMemoryManager()
        self.simulation_running = False
    
    def simulate_physics_parallel(self, physics_objects: List[Dict[str, Any]], 
                                time_steps: int = 100) -> List[List[Dict[str, Any]]]:
        """Simulate physics for multiple objects in parallel."""
        if self.simulation_running:
            print("  ⚠️  Physics simulation already running")
            return []
        
        self.simulation_running = True
        print(f"  🌍 Starting parallel physics simulation for {len(physics_objects)} objects")
        
        # Split objects into chunks
        chunk_size = max(1, len(physics_objects) // self.num_processes)
        chunks = [physics_objects[i:i + chunk_size] for i in range(0, len(physics_objects), chunk_size)]
        
        # Create simulation tasks
        tasks = []
        for i, chunk in enumerate(chunks):
            task_data = {
                "chunk_id": i,
                "objects": chunk,
                "time_steps": time_steps,
                "gravity": (0, -9.81, 0),
                "time_step": 0.016  # 60 FPS
            }
            tasks.append(task_data)
        
        # Run physics simulation in parallel
        start_time = time.time()
        results = self.process_pool.map(self._simulate_chunk, tasks)
        end_time = time.time()
        
        self.simulation_running = False
        
        print(f"  ✅ Physics simulation completed in {end_time - start_time:.3f}s")
        print(f"  📊 Simulated {time_steps} time steps for {len(physics_objects)} objects")
        
        return results
    
    def _simulate_chunk(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate physics for a chunk of objects (runs in separate process)."""
        chunk_id = task_data["chunk_id"]
        objects = task_data["objects"]
        time_steps = task_data["time_steps"]
        gravity = task_data["gravity"]
        dt = task_data["time_step"]
        process_id = os.getpid()
        
        print(f"  🌍 Process {process_id} simulating chunk {chunk_id} with {len(objects)} objects")
        
        # Initialize object states
        for obj in objects:
            obj["position"] = list(obj.get("position", [0, 0, 0]))
            obj["velocity"] = list(obj.get("velocity", [0, 0, 0]))
            obj["mass"] = obj.get("mass", 1.0)
        
        # Simulate physics
        simulation_data = []
        for step in range(time_steps):
            step_data = []
            
            for obj in objects:
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
                
                step_data.append({
                    "object_name": obj.get("name", "unknown"),
                    "position": obj["position"].copy(),
                    "velocity": obj["velocity"].copy(),
                    "step": step
                })
            
            simulation_data.append(step_data)
            
            # Small delay to simulate computation
            time.sleep(0.001)
        
        return simulation_data
    
    def close(self):
        """Close the physics simulator."""
        self.process_pool.close()
        self.process_pool.join()
        self.shared_memory.cleanup()


class ProcessMonitor:
    """Monitor and manage processes in 3D graphics applications."""
    
    def __init__(self):
        self.processes: Dict[str, ProcessInfo] = {}
        self.lock = mp.Lock()
    
    def register_process(self, name: str, pid: int) -> ProcessInfo:
        """Register a process for monitoring."""
        with self.lock:
            process_info = ProcessInfo(name=name, pid=pid)
            self.processes[name] = process_info
            print(f"  📊 Registered process: {name} (PID: {pid})")
            return process_info
    
    def update_process_status(self, name: str, status: ProcessStatus, result: Any = None, 
                            error_message: str = None):
        """Update process status."""
        with self.lock:
            if name in self.processes:
                process_info = self.processes[name]
                process_info.status = status
                if result is not None:
                    process_info.result = result
                if error_message:
                    process_info.error_message = error_message
    
    def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get information about a specific process."""
        with self.lock:
            return self.processes.get(name)
    
    def get_all_processes(self) -> List[ProcessInfo]:
        """Get all registered processes."""
        with self.lock:
            return list(self.processes.values())
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get process statistics."""
        with self.lock:
            total_processes = len(self.processes)
            running_processes = len([p for p in self.processes.values() 
                                   if p.status == ProcessStatus.RUNNING])
            completed_processes = len([p for p in self.processes.values() 
                                     if p.status == ProcessStatus.COMPLETED])
            error_processes = len([p for p in self.processes.values() 
                                 if p.status == ProcessStatus.ERROR])
            
            return {
                "total": total_processes,
                "running": running_processes,
                "completed": completed_processes,
                "error": error_processes,
                "idle": total_processes - running_processes - completed_processes - error_processes
            }


# Example Usage and Demonstration
def demonstrate_multiprocessing_advanced():
    """Demonstrates advanced multiprocessing for 3D graphics."""
    print("=== Advanced Multiprocessing for 3D Graphics ===\n")
    
    # Create process monitor
    monitor = ProcessMonitor()
    
    # Create parallel renderer
    renderer = ParallelRenderer(num_processes=4)
    
    # Create sample scene data
    scene_data = {
        "objects": [
            {"name": f"object_{i}", "vertex_count": random.randint(50, 200), 
             "pixel_count": random.randint(500, 2000)} 
            for i in range(20)
        ],
        "camera": {"position": [0, 0, 5], "target": [0, 0, 0]},
        "lights": [{"position": [1, 1, 1], "intensity": 1.0}],
        "settings": {"resolution": [1920, 1080], "quality": "high"}
    }
    
    # Render scene in parallel
    print("=== Parallel Scene Rendering ===")
    render_results = renderer.render_scene_parallel(scene_data)
    
    # Create parallel computing engine
    computing_engine = ParallelComputingEngine(num_workers=3)
    
    # Add computation tasks
    def compute_transform_matrix(angle: float, axis: str) -> np.ndarray:
        """Compute transformation matrix (simulated computation)."""
        time.sleep(0.01)  # Simulate computation time
        return np.eye(4)  # Simplified result
    
    def compute_lighting_calculation(light_pos: List[float], obj_pos: List[float]) -> float:
        """Compute lighting calculation (simulated)."""
        time.sleep(0.005)  # Simulate computation time
        distance = math.sqrt(sum((l - o) ** 2 for l, o in zip(light_pos, obj_pos)))
        return 1.0 / (1.0 + distance)  # Simplified lighting
    
    def compute_collision_detection(obj1: Dict, obj2: Dict) -> bool:
        """Compute collision detection (simulated)."""
        time.sleep(0.008)  # Simulate computation time
        return random.choice([True, False])  # Simplified collision
    
    # Add tasks to computing engine
    for i in range(10):
        computing_engine.add_computation_task(compute_transform_matrix, 
                                            random.uniform(0, 2*math.pi), "y")
        computing_engine.add_computation_task(compute_lighting_calculation, 
                                            [1, 1, 1], [random.uniform(-5, 5) for _ in range(3)])
        computing_engine.add_computation_task(compute_collision_detection, 
                                            {"pos": [0, 0, 0]}, {"pos": [1, 1, 1]})
    
    # Execute tasks in parallel
    print("\n=== Parallel Computation ===")
    computation_results = computing_engine.execute_tasks_parallel()
    
    # Create physics simulator
    physics_simulator = PhysicsSimulator(num_physics_processes=2)
    
    # Create physics objects
    physics_objects = [
        {"name": f"physics_obj_{i}", "position": [random.uniform(-2, 2), 5, random.uniform(-2, 2)], 
         "velocity": [0, 0, 0], "mass": random.uniform(0.5, 2.0)}
        for i in range(8)
    ]
    
    # Simulate physics in parallel
    print("\n=== Parallel Physics Simulation ===")
    physics_results = physics_simulator.simulate_physics_parallel(physics_objects, time_steps=50)
    
    # Display results
    print("\n=== Results Summary ===")
    print(f"Rendering: {len(render_results)} objects rendered")
    print(f"Computation: {len(computation_results)} tasks completed")
    print(f"Physics: {len(physics_results)} chunks simulated")
    
    # Process statistics
    process_stats = monitor.get_process_stats()
    print(f"Process statistics: {process_stats}")
    
    # Cleanup
    renderer.close()
    computing_engine.close()
    physics_simulator.close()


def demonstrate_shared_memory():
    """Demonstrates shared memory usage in multiprocessing."""
    print("\n=== Shared Memory Demonstration ===\n")
    
    # Create shared memory manager
    shared_memory = SharedMemoryManager()
    
    # Create shared arrays for 3D data
    vertex_array = shared_memory.create_shared_array("vertices", (1000, 3))
    color_array = shared_memory.create_shared_array("colors", (1000, 4))
    transform_matrix = shared_memory.create_shared_array("transform", (4, 4))
    
    # Create shared values
    frame_counter = shared_memory.create_shared_value('i', 0)  # integer
    render_quality = shared_memory.create_shared_value('f', 1.0)  # float
    
    # Initialize shared data
    vertex_array[:] = np.random.rand(1000, 3)
    color_array[:] = np.random.rand(1000, 4)
    transform_matrix[:] = np.eye(4)
    
    print(f"  📊 Shared vertex array shape: {vertex_array.shape}")
    print(f"  📊 Shared color array shape: {color_array.shape}")
    print(f"  📊 Shared transform matrix shape: {transform_matrix.shape}")
    print(f"  📊 Frame counter: {frame_counter.value}")
    print(f"  📊 Render quality: {render_quality.value}")
    
    # Cleanup
    shared_memory.cleanup()


if __name__ == "__main__":
    demonstrate_multiprocessing_advanced()
    demonstrate_shared_memory()
