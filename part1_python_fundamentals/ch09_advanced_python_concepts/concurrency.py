#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Concurrency Example

Demonstrates concurrency including threading, multiprocessing, async/await,
parallel processing, and synchronization for 3D graphics applications.
"""

import time
import threading
import multiprocessing
import asyncio
import queue
import random
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock, Event, Condition, Semaphore, Barrier
import os

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Concurrency"
__description__ = "Concurrency for 3D graphics applications"

# ============================================================================
# THREADING EXAMPLES
# ============================================================================

class ThreadingExample:
    """Demonstrate threading concepts"""
    
    def __init__(self):
        self.counter = 0
        self.lock = Lock()
        self.event = Event()
        self.condition = Condition()
        self.semaphore = Semaphore(3)  # Allow 3 concurrent operations
        self.barrier = Barrier(4)  # Wait for 4 threads
        self.data_queue = queue.Queue()
        self.running = True
    
    def thread_safe_counter(self, thread_id: int, iterations: int):
        """Thread-safe counter using locks"""
        for i in range(iterations):
            with self.lock:
                self.counter += 1
                print(f"Thread {thread_id}: Counter = {self.counter}")
            time.sleep(0.01)
    
    def resource_worker(self, thread_id: int, resource_name: str):
        """Worker that uses a limited resource (semaphore)"""
        for i in range(3):
            with self.semaphore:
                print(f"Thread {thread_id}: Acquired {resource_name}")
                time.sleep(0.1)  # Simulate work
                print(f"Thread {thread_id}: Released {resource_name}")
    
    def producer_consumer(self, is_producer: bool, thread_id: int):
        """Producer-consumer pattern"""
        if is_producer:
            for i in range(5):
                item = f"Item {i} from producer {thread_id}"
                self.data_queue.put(item)
                print(f"Producer {thread_id}: Produced {item}")
                time.sleep(0.1)
        else:
            while self.running:
                try:
                    item = self.data_queue.get(timeout=1)
                    print(f"Consumer {thread_id}: Consumed {item}")
                    self.data_queue.task_done()
                except queue.Empty:
                    break
    
    def barrier_worker(self, thread_id: int):
        """Worker that waits at a barrier"""
        print(f"Thread {thread_id}: Waiting at barrier...")
        self.barrier.wait()
        print(f"Thread {thread_id}: Passed barrier!")
    
    def event_worker(self, thread_id: int):
        """Worker that waits for an event"""
        print(f"Thread {thread_id}: Waiting for event...")
        self.event.wait()
        print(f"Thread {thread_id}: Event received!")

def demonstrate_threading():
    """Demonstrate threading concepts"""
    print("=== Threading Examples ===\n")
    
    example = ThreadingExample()
    
    print("1. Thread-safe counter:")
    threads = []
    for i in range(4):
        thread = threading.Thread(
            target=example.thread_safe_counter,
            args=(i, 5)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    print(f"Final counter value: {example.counter}\n")
    
    print("2. Resource management with semaphore:")
    threads = []
    for i in range(6):
        thread = threading.Thread(
            target=example.resource_worker,
            args=(i, f"Resource-{i}")
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    print()
    
    print("3. Producer-consumer pattern:")
    # Start consumers
    consumer_threads = []
    for i in range(2):
        thread = threading.Thread(
            target=example.producer_consumer,
            args=(False, i)
        )
        consumer_threads.append(thread)
        thread.start()
    
    # Start producers
    producer_threads = []
    for i in range(3):
        thread = threading.Thread(
            target=example.producer_consumer,
            args=(True, i)
        )
        producer_threads.append(thread)
        thread.start()
    
    # Wait for producers to finish
    for thread in producer_threads:
        thread.join()
    
    # Signal consumers to stop
    example.running = False
    
    # Wait for consumers to finish
    for thread in consumer_threads:
        thread.join()
    print()
    
    print("4. Barrier synchronization:")
    threads = []
    for i in range(4):
        thread = threading.Thread(
            target=example.barrier_worker,
            args=(i,)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    print()
    
    print("5. Event synchronization:")
    # Start workers
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=example.event_worker,
            args=(i,)
        )
        threads.append(thread)
        thread.start()
    
    # Wait a bit, then set the event
    time.sleep(1)
    example.event.set()
    
    for thread in threads:
        thread.join()
    print()

# ============================================================================
# MULTIPROCESSING EXAMPLES
# ============================================================================

def cpu_intensive_task(data: List[float]) -> float:
    """CPU-intensive task for multiprocessing"""
    result = 0.0
    for x in data:
        result += math.sqrt(x) * math.sin(x) * math.cos(x)
    return result

def process_worker(worker_id: int, data_chunk: List[float]) -> Tuple[int, float]:
    """Worker function for multiprocessing"""
    print(f"Process {worker_id}: Processing {len(data_chunk)} items")
    result = cpu_intensive_task(data_chunk)
    return worker_id, result

def demonstrate_multiprocessing():
    """Demonstrate multiprocessing concepts"""
    print("=== Multiprocessing Examples ===\n")
    
    print("1. Basic multiprocessing:")
    # Create large dataset
    data = [random.uniform(0, 100) for _ in range(100000)]
    
    # Split data into chunks
    chunk_size = len(data) // 4
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process with multiple processes
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(process_worker, i, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            worker_id, result = future.result()
            results.append((worker_id, result))
            print(f"Process {worker_id} completed with result: {result:.4f}")
    
    total_result = sum(result for _, result in results)
    print(f"Total result: {total_result:.4f}\n")
    
    print("2. Process pool with map:")
    # Simpler approach using map
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, chunks))
    
    print(f"Results: {[f'{r:.4f}' for r in results]}")
    print(f"Total: {sum(results):.4f}\n")
    
    print("3. Process communication:")
    # Demonstrate process communication with shared memory
    from multiprocessing import Value, Array, Manager
    
    # Shared counter
    counter = Value('i', 0)
    
    def increment_counter(process_id: int, lock: Lock):
        for _ in range(1000):
            with lock:
                counter.value += 1
        print(f"Process {process_id}: Finished incrementing")
    
    # Create processes with shared counter
    processes = []
    lock = multiprocessing.Lock()
    
    for i in range(4):
        process = multiprocessing.Process(
            target=increment_counter,
            args=(i, lock)
        )
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    print(f"Final counter value: {counter.value}\n")

# ============================================================================
# ASYNCIO EXAMPLES
# ============================================================================

class Async3DGraphics:
    """Demonstrate async/await for 3D graphics"""
    
    def __init__(self):
        self.scene_objects = []
        self.rendering_queue = asyncio.Queue()
        self.event_loop = asyncio.get_event_loop()
    
    async def load_texture(self, texture_name: str) -> str:
        """Simulate async texture loading"""
        print(f"🖼️  Loading texture: {texture_name}")
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate I/O
        print(f"✅ Texture loaded: {texture_name}")
        return f"texture_{texture_name}"
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Simulate async 3D model loading"""
        print(f"📦 Loading model: {model_name}")
        await asyncio.sleep(random.uniform(0.2, 0.8))  # Simulate I/O
        
        model_data = {
            'name': model_name,
            'vertices': [[random.uniform(-1, 1) for _ in range(3)] for _ in range(100)],
            'faces': [[random.randint(0, 99) for _ in range(3)] for _ in range(50)],
            'textures': []
        }
        
        print(f"✅ Model loaded: {model_name}")
        return model_data
    
    async def render_frame(self, frame_id: int) -> str:
        """Simulate async frame rendering"""
        print(f"🎨 Rendering frame: {frame_id}")
        await asyncio.sleep(random.uniform(0.05, 0.15))  # Simulate rendering
        print(f"✅ Frame rendered: {frame_id}")
        return f"frame_{frame_id}.png"
    
    async def physics_simulation(self, step: int) -> Dict[str, float]:
        """Simulate async physics simulation"""
        print(f"⚙️  Physics step: {step}")
        await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate computation
        
        physics_data = {
            'gravity': 9.81,
            'velocity': random.uniform(0, 10),
            'position': random.uniform(0, 100),
            'step': step
        }
        
        print(f"✅ Physics step completed: {step}")
        return physics_data
    
    async def network_request(self, endpoint: str) -> Dict[str, Any]:
        """Simulate async network requests"""
        print(f"🌐 Network request: {endpoint}")
        await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate network delay
        
        response_data = {
            'endpoint': endpoint,
            'status': 'success',
            'data': f'response_from_{endpoint}',
            'timestamp': time.time()
        }
        
        print(f"✅ Network response: {endpoint}")
        return response_data

async def demonstrate_asyncio():
    """Demonstrate async/await concepts"""
    print("=== Async/Await Examples ===\n")
    
    graphics = Async3DGraphics()
    
    print("1. Sequential async operations:")
    start_time = time.time()
    
    # Sequential execution
    texture1 = await graphics.load_texture("player.png")
    texture2 = await graphics.load_texture("enemy.png")
    model1 = await graphics.load_model("player.obj")
    model2 = await graphics.load_model("enemy.obj")
    
    sequential_time = time.time() - start_time
    print(f"Sequential execution time: {sequential_time:.3f} seconds\n")
    
    print("2. Concurrent async operations:")
    start_time = time.time()
    
    # Concurrent execution
    texture1_task = graphics.load_texture("player.png")
    texture2_task = graphics.load_texture("enemy.png")
    model1_task = graphics.load_model("player.obj")
    model2_task = graphics.load_model("enemy.obj")
    
    # Wait for all tasks to complete
    texture1, texture2, model1, model2 = await asyncio.gather(
        texture1_task, texture2_task, model1_task, model2_task
    )
    
    concurrent_time = time.time() - start_time
    print(f"Concurrent execution time: {concurrent_time:.3f} seconds")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x\n")
    
    print("3. Async frame rendering loop:")
    # Simulate real-time rendering
    async def render_loop():
        frame_count = 0
        while frame_count < 10:
            frame_task = graphics.render_frame(frame_count)
            physics_task = graphics.physics_simulation(frame_count)
            
            # Render frame and update physics concurrently
            frame_result, physics_result = await asyncio.gather(
                frame_task, physics_task
            )
            
            print(f"Frame {frame_count}: {frame_result}, Physics: {physics_result}")
            frame_count += 1
    
    await render_loop()
    print()
    
    print("4. Async network operations:")
    # Simulate multiple network requests
    endpoints = ["/api/player-data", "/api/enemy-data", "/api/world-data"]
    
    async def fetch_all_data():
        tasks = [graphics.network_request(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks)
        return results
    
    network_data = await fetch_all_data()
    for data in network_data:
        print(f"Received: {data}")
    print()
    
    print("5. Async with timeout:")
    # Demonstrate timeout handling
    async def timeout_example():
        try:
            # This will timeout
            result = await asyncio.wait_for(
                graphics.load_texture("large_texture.png"),
                timeout=0.1
            )
            print(f"Result: {result}")
        except asyncio.TimeoutError:
            print("❌ Operation timed out")
    
    await timeout_example()
    print()

# ============================================================================
# PARALLEL PROCESSING EXAMPLES
# ============================================================================

class Parallel3DProcessor:
    """Demonstrate parallel processing for 3D graphics"""
    
    def __init__(self):
        self.vertex_data = []
        self.texture_data = []
        self.light_data = []
    
    def generate_vertex_data(self, count: int) -> List[List[float]]:
        """Generate 3D vertex data"""
        return [[random.uniform(-1, 1) for _ in range(3)] for _ in range(count)]
    
    def process_vertices(self, vertices: List[List[float]]) -> List[List[float]]:
        """Process vertices (simulate transformation)"""
        processed = []
        for vertex in vertices:
            # Simulate vertex transformation
            x, y, z = vertex
            processed_vertex = [
                x * math.cos(0.1) - y * math.sin(0.1),
                x * math.sin(0.1) + y * math.cos(0.1),
                z + 0.1
            ]
            processed.append(processed_vertex)
        return processed
    
    def process_textures(self, texture_chunk: List[str]) -> List[str]:
        """Process texture data"""
        processed = []
        for texture in texture_chunk:
            # Simulate texture processing
            processed_texture = f"processed_{texture}"
            processed.append(processed_texture)
        return processed
    
    def calculate_lighting(self, vertex: List[float], light_pos: List[float]) -> float:
        """Calculate lighting for a vertex"""
        # Simulate lighting calculation
        dx = vertex[0] - light_pos[0]
        dy = vertex[1] - light_pos[1]
        dz = vertex[2] - light_pos[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return max(0, 1.0 - distance / 10.0)

def demonstrate_parallel_processing():
    """Demonstrate parallel processing concepts"""
    print("=== Parallel Processing Examples ===\n")
    
    processor = Parallel3DProcessor()
    
    print("1. Parallel vertex processing:")
    # Generate large vertex dataset
    vertex_count = 10000
    vertices = processor.generate_vertex_data(vertex_count)
    
    # Split into chunks for parallel processing
    chunk_size = vertex_count // 4
    vertex_chunks = [vertices[i:i + chunk_size] for i in range(0, vertex_count, chunk_size)]
    
    # Process vertices in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        start_time = time.time()
        
        # Submit tasks
        futures = [executor.submit(processor.process_vertices, chunk) 
                  for chunk in vertex_chunks]
        
        # Collect results
        processed_chunks = []
        for future in as_completed(futures):
            processed_chunk = future.result()
            processed_chunks.append(processed_chunk)
        
        parallel_time = time.time() - start_time
    
    # Sequential processing for comparison
    start_time = time.time()
    sequential_result = processor.process_vertices(vertices)
    sequential_time = time.time() - start_time
    
    print(f"Parallel processing time: {parallel_time:.4f} seconds")
    print(f"Sequential processing time: {sequential_time:.4f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x\n")
    
    print("2. Parallel lighting calculations:")
    # Generate lighting data
    light_positions = [[random.uniform(-5, 5) for _ in range(3)] for _ in range(10)]
    vertices = processor.generate_vertex_data(1000)
    
    def calculate_lighting_for_vertex(args):
        vertex, light_pos = args
        return processor.calculate_lighting(vertex, light_pos)
    
    # Create all vertex-light combinations
    combinations = [(vertex, light_pos) for vertex in vertices for light_pos in light_positions]
    
    # Calculate lighting in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        start_time = time.time()
        lighting_results = list(executor.map(calculate_lighting_for_vertex, combinations))
        parallel_time = time.time() - start_time
    
    print(f"Calculated {len(lighting_results)} lighting values")
    print(f"Parallel calculation time: {parallel_time:.4f} seconds")
    print(f"Average lighting value: {sum(lighting_results) / len(lighting_results):.4f}\n")
    
    print("3. Pipeline processing:")
    # Demonstrate pipeline processing
    def pipeline_stage1(data):
        """Stage 1: Data preparation"""
        return [f"prepared_{item}" for item in data]
    
    def pipeline_stage2(data):
        """Stage 2: Processing"""
        return [f"processed_{item}" for item in data]
    
    def pipeline_stage3(data):
        """Stage 3: Finalization"""
        return [f"finalized_{item}" for item in data]
    
    # Create pipeline
    input_data = [f"item_{i}" for i in range(100)]
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Execute pipeline stages
        stage1_future = executor.submit(pipeline_stage1, input_data)
        stage1_result = stage1_future.result()
        
        stage2_future = executor.submit(pipeline_stage2, stage1_result)
        stage2_result = stage2_future.result()
        
        stage3_future = executor.submit(pipeline_stage3, stage2_result)
        final_result = stage3_future.result()
    
    print(f"Pipeline processed {len(final_result)} items")
    print(f"Sample results: {final_result[:5]}\n")

# ============================================================================
# SYNCHRONIZATION EXAMPLES
# ============================================================================

class Synchronized3DScene:
    """Demonstrate synchronization in 3D graphics"""
    
    def __init__(self):
        self.scene_objects = {}
        self.object_lock = Lock()
        self.render_queue = queue.Queue()
        self.render_event = Event()
        self.scene_ready = Condition()
        self.resource_semaphore = Semaphore(5)  # Max 5 concurrent operations
    
    def add_object(self, obj_id: str, obj_data: Dict[str, Any]):
        """Thread-safe object addition"""
        with self.object_lock:
            self.scene_objects[obj_id] = obj_data
            print(f"➕ Added object: {obj_id}")
    
    def remove_object(self, obj_id: str):
        """Thread-safe object removal"""
        with self.object_lock:
            if obj_id in self.scene_objects:
                del self.scene_objects[obj_id]
                print(f"➖ Removed object: {obj_id}")
    
    def get_scene_snapshot(self) -> Dict[str, Any]:
        """Thread-safe scene snapshot"""
        with self.object_lock:
            return self.scene_objects.copy()
    
    def render_worker(self, worker_id: int):
        """Worker thread for rendering"""
        while True:
            try:
                # Wait for render event
                if not self.render_event.wait(timeout=1):
                    continue
                
                # Get scene snapshot
                scene_snapshot = self.get_scene_snapshot()
                
                # Acquire rendering resource
                with self.resource_semaphore:
                    print(f"🎨 Render worker {worker_id}: Rendering {len(scene_snapshot)} objects")
                    time.sleep(0.1)  # Simulate rendering
                    print(f"✅ Render worker {worker_id}: Completed")
                
            except Exception as e:
                print(f"❌ Render worker {worker_id}: Error - {e}")
                break
    
    def scene_manager(self):
        """Scene management thread"""
        object_count = 0
        
        while object_count < 10:
            # Add objects
            obj_id = f"object_{object_count}"
            obj_data = {
                'position': [random.uniform(-10, 10) for _ in range(3)],
                'rotation': [random.uniform(0, 360) for _ in range(3)],
                'scale': [random.uniform(0.5, 2.0) for _ in range(3)]
            }
            
            self.add_object(obj_id, obj_data)
            object_count += 1
            
            # Signal render event
            self.render_event.set()
            
            time.sleep(0.2)
        
        # Remove some objects
        for i in range(5):
            obj_id = f"object_{i}"
            self.remove_object(obj_id)
            time.sleep(0.1)

def demonstrate_synchronization():
    """Demonstrate synchronization concepts"""
    print("=== Synchronization Examples ===\n")
    
    scene = Synchronized3DScene()
    
    print("1. Thread-safe scene management:")
    # Start render workers
    render_threads = []
    for i in range(3):
        thread = threading.Thread(target=scene.render_worker, args=(i,))
        render_threads.append(thread)
        thread.start()
    
    # Start scene manager
    manager_thread = threading.Thread(target=scene.scene_manager)
    manager_thread.start()
    
    # Wait for completion
    manager_thread.join()
    
    # Stop render workers
    scene.render_event.clear()
    for thread in render_threads:
        thread.join()
    
    print(f"Final scene objects: {len(scene.get_scene_snapshot())}\n")
    
    print("2. Resource pool management:")
    # Demonstrate resource pool with semaphore
    def resource_user(user_id: int, semaphore: Semaphore):
        for i in range(3):
            with semaphore:
                print(f"👤 User {user_id}: Using resource (iteration {i})")
                time.sleep(0.1)  # Simulate resource usage
                print(f"✅ User {user_id}: Released resource")
    
    # Create multiple users
    users = []
    semaphore = Semaphore(3)  # Only 3 resources available
    
    for i in range(6):
        thread = threading.Thread(target=resource_user, args=(i, semaphore))
        users.append(thread)
        thread.start()
    
    for thread in users:
        thread.join()
    print()

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

def performance_comparison():
    """Compare different concurrency approaches"""
    print("=== Performance Comparison ===\n")
    
    def cpu_bound_task(n: int) -> int:
        """CPU-bound task"""
        result = 0
        for i in range(n):
            result += math.sqrt(i) * math.sin(i)
        return result
    
    def io_bound_task(duration: float) -> str:
        """I/O-bound task"""
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    # Test data
    cpu_tasks = [1000000, 2000000, 3000000, 4000000]
    io_tasks = [0.1, 0.2, 0.3, 0.4]
    
    print("1. CPU-bound tasks comparison:")
    
    # Sequential execution
    start_time = time.time()
    sequential_results = [cpu_bound_task(n) for n in cpu_tasks]
    sequential_time = time.time() - start_time
    
    # Threading
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threading_results = list(executor.map(cpu_bound_task, cpu_tasks))
    threading_time = time.time() - start_time
    
    # Multiprocessing
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        multiprocessing_results = list(executor.map(cpu_bound_task, cpu_tasks))
    multiprocessing_time = time.time() - start_time
    
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Threading: {threading_time:.4f}s (speedup: {sequential_time/threading_time:.2f}x)")
    print(f"Multiprocessing: {multiprocessing_time:.4f}s (speedup: {sequential_time/multiprocessing_time:.2f}x)\n")
    
    print("2. I/O-bound tasks comparison:")
    
    # Sequential execution
    start_time = time.time()
    sequential_results = [io_bound_task(d) for d in io_tasks]
    sequential_time = time.time() - start_time
    
    # Threading
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threading_results = list(executor.map(io_bound_task, io_tasks))
    threading_time = time.time() - start_time
    
    # Async/await
    async def async_io_tasks():
        tasks = [io_bound_task(d) for d in io_tasks]
        return await asyncio.gather(*tasks)
    
    start_time = time.time()
    async_results = asyncio.run(async_io_tasks())
    async_time = time.time() - start_time
    
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Threading: {threading_time:.4f}s (speedup: {sequential_time/threading_time:.2f}x)")
    print(f"Async/Await: {async_time:.4f}s (speedup: {sequential_time/async_time:.2f}x)\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate concurrency"""
    print("=== Concurrency Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all concurrency concepts
    demonstrate_threading()
    demonstrate_multiprocessing()
    asyncio.run(demonstrate_asyncio())
    demonstrate_parallel_processing()
    demonstrate_synchronization()
    performance_comparison()
    
    print("="*60)
    print("Concurrency demo completed successfully!")
    print("\nKey concepts demonstrated:")
    print("✓ Threading: Thread-safe operations, synchronization primitives")
    print("✓ Multiprocessing: CPU-intensive tasks, process pools")
    print("✓ Async/Await: I/O-bound operations, concurrent execution")
    print("✓ Parallel Processing: Data parallelism, pipeline processing")
    print("✓ Synchronization: Locks, events, conditions, semaphores")
    print("✓ Performance: Comparison of different concurrency approaches")
    
    print("\nBest practices:")
    print("• Use threading for I/O-bound tasks")
    print("• Use multiprocessing for CPU-bound tasks")
    print("• Use async/await for high-concurrency I/O")
    print("• Always use proper synchronization for shared resources")
    print("• Consider the GIL when choosing between threading and multiprocessing")

if __name__ == "__main__":
    main()
