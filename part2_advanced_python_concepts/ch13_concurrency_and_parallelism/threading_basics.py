"""
Chapter 13: Concurrency and Parallelism - Threading Basics
==========================================================

This module demonstrates basic threading concepts for 3D graphics applications,
including thread creation, synchronization, and resource management.

Key Concepts:
- Thread creation and management
- Thread synchronization (locks, semaphores)
- Resource sharing and thread safety
- Thread communication and coordination
- Performance considerations for 3D graphics
"""

import threading
import time
import queue
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math


class ThreadStatus(Enum):
    """Status of a thread."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ThreadInfo:
    """Information about a thread."""
    name: str
    thread: threading.Thread
    status: ThreadStatus = ThreadStatus.IDLE
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def get_duration(self) -> float:
        """Get thread execution duration."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class ThreadSafeCounter:
    """Thread-safe counter for 3D graphics operations."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment the counter safely."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement the counter safely."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get_value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> int:
        """Reset counter to zero."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class RenderQueue:
    """Thread-safe queue for 3D rendering tasks."""
    
    def __init__(self, max_size: int = 100):
        self.queue = queue.Queue(maxsize=max_size)
        self.completed_tasks = ThreadSafeCounter()
        self.failed_tasks = ThreadSafeCounter()
    
    def add_task(self, task: Dict[str, Any]) -> bool:
        """Add a rendering task to the queue."""
        try:
            self.queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            print(f"  ⚠️  Render queue is full, task dropped: {task.get('name', 'unknown')}")
            return False
    
    def get_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from the queue."""
        try:
            return self.queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def mark_completed(self):
        """Mark a task as completed."""
        self.completed_tasks.increment()
    
    def mark_failed(self):
        """Mark a task as failed."""
        self.failed_tasks.increment()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "queue_size": self.queue.qsize(),
            "completed": self.completed_tasks.get_value(),
            "failed": self.failed_tasks.get_value(),
            "total": self.completed_tasks.get_value() + self.failed_tasks.get_value()
        }


class ThreadPool:
    """Simple thread pool for 3D graphics operations."""
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.threads: List[ThreadInfo] = []
        self.running = False
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.lock = threading.Lock()
    
    def start(self):
        """Start the thread pool."""
        if self.running:
            return
        
        self.running = True
        print(f"  🚀 Starting thread pool with {self.num_threads} threads")
        
        for i in range(self.num_threads):
            thread_name = f"Worker-{i+1}"
            thread = threading.Thread(target=self._worker, name=thread_name, daemon=True)
            thread_info = ThreadInfo(name=thread_name, thread=thread)
            self.threads.append(thread_info)
            thread.start()
    
    def stop(self):
        """Stop the thread pool."""
        if not self.running:
            return
        
        print("  🛑 Stopping thread pool...")
        self.running = False
        
        # Wait for all threads to complete
        for thread_info in self.threads:
            thread_info.thread.join(timeout=2.0)
            if thread_info.thread.is_alive():
                print(f"  ⚠️  Thread {thread_info.name} did not stop gracefully")
        
        self.threads.clear()
        print("  ✅ Thread pool stopped")
    
    def submit_task(self, task_func: Callable, *args, **kwargs) -> bool:
        """Submit a task to the thread pool."""
        if not self.running:
            return False
        
        task = {
            "func": task_func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": time.time()
        }
        
        try:
            self.task_queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            print("  ⚠️  Task queue is full")
            return False
    
    def get_result(self, timeout: float = 1.0) -> Optional[Any]:
        """Get a result from the thread pool."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _worker(self):
        """Worker thread function."""
        thread_name = threading.current_thread().name
        thread_info = next((t for t in self.threads if t.name == thread_name), None)
        
        if thread_info:
            thread_info.status = ThreadStatus.RUNNING
            thread_info.start_time = time.time()
        
        while self.running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=0.1)
                
                if thread_info:
                    thread_info.status = ThreadStatus.RUNNING
                
                # Execute task
                func = task["func"]
                args = task.get("args", ())
                kwargs = task.get("kwargs", {})
                
                result = func(*args, **kwargs)
                
                # Put result in result queue
                self.result_queue.put({
                    "result": result,
                    "worker": thread_name,
                    "completed_at": time.time(),
                    "submitted_at": task.get("submitted_at", 0)
                })
                
                if thread_info:
                    thread_info.status = ThreadStatus.IDLE
                
            except queue.Empty:
                if thread_info:
                    thread_info.status = ThreadStatus.WAITING
                continue
            except Exception as e:
                if thread_info:
                    thread_info.status = ThreadStatus.ERROR
                    thread_info.error_message = str(e)
                print(f"  ❌ Worker {thread_name} encountered error: {e}")
        
        if thread_info:
            thread_info.status = ThreadStatus.COMPLETED
            thread_info.end_time = time.time()


class SceneRenderer:
    """Thread-safe 3D scene renderer."""
    
    def __init__(self, num_render_threads: int = 2):
        self.render_queue = RenderQueue()
        self.thread_pool = ThreadPool(num_render_threads)
        self.scene_objects: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.rendering = False
    
    def add_object(self, obj: Dict[str, Any]):
        """Add an object to the scene."""
        with self.lock:
            self.scene_objects.append(obj)
            print(f"  📦 Added object: {obj.get('name', 'unnamed')}")
    
    def remove_object(self, obj_name: str):
        """Remove an object from the scene."""
        with self.lock:
            self.scene_objects = [obj for obj in self.scene_objects 
                                if obj.get('name') != obj_name]
            print(f"  🗑️  Removed object: {obj_name}")
    
    def start_rendering(self):
        """Start the rendering process."""
        if self.rendering:
            return
        
        self.rendering = True
        self.thread_pool.start()
        print("  🎨 Started scene rendering")
    
    def stop_rendering(self):
        """Stop the rendering process."""
        if not self.rendering:
            return
        
        self.rendering = False
        self.thread_pool.stop()
        print("  🛑 Stopped scene rendering")
    
    def render_frame(self, frame_number: int):
        """Render a single frame."""
        if not self.rendering:
            return False
        
        # Create rendering tasks for each object
        with self.lock:
            objects_to_render = self.scene_objects.copy()
        
        for obj in objects_to_render:
            task = {
                "name": f"render_{obj.get('name', 'object')}_{frame_number}",
                "object": obj,
                "frame": frame_number,
                "priority": obj.get('priority', 1)
            }
            
            # Submit rendering task
            self.thread_pool.submit_task(self._render_object, task)
        
        return True
    
    def _render_object(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single object (simulated)."""
        obj = task["object"]
        frame = task["frame"]
        
        # Simulate rendering time
        render_time = random.uniform(0.01, 0.1)
        time.sleep(render_time)
        
        # Simulate rendering result
        result = {
            "object_name": obj.get('name', 'unknown'),
            "frame": frame,
            "render_time": render_time,
            "vertices_rendered": obj.get('vertex_count', 100),
            "textures_loaded": obj.get('texture_count', 1)
        }
        
        print(f"  🎨 Rendered {obj.get('name', 'object')} (frame {frame}) in {render_time:.3f}s")
        return result
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        pool_stats = self.thread_pool.get_result()
        queue_stats = self.render_queue.get_stats()
        
        return {
            "rendering": self.rendering,
            "scene_objects": len(self.scene_objects),
            "queue_stats": queue_stats,
            "pool_stats": pool_stats
        }


class ThreadMonitor:
    """Monitor and manage threads in 3D graphics applications."""
    
    def __init__(self):
        self.threads: Dict[str, ThreadInfo] = {}
        self.lock = threading.Lock()
    
    def register_thread(self, name: str, thread: threading.Thread) -> ThreadInfo:
        """Register a thread for monitoring."""
        with self.lock:
            thread_info = ThreadInfo(name=name, thread=thread)
            self.threads[name] = thread_info
            print(f"  📊 Registered thread: {name}")
            return thread_info
    
    def update_status(self, name: str, status: ThreadStatus, error_message: str = None):
        """Update thread status."""
        with self.lock:
            if name in self.threads:
                thread_info = self.threads[name]
                thread_info.status = status
                if error_message:
                    thread_info.error_message = error_message
    
    def get_thread_info(self, name: str) -> Optional[ThreadInfo]:
        """Get information about a specific thread."""
        with self.lock:
            return self.threads.get(name)
    
    def get_all_threads(self) -> List[ThreadInfo]:
        """Get all registered threads."""
        with self.lock:
            return list(self.threads.values())
    
    def get_active_threads(self) -> List[ThreadInfo]:
        """Get all active threads."""
        with self.lock:
            return [t for t in self.threads.values() 
                   if t.status in [ThreadStatus.RUNNING, ThreadStatus.WAITING]]
    
    def get_thread_stats(self) -> Dict[str, Any]:
        """Get thread statistics."""
        with self.lock:
            total_threads = len(self.threads)
            active_threads = len([t for t in self.threads.values() 
                                if t.status in [ThreadStatus.RUNNING, ThreadStatus.WAITING]])
            completed_threads = len([t for t in self.threads.values() 
                                   if t.status == ThreadStatus.COMPLETED])
            error_threads = len([t for t in self.threads.values() 
                               if t.status == ThreadStatus.ERROR])
            
            return {
                "total": total_threads,
                "active": active_threads,
                "completed": completed_threads,
                "error": error_threads,
                "idle": total_threads - active_threads - completed_threads - error_threads
            }


# Example Usage and Demonstration
def demonstrate_threading_basics():
    """Demonstrates threading basics for 3D graphics."""
    print("=== Threading Basics for 3D Graphics ===\n")
    
    # Create thread monitor
    monitor = ThreadMonitor()
    
    # Create scene renderer
    renderer = SceneRenderer(num_render_threads=3)
    
    # Add some 3D objects to the scene
    objects = [
        {"name": "cube", "vertex_count": 24, "texture_count": 1, "priority": 1},
        {"name": "sphere", "vertex_count": 100, "texture_count": 2, "priority": 2},
        {"name": "cylinder", "vertex_count": 50, "texture_count": 1, "priority": 1},
        {"name": "pyramid", "vertex_count": 16, "texture_count": 1, "priority": 3}
    ]
    
    for obj in objects:
        renderer.add_object(obj)
    
    # Start rendering
    renderer.start_rendering()
    
    # Render several frames
    print("\n=== Rendering Frames ===")
    for frame in range(1, 4):
        print(f"\nRendering frame {frame}:")
        renderer.render_frame(frame)
        
        # Wait a bit for rendering to complete
        time.sleep(0.5)
        
        # Get and display stats
        stats = renderer.get_render_stats()
        print(f"  📊 Frame {frame} stats: {stats['queue_stats']}")
    
    # Monitor threads
    print("\n=== Thread Monitoring ===")
    thread_stats = monitor.get_thread_stats()
    print(f"Thread statistics: {thread_stats}")
    
    # Stop rendering
    renderer.stop_rendering()
    
    # Final statistics
    print("\n=== Final Statistics ===")
    final_stats = renderer.get_render_stats()
    print(f"Final render stats: {final_stats['queue_stats']}")


def demonstrate_thread_safety():
    """Demonstrates thread safety concepts."""
    print("\n=== Thread Safety Demonstration ===\n")
    
    # Create thread-safe counter
    counter = ThreadSafeCounter(0)
    
    def increment_worker(worker_id: int, iterations: int):
        """Worker function that increments counter."""
        for i in range(iterations):
            value = counter.increment(1)
            if i % 100 == 0:
                print(f"  Worker {worker_id}: counter = {value}")
            time.sleep(0.001)  # Small delay to simulate work
    
    # Create multiple threads
    threads = []
    num_threads = 4
    iterations_per_thread = 1000
    
    print(f"Starting {num_threads} threads, each incrementing {iterations_per_thread} times...")
    
    for i in range(num_threads):
        thread = threading.Thread(target=increment_worker, args=(i+1, iterations_per_thread))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check final value
    final_value = counter.get_value()
    expected_value = num_threads * iterations_per_thread
    
    print(f"\nFinal counter value: {final_value}")
    print(f"Expected value: {expected_value}")
    print(f"Thread safety test: {'✅ PASSED' if final_value == expected_value else '❌ FAILED'}")


if __name__ == "__main__":
    demonstrate_threading_basics()
    demonstrate_thread_safety()
