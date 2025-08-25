#!/usr/bin/env python3
"""
Chapter 15: Advanced 3D Graphics Libraries and Tools
Performance Profiling Tool

Creates a comprehensive profiling system for analyzing and optimizing
3D graphics applications with detailed performance metrics.
"""

import time
import psutil
import threading
import queue
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

class ProfilerEvent(Enum):
    """Types of profiling events"""
    FRAME_START = "frame_start"
    FRAME_END = "frame_end"
    RENDER_START = "render_start"
    RENDER_END = "render_end"
    SHADER_COMPILE = "shader_compile"
    TEXTURE_LOAD = "texture_load"
    BUFFER_UPDATE = "buffer_update"
    DRAW_CALL = "draw_call"
    MEMORY_ALLOC = "memory_alloc"
    MEMORY_FREE = "memory_free"
    GPU_SYNC = "gpu_sync"

@dataclass
class ProfilerSample:
    """A single profiling sample"""
    timestamp: float
    event_type: ProfilerEvent
    duration: float = 0.0
    metadata: Dict[str, Any] = None
    thread_id: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    fps: float = 0.0
    frame_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    draw_calls: int = 0
    triangles: int = 0
    vertices: int = 0
    shader_switches: int = 0
    texture_binds: int = 0
    buffer_updates: int = 0

class PerformanceProfiler:
    """Main performance profiler class"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        
        # Event tracking
        self.active_events: Dict[str, float] = {}
        self.event_counts: Dict[ProfilerEvent, int] = defaultdict(int)
        
        # Performance monitoring
        self.frame_times: deque = deque(maxlen=60)
        self.fps_history: deque = deque(maxlen=60)
        
        # Threading
        self.sample_queue = queue.Queue()
        self.profiling_active = False
        self.profiler_thread = None
        
        # System information
        self.system_info = self.get_system_info()
        
        # Performance thresholds
        self.thresholds = {
            'fps_min': 30.0,
            'frame_time_max': 33.33,  # ms
            'cpu_usage_max': 80.0,
            'memory_usage_max': 85.0,
            'gpu_usage_max': 90.0
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def start_profiling(self):
        """Start the profiling system"""
        if not self.profiling_active:
            self.profiling_active = True
            self.profiler_thread = threading.Thread(target=self._profiler_loop, daemon=True)
            self.profiler_thread.start()
            print("Performance profiling started")
    
    def stop_profiling(self):
        """Stop the profiling system"""
        self.profiling_active = False
        if self.profiler_thread:
            self.profiler_thread.join()
        print("Performance profiling stopped")
    
    def _profiler_loop(self):
        """Main profiler loop running in separate thread"""
        while self.profiling_active:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent
                
                # Create system sample
                sample = ProfilerSample(
                    timestamp=time.time(),
                    event_type=ProfilerEvent.FRAME_END,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    thread_id=threading.get_ident()
                )
                
                self.sample_queue.put(sample)
                
            except Exception as e:
                print(f"Profiler error: {e}")
    
    def begin_event(self, event_type: ProfilerEvent, metadata: Dict[str, Any] = None):
        """Begin a profiling event"""
        event_id = f"{event_type.value}_{time.time()}"
        self.active_events[event_id] = time.time()
        
        sample = ProfilerSample(
            timestamp=time.time(),
            event_type=event_type,
            metadata=metadata or {},
            thread_id=threading.get_ident(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent
        )
        
        self.samples.append(sample)
        self.event_counts[event_type] += 1
        
        return event_id
    
    def end_event(self, event_id: str, metadata: Dict[str, Any] = None):
        """End a profiling event"""
        if event_id in self.active_events:
            start_time = self.active_events.pop(event_id)
            duration = time.time() - start_time
            
            sample = ProfilerSample(
                timestamp=time.time(),
                event_type=ProfilerEvent.FRAME_END,  # Generic end event
                duration=duration,
                metadata=metadata or {},
                thread_id=threading.get_ident(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent
            )
            
            self.samples.append(sample)
    
    def record_frame(self, frame_time: float, draw_calls: int = 0, 
                    triangles: int = 0, vertices: int = 0):
        """Record frame performance data"""
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.fps_history.append(fps)
        
        # Update current metrics
        self.current_metrics.fps = fps
        self.current_metrics.frame_time = frame_time * 1000  # Convert to ms
        self.current_metrics.draw_calls = draw_calls
        self.current_metrics.triangles = triangles
        self.current_metrics.vertices = vertices
        self.current_metrics.cpu_usage = psutil.cpu_percent()
        self.current_metrics.memory_usage = psutil.virtual_memory().percent
        
        # Store metrics history
        self.metrics_history.append(self.current_metrics)
        
        # Check performance thresholds
        self.check_performance_thresholds()
    
    def check_performance_thresholds(self):
        """Check if performance is below thresholds"""
        warnings = []
        
        if self.current_metrics.fps < self.thresholds['fps_min']:
            warnings.append(f"Low FPS: {self.current_metrics.fps:.1f}")
        
        if self.current_metrics.frame_time > self.thresholds['frame_time_max']:
            warnings.append(f"High frame time: {self.current_metrics.frame_time:.1f}ms")
        
        if self.current_metrics.cpu_usage > self.thresholds['cpu_usage_max']:
            warnings.append(f"High CPU usage: {self.current_metrics.cpu_usage:.1f}%")
        
        if self.current_metrics.memory_usage > self.thresholds['memory_usage_max']:
            warnings.append(f"High memory usage: {self.current_metrics.memory_usage:.1f}%")
        
        if warnings:
            print(f"Performance warnings: {', '.join(warnings)}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        # Calculate statistics
        fps_values = [m.fps for m in self.metrics_history]
        frame_times = [m.frame_time for m in self.metrics_history]
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "summary": {
                "total_frames": len(self.metrics_history),
                "avg_fps": np.mean(fps_values),
                "min_fps": np.min(fps_values),
                "max_fps": np.max(fps_values),
                "avg_frame_time": np.mean(frame_times),
                "avg_cpu_usage": np.mean(cpu_usage),
                "avg_memory_usage": np.mean(memory_usage),
                "total_draw_calls": sum(m.draw_calls for m in self.metrics_history),
                "total_triangles": sum(m.triangles for m in self.metrics_history),
                "total_vertices": sum(m.vertices for m in self.metrics_history)
            },
            "event_counts": {event.value: count for event, count in self.event_counts.items()},
            "performance_thresholds": self.thresholds,
            "current_metrics": asdict(self.current_metrics)
        }
        
        return report
    
    def save_report(self, filename: str = None):
        """Save performance report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = self.get_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {filename}")
        return filename
    
    def create_visualizations(self, save_plots: bool = True):
        """Create performance visualization plots"""
        if not self.metrics_history:
            print("No performance data available for visualization")
            return
        
        # Prepare data
        timestamps = list(range(len(self.metrics_history)))
        fps_values = [m.fps for m in self.metrics_history]
        frame_times = [m.frame_time for m in self.metrics_history]
        cpu_usage = [m.cpu_usage for m in self.metrics_history]
        memory_usage = [m.memory_usage for m in self.metrics_history]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Profiling Results', fontsize=16)
        
        # FPS over time
        ax1.plot(timestamps, fps_values, 'b-', linewidth=2)
        ax1.axhline(y=self.thresholds['fps_min'], color='r', linestyle='--', label=f"Min FPS ({self.thresholds['fps_min']})")
        ax1.set_title('Frame Rate Over Time')
        ax1.set_ylabel('FPS')
        ax1.set_xlabel('Frame')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frame time over time
        ax2.plot(timestamps, frame_times, 'g-', linewidth=2)
        ax2.axhline(y=self.thresholds['frame_time_max'], color='r', linestyle='--', label=f"Max Frame Time ({self.thresholds['frame_time_max']}ms)")
        ax2.set_title('Frame Time Over Time')
        ax2.set_ylabel('Frame Time (ms)')
        ax2.set_xlabel('Frame')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # CPU usage over time
        ax3.plot(timestamps, cpu_usage, 'orange', linewidth=2)
        ax3.axhline(y=self.thresholds['cpu_usage_max'], color='r', linestyle='--', label=f"Max CPU ({self.thresholds['cpu_usage_max']}%)")
        ax3.set_title('CPU Usage Over Time')
        ax3.set_ylabel('CPU Usage (%)')
        ax3.set_xlabel('Frame')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Memory usage over time
        ax4.plot(timestamps, memory_usage, 'purple', linewidth=2)
        ax4.axhline(y=self.thresholds['memory_usage_max'], color='r', linestyle='--', label=f"Max Memory ({self.thresholds['memory_usage_max']}%)")
        ax4.set_title('Memory Usage Over Time')
        ax4.set_ylabel('Memory Usage (%)')
        ax4.set_xlabel('Frame')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_plots_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to {filename}")
        
        plt.show()

class PerformanceOptimizer:
    """Performance optimization recommendations"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.recommendations = []
    
    def analyze_performance(self) -> List[str]:
        """Analyze performance and generate recommendations"""
        self.recommendations = []
        
        if not self.profiler.metrics_history:
            return ["No performance data available for analysis"]
        
        # Analyze FPS
        avg_fps = np.mean([m.fps for m in self.profiler.metrics_history])
        if avg_fps < 30:
            self.recommendations.append("Low FPS detected. Consider:")
            self.recommendations.append("  - Reducing polygon count")
            self.recommendations.append("  - Optimizing shaders")
            self.recommendations.append("  - Using LOD (Level of Detail)")
            self.recommendations.append("  - Implementing frustum culling")
        
        # Analyze frame time
        avg_frame_time = np.mean([m.frame_time for m in self.profiler.metrics_history])
        if avg_frame_time > 33.33:
            self.recommendations.append("High frame time detected. Consider:")
            self.recommendations.append("  - Reducing draw calls")
            self.recommendations.append("  - Using instanced rendering")
            self.recommendations.append("  - Optimizing texture usage")
            self.recommendations.append("  - Implementing occlusion culling")
        
        # Analyze CPU usage
        avg_cpu = np.mean([m.cpu_usage for m in self.profiler.metrics_history])
        if avg_cpu > 80:
            self.recommendations.append("High CPU usage detected. Consider:")
            self.recommendations.append("  - Moving work to GPU")
            self.recommendations.append("  - Using multi-threading")
            self.recommendations.append("  - Optimizing algorithms")
            self.recommendations.append("  - Reducing update frequency")
        
        # Analyze memory usage
        avg_memory = np.mean([m.memory_usage for m in self.profiler.metrics_history])
        if avg_memory > 85:
            self.recommendations.append("High memory usage detected. Consider:")
            self.recommendations.append("  - Implementing texture streaming")
            self.recommendations.append("  - Using object pooling")
            self.recommendations.append("  - Reducing texture resolution")
            self.recommendations.append("  - Implementing garbage collection")
        
        # Analyze draw calls
        total_draw_calls = sum(m.draw_calls for m in self.profiler.metrics_history)
        avg_draw_calls = total_draw_calls / len(self.profiler.metrics_history)
        if avg_draw_calls > 1000:
            self.recommendations.append("High draw call count detected. Consider:")
            self.recommendations.append("  - Using batch rendering")
            self.recommendations.append("  - Combining meshes")
            self.recommendations.append("  - Using instanced rendering")
            self.recommendations.append("  - Implementing culling")
        
        if not self.recommendations:
            self.recommendations.append("Performance appears to be good!")
            self.recommendations.append("Consider monitoring for longer periods")
        
        return self.recommendations
    
    def get_optimization_score(self) -> float:
        """Calculate an optimization score (0-100, higher is better)"""
        if not self.profiler.metrics_history:
            return 0.0
        
        score = 100.0
        
        # FPS penalty
        avg_fps = np.mean([m.fps for m in self.profiler.metrics_history])
        if avg_fps < 60:
            score -= (60 - avg_fps) * 2
        
        # Frame time penalty
        avg_frame_time = np.mean([m.frame_time for m in self.profiler.metrics_history])
        if avg_frame_time > 16.67:  # 60 FPS target
            score -= (avg_frame_time - 16.67) * 2
        
        # CPU usage penalty
        avg_cpu = np.mean([m.cpu_usage for m in self.profiler.metrics_history])
        if avg_cpu > 50:
            score -= (avg_cpu - 50) * 0.5
        
        # Memory usage penalty
        avg_memory = np.mean([m.memory_usage for m in self.profiler.metrics_history])
        if avg_memory > 70:
            score -= (avg_memory - 70) * 0.5
        
        return max(0.0, min(100.0, score))

class PerformanceProfilingTool:
    """Main profiling tool interface"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.optimizer = PerformanceOptimizer(self.profiler)
        self.running = False
    
    def start(self):
        """Start the profiling tool"""
        self.profiler.start_profiling()
        self.running = True
        print("Performance Profiling Tool started")
        print("Press Ctrl+C to stop and generate report")
    
    def stop(self):
        """Stop the profiling tool and generate report"""
        if self.running:
            self.profiler.stop_profiling()
            self.running = False
            
            # Generate report
            report = self.profiler.get_performance_report()
            print("\n" + "="*50)
            print("PERFORMANCE REPORT")
            print("="*50)
            
            summary = report['summary']
            print(f"Total Frames: {summary['total_frames']}")
            print(f"Average FPS: {summary['avg_fps']:.1f}")
            print(f"Min/Max FPS: {summary['min_fps']:.1f}/{summary['max_fps']:.1f}")
            print(f"Average Frame Time: {summary['avg_frame_time']:.1f}ms")
            print(f"Average CPU Usage: {summary['avg_cpu_usage']:.1f}%")
            print(f"Average Memory Usage: {summary['avg_memory_usage']:.1f}%")
            print(f"Total Draw Calls: {summary['total_draw_calls']}")
            print(f"Total Triangles: {summary['total_triangles']}")
            print(f"Total Vertices: {summary['total_vertices']}")
            
            # Save report
            self.profiler.save_report()
            
            # Generate visualizations
            self.profiler.create_visualizations()
            
            # Get optimization recommendations
            print("\n" + "="*50)
            print("OPTIMIZATION RECOMMENDATIONS")
            print("="*50)
            
            recommendations = self.optimizer.analyze_performance()
            for rec in recommendations:
                print(rec)
            
            # Show optimization score
            score = self.optimizer.get_optimization_score()
            print(f"\nOptimization Score: {score:.1f}/100")
            
            if score >= 80:
                print("Excellent performance!")
            elif score >= 60:
                print("Good performance with room for improvement")
            elif score >= 40:
                print("Moderate performance, optimization recommended")
            else:
                print("Poor performance, significant optimization needed")
    
    def simulate_performance_data(self, duration: float = 10.0):
        """Simulate performance data for testing"""
        print(f"Simulating performance data for {duration} seconds...")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            # Simulate frame rendering
            frame_time = np.random.uniform(0.01, 0.05)  # 20-100 FPS
            time.sleep(frame_time)
            
            # Record frame with simulated data
            draw_calls = np.random.randint(100, 1000)
            triangles = np.random.randint(1000, 10000)
            vertices = np.random.randint(5000, 50000)
            
            self.profiler.record_frame(frame_time, draw_calls, triangles, vertices)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Simulated {frame_count} frames...")
        
        print(f"Simulation complete: {frame_count} frames recorded")

def main():
    print("=== Performance Profiling Tool ===\n")
    print("This tool provides comprehensive performance analysis for 3D graphics applications.")
    print("Features:")
    print("  • Real-time performance monitoring")
    print("  • Detailed metrics collection")
    print("  • Performance visualization")
    print("  • Optimization recommendations")
    print("  • Automated reporting")
    
    try:
        # Create profiling tool
        profiling_tool = PerformanceProfilingTool()
        
        # Start profiling
        profiling_tool.start()
        
        # Simulate some performance data
        profiling_tool.simulate_performance_data(5.0)
        
        # Stop and generate report
        profiling_tool.stop()
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        if 'profiling_tool' in locals():
            profiling_tool.stop()
    except Exception as e:
        print(f"✗ Performance profiling tool failed: {e}")

if __name__ == "__main__":
    main()
