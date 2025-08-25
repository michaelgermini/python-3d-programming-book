"""
Chapter 14: Testing and Debugging Python Code - Debugging Tools
==============================================================

This module demonstrates debugging tools and techniques for 3D graphics
applications, including logging, profiling, and error handling.

Key Concepts:
- Debugging fundamentals and strategies
- Logging and error tracking
- Performance profiling and optimization
- Memory debugging and leak detection
- Error handling and recovery
- Debugging 3D graphics applications
"""

import logging
import time
import traceback
import sys
import gc
import psutil
import os
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from contextlib import contextmanager
from functools import wraps


class DebugLevel(Enum):
    """Debug levels for logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DebugInfo:
    """Information about a debug session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    memory_usage: List[float] = field(default_factory=list)


class DebugLogger:
    """Advanced logging system for 3D graphics debugging."""
    
    def __init__(self, name: str = "3DGraphics", level: DebugLevel = DebugLevel.INFO):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Create handlers
        self._setup_handlers()
        
        # Debug session tracking
        self.debug_sessions: Dict[str, DebugInfo] = {}
        self.current_session: Optional[str] = None
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(f"{self.name}_debug.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def start_debug_session(self, session_id: str) -> str:
        """Start a new debug session."""
        if self.current_session:
            self.end_debug_session()
        
        self.current_session = session_id
        debug_info = DebugInfo(session_id=session_id, start_time=time.time())
        self.debug_sessions[session_id] = debug_info
        
        self.logger.info(f"Started debug session: {session_id}")
        return session_id
    
    def end_debug_session(self):
        """End the current debug session."""
        if not self.current_session:
            return
        
        session = self.debug_sessions[self.current_session]
        session.end_time = time.time()
        duration = session.end_time - session.start_time
        
        self.logger.info(f"Ended debug session: {self.current_session} (duration: {duration:.3f}s)")
        self.logger.info(f"Session summary - Errors: {len(session.errors)}, Warnings: {len(session.warnings)}")
        
        self.current_session = None
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error with optional exception details."""
        if exception:
            error_msg = f"{message}: {str(exception)}"
            self.logger.error(error_msg, exc_info=True)
        else:
            self.logger.error(message)
        
        if self.current_session:
            self.debug_sessions[self.current_session].errors.append(message)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
        
        if self.current_session:
            self.debug_sessions[self.current_session].warnings.append(message)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a debug session."""
        if session_id not in self.debug_sessions:
            return None
        
        session = self.debug_sessions[session_id]
        duration = (session.end_time or time.time()) - session.start_time
        
        return {
            "session_id": session_id,
            "duration": duration,
            "error_count": len(session.errors),
            "warning_count": len(session.warnings),
            "performance_metrics": session.performance_metrics,
            "memory_usage": session.memory_usage
        }


class PerformanceProfiler:
    """Performance profiling tool for 3D graphics applications."""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.active_profiles: Dict[str, float] = {}
    
    def start_profile(self, profile_name: str):
        """Start profiling a section of code."""
        if profile_name in self.active_profiles:
            self.logger.log_warning(f"Profile {profile_name} already active")
            return
        
        self.active_profiles[profile_name] = time.time()
        self.logger.log_debug(f"Started profiling: {profile_name}")
    
    def end_profile(self, profile_name: str) -> float:
        """End profiling and return duration."""
        if profile_name not in self.active_profiles:
            self.logger.log_warning(f"Profile {profile_name} not found")
            return 0.0
        
        start_time = self.active_profiles.pop(profile_name)
        duration = time.time() - start_time
        
        # Store profile data
        if profile_name not in self.profiles:
            self.profiles[profile_name] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "avg_time": 0.0
            }
        
        profile = self.profiles[profile_name]
        profile["count"] += 1
        profile["total_time"] += duration
        profile["min_time"] = min(profile["min_time"], duration)
        profile["max_time"] = max(profile["max_time"], duration)
        profile["avg_time"] = profile["total_time"] / profile["count"]
        
        self.logger.log_debug(f"Ended profiling: {profile_name} (duration: {duration:.6f}s)")
        return duration
    
    @contextmanager
    def profile_section(self, profile_name: str):
        """Context manager for profiling code sections."""
        self.start_profile(profile_name)
        try:
            yield
        finally:
            self.end_profile(profile_name)
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        return {
            "profiles": self.profiles,
            "active_profiles": list(self.active_profiles.keys())
        }
    
    def clear_profiles(self):
        """Clear all profile data."""
        self.profiles.clear()
        self.active_profiles.clear()
        self.logger.log_info("Cleared all profile data")


class MemoryDebugger:
    """Memory debugging and leak detection tool."""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.process = psutil.Process()
    
    def take_memory_snapshot(self, label: str = ""):
        """Take a memory usage snapshot."""
        memory_info = self.process.memory_info()
        gc.collect()  # Force garbage collection
        
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": self.process.memory_percent(),
            "objects": len(gc.get_objects())
        }
        
        self.memory_snapshots.append(snapshot)
        self.logger.log_debug(f"Memory snapshot '{label}': {snapshot['rss'] / 1024 / 1024:.2f} MB")
        
        return snapshot
    
    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int) -> Dict[str, Any]:
        """Compare two memory snapshots."""
        if snapshot1_idx >= len(self.memory_snapshots) or snapshot2_idx >= len(self.memory_snapshots):
            return {}
        
        snap1 = self.memory_snapshots[snapshot1_idx]
        snap2 = self.memory_snapshots[snapshot2_idx]
        
        rss_diff = snap2["rss"] - snap1["rss"]
        vms_diff = snap2["vms"] - snap1["vms"]
        objects_diff = snap2["objects"] - snap1["objects"]
        
        comparison = {
            "rss_change_mb": rss_diff / 1024 / 1024,
            "vms_change_mb": vms_diff / 1024 / 1024,
            "objects_change": objects_diff,
            "time_diff": snap2["timestamp"] - snap1["timestamp"]
        }
        
        self.logger.log_info(f"Memory comparison: RSS change: {comparison['rss_change_mb']:.2f} MB")
        
        return comparison
    
    def detect_memory_leak(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        for i in range(1, len(self.memory_snapshots)):
            comparison = self.compare_snapshots(i-1, i)
            
            if comparison.get("rss_change_mb", 0) > threshold_mb:
                leak_info = {
                    "snapshot1": self.memory_snapshots[i-1]["label"],
                    "snapshot2": self.memory_snapshots[i]["label"],
                    "rss_increase_mb": comparison["rss_change_mb"],
                    "objects_increase": comparison["objects_change"]
                }
                leaks.append(leak_info)
                self.logger.log_warning(f"Potential memory leak detected: {leak_info}")
        
        return leaks
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_snapshots:
            return {}
        
        latest = self.memory_snapshots[-1]
        first = self.memory_snapshots[0]
        
        return {
            "current_rss_mb": latest["rss"] / 1024 / 1024,
            "current_vms_mb": latest["vms"] / 1024 / 1024,
            "total_rss_increase_mb": (latest["rss"] - first["rss"]) / 1024 / 1024,
            "total_objects_increase": latest["objects"] - first["objects"],
            "snapshot_count": len(self.memory_snapshots)
        }


class ErrorHandler:
    """Advanced error handling and recovery system."""
    
    def __init__(self, logger: DebugLogger):
        self.logger = logger
        self.error_handlers: Dict[type, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.error_count = 0
        self.recovery_count = 0
    
    def register_error_handler(self, exception_type: type, handler: Callable):
        """Register a custom error handler."""
        self.error_handlers[exception_type] = handler
        self.logger.log_info(f"Registered error handler for {exception_type.__name__}")
    
    def register_recovery_strategy(self, strategy_name: str, strategy: Callable):
        """Register a recovery strategy."""
        self.recovery_strategies[strategy_name] = strategy
        self.logger.log_info(f"Registered recovery strategy: {strategy_name}")
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle an error with registered handlers and recovery strategies."""
        self.error_count += 1
        
        # Log the error
        self.logger.log_error(f"Error in {context}: {str(error)}", error)
        
        # Try to find a specific handler
        error_type = type(error)
        if error_type in self.error_handlers:
            try:
                self.error_handlers[error_type](error, context)
                return True
            except Exception as handler_error:
                self.logger.log_error(f"Error handler failed: {str(handler_error)}")
        
        # Try recovery strategies
        for strategy_name, strategy in self.recovery_strategies.items():
            try:
                if strategy(error, context):
                    self.recovery_count += 1
                    self.logger.log_info(f"Recovery strategy '{strategy_name}' succeeded")
                    return True
            except Exception as recovery_error:
                self.logger.log_error(f"Recovery strategy '{strategy_name}' failed: {str(recovery_error)}")
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        return {
            "total_errors": self.error_count,
            "successful_recoveries": self.recovery_count,
            "recovery_rate": self.recovery_count / max(self.error_count, 1) * 100
        }


class DebugDecorators:
    """Decorators for debugging and profiling."""
    
    def __init__(self, logger: DebugLogger, profiler: PerformanceProfiler):
        self.logger = logger
        self.profiler = profiler
    
    def debug_function(self, func: Callable) -> Callable:
        """Decorator to debug function calls."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            self.logger.log_debug(f"Entering function: {function_name}")
            
            try:
                with self.profiler.profile_section(f"function_{function_name}"):
                    result = func(*args, **kwargs)
                
                self.logger.log_debug(f"Exiting function: {function_name}")
                return result
            except Exception as e:
                self.logger.log_error(f"Error in function {function_name}: {str(e)}", e)
                raise
        
        return wrapper
    
    def performance_monitor(self, threshold_seconds: float = 1.0):
        """Decorator to monitor function performance."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                function_name = func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if duration > threshold_seconds:
                        self.logger.log_warning(
                            f"Function {function_name} took {duration:.3f}s (threshold: {threshold_seconds}s)"
                        )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.log_error(
                        f"Function {function_name} failed after {duration:.3f}s: {str(e)}", e
                    )
                    raise
            
            return wrapper
        return decorator
    
    def memory_monitor(self, func: Callable) -> Callable:
        """Decorator to monitor memory usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            
            # Take snapshot before function call
            memory_before = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                
                # Take snapshot after function call
                memory_after = psutil.Process().memory_info().rss
                memory_diff = memory_after - memory_before
                
                if memory_diff > 1024 * 1024:  # 1 MB threshold
                    self.logger.log_warning(
                        f"Function {function_name} used {memory_diff / 1024 / 1024:.2f} MB of memory"
                    )
                
                return result
            except Exception as e:
                self.logger.log_error(f"Error in function {function_name}: {str(e)}", e)
                raise
        
        return wrapper


# Example 3D Graphics Components for Debugging
class DebuggableRenderer:
    """3D renderer with built-in debugging capabilities."""
    
    def __init__(self):
        self.logger = DebugLogger("Renderer")
        self.profiler = PerformanceProfiler(self.logger)
        self.memory_debugger = MemoryDebugger(self.logger)
        self.error_handler = ErrorHandler(self.logger)
        self.debug_decorators = DebugDecorators(self.logger, self.profiler)
        
        # Register error handlers
        self.error_handler.register_error_handler(
            ValueError, 
            lambda e, ctx: self.logger.log_warning(f"Value error in {ctx}: {e}")
        )
        
        # Register recovery strategies
        self.error_handler.register_recovery_strategy(
            "reset_renderer",
            lambda e, ctx: self._reset_renderer()
        )
        
        self.objects = []
        self.rendering_enabled = True
    
    def _reset_renderer(self) -> bool:
        """Reset the renderer as a recovery strategy."""
        try:
            self.objects.clear()
            self.rendering_enabled = True
            self.logger.log_info("Renderer reset successfully")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to reset renderer: {str(e)}", e)
            return False
    
    @property
    def debug_function(self):
        return self.debug_decorators.debug_function
    
    @property
    def performance_monitor(self):
        return self.debug_decorators.performance_monitor
    
    @property
    def memory_monitor(self):
        return self.debug_decorators.memory_monitor
    
    def start_debug_session(self, session_id: str):
        """Start a debug session."""
        self.logger.start_debug_session(session_id)
        self.memory_debugger.take_memory_snapshot("session_start")
    
    def end_debug_session(self):
        """End the debug session."""
        self.memory_debugger.take_memory_snapshot("session_end")
        self.logger.end_debug_session()
    
    @debug_function
    @performance_monitor(threshold_seconds=0.1)
    @memory_monitor
    def add_object(self, obj: Dict[str, Any]):
        """Add an object to the renderer."""
        if not self.rendering_enabled:
            raise ValueError("Renderer is disabled")
        
        self.objects.append(obj)
        self.logger.log_info(f"Added object: {obj.get('name', 'unknown')}")
    
    @debug_function
    @performance_monitor(threshold_seconds=0.5)
    @memory_monitor
    def render_scene(self) -> List[Dict[str, Any]]:
        """Render the scene with debugging."""
        if not self.rendering_enabled:
            raise ValueError("Renderer is disabled")
        
        with self.profiler.profile_section("render_scene"):
            results = []
            for i, obj in enumerate(self.objects):
                try:
                    with self.profiler.profile_section(f"render_object_{i}"):
                        result = self._render_object(obj)
                        results.append(result)
                except Exception as e:
                    if not self.error_handler.handle_error(e, f"rendering object {i}"):
                        raise
            
            self.logger.log_info(f"Rendered {len(results)} objects")
            return results
    
    def _render_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single object."""
        # Simulate rendering with potential errors
        if random.random() < 0.1:  # 10% chance of error
            raise RuntimeError("Simulated rendering error")
        
        return {
            "object_name": obj.get("name", "unknown"),
            "rendered": True,
            "pixels_processed": random.randint(100, 1000)
        }
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get comprehensive debug summary."""
        return {
            "renderer_stats": {
                "object_count": len(self.objects),
                "rendering_enabled": self.rendering_enabled
            },
            "performance_summary": self.profiler.get_profile_summary(),
            "memory_summary": self.memory_debugger.get_memory_summary(),
            "error_summary": self.error_handler.get_error_summary(),
            "session_summaries": {
                session_id: self.logger.get_session_summary(session_id)
                for session_id in self.logger.debug_sessions
            }
        }


# Example Usage and Demonstration
def demonstrate_debugging_tools():
    """Demonstrate debugging tools and techniques."""
    print("=== Debugging Tools Demonstration ===\n")
    
    # Create debuggable renderer
    renderer = DebuggableRenderer()
    
    # Start debug session
    renderer.start_debug_session("demo_session")
    
    try:
        # Add objects with potential errors
        objects = [
            {"name": "cube", "position": [0, 0, 0]},
            {"name": "sphere", "position": [1, 1, 1]},
            {"name": "cylinder", "position": [2, 2, 2]},
            {"name": "error_object", "position": None},  # This will cause an error
        ]
        
        for obj in objects:
            try:
                renderer.add_object(obj)
            except Exception as e:
                print(f"  ❌ Failed to add object {obj.get('name', 'unknown')}: {e}")
        
        # Render scene multiple times to test performance
        for i in range(5):
            try:
                results = renderer.render_scene()
                print(f"  ✅ Render {i+1}: {len(results)} objects rendered")
            except Exception as e:
                print(f"  ❌ Render {i+1} failed: {e}")
        
        # Test memory leak detection
        for i in range(10):
            # Simulate memory usage
            large_data = [random.random() for _ in range(10000)]
            renderer.render_scene()
            del large_data
        
    finally:
        # End debug session
        renderer.end_debug_session()
    
    # Print debug summary
    print("\n=== Debug Summary ===")
    summary = renderer.get_debug_summary()
    
    print(f"Performance Profiles: {len(summary['performance_summary']['profiles'])}")
    print(f"Memory Usage: {summary['memory_summary'].get('current_rss_mb', 0):.2f} MB")
    print(f"Error Recovery Rate: {summary['error_summary']['recovery_rate']:.1f}%")
    
    # Check for memory leaks
    leaks = renderer.memory_debugger.detect_memory_leak(threshold_mb=5.0)
    if leaks:
        print(f"  ⚠️  Detected {len(leaks)} potential memory leaks")
    else:
        print("  ✅ No memory leaks detected")


def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Error Handling Demonstration ===\n")
    
    logger = DebugLogger("ErrorDemo")
    error_handler = ErrorHandler(logger)
    
    # Register custom error handlers
    def handle_value_error(error: ValueError, context: str):
        logger.log_warning(f"Handling ValueError in {context}: {error}")
        return True
    
    def handle_type_error(error: TypeError, context: str):
        logger.log_warning(f"Handling TypeError in {context}: {error}")
        return True
    
    error_handler.register_error_handler(ValueError, handle_value_error)
    error_handler.register_error_handler(TypeError, handle_type_error)
    
    # Register recovery strategies
    def retry_strategy(error: Exception, context: str) -> bool:
        logger.log_info(f"Retry strategy for {context}")
        return random.random() < 0.5  # 50% success rate
    
    def fallback_strategy(error: Exception, context: str) -> bool:
        logger.log_info(f"Fallback strategy for {context}")
        return True
    
    error_handler.register_recovery_strategy("retry", retry_strategy)
    error_handler.register_recovery_strategy("fallback", fallback_strategy)
    
    # Test error handling
    test_errors = [
        ValueError("Invalid value"),
        TypeError("Invalid type"),
        RuntimeError("Runtime error"),
        KeyError("Missing key")
    ]
    
    for error in test_errors:
        print(f"Testing error: {type(error).__name__}")
        success = error_handler.handle_error(error, "test_context")
        print(f"  {'✅ Handled' if success else '❌ Failed'}")
    
    # Print error summary
    summary = error_handler.get_error_summary()
    print(f"\nError Summary:")
    print(f"  Total Errors: {summary['total_errors']}")
    print(f"  Successful Recoveries: {summary['successful_recoveries']}")
    print(f"  Recovery Rate: {summary['recovery_rate']:.1f}%")


if __name__ == "__main__":
    demonstrate_debugging_tools()
    demonstrate_error_handling()
