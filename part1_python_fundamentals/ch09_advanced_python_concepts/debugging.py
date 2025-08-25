#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Debugging Example

Demonstrates debugging including debugging tools, techniques, logging,
profiling, breakpoints, and debugging strategies for 3D graphics applications.
"""

import sys
import time
import math
import random
import logging
import traceback
import pdb
import cProfile
import pstats
import io
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from functools import wraps
import os

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Debugging"
__description__ = "Debugging for 3D graphics applications"

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# DEBUGGING DECORATORS
# ============================================================================

def debug_function(func):
    """Decorator to add debugging information to functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} with result={result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.6f} seconds")
        return result
    return wrapper

def validate_inputs(func):
    """Decorator to validate function inputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Validating inputs for {func.__name__}")
        # Add input validation logic here
        result = func(*args, **kwargs)
        logger.debug(f"Input validation passed for {func.__name__}")
        return result
    return wrapper

# ============================================================================
# DEBUGGING CLASSES
# ============================================================================

class DebugVector3D:
    """3D vector class with debugging capabilities"""
    
    def __init__(self, x: float, y: float, z: float):
        logger.debug(f"Creating Vector3D: x={x}, y={y}, z={z}")
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'DebugVector3D') -> 'DebugVector3D':
        logger.debug(f"Adding vectors: {self} + {other}")
        result = DebugVector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        logger.debug(f"Result: {result}")
        return result
    
    def __sub__(self, other: 'DebugVector3D') -> 'DebugVector3D':
        logger.debug(f"Subtracting vectors: {self} - {other}")
        result = DebugVector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        logger.debug(f"Result: {result}")
        return result
    
    def magnitude(self) -> float:
        logger.debug(f"Calculating magnitude of {self}")
        result = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        logger.debug(f"Magnitude: {result}")
        return result
    
    def normalize(self) -> 'DebugVector3D':
        logger.debug(f"Normalizing vector: {self}")
        mag = self.magnitude()
        if mag == 0:
            logger.warning("Attempting to normalize zero vector")
            return DebugVector3D(0, 0, 0)
        result = DebugVector3D(self.x / mag, self.y / mag, self.z / mag)
        logger.debug(f"Normalized result: {result}")
        return result
    
    def __str__(self):
        return f"Vector3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

class DebugGraphicsObject:
    """Graphics object with debugging capabilities"""
    
    def __init__(self, name: str, position: DebugVector3D):
        logger.debug(f"Creating GraphicsObject: {name} at {position}")
        self.name = name
        self.position = position
        self.visible = True
        self.debug_info = {}
    
    def move_to(self, new_position: DebugVector3D):
        logger.debug(f"Moving {self.name} from {self.position} to {new_position}")
        old_position = self.position
        self.position = new_position
        self.debug_info['last_move'] = {
            'from': str(old_position),
            'to': str(new_position),
            'timestamp': time.time()
        }
    
    def set_visibility(self, visible: bool):
        logger.debug(f"Setting visibility of {self.name} to {visible}")
        self.visible = visible
        self.debug_info['visibility_changed'] = {
            'new_state': visible,
            'timestamp': time.time()
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'position': str(self.position),
            'visible': self.visible,
            'debug_info': self.debug_info
        }

# ============================================================================
# DEBUGGING TOOLS
# ============================================================================

class Debugger:
    """Custom debugging tool"""
    
    def __init__(self):
        self.breakpoints = set()
        self.watch_variables = {}
        self.debug_history = []
    
    def add_breakpoint(self, line_number: int, condition: str = None):
        """Add a breakpoint at a specific line"""
        self.breakpoints.add((line_number, condition))
        logger.info(f"Added breakpoint at line {line_number}")
    
    def watch_variable(self, name: str, value: Any):
        """Watch a variable for changes"""
        if name not in self.watch_variables:
            self.watch_variables[name] = []
        self.watch_variables[name].append({
            'value': value,
            'timestamp': time.time()
        })
        logger.debug(f"Watching variable {name}: {value}")
    
    def log_debug_info(self, info: str, level: str = "DEBUG"):
        """Log debug information"""
        self.debug_history.append({
            'message': info,
            'level': level,
            'timestamp': time.time()
        })
        logger.debug(info)
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """Get a summary of debug information"""
        return {
            'breakpoints': len(self.breakpoints),
            'watched_variables': len(self.watch_variables),
            'debug_history_count': len(self.debug_history),
            'recent_history': self.debug_history[-10:] if self.debug_history else []
        }

class PerformanceProfiler:
    """Performance profiling tool"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats = None
    
    def start_profiling(self):
        """Start performance profiling"""
        logger.info("Starting performance profiling")
        self.profiler.enable()
    
    def stop_profiling(self):
        """Stop performance profiling"""
        self.profiler.disable()
        logger.info("Stopped performance profiling")
    
    def get_stats(self) -> str:
        """Get profiling statistics"""
        s = io.StringIO()
        self.stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        self.stats.print_stats(20)  # Print top 20 functions
        return s.getvalue()
    
    def save_stats(self, filename: str):
        """Save profiling statistics to file"""
        if self.stats:
            self.stats.dump_stats(filename)
            logger.info(f"Profiling stats saved to {filename}")

# ============================================================================
# DEBUGGING FUNCTIONS
# ============================================================================

@debug_function
@performance_monitor
def complex_vector_calculation(vectors: List[DebugVector3D]) -> List[DebugVector3D]:
    """Complex vector calculation with debugging"""
    logger.info(f"Starting complex vector calculation with {len(vectors)} vectors")
    
    results = []
    for i, vector in enumerate(vectors):
        logger.debug(f"Processing vector {i}: {vector}")
        
        # Simulate complex calculation
        normalized = vector.normalize()
        scaled = DebugVector3D(normalized.x * 2, normalized.y * 2, normalized.z * 2)
        results.append(scaled)
        
        logger.debug(f"Result for vector {i}: {scaled}")
    
    logger.info(f"Completed complex vector calculation, returning {len(results)} results")
    return results

@debug_function
def problematic_function(x: float, y: float) -> float:
    """Function with potential issues for debugging"""
    logger.debug(f"Entering problematic_function with x={x}, y={y}")
    
    # Simulate a potential issue
    if x == 0:
        logger.warning("x is zero, this might cause issues")
    
    if y < 0:
        logger.error("y is negative, this will cause an error")
        raise ValueError("y cannot be negative")
    
    # Simulate some computation
    result = math.sqrt(x**2 + y**2)
    
    # Add a breakpoint for debugging
    if x > 10 and y > 10:
        logger.info("Large values detected, adding breakpoint")
        # pdb.set_trace()  # Uncomment to add breakpoint
    
    logger.debug(f"Exiting problematic_function with result={result}")
    return result

def interactive_debugging_example():
    """Example of interactive debugging"""
    logger.info("Starting interactive debugging example")
    
    # Create some test data
    vectors = [
        DebugVector3D(1, 2, 3),
        DebugVector3D(4, 5, 6),
        DebugVector3D(0, 0, 0),  # This might cause issues
        DebugVector3D(-1, -2, -3)
    ]
    
    # Process vectors with potential issues
    for i, vector in enumerate(vectors):
        try:
            logger.debug(f"Processing vector {i}: {vector}")
            
            # Check for potential issues
            if vector.x == 0 and vector.y == 0 and vector.z == 0:
                logger.warning(f"Vector {i} is zero vector")
            
            # Perform calculation
            magnitude = vector.magnitude()
            logger.debug(f"Vector {i} magnitude: {magnitude}")
            
            if magnitude > 10:
                logger.info(f"Vector {i} has large magnitude: {magnitude}")
            
        except Exception as e:
            logger.error(f"Error processing vector {i}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

# ============================================================================
# DEBUGGING TECHNIQUES
# ============================================================================

def demonstrate_logging_levels():
    """Demonstrate different logging levels"""
    logger.debug("This is a DEBUG message - detailed information")
    logger.info("This is an INFO message - general information")
    logger.warning("This is a WARNING message - something might be wrong")
    logger.error("This is an ERROR message - something is wrong")
    logger.critical("This is a CRITICAL message - program may not be able to continue")

def demonstrate_exception_handling():
    """Demonstrate exception handling and debugging"""
    logger.info("Demonstrating exception handling")
    
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"Caught ZeroDivisionError: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    try:
        # Simulate another error
        undefined_variable + 1
    except NameError as e:
        logger.error(f"Caught NameError: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

def demonstrate_conditional_debugging():
    """Demonstrate conditional debugging"""
    debug_mode = True  # This could be set from environment or config
    
    if debug_mode:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - showing detailed information")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode disabled - showing only important information")
    
    # Some operations
    for i in range(5):
        if debug_mode:
            logger.debug(f"Processing item {i}")
        else:
            logger.info(f"Processing item {i}")
        
        # Simulate work
        time.sleep(0.1)

# ============================================================================
# PROFILING EXAMPLES
# ============================================================================

def demonstrate_profiling():
    """Demonstrate performance profiling"""
    logger.info("Starting profiling demonstration")
    
    profiler = PerformanceProfiler()
    
    # Start profiling
    profiler.start_profiling()
    
    # Perform some operations
    vectors = [DebugVector3D(random.uniform(-10, 10), 
                            random.uniform(-10, 10), 
                            random.uniform(-10, 10)) for _ in range(1000)]
    
    # Complex calculation
    results = complex_vector_calculation(vectors)
    
    # More operations
    for vector in vectors[:100]:
        vector.normalize()
        vector.magnitude()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Get and display stats
    stats = profiler.get_stats()
    logger.info("Profiling results:")
    print(stats)
    
    # Save stats to file
    profiler.save_stats('profiling_results.prof')
    
    return results

def demonstrate_memory_debugging():
    """Demonstrate memory debugging techniques"""
    logger.info("Demonstrating memory debugging")
    
    import gc
    import sys
    
    # Check initial memory usage
    initial_memory = sys.getsizeof([])
    logger.info(f"Initial memory usage: {initial_memory} bytes")
    
    # Create some objects
    objects = []
    for i in range(1000):
        obj = DebugGraphicsObject(f"obj_{i}", DebugVector3D(i, i, i))
        objects.append(obj)
    
    # Check memory after creating objects
    current_memory = sys.getsizeof(objects)
    logger.info(f"Memory after creating objects: {current_memory} bytes")
    
    # Force garbage collection
    collected = gc.collect()
    logger.info(f"Garbage collected {collected} objects")
    
    # Clear objects
    objects.clear()
    gc.collect()
    
    # Check final memory
    final_memory = sys.getsizeof([])
    logger.info(f"Final memory usage: {final_memory} bytes")

# ============================================================================
# DEBUGGING STRATEGIES
# ============================================================================

class DebuggingStrategies:
    """Demonstrate different debugging strategies"""
    
    def __init__(self):
        self.debugger = Debugger()
        self.profiler = PerformanceProfiler()
    
    def strategy_1_binary_search(self, data: List[int], target: int) -> int:
        """Binary search with debugging"""
        logger.info(f"Binary search for {target} in list of {len(data)} elements")
        
        left, right = 0, len(data) - 1
        
        while left <= right:
            mid = (left + right) // 2
            logger.debug(f"Checking index {mid}, value {data[mid]}")
            
            if data[mid] == target:
                logger.info(f"Found {target} at index {mid}")
                return mid
            elif data[mid] < target:
                logger.debug(f"Target is greater than {data[mid]}, moving right")
                left = mid + 1
            else:
                logger.debug(f"Target is less than {data[mid]}, moving left")
                right = mid - 1
        
        logger.warning(f"Target {target} not found")
        return -1
    
    def strategy_2_state_tracking(self, initial_state: Dict[str, Any]):
        """State tracking debugging strategy"""
        logger.info("Starting state tracking debugging")
        
        current_state = initial_state.copy()
        state_history = [current_state.copy()]
        
        # Simulate state changes
        operations = [
            ('add', 'x', 5),
            ('multiply', 'y', 2),
            ('subtract', 'z', 1),
            ('divide', 'x', 2)
        ]
        
        for operation, key, value in operations:
            logger.debug(f"Applying operation: {operation} {key} by {value}")
            
            if operation == 'add':
                current_state[key] = current_state.get(key, 0) + value
            elif operation == 'multiply':
                current_state[key] = current_state.get(key, 1) * value
            elif operation == 'subtract':
                current_state[key] = current_state.get(key, 0) - value
            elif operation == 'divide':
                if value != 0:
                    current_state[key] = current_state.get(key, 0) / value
                else:
                    logger.error("Division by zero attempted")
                    break
            
            state_history.append(current_state.copy())
            logger.debug(f"New state: {current_state}")
        
        logger.info(f"State tracking completed. Final state: {current_state}")
        return current_state, state_history
    
    def strategy_3_assertion_debugging(self, data: List[float]):
        """Assertion-based debugging strategy"""
        logger.info("Starting assertion-based debugging")
        
        # Preconditions
        assert len(data) > 0, "Data list cannot be empty"
        assert all(isinstance(x, (int, float)) for x in data), "All elements must be numbers"
        
        # Process data
        total = sum(data)
        average = total / len(data)
        
        # Postconditions
        assert total >= 0, "Total should be non-negative"
        assert min(data) <= average <= max(data), "Average should be between min and max"
        
        logger.info(f"Data processing completed. Total: {total}, Average: {average}")
        return total, average

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def create_debug_report():
    """Create a comprehensive debug report"""
    logger.info("Creating debug report")
    
    report = {
        'timestamp': time.time(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform,
            'executable': sys.executable
        },
        'memory_info': {
            'heap_size': sys.getsizeof([]),
            'gc_stats': gc.get_stats() if 'gc' in globals() else None
        },
        'logging_info': {
            'log_level': logger.level,
            'handlers': len(logger.handlers)
        }
    }
    
    logger.info("Debug report created")
    return report

def demonstrate_breakpoint_usage():
    """Demonstrate breakpoint usage"""
    logger.info("Demonstrating breakpoint usage")
    
    x = 10
    y = 20
    
    # Set a conditional breakpoint
    if x > 5 and y > 15:
        logger.info("Breakpoint condition met")
        # pdb.set_trace()  # Uncomment to add breakpoint
    
    result = x + y
    logger.info(f"Result: {result}")
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate debugging"""
    print("=== Debugging Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate logging levels
    print("1. Logging Levels:")
    demonstrate_logging_levels()
    print()
    
    # Demonstrate exception handling
    print("2. Exception Handling:")
    demonstrate_exception_handling()
    print()
    
    # Demonstrate conditional debugging
    print("3. Conditional Debugging:")
    demonstrate_conditional_debugging()
    print()
    
    # Demonstrate interactive debugging
    print("4. Interactive Debugging:")
    interactive_debugging_example()
    print()
    
    # Demonstrate profiling
    print("5. Performance Profiling:")
    results = demonstrate_profiling()
    print(f"   Generated {len(results)} results")
    print()
    
    # Demonstrate memory debugging
    print("6. Memory Debugging:")
    demonstrate_memory_debugging()
    print()
    
    # Demonstrate debugging strategies
    print("7. Debugging Strategies:")
    strategies = DebuggingStrategies()
    
    # Binary search
    data = list(range(100))
    target = 42
    result = strategies.strategy_1_binary_search(data, target)
    print(f"   Binary search result: {result}")
    
    # State tracking
    initial_state = {'x': 0, 'y': 1, 'z': 2}
    final_state, history = strategies.strategy_2_state_tracking(initial_state)
    print(f"   State tracking completed: {final_state}")
    
    # Assertion debugging
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    total, average = strategies.strategy_3_assertion_debugging(test_data)
    print(f"   Assertion debugging: total={total}, average={average}")
    print()
    
    # Create debug report
    print("8. Debug Report:")
    report = create_debug_report()
    print(f"   Report created with {len(report)} sections")
    print()
    
    # Demonstrate breakpoint usage
    print("9. Breakpoint Usage:")
    result = demonstrate_breakpoint_usage()
    print(f"   Breakpoint demonstration completed: {result}")
    print()
    
    print("="*60)
    print("Debugging demo completed successfully!")
    print("\nKey debugging concepts demonstrated:")
    print("✓ Logging: Different log levels and configurations")
    print("✓ Exception Handling: Error catching and traceback analysis")
    print("✓ Interactive Debugging: Breakpoints and step-through debugging")
    print("✓ Performance Profiling: CPU and memory profiling")
    print("✓ Debugging Strategies: Binary search, state tracking, assertions")
    print("✓ Debugging Tools: Custom debugger and profiler classes")
    print("✓ Memory Debugging: Memory usage tracking and garbage collection")
    print("✓ Debug Reports: Comprehensive debugging information")
    
    print("\nBest practices:")
    print("• Use appropriate log levels for different types of information")
    print("• Add debug information to complex calculations")
    print("• Use profiling to identify performance bottlenecks")
    print("• Implement proper exception handling with detailed logging")
    print("• Use assertions to catch logical errors early")
    print("• Create comprehensive debug reports for troubleshooting")
    print("• Use conditional debugging to control debug output")

if __name__ == "__main__":
    main()
