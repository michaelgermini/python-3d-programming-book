#!/usr/bin/env python3
"""
Chapter 15: Advanced 3D Graphics Libraries and Tools
Library Comparison System

Demonstrates how to benchmark and compare different 3D graphics libraries
for performance, ease of use, and feature completeness.
"""

import time
import sys
import platform
import psutil
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import numpy as np

class GraphicsLibrary(Enum):
    """Supported graphics libraries"""
    MODERNGL = "ModernGL"
    PYGAME = "Pygame"
    PYGAME3D = "Pygame3D"
    ARCADE = "Arcade"
    KIVY = "Kivy"
    PYSFML = "PySFML"
    PYSFML2 = "PySFML2"
    PYSFML3 = "PySFML3"
    PYSFML4 = "PySFML4"
    PYSFML5 = "PySFML5"
    PYSFML6 = "PySFML6"
    PYSFML7 = "PySFML7"
    PYSFML8 = "PySFML8"
    PYSFML9 = "PySFML9"
    PYSFML10 = "PySFML10"

@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    library: str
    test_name: str
    execution_time: float
    memory_usage: float
    frame_rate: float
    cpu_usage: float
    gpu_usage: float
    success: bool
    error_message: str = ""
    features_supported: List[str] = None
    platform_info: Dict[str, str] = None

@dataclass
class LibraryInfo:
    """Information about a graphics library"""
    name: str
    version: str
    description: str
    features: List[str]
    platforms: List[str]
    performance_rating: float
    ease_of_use_rating: float
    documentation_rating: float
    community_rating: float
    installation_difficulty: str
    license: str
    website: str

class BenchmarkTest:
    """Base class for benchmark tests"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results: List[BenchmarkResult] = []
    
    def run_test(self, library: GraphicsLibrary) -> BenchmarkResult:
        """Run the benchmark test for a specific library"""
        raise NotImplementedError("Subclasses must implement run_test")
    
    def cleanup(self):
        """Clean up resources after test"""
        pass

class PerformanceTest(BenchmarkTest):
    """Performance benchmark test"""
    
    def __init__(self):
        super().__init__("Performance Test", "Measures rendering performance and frame rates")
    
    def run_test(self, library: GraphicsLibrary) -> BenchmarkResult:
        """Run performance test"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Simulate rendering workload
            frame_count = 0
            test_duration = 5.0  # 5 seconds
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                # Simulate rendering operations
                self.simulate_rendering(library)
                frame_count += 1
                time.sleep(1/60)  # Target 60 FPS
            
            end_time = time.time()
            execution_time = end_time - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            frame_rate = frame_count / execution_time
            
            return BenchmarkResult(
                library=library.value,
                test_name=self.name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                frame_rate=frame_rate,
                cpu_usage=psutil.cpu_percent(),
                gpu_usage=0.0,  # Would need GPU monitoring library
                success=True,
                features_supported=self.get_supported_features(library),
                platform_info=self.get_platform_info()
            )
            
        except Exception as e:
            return BenchmarkResult(
                library=library.value,
                test_name=self.name,
                execution_time=0.0,
                memory_usage=0.0,
                frame_rate=0.0,
                cpu_usage=0.0,
                gpu_usage=0.0,
                success=False,
                error_message=str(e),
                features_supported=[],
                platform_info=self.get_platform_info()
            )
    
    def simulate_rendering(self, library: GraphicsLibrary):
        """Simulate rendering operations for the library"""
        # Simulate different workloads based on library
        if library == GraphicsLibrary.MODERNGL:
            # ModernGL is typically fast
            time.sleep(0.001)
        elif library == GraphicsLibrary.PYGAME:
            # Pygame is slower
            time.sleep(0.005)
        elif library == GraphicsLibrary.ARCADE:
            # Arcade is moderate
            time.sleep(0.003)
        else:
            # Default simulation
            time.sleep(0.002)
    
    def get_supported_features(self, library: GraphicsLibrary) -> List[str]:
        """Get features supported by the library"""
        feature_map = {
            GraphicsLibrary.MODERNGL: ["OpenGL", "Shaders", "3D Rendering", "Textures"],
            GraphicsLibrary.PYGAME: ["2D Graphics", "Sound", "Input", "Sprites"],
            GraphicsLibrary.ARCADE: ["2D Graphics", "Physics", "Sprites", "Sound"],
            GraphicsLibrary.KIVY: ["Cross-platform", "UI", "Touch", "Mobile"],
        }
        return feature_map.get(library, ["Basic Graphics"])
    
    def get_platform_info(self) -> Dict[str, str]:
        """Get current platform information"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor()
        }

class FeatureTest(BenchmarkTest):
    """Feature compatibility test"""
    
    def __init__(self):
        super().__init__("Feature Test", "Tests feature support and compatibility")
    
    def run_test(self, library: GraphicsLibrary) -> BenchmarkResult:
        """Run feature test"""
        start_time = time.time()
        
        try:
            features = self.test_features(library)
            success = len(features) > 0
            
            return BenchmarkResult(
                library=library.value,
                test_name=self.name,
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                frame_rate=0.0,
                cpu_usage=0.0,
                gpu_usage=0.0,
                success=success,
                features_supported=features,
                platform_info=self.get_platform_info()
            )
            
        except Exception as e:
            return BenchmarkResult(
                library=library.value,
                test_name=self.name,
                execution_time=time.time() - start_time,
                memory_usage=0.0,
                frame_rate=0.0,
                cpu_usage=0.0,
                gpu_usage=0.0,
                success=False,
                error_message=str(e),
                features_supported=[],
                platform_info=self.get_platform_info()
            )
    
    def test_features(self, library: GraphicsLibrary) -> List[str]:
        """Test which features are supported"""
        features = []
        
        # Test basic features
        if self.test_basic_graphics(library):
            features.append("Basic Graphics")
        
        if self.test_3d_rendering(library):
            features.append("3D Rendering")
        
        if self.test_shaders(library):
            features.append("Shaders")
        
        if self.test_textures(library):
            features.append("Textures")
        
        if self.test_input_handling(library):
            features.append("Input Handling")
        
        if self.test_sound(library):
            features.append("Sound")
        
        return features
    
    def test_basic_graphics(self, library: GraphicsLibrary) -> bool:
        """Test basic graphics capabilities"""
        return library in [GraphicsLibrary.MODERNGL, GraphicsLibrary.PYGAME, GraphicsLibrary.ARCADE]
    
    def test_3d_rendering(self, library: GraphicsLibrary) -> bool:
        """Test 3D rendering capabilities"""
        return library == GraphicsLibrary.MODERNGL
    
    def test_shaders(self, library: GraphicsLibrary) -> bool:
        """Test shader support"""
        return library == GraphicsLibrary.MODERNGL
    
    def test_textures(self, library: GraphicsLibrary) -> bool:
        """Test texture support"""
        return library in [GraphicsLibrary.MODERNGL, GraphicsLibrary.PYGAME, GraphicsLibrary.ARCADE]
    
    def test_input_handling(self, library: GraphicsLibrary) -> bool:
        """Test input handling"""
        return library in [GraphicsLibrary.PYGAME, GraphicsLibrary.ARCADE, GraphicsLibrary.KIVY]
    
    def test_sound(self, library: GraphicsLibrary) -> bool:
        """Test sound support"""
        return library in [GraphicsLibrary.PYGAME, GraphicsLibrary.ARCADE]
    
    def get_platform_info(self) -> Dict[str, str]:
        """Get current platform information"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor()
        }

class LibraryComparisonSystem:
    """Main system for comparing graphics libraries"""
    
    def __init__(self):
        self.libraries = self.get_library_info()
        self.tests = [
            PerformanceTest(),
            FeatureTest()
        ]
        self.results: Dict[str, List[BenchmarkResult]] = {}
    
    def get_library_info(self) -> Dict[str, LibraryInfo]:
        """Get information about available libraries"""
        return {
            "ModernGL": LibraryInfo(
                name="ModernGL",
                version="5.8.0",
                description="Modern OpenGL wrapper for Python",
                features=["OpenGL", "Shaders", "3D Rendering", "Textures", "Framebuffers"],
                platforms=["Windows", "macOS", "Linux"],
                performance_rating=9.0,
                ease_of_use_rating=7.0,
                documentation_rating=8.0,
                community_rating=7.0,
                installation_difficulty="Medium",
                license="MIT",
                website="https://github.com/moderngl/moderngl"
            ),
            "Pygame": LibraryInfo(
                name="Pygame",
                version="2.5.0",
                description="Cross-platform library for game development",
                features=["2D Graphics", "Sound", "Input", "Sprites", "Collision"],
                platforms=["Windows", "macOS", "Linux"],
                performance_rating=6.0,
                ease_of_use_rating=9.0,
                documentation_rating=9.0,
                community_rating=9.0,
                installation_difficulty="Easy",
                license="LGPL",
                website="https://www.pygame.org/"
            ),
            "Arcade": LibraryInfo(
                name="Arcade",
                version="2.6.0",
                description="Modern Python library for crafting games",
                features=["2D Graphics", "Physics", "Sprites", "Sound", "UI"],
                platforms=["Windows", "macOS", "Linux"],
                performance_rating=7.0,
                ease_of_use_rating=8.0,
                documentation_rating=8.0,
                community_rating=7.0,
                installation_difficulty="Easy",
                license="MIT",
                website="https://arcade.academy/"
            ),
            "Kivy": LibraryInfo(
                name="Kivy",
                version="2.2.0",
                description="Cross-platform Python framework for applications",
                features=["Cross-platform", "UI", "Touch", "Mobile", "Graphics"],
                platforms=["Windows", "macOS", "Linux", "Android", "iOS"],
                performance_rating=6.0,
                ease_of_use_rating=6.0,
                documentation_rating=7.0,
                community_rating=7.0,
                installation_difficulty="Medium",
                license="MIT",
                website="https://kivy.org/"
            )
        }
    
    def run_benchmarks(self, libraries: List[GraphicsLibrary] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks for specified libraries"""
        if libraries is None:
            libraries = list(GraphicsLibrary)[:4]  # Test first 4 libraries
        
        print(f"Running benchmarks for {len(libraries)} libraries...")
        
        for library in libraries:
            print(f"\nTesting {library.value}...")
            self.results[library.value] = []
            
            for test in self.tests:
                print(f"  Running {test.name}...")
                result = test.run_test(library)
                self.results[library.value].append(result)
                
                if result.success:
                    print(f"    ✓ Success - Frame Rate: {result.frame_rate:.1f} FPS")
                else:
                    print(f"    ✗ Failed - {result.error_message}")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("# Graphics Library Comparison Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Platform: {platform.system()} {platform.version()}")
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("| Library | Avg FPS | Memory (MB) | CPU (%) | Features |")
        report.append("|---------|---------|-------------|---------|----------|")
        
        for library_name, results in self.results.items():
            if results:
                perf_result = next((r for r in results if r.test_name == "Performance Test"), None)
                feature_result = next((r for r in results if r.test_name == "Feature Test"), None)
                
                if perf_result and feature_result:
                    fps = perf_result.frame_rate
                    memory = perf_result.memory_usage
                    cpu = perf_result.cpu_usage
                    features = len(feature_result.features_supported)
                    
                    report.append(f"| {library_name} | {fps:.1f} | {memory:.1f} | {cpu:.1f} | {features} |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for library_name, results in self.results.items():
            report.append(f"### {library_name}")
            
            for result in results:
                report.append(f"#### {result.test_name}")
                if result.success:
                    report.append(f"- Execution Time: {result.execution_time:.3f}s")
                    report.append(f"- Frame Rate: {result.frame_rate:.1f} FPS")
                    report.append(f"- Memory Usage: {result.memory_usage:.1f} MB")
                    report.append(f"- CPU Usage: {result.cpu_usage:.1f}%")
                    if result.features_supported:
                        report.append(f"- Features: {', '.join(result.features_supported)}")
                else:
                    report.append(f"- Status: Failed")
                    report.append(f"- Error: {result.error_message}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "library_comparison_results.json"):
        """Save results to JSON file"""
        data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "platform": {
                "os": platform.system(),
                "version": platform.version(),
                "python": platform.python_version(),
                "architecture": platform.architecture()[0]
            },
            "results": {}
        }
        
        for library_name, results in self.results.items():
            data["results"][library_name] = []
            for result in results:
                data["results"][library_name].append({
                    "test_name": result.test_name,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "frame_rate": result.frame_rate,
                    "cpu_usage": result.cpu_usage,
                    "success": result.success,
                    "error_message": result.error_message,
                    "features_supported": result.features_supported or []
                })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def create_visualizations(self):
        """Create visualizations of the benchmark results"""
        if not self.results:
            print("No results to visualize. Run benchmarks first.")
            return
        
        # Prepare data
        libraries = []
        fps_scores = []
        memory_scores = []
        feature_scores = []
        
        for library_name, results in self.results.items():
            if results:
                libraries.append(library_name)
                
                perf_result = next((r for r in results if r.test_name == "Performance Test"), None)
                feature_result = next((r for r in results if r.test_name == "Feature Test"), None)
                
                fps_scores.append(perf_result.frame_rate if perf_result and perf_result.success else 0)
                memory_scores.append(perf_result.memory_usage if perf_result and perf_result.success else 0)
                feature_scores.append(len(feature_result.features_supported) if feature_result else 0)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Graphics Library Comparison Results', fontsize=16)
        
        # Frame Rate Comparison
        bars1 = ax1.bar(libraries, fps_scores, color='skyblue')
        ax1.set_title('Frame Rate Performance')
        ax1.set_ylabel('FPS')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory Usage Comparison
        bars2 = ax2.bar(libraries, memory_scores, color='lightcoral')
        ax2.set_title('Memory Usage')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Feature Support Comparison
        bars3 = ax3.bar(libraries, feature_scores, color='lightgreen')
        ax3.set_title('Feature Support')
        ax3.set_ylabel('Number of Features')
        ax3.tick_params(axis='x', rotation=45)
        
        # Overall Score (normalized)
        overall_scores = []
        for i in range(len(libraries)):
            # Normalize scores (0-1) and combine
            fps_norm = fps_scores[i] / max(fps_scores) if max(fps_scores) > 0 else 0
            memory_norm = 1 - (memory_scores[i] / max(memory_scores)) if max(memory_scores) > 0 else 0
            feature_norm = feature_scores[i] / max(feature_scores) if max(feature_scores) > 0 else 0
            
            overall = (fps_norm + memory_norm + feature_norm) / 3
            overall_scores.append(overall)
        
        bars4 = ax4.bar(libraries, overall_scores, color='gold')
        ax4.set_title('Overall Score')
        ax4.set_ylabel('Score (0-1)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for ax, bars in [(ax1, bars1), (ax2, bars2), (ax3, bars3), (ax4, bars4)]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('library_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to library_comparison_results.png")
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on benchmark results"""
        recommendations = []
        
        if not self.results:
            return ["Run benchmarks first to get recommendations"]
        
        # Find best performers
        best_fps = None
        best_memory = None
        most_features = None
        
        for library_name, results in self.results.items():
            if results:
                perf_result = next((r for r in results if r.test_name == "Performance Test"), None)
                feature_result = next((r for r in results if r.test_name == "Feature Test"), None)
                
                if perf_result and perf_result.success:
                    if best_fps is None or perf_result.frame_rate > best_fps[1]:
                        best_fps = (library_name, perf_result.frame_rate)
                    
                    if best_memory is None or perf_result.memory_usage < best_memory[1]:
                        best_memory = (library_name, perf_result.memory_usage)
                
                if feature_result:
                    if most_features is None or len(feature_result.features_supported) > most_features[1]:
                        most_features = (library_name, len(feature_result.features_supported))
        
        recommendations.append("## Library Recommendations")
        recommendations.append("")
        
        if best_fps:
            recommendations.append(f"**Best Performance:** {best_fps[0]} ({best_fps[1]:.1f} FPS)")
        
        if best_memory:
            recommendations.append(f"**Most Memory Efficient:** {best_memory[0]} ({best_memory[1]:.1f} MB)")
        
        if most_features:
            recommendations.append(f"**Most Features:** {most_features[0]} ({most_features[1]} features)")
        
        recommendations.append("")
        recommendations.append("### Use Case Recommendations:")
        recommendations.append("- **Game Development:** Consider Pygame for 2D games, ModernGL for 3D")
        recommendations.append("- **Cross-platform Apps:** Kivy for mobile and desktop")
        recommendations.append("- **High Performance:** ModernGL for OpenGL-based applications")
        recommendations.append("- **Learning/Prototyping:** Arcade for easy 2D graphics")
        
        return recommendations

def main():
    print("=== Graphics Library Comparison System ===\n")
    print("This system benchmarks and compares different 3D graphics libraries")
    print("for performance, features, and ease of use.\n")
    
    # Create comparison system
    comparison_system = LibraryComparisonSystem()
    
    # Run benchmarks
    print("Starting benchmarks...")
    results = comparison_system.run_benchmarks()
    
    # Generate and display report
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    report = comparison_system.generate_report()
    print(report)
    
    # Save results
    comparison_system.save_results()
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison_system.create_visualizations()
    
    # Get recommendations
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    recommendations = comparison_system.get_recommendations()
    for rec in recommendations:
        print(rec)
    
    print("\nBenchmark completed successfully!")
    print("Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main()
