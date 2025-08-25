"""
Chapter 14: Testing and Debugging Python Code - Test Coverage and Quality
=========================================================================

This module demonstrates test coverage analysis and code quality metrics
for 3D graphics applications.

Key Concepts:
- Test coverage analysis and reporting
- Code quality metrics and standards
- Coverage-driven development
- Quality assurance practices
"""

import coverage
import ast
import inspect
import time
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class CoverageType(Enum):
    """Types of code coverage."""
    STATEMENT = "statement"
    BRANCH = "branch"
    FUNCTION = "function"
    LINE = "line"


@dataclass
class CoverageMetrics:
    """Coverage metrics for a code module."""
    module_name: str
    statements: int = 0
    executed_statements: int = 0
    branches: int = 0
    executed_branches: int = 0
    functions: int = 0
    executed_functions: int = 0
    
    @property
    def statement_coverage(self) -> float:
        """Calculate statement coverage percentage."""
        if self.statements == 0:
            return 100.0
        return (self.executed_statements / self.statements) * 100
    
    @property
    def overall_coverage(self) -> float:
        """Calculate overall coverage percentage."""
        total_items = self.statements + self.branches + self.functions
        executed_items = (self.executed_statements + self.executed_branches + 
                         self.executed_functions)
        
        if total_items == 0:
            return 100.0
        return (executed_items / total_items) * 100


@dataclass
class CodeQualityMetrics:
    """Code quality metrics for analysis."""
    module_name: str
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    function_count: int = 0
    class_count: int = 0
    comment_ratio: float = 0.0
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        complexity_score = max(0, 100 - self.cyclomatic_complexity * 5)
        comment_score = min(100, self.comment_ratio * 100)
        return (complexity_score + comment_score) / 2


class CoverageAnalyzer:
    """Analyzer for test coverage in 3D graphics applications."""
    
    def __init__(self):
        self.cov = coverage.Coverage()
        self.coverage_data: Dict[str, CoverageMetrics] = {}
    
    def start_coverage(self):
        """Start coverage measurement."""
        self.cov.start()
        print("  📊 Started coverage measurement")
    
    def stop_coverage(self):
        """Stop coverage measurement."""
        self.cov.stop()
        print("  📊 Stopped coverage measurement")
    
    def analyze_coverage(self, modules: List[str]) -> Dict[str, CoverageMetrics]:
        """Analyze coverage for specified modules."""
        self.cov.save()
        
        for module in modules:
            try:
                analysis = self.cov.analysis(module)
                if analysis:
                    missing, executable, missing_branches, executable_branches = analysis
                    
                    metrics = CoverageMetrics(
                        module_name=module,
                        statements=len(executable),
                        executed_statements=len(executable) - len(missing),
                        branches=len(executable_branches) if executable_branches else 0,
                        executed_branches=len(executable_branches) - len(missing_branches) if executable_branches else 0,
                        functions=self._count_functions(module),
                        executed_functions=self._count_functions(module)
                    )
                    
                    self.coverage_data[module] = metrics
                    print(f"  📊 {module}: {metrics.overall_coverage:.1f}% coverage")
                
            except Exception as e:
                print(f"  ⚠️  Could not analyze {module}: {e}")
        
        return self.coverage_data
    
    def _count_functions(self, module_name: str) -> int:
        """Count functions in a module."""
        try:
            module = __import__(module_name, fromlist=[''])
            return len([name for name, obj in inspect.getmembers(module) 
                       if inspect.isfunction(obj)])
        except:
            return 0
    
    def generate_coverage_report(self, output_file: str = "coverage_report.json"):
        """Generate a coverage report."""
        report_data = {
            "timestamp": time.time(),
            "modules": {},
            "summary": {
                "total_modules": len(self.coverage_data),
                "average_coverage": 0.0
            }
        }
        
        total_coverage = 0.0
        
        for module_name, metrics in self.coverage_data.items():
            report_data["modules"][module_name] = {
                "statement_coverage": metrics.statement_coverage,
                "overall_coverage": metrics.overall_coverage,
                "statements": metrics.statements,
                "executed_statements": metrics.executed_statements
            }
            total_coverage += metrics.overall_coverage
        
        if self.coverage_data:
            report_data["summary"]["average_coverage"] = total_coverage / len(self.coverage_data)
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"  📊 Coverage report saved to {output_file}")
        return report_data


class CodeQualityAnalyzer:
    """Analyzer for code quality metrics."""
    
    def __init__(self):
        self.quality_data: Dict[str, CodeQualityMetrics] = {}
    
    def analyze_module(self, module_path: str) -> CodeQualityMetrics:
        """Analyze code quality for a module."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            metrics = CodeQualityMetrics(
                module_name=os.path.basename(module_path),
                lines_of_code=len(source_code.split('\n')),
                cyclomatic_complexity=self._calculate_cyclomatic_complexity(tree),
                function_count=self._count_functions_ast(tree),
                class_count=self._count_classes_ast(tree),
                comment_ratio=self._calculate_comment_ratio(source_code)
            )
            
            self.quality_data[module_path] = metrics
            print(f"  🔍 {module_path}: Quality score {metrics.quality_score:.1f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ⚠️  Could not analyze {module_path}: {e}")
            return CodeQualityMetrics(module_name=os.path.basename(module_path))
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
        return complexity
    
    def _count_functions_ast(self, tree: ast.AST) -> int:
        """Count functions using AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
    
    def _count_classes_ast(self, tree: ast.AST) -> int:
        """Count classes using AST."""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _calculate_comment_ratio(self, source_code: str) -> float:
        """Calculate comment ratio."""
        lines = source_code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return comment_lines / max(len(lines), 1)


class TestRunner:
    """Test runner with coverage and quality analysis."""
    
    def __init__(self):
        self.coverage_analyzer = CoverageAnalyzer()
        self.quality_analyzer = CodeQualityAnalyzer()
    
    def run_tests_with_coverage(self, test_modules: List[str], 
                               source_modules: List[str]) -> Dict[str, Any]:
        """Run tests with coverage analysis."""
        print("=== Running Tests with Coverage Analysis ===\n")
        
        self.coverage_analyzer.start_coverage()
        
        # Run test functions
        test_functions = [test_vector_creation, test_vector_addition, 
                         test_vector_magnitude, test_matrix_creation]
        
        tests_run = 0
        failures = 0
        errors = 0
        
        for test_func in test_functions:
            tests_run += 1
            try:
                test_func()
                print(f"  ✅ {test_func.__name__}: Passed")
            except AssertionError as e:
                failures += 1
                print(f"  ❌ {test_func.__name__}: Failed - {e}")
            except Exception as e:
                errors += 1
                print(f"  ❌ {test_func.__name__}: Error - {e}")
        
        self.coverage_analyzer.stop_coverage()
        
        coverage_data = self.coverage_analyzer.analyze_coverage(source_modules)
        coverage_report = self.coverage_analyzer.generate_coverage_report()
        
        return {
            "tests_run": tests_run,
            "failures": failures,
            "errors": errors,
            "coverage_data": coverage_data,
            "coverage_report": coverage_report
        }
    
    def analyze_code_quality(self, source_files: List[str]) -> Dict[str, Any]:
        """Analyze code quality for source files."""
        print("\n=== Analyzing Code Quality ===\n")
        
        for source_file in source_files:
            if os.path.exists(source_file):
                self.quality_analyzer.analyze_module(source_file)
        
        return {"quality_data": self.quality_analyzer.quality_data}


# Example 3D Graphics Components
class Vector3D:
    """3D vector for testing purposes."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


class Matrix4x4:
    """4x4 transformation matrix for testing purposes."""
    
    def __init__(self, data: Optional[List[List[float]]] = None):
        if data is None:
            self.data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        else:
            self.data = data


# Test Functions
def test_vector_creation():
    """Test vector creation."""
    v = Vector3D(1, 2, 3)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3


def test_vector_addition():
    """Test vector addition."""
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)
    result = v1 + v2
    assert result.x == 5
    assert result.y == 7
    assert result.z == 9


def test_vector_magnitude():
    """Test vector magnitude calculation."""
    v = Vector3D(3, 4, 0)
    assert v.magnitude() == 5.0


def test_matrix_creation():
    """Test matrix creation."""
    m = Matrix4x4()
    assert m.data[0][0] == 1
    assert m.data[1][1] == 1


# Example Usage
def demonstrate_coverage_and_quality():
    """Demonstrate coverage and quality analysis."""
    print("=== Coverage and Quality Analysis ===\n")
    
    test_runner = TestRunner()
    
    # Run tests with coverage
    results = test_runner.run_tests_with_coverage(["__main__"], ["__main__"])
    
    # Analyze code quality
    current_file = __file__
    quality_results = test_runner.analyze_code_quality([current_file])
    
    # Print summary
    print("\n=== Analysis Summary ===")
    print(f"Tests: {results['tests_run']} run, {results['failures']} failures, {results['errors']} errors")
    
    coverage_data = results["coverage_data"]
    if coverage_data:
        avg_coverage = sum(m.overall_coverage for m in coverage_data.values()) / len(coverage_data)
        print(f"Average Coverage: {avg_coverage:.1f}%")


if __name__ == "__main__":
    demonstrate_coverage_and_quality()
