"""
Chapter 14: Testing and Debugging Python Code - Unit Testing Framework
=====================================================================

This module demonstrates unit testing concepts for 3D graphics applications,
including test frameworks, test cases, and testing strategies.

Key Concepts:
- Unit testing fundamentals
- Test frameworks and tools
- Test case design and organization
- Mocking and test doubles
- Test coverage and quality metrics
- Testing 3D graphics components
"""

import unittest
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import time
import random


@dataclass
class Vector3D:
    """3D vector for testing purposes."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


@dataclass
class Matrix4x4:
    """4x4 transformation matrix for testing purposes."""
    data: List[List[float]]
    
    def __init__(self, data: Optional[List[List[float]]] = None):
        if data is None:
            self.data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        else:
            self.data = data
    
    def __mul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        result = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix4x4(result)
    
    def transform_point(self, point: Vector3D) -> Vector3D:
        x = self.data[0][0] * point.x + self.data[0][1] * point.y + self.data[0][2] * point.z + self.data[0][3]
        y = self.data[1][0] * point.x + self.data[1][1] * point.y + self.data[1][2] * point.z + self.data[1][3]
        z = self.data[2][0] * point.x + self.data[2][1] * point.y + self.data[2][2] * point.z + self.data[2][3]
        return Vector3D(x, y, z)
    
    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Matrix4x4':
        return Matrix4x4([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    
    @staticmethod
    def rotation_x(angle: float) -> 'Matrix4x4':
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Matrix4x4([[1, 0, 0, 0], [0, cos_a, -sin_a, 0], [0, sin_a, cos_a, 0], [0, 0, 0, 1]])


class Renderer:
    """Simple 3D renderer for testing purposes."""
    
    def __init__(self):
        self.objects: List[Dict[str, Any]] = []
        self.camera_position = Vector3D(0, 0, 5)
        self.camera_target = Vector3D(0, 0, 0)
    
    def add_object(self, obj: Dict[str, Any]):
        """Add an object to the scene."""
        self.objects.append(obj)
    
    def render_scene(self) -> List[Dict[str, Any]]:
        """Render the scene and return results."""
        results = []
        for obj in self.objects:
            # Simulate rendering
            result = {
                "object_name": obj.get("name", "unknown"),
                "position": obj.get("position", Vector3D(0, 0, 0)),
                "rendered": True,
                "pixels_processed": random.randint(100, 1000)
            }
            results.append(result)
        return results
    
    def get_scene_stats(self) -> Dict[str, Any]:
        """Get scene statistics."""
        return {
            "object_count": len(self.objects),
            "camera_position": self.camera_position,
            "camera_target": self.camera_target
        }


class PhysicsEngine:
    """Simple physics engine for testing purposes."""
    
    def __init__(self):
        self.gravity = Vector3D(0, -9.81, 0)
        self.objects: List[Dict[str, Any]] = []
    
    def add_object(self, obj: Dict[str, Any]):
        """Add a physics object."""
        self.objects.append(obj)
    
    def update_physics(self, delta_time: float):
        """Update physics simulation."""
        for obj in self.objects:
            if "velocity" in obj and "position" in obj:
                # Apply gravity
                obj["velocity"] = obj["velocity"] + self.gravity * delta_time
                # Update position
                obj["position"] = obj["position"] + obj["velocity"] * delta_time
    
    def get_physics_stats(self) -> Dict[str, Any]:
        """Get physics statistics."""
        return {
            "object_count": len(self.objects),
            "gravity": self.gravity
        }


# Test Cases
class TestVector3D(unittest.TestCase):
    """Test cases for Vector3D class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.v1 = Vector3D(1, 2, 3)
        self.v2 = Vector3D(4, 5, 6)
        self.zero = Vector3D(0, 0, 0)
    
    def test_vector_creation(self):
        """Test vector creation."""
        self.assertEqual(self.v1.x, 1)
        self.assertEqual(self.v1.y, 2)
        self.assertEqual(self.v1.z, 3)
    
    def test_vector_addition(self):
        """Test vector addition."""
        result = self.v1 + self.v2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 7)
        self.assertEqual(result.z, 9)
    
    def test_vector_subtraction(self):
        """Test vector subtraction."""
        result = self.v2 - self.v1
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 3)
        self.assertEqual(result.z, 3)
    
    def test_vector_scalar_multiplication(self):
        """Test vector scalar multiplication."""
        result = self.v1 * 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 4)
        self.assertEqual(result.z, 6)
    
    def test_vector_magnitude(self):
        """Test vector magnitude calculation."""
        magnitude = self.v1.magnitude()
        expected = math.sqrt(1**2 + 2**2 + 3**2)
        self.assertAlmostEqual(magnitude, expected, places=6)
    
    def test_vector_normalization(self):
        """Test vector normalization."""
        normalized = self.v1.normalize()
        magnitude = normalized.magnitude()
        self.assertAlmostEqual(magnitude, 1.0, places=6)
    
    def test_zero_vector_normalization(self):
        """Test normalization of zero vector."""
        normalized = self.zero.normalize()
        self.assertEqual(normalized.x, 0)
        self.assertEqual(normalized.y, 0)
        self.assertEqual(normalized.z, 0)
    
    def test_vector_dot_product(self):
        """Test vector dot product."""
        dot_product = self.v1.dot(self.v2)
        expected = 1*4 + 2*5 + 3*6
        self.assertEqual(dot_product, expected)


class TestMatrix4x4(unittest.TestCase):
    """Test cases for Matrix4x4 class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identity = Matrix4x4()
        self.translation_matrix = Matrix4x4.translation(1, 2, 3)
        self.rotation_matrix = Matrix4x4.rotation_x(math.pi / 2)
        self.point = Vector3D(1, 0, 0)
    
    def test_identity_matrix_creation(self):
        """Test identity matrix creation."""
        self.assertEqual(self.identity.data[0][0], 1)
        self.assertEqual(self.identity.data[1][1], 1)
        self.assertEqual(self.identity.data[2][2], 1)
        self.assertEqual(self.identity.data[3][3], 1)
    
    def test_translation_matrix_creation(self):
        """Test translation matrix creation."""
        self.assertEqual(self.translation_matrix.data[0][3], 1)
        self.assertEqual(self.translation_matrix.data[1][3], 2)
        self.assertEqual(self.translation_matrix.data[2][3], 3)
    
    def test_rotation_matrix_creation(self):
        """Test rotation matrix creation."""
        cos_pi_2 = math.cos(math.pi / 2)
        sin_pi_2 = math.sin(math.pi / 2)
        self.assertAlmostEqual(self.rotation_matrix.data[1][1], cos_pi_2, places=6)
        self.assertAlmostEqual(self.rotation_matrix.data[1][2], -sin_pi_2, places=6)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        result = self.translation_matrix * self.rotation_matrix
        # Verify result is a valid matrix
        self.assertEqual(len(result.data), 4)
        self.assertEqual(len(result.data[0]), 4)
    
    def test_point_transformation(self):
        """Test point transformation."""
        transformed = self.translation_matrix.transform_point(self.point)
        self.assertEqual(transformed.x, 2)  # 1 + 1
        self.assertEqual(transformed.y, 2)  # 0 + 2
        self.assertEqual(transformed.z, 3)  # 0 + 3


class TestRenderer(unittest.TestCase):
    """Test cases for Renderer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = Renderer()
        self.test_object = {
            "name": "test_cube",
            "position": Vector3D(0, 0, 0),
            "vertices": [Vector3D(0, 0, 0), Vector3D(1, 0, 0)]
        }
    
    def test_renderer_initialization(self):
        """Test renderer initialization."""
        self.assertEqual(len(self.renderer.objects), 0)
        self.assertEqual(self.renderer.camera_position.x, 0)
        self.assertEqual(self.renderer.camera_position.y, 0)
        self.assertEqual(self.renderer.camera_position.z, 5)
    
    def test_add_object(self):
        """Test adding objects to renderer."""
        self.renderer.add_object(self.test_object)
        self.assertEqual(len(self.renderer.objects), 1)
        self.assertEqual(self.renderer.objects[0]["name"], "test_cube")
    
    def test_render_scene(self):
        """Test scene rendering."""
        self.renderer.add_object(self.test_object)
        results = self.renderer.render_scene()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object_name"], "test_cube")
        self.assertTrue(results[0]["rendered"])
        self.assertIn("pixels_processed", results[0])
    
    def test_scene_stats(self):
        """Test scene statistics."""
        self.renderer.add_object(self.test_object)
        stats = self.renderer.get_scene_stats()
        
        self.assertEqual(stats["object_count"], 1)
        self.assertEqual(stats["camera_position"], self.renderer.camera_position)
        self.assertEqual(stats["camera_target"], self.renderer.camera_target)


class TestPhysicsEngine(unittest.TestCase):
    """Test cases for PhysicsEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.physics = PhysicsEngine()
        self.test_object = {
            "name": "test_sphere",
            "position": Vector3D(0, 10, 0),
            "velocity": Vector3D(0, 0, 0),
            "mass": 1.0
        }
    
    def test_physics_engine_initialization(self):
        """Test physics engine initialization."""
        self.assertEqual(len(self.physics.objects), 0)
        self.assertEqual(self.physics.gravity.y, -9.81)
    
    def test_add_physics_object(self):
        """Test adding physics objects."""
        self.physics.add_object(self.test_object)
        self.assertEqual(len(self.physics.objects), 1)
        self.assertEqual(self.physics.objects[0]["name"], "test_sphere")
    
    def test_physics_update(self):
        """Test physics simulation update."""
        self.physics.add_object(self.test_object)
        initial_position = self.test_object["position"]
        
        self.physics.update_physics(0.1)  # 100ms
        
        # Position should have changed due to gravity
        self.assertNotEqual(self.test_object["position"], initial_position)
    
    def test_physics_stats(self):
        """Test physics statistics."""
        self.physics.add_object(self.test_object)
        stats = self.physics.get_physics_stats()
        
        self.assertEqual(stats["object_count"], 1)
        self.assertEqual(stats["gravity"], self.physics.gravity)


class TestIntegration(unittest.TestCase):
    """Integration tests for 3D graphics components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.renderer = Renderer()
        self.physics = PhysicsEngine()
        self.test_object = {
            "name": "physics_cube",
            "position": Vector3D(0, 5, 0),
            "velocity": Vector3D(0, 0, 0),
            "mass": 1.0
        }
    
    def test_physics_to_rendering_integration(self):
        """Test integration between physics and rendering."""
        # Add object to physics engine
        self.physics.add_object(self.test_object)
        
        # Simulate physics for a few steps
        for _ in range(10):
            self.physics.update_physics(0.016)  # 60 FPS
        
        # Add object to renderer
        self.renderer.add_object(self.test_object)
        
        # Render the scene
        results = self.renderer.render_scene()
        
        # Verify integration
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object_name"], "physics_cube")
        self.assertTrue(results[0]["rendered"])
    
    def test_vector_matrix_integration(self):
        """Test integration between vectors and matrices."""
        point = Vector3D(1, 0, 0)
        translation = Matrix4x4.translation(2, 3, 4)
        rotation = Matrix4x4.rotation_x(math.pi)
        
        # Apply transformations
        transformed = translation.transform_point(point)
        rotated = rotation.transform_point(transformed)
        
        # Verify transformations
        self.assertEqual(transformed.x, 3)  # 1 + 2
        self.assertEqual(transformed.y, 3)  # 0 + 3
        self.assertEqual(transformed.z, 4)  # 0 + 4


class TestMocking(unittest.TestCase):
    """Test cases demonstrating mocking techniques."""
    
    def test_mock_renderer(self):
        """Test using mock renderer."""
        mock_renderer = Mock()
        mock_renderer.render_scene.return_value = [
            {"object_name": "mock_cube", "rendered": True}
        ]
        
        results = mock_renderer.render_scene()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["object_name"], "mock_cube")
        mock_renderer.render_scene.assert_called_once()
    
    def test_patch_physics_gravity(self):
        """Test patching physics gravity."""
        physics = PhysicsEngine()
        
        with patch.object(physics, 'gravity', Vector3D(0, -5, 0)):
            self.assertEqual(physics.gravity.y, -5)
        
        # Original gravity should be restored
        self.assertEqual(physics.gravity.y, -9.81)
    
    def test_mock_file_operations(self):
        """Test mocking file operations."""
        with patch('builtins.open', mock_open(read_data='test data')):
            with open('test.txt', 'r') as f:
                content = f.read()
                self.assertEqual(content, 'test_data')


class TestPerformance(unittest.TestCase):
    """Performance tests for 3D graphics components."""
    
    def test_vector_operations_performance(self):
        """Test performance of vector operations."""
        vectors = [Vector3D(i, i+1, i+2) for i in range(1000)]
        
        start_time = time.time()
        for i in range(len(vectors) - 1):
            result = vectors[i] + vectors[i+1]
            magnitude = result.magnitude()
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second
    
    def test_matrix_multiplication_performance(self):
        """Test performance of matrix multiplication."""
        matrices = [Matrix4x4.translation(i, i, i) for i in range(100)]
        
        start_time = time.time()
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result * matrix
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second


# Test Runner
def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVector3D,
        TestMatrix4x4,
        TestRenderer,
        TestPhysicsEngine,
        TestIntegration,
        TestMocking,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    # Run tests
    result = run_tests()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Print failures and errors
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
