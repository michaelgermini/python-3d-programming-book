#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
Testing Example

Demonstrates testing including unit tests, test frameworks, mocking,
test-driven development, and testing strategies for 3D graphics applications.
"""

import unittest
import math
import random
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Testing"
__description__ = "Testing for 3D graphics applications"

# ============================================================================
# CLASSES TO TEST
# ============================================================================

@dataclass
class Vector3D:
    """3D vector class for testing"""
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
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Matrix3D:
    """3D matrix class for testing"""
    
    def __init__(self, data: List[List[float]]):
        if len(data) != 3 or any(len(row) != 3 for row in data):
            raise ValueError("Matrix must be 3x3")
        self.data = data
    
    def __mul__(self, other: Union['Matrix3D', Vector3D]) -> Union['Matrix3D', Vector3D]:
        if isinstance(other, Vector3D):
            # Matrix * Vector
            result = [0, 0, 0]
            for i in range(3):
                for j in range(3):
                    result[i] += self.data[i][j] * [other.x, other.y, other.z][j]
            return Vector3D(result[0], result[1], result[2])
        elif isinstance(other, Matrix3D):
            # Matrix * Matrix
            result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        result[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix3D(result)
        else:
            raise TypeError("Can only multiply Matrix3D with Matrix3D or Vector3D")
    
    def transpose(self) -> 'Matrix3D':
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                result[i][j] = self.data[j][i]
        return Matrix3D(result)
    
    def determinant(self) -> float:
        return (self.data[0][0] * (self.data[1][1] * self.data[2][2] - self.data[1][2] * self.data[2][1]) -
                self.data[0][1] * (self.data[1][0] * self.data[2][2] - self.data[1][2] * self.data[2][0]) +
                self.data[0][2] * (self.data[1][0] * self.data[2][1] - self.data[1][1] * self.data[2][0]))

class Transform3D:
    """3D transformation class for testing"""
    
    def __init__(self):
        self.translation = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.scale = Vector3D(1, 1, 1)
    
    def translate(self, x: float, y: float, z: float):
        self.translation = Vector3D(x, y, z)
    
    def rotate(self, x: float, y: float, z: float):
        self.rotation = Vector3D(x, y, z)
    
    def scale(self, x: float, y: float, z: float):
        self.scale = Vector3D(x, y, z)
    
    def apply_to_point(self, point: Vector3D) -> Vector3D:
        # Apply scale
        scaled = point * self.scale.x  # Simplified for testing
        
        # Apply rotation (simplified)
        rotated = Vector3D(
            scaled.x * math.cos(self.rotation.x),
            scaled.y * math.cos(self.rotation.y),
            scaled.z * math.cos(self.rotation.z)
        )
        
        # Apply translation
        return rotated + self.translation

class GraphicsObject:
    """Graphics object class for testing"""
    
    def __init__(self, name: str, position: Vector3D):
        self.name = name
        self.position = position
        self.visible = True
        self.transform = Transform3D()
    
    def move_to(self, new_position: Vector3D):
        self.position = new_position
    
    def set_visibility(self, visible: bool):
        self.visible = visible
    
    def get_bounding_box(self) -> Tuple[Vector3D, Vector3D]:
        # Simplified bounding box calculation
        min_point = Vector3D(self.position.x - 1, self.position.y - 1, self.position.z - 1)
        max_point = Vector3D(self.position.x + 1, self.position.y + 1, self.position.z + 1)
        return min_point, max_point

class SceneManager:
    """Scene manager class for testing"""
    
    def __init__(self):
        self.objects = {}
        self.camera_position = Vector3D(0, 0, -10)
        self.camera_target = Vector3D(0, 0, 0)
    
    def add_object(self, obj: GraphicsObject):
        self.objects[obj.name] = obj
    
    def remove_object(self, name: str):
        if name in self.objects:
            del self.objects[name]
    
    def get_object(self, name: str) -> Optional[GraphicsObject]:
        return self.objects.get(name)
    
    def get_visible_objects(self) -> List[GraphicsObject]:
        return [obj for obj in self.objects.values() if obj.visible]
    
    def set_camera(self, position: Vector3D, target: Vector3D):
        self.camera_position = position
        self.camera_target = target
    
    def get_camera_direction(self) -> Vector3D:
        return (self.camera_target - self.camera_position).normalize()

# ============================================================================
# UNIT TESTS
# ============================================================================

class TestVector3D(unittest.TestCase):
    """Test cases for Vector3D class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.v1 = Vector3D(1, 2, 3)
        self.v2 = Vector3D(4, 5, 6)
        self.v3 = Vector3D(0, 0, 0)
    
    def test_vector_creation(self):
        """Test vector creation"""
        self.assertEqual(self.v1.x, 1)
        self.assertEqual(self.v1.y, 2)
        self.assertEqual(self.v1.z, 3)
    
    def test_vector_addition(self):
        """Test vector addition"""
        result = self.v1 + self.v2
        expected = Vector3D(5, 7, 9)
        self.assertEqual(result.x, expected.x)
        self.assertEqual(result.y, expected.y)
        self.assertEqual(result.z, expected.z)
    
    def test_vector_subtraction(self):
        """Test vector subtraction"""
        result = self.v2 - self.v1
        expected = Vector3D(3, 3, 3)
        self.assertEqual(result.x, expected.x)
        self.assertEqual(result.y, expected.y)
        self.assertEqual(result.z, expected.z)
    
    def test_vector_scalar_multiplication(self):
        """Test vector scalar multiplication"""
        result = self.v1 * 2
        expected = Vector3D(2, 4, 6)
        self.assertEqual(result.x, expected.x)
        self.assertEqual(result.y, expected.y)
        self.assertEqual(result.z, expected.z)
    
    def test_vector_magnitude(self):
        """Test vector magnitude calculation"""
        result = self.v1.magnitude()
        expected = math.sqrt(1**2 + 2**2 + 3**2)
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_vector_normalization(self):
        """Test vector normalization"""
        result = self.v1.normalize()
        magnitude = result.magnitude()
        self.assertAlmostEqual(magnitude, 1.0, places=6)
    
    def test_zero_vector_normalization(self):
        """Test normalization of zero vector"""
        result = self.v3.normalize()
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)
    
    def test_vector_dot_product(self):
        """Test vector dot product"""
        result = self.v1.dot(self.v2)
        expected = 1*4 + 2*5 + 3*6
        self.assertEqual(result, expected)
    
    def test_vector_cross_product(self):
        """Test vector cross product"""
        result = self.v1.cross(self.v2)
        expected = Vector3D(
            2*6 - 3*5,
            3*4 - 1*6,
            1*5 - 2*4
        )
        self.assertEqual(result.x, expected.x)
        self.assertEqual(result.y, expected.y)
        self.assertEqual(result.z, expected.z)

class TestMatrix3D(unittest.TestCase):
    """Test cases for Matrix3D class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.m1 = Matrix3D([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.m2 = Matrix3D([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        self.v = Vector3D(1, 2, 3)
    
    def test_matrix_creation(self):
        """Test matrix creation"""
        self.assertEqual(self.m1.data[0][0], 1)
        self.assertEqual(self.m1.data[1][1], 5)
        self.assertEqual(self.m1.data[2][2], 9)
    
    def test_matrix_creation_invalid_size(self):
        """Test matrix creation with invalid size"""
        with self.assertRaises(ValueError):
            Matrix3D([[1, 2], [3, 4]])  # 2x2 matrix
    
    def test_matrix_vector_multiplication(self):
        """Test matrix-vector multiplication"""
        result = self.m1 * self.v
        expected = Vector3D(1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3, 7*1 + 8*2 + 9*3)
        self.assertEqual(result.x, expected.x)
        self.assertEqual(result.y, expected.y)
        self.assertEqual(result.z, expected.z)
    
    def test_matrix_matrix_multiplication(self):
        """Test matrix-matrix multiplication"""
        result = self.m1 * self.m2
        # First element: 1*9 + 2*6 + 3*3 = 9 + 12 + 9 = 30
        self.assertEqual(result.data[0][0], 30)
    
    def test_matrix_transpose(self):
        """Test matrix transpose"""
        result = self.m1.transpose()
        self.assertEqual(result.data[0][1], self.m1.data[1][0])
        self.assertEqual(result.data[1][0], self.m1.data[0][1])
        self.assertEqual(result.data[2][0], self.m1.data[0][2])
    
    def test_matrix_determinant(self):
        """Test matrix determinant calculation"""
        # Identity matrix should have determinant 1
        identity = Matrix3D([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(identity.determinant(), 1)
        
        # Zero matrix should have determinant 0
        zero = Matrix3D([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(zero.determinant(), 0)

class TestTransform3D(unittest.TestCase):
    """Test cases for Transform3D class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.transform = Transform3D()
        self.point = Vector3D(1, 1, 1)
    
    def test_transform_creation(self):
        """Test transform creation"""
        self.assertEqual(self.transform.translation.x, 0)
        self.assertEqual(self.transform.translation.y, 0)
        self.assertEqual(self.transform.translation.z, 0)
        self.assertEqual(self.transform.scale.x, 1)
        self.assertEqual(self.transform.scale.y, 1)
        self.assertEqual(self.transform.scale.z, 1)
    
    def test_translate(self):
        """Test translation"""
        self.transform.translate(1, 2, 3)
        self.assertEqual(self.transform.translation.x, 1)
        self.assertEqual(self.transform.translation.y, 2)
        self.assertEqual(self.transform.translation.z, 3)
    
    def test_rotate(self):
        """Test rotation"""
        self.transform.rotate(0.5, 1.0, 1.5)
        self.assertEqual(self.transform.rotation.x, 0.5)
        self.assertEqual(self.transform.rotation.y, 1.0)
        self.assertEqual(self.transform.rotation.z, 1.5)
    
    def test_scale(self):
        """Test scaling"""
        self.transform.scale(2, 3, 4)
        self.assertEqual(self.transform.scale.x, 2)
        self.assertEqual(self.transform.scale.y, 3)
        self.assertEqual(self.transform.scale.z, 4)
    
    def test_apply_to_point(self):
        """Test applying transform to point"""
        self.transform.translate(1, 1, 1)
        result = self.transform.apply_to_point(self.point)
        # Should apply translation
        self.assertGreater(result.x, self.point.x)
        self.assertGreater(result.y, self.point.y)
        self.assertGreater(result.z, self.point.z)

class TestGraphicsObject(unittest.TestCase):
    """Test cases for GraphicsObject class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.position = Vector3D(0, 0, 0)
        self.obj = GraphicsObject("test_object", self.position)
    
    def test_object_creation(self):
        """Test object creation"""
        self.assertEqual(self.obj.name, "test_object")
        self.assertEqual(self.obj.position.x, 0)
        self.assertEqual(self.obj.position.y, 0)
        self.assertEqual(self.obj.position.z, 0)
        self.assertTrue(self.obj.visible)
    
    def test_move_to(self):
        """Test moving object"""
        new_position = Vector3D(1, 2, 3)
        self.obj.move_to(new_position)
        self.assertEqual(self.obj.position.x, 1)
        self.assertEqual(self.obj.position.y, 2)
        self.assertEqual(self.obj.position.z, 3)
    
    def test_set_visibility(self):
        """Test setting visibility"""
        self.obj.set_visibility(False)
        self.assertFalse(self.obj.visible)
        
        self.obj.set_visibility(True)
        self.assertTrue(self.obj.visible)
    
    def test_get_bounding_box(self):
        """Test bounding box calculation"""
        min_point, max_point = self.obj.get_bounding_box()
        self.assertEqual(min_point.x, -1)
        self.assertEqual(min_point.y, -1)
        self.assertEqual(min_point.z, -1)
        self.assertEqual(max_point.x, 1)
        self.assertEqual(max_point.y, 1)
        self.assertEqual(max_point.z, 1)

class TestSceneManager(unittest.TestCase):
    """Test cases for SceneManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scene = SceneManager()
        self.obj1 = GraphicsObject("obj1", Vector3D(0, 0, 0))
        self.obj2 = GraphicsObject("obj2", Vector3D(1, 1, 1))
    
    def test_scene_creation(self):
        """Test scene creation"""
        self.assertEqual(len(self.scene.objects), 0)
        self.assertEqual(self.scene.camera_position.z, -10)
    
    def test_add_object(self):
        """Test adding object to scene"""
        self.scene.add_object(self.obj1)
        self.assertEqual(len(self.scene.objects), 1)
        self.assertIn("obj1", self.scene.objects)
    
    def test_remove_object(self):
        """Test removing object from scene"""
        self.scene.add_object(self.obj1)
        self.scene.remove_object("obj1")
        self.assertEqual(len(self.scene.objects), 0)
        self.assertNotIn("obj1", self.scene.objects)
    
    def test_get_object(self):
        """Test getting object from scene"""
        self.scene.add_object(self.obj1)
        retrieved = self.scene.get_object("obj1")
        self.assertEqual(retrieved, self.obj1)
        
        # Test getting non-existent object
        retrieved = self.scene.get_object("nonexistent")
        self.assertIsNone(retrieved)
    
    def test_get_visible_objects(self):
        """Test getting visible objects"""
        self.obj2.set_visibility(False)
        self.scene.add_object(self.obj1)
        self.scene.add_object(self.obj2)
        
        visible_objects = self.scene.get_visible_objects()
        self.assertEqual(len(visible_objects), 1)
        self.assertEqual(visible_objects[0].name, "obj1")
    
    def test_set_camera(self):
        """Test setting camera"""
        position = Vector3D(0, 0, -5)
        target = Vector3D(0, 0, 0)
        self.scene.set_camera(position, target)
        
        self.assertEqual(self.scene.camera_position.z, -5)
        self.assertEqual(self.scene.camera_target.z, 0)
    
    def test_get_camera_direction(self):
        """Test getting camera direction"""
        position = Vector3D(0, 0, -10)
        target = Vector3D(0, 0, 0)
        self.scene.set_camera(position, target)
        
        direction = self.scene.get_camera_direction()
        self.assertAlmostEqual(direction.z, 1.0, places=6)

# ============================================================================
# MOCKING EXAMPLES
# ============================================================================

class TestWithMocking(unittest.TestCase):
    """Test cases demonstrating mocking"""
    
    def test_mock_vector_operations(self):
        """Test mocking vector operations"""
        # Create a mock vector
        mock_vector = Mock(spec=Vector3D)
        mock_vector.x = 1
        mock_vector.y = 2
        mock_vector.z = 3
        mock_vector.magnitude.return_value = 3.7416573867739413
        
        # Test the mock
        self.assertEqual(mock_vector.x, 1)
        self.assertEqual(mock_vector.y, 2)
        self.assertEqual(mock_vector.z, 3)
        self.assertAlmostEqual(mock_vector.magnitude(), 3.7416573867739413)
        
        # Verify the method was called
        mock_vector.magnitude.assert_called_once()
    
    def test_mock_scene_manager(self):
        """Test mocking scene manager"""
        # Create a mock scene manager
        mock_scene = Mock(spec=SceneManager)
        mock_scene.get_object.return_value = GraphicsObject("mocked_obj", Vector3D(0, 0, 0))
        
        # Test the mock
        obj = mock_scene.get_object("any_name")
        self.assertEqual(obj.name, "mocked_obj")
        
        # Verify the method was called with correct argument
        mock_scene.get_object.assert_called_once_with("any_name")
    
    @patch('random.uniform')
    def test_mock_random_function(self, mock_uniform):
        """Test mocking random function"""
        # Set up the mock to return a specific value
        mock_uniform.return_value = 0.5
        
        # Call the function that uses random.uniform
        result = random.uniform(0, 1)
        
        # Verify the result
        self.assertEqual(result, 0.5)
        mock_uniform.assert_called_once_with(0, 1)
    
    def test_mock_file_operations(self):
        """Test mocking file operations"""
        with patch('builtins.open', mock_open(read_data='test data')) as mock_file:
            with open('test.txt', 'r') as f:
                content = f.read()
            
            self.assertEqual(content, 'test data')
            mock_file.assert_called_once_with('test.txt', 'r')

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration test cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scene = SceneManager()
        self.transform = Transform3D()
    
    def test_scene_with_transformed_objects(self):
        """Test scene with transformed objects"""
        # Create objects with transforms
        obj1 = GraphicsObject("obj1", Vector3D(0, 0, 0))
        obj2 = GraphicsObject("obj2", Vector3D(1, 1, 1))
        
        # Apply transforms
        obj1.transform.translate(1, 1, 1)
        obj2.transform.scale(2, 2, 2)
        
        # Add to scene
        self.scene.add_object(obj1)
        self.scene.add_object(obj2)
        
        # Test scene operations
        self.assertEqual(len(self.scene.objects), 2)
        self.assertIn("obj1", self.scene.objects)
        self.assertIn("obj2", self.scene.objects)
        
        # Test object retrieval and transform application
        retrieved_obj1 = self.scene.get_object("obj1")
        transformed_point = retrieved_obj1.transform.apply_to_point(Vector3D(0, 0, 0))
        self.assertGreater(transformed_point.x, 0)
    
    def test_vector_matrix_operations(self):
        """Test vector and matrix operations together"""
        # Create vector and matrix
        vector = Vector3D(1, 2, 3)
        matrix = Matrix3D([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity matrix
        
        # Test matrix-vector multiplication
        result = matrix * vector
        self.assertEqual(result.x, vector.x)
        self.assertEqual(result.y, vector.y)
        self.assertEqual(result.z, vector.z)
        
        # Test with non-identity matrix
        scale_matrix = Matrix3D([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        scaled_result = scale_matrix * vector
        self.assertEqual(scaled_result.x, vector.x * 2)
        self.assertEqual(scaled_result.y, vector.y * 2)
        self.assertEqual(scaled_result.z, vector.z * 2)

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Performance test cases"""
    
    def test_vector_operations_performance(self):
        """Test performance of vector operations"""
        vectors = [Vector3D(random.uniform(-10, 10), 
                           random.uniform(-10, 10), 
                           random.uniform(-10, 10)) for _ in range(1000)]
        
        start_time = time.time()
        
        # Perform operations
        for v in vectors:
            v.normalize()
            v.magnitude()
            v * 2.0
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertion (should complete within reasonable time)
        self.assertLess(duration, 1.0, f"Vector operations took {duration:.4f} seconds")
    
    def test_matrix_operations_performance(self):
        """Test performance of matrix operations"""
        matrices = [Matrix3D([[random.uniform(-10, 10) for _ in range(3)] 
                             for _ in range(3)]) for _ in range(100)]
        
        start_time = time.time()
        
        # Perform operations
        for m in matrices:
            m.transpose()
            m.determinant()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertion
        self.assertLess(duration, 1.0, f"Matrix operations took {duration:.4f} seconds")

# ============================================================================
# TEST SUITES
# ============================================================================

def create_test_suite():
    """Create a comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestVector3D))
    suite.addTest(unittest.makeSuite(TestMatrix3D))
    suite.addTest(unittest.makeSuite(TestTransform3D))
    suite.addTest(unittest.makeSuite(TestGraphicsObject))
    suite.addTest(unittest.makeSuite(TestSceneManager))
    suite.addTest(unittest.makeSuite(TestWithMocking))
    suite.addTest(unittest.makeSuite(TestIntegration))
    suite.addTest(unittest.makeSuite(TestPerformance))
    
    return suite

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all tests"""
    print("=== Running 3D Graphics Tests ===\n")
    
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

# ============================================================================
# TEST-DRIVEN DEVELOPMENT EXAMPLE
# ============================================================================

class TestDrivenDevelopment:
    """Example of test-driven development approach"""
    
    def test_quaternion_creation(self):
        """Test quaternion creation (TDD example)"""
        # First, write the test for functionality that doesn't exist yet
        # This will fail initially, which is expected in TDD
        
        # Test quaternion creation
        q = Quaternion(1, 2, 3, 4)  # This class doesn't exist yet
        self.assertEqual(q.w, 1)
        self.assertEqual(q.x, 2)
        self.assertEqual(q.y, 3)
        self.assertEqual(q.z, 4)
    
    def test_quaternion_multiplication(self):
        """Test quaternion multiplication (TDD example)"""
        q1 = Quaternion(1, 0, 0, 0)  # Identity quaternion
        q2 = Quaternion(0, 1, 0, 0)  # i quaternion
        
        result = q1 * q2
        self.assertEqual(result.w, 0)
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate testing"""
    print("=== Testing Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Run all tests
    success = run_tests()
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    print("\nKey testing concepts demonstrated:")
    print("✓ Unit Tests: Individual component testing")
    print("✓ Test Fixtures: setUp and tearDown methods")
    print("✓ Assertions: Various assertion methods")
    print("✓ Mocking: Mock objects and patching")
    print("✓ Integration Tests: Component interaction testing")
    print("✓ Performance Tests: Performance benchmarking")
    print("✓ Test Suites: Organized test collections")
    print("✓ Test-Driven Development: TDD approach")
    
    print("\nBest practices:")
    print("• Write tests before implementing features (TDD)")
    print("• Use descriptive test method names")
    print("• Test both valid and invalid inputs")
    print("• Mock external dependencies")
    print("• Keep tests independent and isolated")
    print("• Use appropriate assertions for the data type")
    print("• Test edge cases and boundary conditions")

if __name__ == "__main__":
    main()
