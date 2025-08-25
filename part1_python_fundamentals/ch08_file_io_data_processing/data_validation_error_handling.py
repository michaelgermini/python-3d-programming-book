#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
Data Validation and Error Handling Example

Demonstrates data validation, error handling, and data integrity techniques
for 3D graphics applications.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import math
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Data Validator"
__description__ = "Data validation and error handling for 3D graphics applications"

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class ValidationError(Exception):
    """Base exception for validation errors"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)

class DataTypeError(ValidationError):
    """Exception for data type validation errors"""
    pass

class RangeError(ValidationError):
    """Exception for value range validation errors"""
    pass

class FormatError(ValidationError):
    """Exception for format validation errors"""
    pass

class FileError(Exception):
    """Base exception for file operation errors"""
    def __init__(self, message: str, file_path: str = None, operation: str = None):
        self.message = message
        self.file_path = file_path
        self.operation = operation
        super().__init__(self.message)

class FileNotFoundError(FileError):
    """Exception for file not found errors"""
    pass

class FilePermissionError(FileError):
    """Exception for file permission errors"""
    pass

class FileCorruptionError(FileError):
    """Exception for corrupted file errors"""
    pass

# ============================================================================
# VALIDATION RULES
# ============================================================================

class ValidationRule:
    """Base class for validation rules"""
    
    def __init__(self, field_name: str, required: bool = True):
        self.field_name = field_name
        self.required = required
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value and return (is_valid, error_message)"""
        raise NotImplementedError("Subclasses must implement validate method")

class TypeRule(ValidationRule):
    """Rule for validating data types"""
    
    def __init__(self, field_name: str, expected_type: type, required: bool = True):
        super().__init__(field_name, required)
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.required:
                return False, f"Field '{self.field_name}' is required"
            return True, None
        
        if not isinstance(value, self.expected_type):
            return False, f"Field '{self.field_name}' must be of type {self.expected_type.__name__}, got {type(value).__name__}"
        
        return True, None

class RangeRule(ValidationRule):
    """Rule for validating value ranges"""
    
    def __init__(self, field_name: str, min_value: float = None, max_value: float = None, required: bool = True):
        super().__init__(field_name, required)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.required:
                return False, f"Field '{self.field_name}' is required"
            return True, None
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, f"Field '{self.field_name}' must be a number"
        
        if self.min_value is not None and num_value < self.min_value:
            return False, f"Field '{self.field_name}' must be >= {self.min_value}"
        
        if self.max_value is not None and num_value > self.max_value:
            return False, f"Field '{self.field_name}' must be <= {self.max_value}"
        
        return True, None

class LengthRule(ValidationRule):
    """Rule for validating string/list lengths"""
    
    def __init__(self, field_name: str, min_length: int = None, max_length: int = None, required: bool = True):
        super().__init__(field_name, required)
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.required:
                return False, f"Field '{self.field_name}' is required"
            return True, None
        
        if not hasattr(value, '__len__'):
            return False, f"Field '{self.field_name}' must have a length"
        
        length = len(value)
        
        if self.min_length is not None and length < self.min_length:
            return False, f"Field '{self.field_name}' must have at least {self.min_length} elements"
        
        if self.max_length is not None and length > self.max_length:
            return False, f"Field '{self.field_name}' must have at most {self.max_length} elements"
        
        return True, None

class PatternRule(ValidationRule):
    """Rule for validating string patterns"""
    
    def __init__(self, field_name: str, pattern: str, required: bool = True):
        super().__init__(field_name, required)
        self.pattern = re.compile(pattern)
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.required:
                return False, f"Field '{self.field_name}' is required"
            return True, None
        
        if not isinstance(value, str):
            return False, f"Field '{self.field_name}' must be a string"
        
        if not self.pattern.match(value):
            return False, f"Field '{self.field_name}' does not match pattern '{self.pattern.pattern}'"
        
        return True, None

class Vector3Rule(ValidationRule):
    """Rule for validating 3D vectors"""
    
    def __init__(self, field_name: str, min_magnitude: float = None, max_magnitude: float = None, required: bool = True):
        super().__init__(field_name, required)
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        if value is None:
            if self.required:
                return False, f"Field '{self.field_name}' is required"
            return True, None
        
        # Check if it's a list/tuple with 3 elements
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False, f"Field '{self.field_name}' must be a 3D vector (list/tuple with 3 elements)"
        
        # Check if all elements are numbers
        try:
            x, y, z = float(value[0]), float(value[1]), float(value[2])
        except (ValueError, TypeError):
            return False, f"Field '{self.field_name}' must contain numeric values"
        
        # Check magnitude if specified
        if self.min_magnitude is not None or self.max_magnitude is not None:
            magnitude = math.sqrt(x*x + y*y + z*z)
            
            if self.min_magnitude is not None and magnitude < self.min_magnitude:
                return False, f"Field '{self.field_name}' magnitude must be >= {self.min_magnitude}"
            
            if self.max_magnitude is not None and magnitude > self.max_magnitude:
                return False, f"Field '{self.field_name}' magnitude must be <= {self.max_magnitude}"
        
        return True, None

# ============================================================================
# DATA VALIDATOR CLASS
# ============================================================================

class DataValidator:
    """Class for validating data structures"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.errors: List[str] = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against all rules"""
        self.errors.clear()
        is_valid = True
        
        for rule in self.rules:
            value = data.get(rule.field_name)
            valid, error = rule.validate(value)
            
            if not valid:
                self.errors.append(error)
                is_valid = False
        
        return is_valid
    
    def get_errors(self) -> List[str]:
        """Get all validation errors"""
        return self.errors.copy()
    
    def clear_errors(self):
        """Clear all validation errors"""
        self.errors.clear()

# ============================================================================
# 3D GRAPHICS DATA VALIDATORS
# ============================================================================

class Vector3Validator(DataValidator):
    """Validator for 3D vector data"""
    
    def __init__(self):
        super().__init__()
        self.add_rule(Vector3Rule("position", required=True))
        self.add_rule(Vector3Rule("rotation", required=True))
        self.add_rule(Vector3Rule("scale", min_magnitude=0.001, required=True))

class MaterialValidator(DataValidator):
    """Validator for material data"""
    
    def __init__(self):
        super().__init__()
        self.add_rule(TypeRule("name", str, required=True))
        self.add_rule(Vector3Rule("diffuse_color", required=True))
        self.add_rule(Vector3Rule("specular_color", required=True))
        self.add_rule(RangeRule("shininess", min_value=0.0, max_value=1000.0, required=True))
        self.add_rule(RangeRule("opacity", min_value=0.0, max_value=1.0, required=False))

class MeshValidator(DataValidator):
    """Validator for mesh data"""
    
    def __init__(self):
        super().__init__()
        self.add_rule(TypeRule("name", str, required=True))
        self.add_rule(TypeRule("vertices", list, required=True))
        self.add_rule(TypeRule("indices", list, required=True))
        self.add_rule(LengthRule("vertices", min_length=3, required=True))
        self.add_rule(LengthRule("indices", min_length=3, required=True))

class SceneValidator(DataValidator):
    """Validator for scene data"""
    
    def __init__(self):
        super().__init__()
        self.add_rule(TypeRule("name", str, required=True))
        self.add_rule(TypeRule("objects", list, required=True))
        self.add_rule(TypeRule("camera", dict, required=True))
        self.add_rule(TypeRule("lights", list, required=True))

# ============================================================================
# ERROR HANDLER CLASS
# ============================================================================

class ErrorHandler:
    """Class for handling and logging errors"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.error_count = 0
        self.warning_count = 0
    
    def handle_error(self, error: Exception, context: str = None) -> bool:
        """Handle an error and return whether to continue"""
        self.error_count += 1
        error_message = self._format_error(error, context)
        
        print(f"ERROR: {error_message}")
        
        if self.log_file:
            self._log_error(error_message)
        
        # Return True to continue, False to stop
        return isinstance(error, (ValidationError, FileError))
    
    def handle_warning(self, message: str, context: str = None) -> None:
        """Handle a warning message"""
        self.warning_count += 1
        warning_message = self._format_warning(message, context)
        
        print(f"WARNING: {warning_message}")
        
        if self.log_file:
            self._log_warning(warning_message)
    
    def _format_error(self, error: Exception, context: str = None) -> str:
        """Format an error message"""
        timestamp = datetime.now().isoformat()
        error_type = type(error).__name__
        message = str(error)
        
        if context:
            return f"[{timestamp}] {error_type} in {context}: {message}"
        else:
            return f"[{timestamp}] {error_type}: {message}"
    
    def _format_warning(self, message: str, context: str = None) -> str:
        """Format a warning message"""
        timestamp = datetime.now().isoformat()
        
        if context:
            return f"[{timestamp}] WARNING in {context}: {message}"
        else:
            return f"[{timestamp}] WARNING: {message}"
    
    def _log_error(self, message: str):
        """Log an error to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"Failed to log error: {e}")
    
    def _log_warning(self, message: str):
        """Log a warning to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"Failed to log warning: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get error and warning statistics"""
        return {
            "errors": self.error_count,
            "warnings": self.warning_count
        }

# ============================================================================
# SAFE FILE OPERATIONS
# ============================================================================

class SafeFileOperations:
    """Class for safe file operations with error handling"""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
    
    def safe_read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Safely read a JSON file with error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not os.access(file_path, os.R_OK):
                raise FilePermissionError(f"Cannot read file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except json.JSONDecodeError as e:
            raise FileCorruptionError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            if not self.error_handler.handle_error(e, f"Reading JSON file: {file_path}"):
                raise
            return None
    
    def safe_write_json(self, file_path: str, data: Dict[str, Any]) -> bool:
        """Safely write a JSON file with error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Check if we can write to the directory
            if not os.access(os.path.dirname(file_path), os.W_OK):
                raise FilePermissionError(f"Cannot write to directory: {os.path.dirname(file_path)}")
            
            # Write to temporary file first
            temp_path = file_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Move temporary file to final location
            os.replace(temp_path, file_path)
            
            return True
            
        except Exception as e:
            if not self.error_handler.handle_error(e, f"Writing JSON file: {file_path}"):
                raise
            return False
    
    def safe_validate_and_save(self, file_path: str, data: Dict[str, Any], 
                              validator: DataValidator) -> bool:
        """Safely validate and save data"""
        try:
            # Validate data
            if not validator.validate(data):
                errors = validator.get_errors()
                raise ValidationError(f"Data validation failed: {'; '.join(errors)}")
            
            # Save data
            return self.safe_write_json(file_path, data)
            
        except Exception as e:
            if not self.error_handler.handle_error(e, f"Validating and saving: {file_path}"):
                raise
            return False

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_validation_rules():
    """Demonstrate validation rules"""
    print("=== Validation Rules Demo ===\n")
    
    # Create a validator for 3D object data
    validator = DataValidator()
    validator.add_rule(TypeRule("name", str, required=True))
    validator.add_rule(Vector3Rule("position", required=True))
    validator.add_rule(Vector3Rule("rotation", required=True))
    validator.add_rule(Vector3Rule("scale", min_magnitude=0.001, required=True))
    validator.add_rule(RangeRule("opacity", min_value=0.0, max_value=1.0, required=False))
    
    # Test valid data
    print("1. Testing valid data...")
    valid_data = {
        "name": "test_object",
        "position": [1.0, 2.0, 3.0],
        "rotation": [0.0, 45.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "opacity": 0.8
    }
    
    if validator.validate(valid_data):
        print("  ✓ Data is valid")
    else:
        print(f"  ✗ Validation failed: {validator.get_errors()}")
    
    # Test invalid data
    print("\n2. Testing invalid data...")
    invalid_data = {
        "name": 123,  # Wrong type
        "position": [1.0, 2.0],  # Not 3D vector
        "rotation": [0.0, 45.0, 0.0],
        "scale": [0.0, 0.0, 0.0],  # Zero magnitude
        "opacity": 1.5  # Out of range
    }
    
    if validator.validate(invalid_data):
        print("  ✓ Data is valid")
    else:
        print("  ✗ Validation failed:")
        for error in validator.get_errors():
            print(f"    - {error}")
    
    print()

def demonstrate_3d_graphics_validators():
    """Demonstrate 3D graphics specific validators"""
    print("=== 3D Graphics Validators Demo ===\n")
    
    # Test Vector3 validator
    print("1. Testing Vector3 validator...")
    vector_validator = Vector3Validator()
    
    valid_vector_data = {
        "position": [1.0, 2.0, 3.0],
        "rotation": [0.0, 45.0, 0.0],
        "scale": [1.0, 1.0, 1.0]
    }
    
    if vector_validator.validate(valid_vector_data):
        print("  ✓ Vector3 data is valid")
    else:
        print(f"  ✗ Vector3 validation failed: {vector_validator.get_errors()}")
    
    # Test Material validator
    print("\n2. Testing Material validator...")
    material_validator = MaterialValidator()
    
    valid_material_data = {
        "name": "metal_material",
        "diffuse_color": [0.8, 0.8, 0.8],
        "specular_color": [1.0, 1.0, 1.0],
        "shininess": 128.0,
        "opacity": 1.0
    }
    
    if material_validator.validate(valid_material_data):
        print("  ✓ Material data is valid")
    else:
        print(f"  ✗ Material validation failed: {material_validator.get_errors()}")
    
    # Test Mesh validator
    print("\n3. Testing Mesh validator...")
    mesh_validator = MeshValidator()
    
    valid_mesh_data = {
        "name": "cube_mesh",
        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        "indices": [0, 1, 2, 0, 2, 3]
    }
    
    if mesh_validator.validate(valid_mesh_data):
        print("  ✓ Mesh data is valid")
    else:
        print(f"  ✗ Mesh validation failed: {mesh_validator.get_errors()}")
    
    print()

def demonstrate_error_handling():
    """Demonstrate error handling"""
    print("=== Error Handling Demo ===\n")
    
    # Create error handler
    error_handler = ErrorHandler("error_log.txt")
    
    # Test different types of errors
    print("1. Testing validation errors...")
    try:
        raise ValidationError("Invalid data format", "position", [1, 2])
    except Exception as e:
        should_continue = error_handler.handle_error(e, "Data validation")
        print(f"  Should continue: {should_continue}")
    
    print("\n2. Testing file errors...")
    try:
        raise FileNotFoundError("Configuration file not found", "config.json", "read")
    except Exception as e:
        should_continue = error_handler.handle_error(e, "File operations")
        print(f"  Should continue: {should_continue}")
    
    print("\n3. Testing warnings...")
    error_handler.handle_warning("Using default material settings", "Material loading")
    error_handler.handle_warning("Texture resolution is low", "Texture loading")
    
    # Show statistics
    print("\n4. Error statistics:")
    stats = error_handler.get_statistics()
    print(f"  Errors: {stats['errors']}")
    print(f"  Warnings: {stats['warnings']}")
    
    print()

def demonstrate_safe_file_operations():
    """Demonstrate safe file operations"""
    print("=== Safe File Operations Demo ===\n")
    
    # Create error handler and safe file operations
    error_handler = ErrorHandler("file_operations_log.txt")
    safe_ops = SafeFileOperations(error_handler)
    
    # Create a validator
    validator = MaterialValidator()
    
    # Test data
    material_data = {
        "name": "test_material",
        "diffuse_color": [1.0, 0.5, 0.2],
        "specular_color": [1.0, 1.0, 1.0],
        "shininess": 32.0,
        "opacity": 1.0
    }
    
    # Test safe validation and save
    print("1. Testing safe validation and save...")
    success = safe_ops.safe_validate_and_save("test_material.json", material_data, validator)
    print(f"  Save successful: {success}")
    
    # Test safe read
    print("\n2. Testing safe read...")
    loaded_data = safe_ops.safe_read_json("test_material.json")
    if loaded_data:
        print(f"  Loaded data: {loaded_data['name']}")
    
    # Test invalid data
    print("\n3. Testing invalid data...")
    invalid_material = {
        "name": 123,  # Wrong type
        "diffuse_color": [1.0, 0.5],  # Not 3D vector
        "specular_color": [1.0, 1.0, 1.0],
        "shininess": -5.0  # Invalid range
    }
    
    success = safe_ops.safe_validate_and_save("invalid_material.json", invalid_material, validator)
    print(f"  Save successful: {success}")
    
    # Clean up
    try:
        os.remove("test_material.json")
        if os.path.exists("invalid_material.json"):
            os.remove("invalid_material.json")
    except:
        pass
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate data validation and error handling"""
    print("=== Data Validation and Error Handling Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_validation_rules()
    demonstrate_3d_graphics_validators()
    demonstrate_error_handling()
    demonstrate_safe_file_operations()
    
    print("="*60)
    print("Data Validation and Error Handling demo completed successfully!")
    print("\nKey features:")
    print("✓ Validation rules: Type, range, length, pattern, vector validation")
    print("✓ 3D graphics validators: Vector3, Material, Mesh, Scene validation")
    print("✓ Error handling: Custom exceptions and error logging")
    print("✓ Safe file operations: Error-safe file I/O with validation")
    print("✓ Data integrity: Ensuring data quality and consistency")

if __name__ == "__main__":
    main()
