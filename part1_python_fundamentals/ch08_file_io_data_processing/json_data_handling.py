#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
JSON Data Handling Example

Demonstrates JSON file processing for configurations, data serialization,
and structured data handling in 3D graphics applications.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import math

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics JSON Handler"
__description__ = "JSON data handling for 3D graphics applications"

# ============================================================================
# JSON HANDLER CLASS
# ============================================================================

class JSONHandler:
    """Class for handling JSON file operations"""
    
    def __init__(self, base_directory: str = "json_data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.encoding = "utf-8"
    
    def save_json(self, filename: str, data: Any, indent: int = 2, 
                  ensure_ascii: bool = False) -> bool:
        """Save data to JSON file"""
        try:
            file_path = self.base_directory / filename
            
            with open(file_path, 'w', encoding=self.encoding) as file:
                json.dump(data, file, indent=indent, ensure_ascii=ensure_ascii)
            
            print(f"Successfully saved JSON file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON file {filename}: {e}")
            return False
    
    def load_json(self, filename: str) -> Optional[Any]:
        """Load data from JSON file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"JSON file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding=self.encoding) as file:
                data = json.load(file)
            
            print(f"Successfully loaded JSON file: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in {filename}: {e}")
            return None
        except Exception as e:
            print(f"Error loading JSON file {filename}: {e}")
            return None
    
    def validate_json(self, filename: str) -> bool:
        """Validate JSON file syntax"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"JSON file not found: {file_path}")
                return False
            
            with open(file_path, 'r', encoding=self.encoding) as file:
                json.load(file)
            
            print(f"JSON file is valid: {file_path}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {filename}: {e}")
            return False
        except Exception as e:
            print(f"Error validating JSON file {filename}: {e}")
            return False
    
    def pretty_print_json(self, filename: str) -> bool:
        """Pretty print JSON file content"""
        try:
            data = self.load_json(filename)
            if data is not None:
                print(f"\nPretty-printed JSON from {filename}:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return True
            return False
            
        except Exception as e:
            print(f"Error pretty printing JSON file {filename}: {e}")
            return False

# ============================================================================
# 3D GRAPHICS JSON HANDLERS
# ============================================================================

class GraphicsJSONHandler:
    """Specialized JSON handler for 3D graphics applications"""
    
    def __init__(self, base_directory: str = "graphics_json"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.json_handler = JSONHandler(str(self.base_directory))
    
    def save_scene_config(self, scene_name: str, config_data: Dict[str, Any]) -> bool:
        """Save 3D scene configuration to JSON"""
        try:
            # Add metadata
            config_with_metadata = {
                "metadata": {
                    "scene_name": scene_name,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": f"3D scene configuration for {scene_name}"
                },
                "configuration": config_data
            }
            
            filename = f"{scene_name}_config.json"
            return self.json_handler.save_json(filename, config_with_metadata)
            
        except Exception as e:
            print(f"Error saving scene config: {e}")
            return False
    
    def load_scene_config(self, scene_name: str) -> Optional[Dict[str, Any]]:
        """Load 3D scene configuration from JSON"""
        try:
            filename = f"{scene_name}_config.json"
            data = self.json_handler.load_json(filename)
            
            if data and "configuration" in data:
                return data["configuration"]
            return data
            
        except Exception as e:
            print(f"Error loading scene config: {e}")
            return None
    
    def save_material_library(self, library_name: str, materials: Dict[str, Any]) -> bool:
        """Save material library to JSON"""
        try:
            # Add metadata
            library_with_metadata = {
                "metadata": {
                    "library_name": library_name,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "material_count": len(materials),
                    "description": f"Material library: {library_name}"
                },
                "materials": materials
            }
            
            filename = f"{library_name}_materials.json"
            return self.json_handler.save_json(filename, library_with_metadata)
            
        except Exception as e:
            print(f"Error saving material library: {e}")
            return False
    
    def load_material_library(self, library_name: str) -> Optional[Dict[str, Any]]:
        """Load material library from JSON"""
        try:
            filename = f"{library_name}_materials.json"
            data = self.json_handler.load_json(filename)
            
            if data and "materials" in data:
                return data["materials"]
            return data
            
        except Exception as e:
            print(f"Error loading material library: {e}")
            return None
    
    def save_animation_data(self, animation_name: str, animation_data: Dict[str, Any]) -> bool:
        """Save animation data to JSON"""
        try:
            # Add metadata
            animation_with_metadata = {
                "metadata": {
                    "animation_name": animation_name,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": f"Animation data: {animation_name}"
                },
                "animation": animation_data
            }
            
            filename = f"{animation_name}_animation.json"
            return self.json_handler.save_json(filename, animation_with_metadata)
            
        except Exception as e:
            print(f"Error saving animation data: {e}")
            return False
    
    def load_animation_data(self, animation_name: str) -> Optional[Dict[str, Any]]:
        """Load animation data from JSON"""
        try:
            filename = f"{animation_name}_animation.json"
            data = self.json_handler.load_json(filename)
            
            if data and "animation" in data:
                return data["animation"]
            return data
            
        except Exception as e:
            print(f"Error loading animation data: {e}")
            return None
    
    def save_project_settings(self, project_name: str, settings: Dict[str, Any]) -> bool:
        """Save project settings to JSON"""
        try:
            # Add metadata
            settings_with_metadata = {
                "metadata": {
                    "project_name": project_name,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": f"Project settings: {project_name}"
                },
                "settings": settings
            }
            
            filename = f"{project_name}_settings.json"
            return self.json_handler.save_json(filename, settings_with_metadata)
            
        except Exception as e:
            print(f"Error saving project settings: {e}")
            return False
    
    def load_project_settings(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Load project settings from JSON"""
        try:
            filename = f"{project_name}_settings.json"
            data = self.json_handler.load_json(filename)
            
            if data and "settings" in data:
                return data["settings"]
            return data
            
        except Exception as e:
            print(f"Error loading project settings: {e}")
            return None

# ============================================================================
# JSON SCHEMA VALIDATOR
# ============================================================================

class JSONSchemaValidator:
    """Simple JSON schema validator for 3D graphics data"""
    
    @staticmethod
    def validate_scene_config(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scene configuration schema"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["camera", "lighting", "rendering"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Camera validation
        if "camera" in data:
            camera = data["camera"]
            if not isinstance(camera, dict):
                errors.append("Camera must be an object")
            else:
                if "position" not in camera:
                    errors.append("Camera missing position")
                elif not isinstance(camera["position"], list) or len(camera["position"]) != 3:
                    errors.append("Camera position must be a 3D vector")
        
        # Lighting validation
        if "lighting" in data:
            lighting = data["lighting"]
            if not isinstance(lighting, dict):
                errors.append("Lighting must be an object")
            else:
                if "ambient" not in lighting:
                    warnings.append("Lighting missing ambient light")
        
        # Rendering validation
        if "rendering" in data:
            rendering = data["rendering"]
            if not isinstance(rendering, dict):
                errors.append("Rendering must be an object")
            else:
                if "resolution" not in rendering:
                    warnings.append("Rendering missing resolution")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def validate_material(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate material schema"""
        errors = []
        warnings = []
        
        # Required fields
        if "name" not in data:
            errors.append("Material missing name")
        
        if "diffuse_color" not in data:
            errors.append("Material missing diffuse_color")
        elif not isinstance(data["diffuse_color"], list) or len(data["diffuse_color"]) != 3:
            errors.append("Diffuse color must be a 3D vector")
        
        # Optional fields validation
        if "specular_color" in data:
            if not isinstance(data["specular_color"], list) or len(data["specular_color"]) != 3:
                errors.append("Specular color must be a 3D vector")
        
        if "shininess" in data:
            if not isinstance(data["shininess"], (int, float)) or data["shininess"] < 0:
                errors.append("Shininess must be a positive number")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_json_operations():
    """Demonstrate basic JSON operations"""
    print("=== Basic JSON Operations Demo ===\n")
    
    # Create JSON handler
    json_handler = JSONHandler("json_demo")
    
    # Create sample data
    sample_data = {
        "name": "demo_object",
        "position": [1.0, 2.0, 3.0],
        "rotation": [0.0, 45.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "properties": {
            "visible": True,
            "cast_shadow": True,
            "receive_shadow": True
        },
        "tags": ["demo", "3d", "object"]
    }
    
    # Save JSON file
    print("1. Saving JSON file...")
    json_handler.save_json("sample_data.json", sample_data)
    
    # Load JSON file
    print("\n2. Loading JSON file...")
    loaded_data = json_handler.load_json("sample_data.json")
    if loaded_data:
        print("Loaded data:")
        print(f"  Name: {loaded_data['name']}")
        print(f"  Position: {loaded_data['position']}")
        print(f"  Properties: {loaded_data['properties']}")
    
    # Validate JSON file
    print("\n3. Validating JSON file...")
    is_valid = json_handler.validate_json("sample_data.json")
    print(f"JSON is valid: {is_valid}")
    
    # Pretty print JSON
    print("\n4. Pretty printing JSON...")
    json_handler.pretty_print_json("sample_data.json")
    
    print()

def demonstrate_graphics_json_handling():
    """Demonstrate 3D graphics JSON handling"""
    print("=== 3D Graphics JSON Handling Demo ===\n")
    
    # Create graphics JSON handler
    graphics_handler = GraphicsJSONHandler("graphics_json_demo")
    
    # Save scene configuration
    print("1. Saving scene configuration...")
    scene_config = {
        "camera": {
            "position": [0.0, 5.0, -10.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov": 60.0,
            "near": 0.1,
            "far": 1000.0
        },
        "lighting": {
            "ambient": [0.1, 0.1, 0.1],
            "directional_lights": [
                {
                    "direction": [-1.0, -1.0, -1.0],
                    "color": [1.0, 1.0, 1.0],
                    "intensity": 1.0
                }
            ],
            "point_lights": [
                {
                    "position": [10.0, 10.0, 10.0],
                    "color": [1.0, 0.8, 0.6],
                    "intensity": 0.5,
                    "range": 50.0
                }
            ]
        },
        "rendering": {
            "resolution": [1920, 1080],
            "antialiasing": True,
            "shadows": True,
            "fog": {
                "enabled": True,
                "color": [0.5, 0.5, 0.5],
                "density": 0.01
            }
        },
        "objects": [
            {
                "name": "ground",
                "type": "plane",
                "position": [0.0, 0.0, 0.0],
                "scale": [10.0, 1.0, 10.0],
                "material": "ground_material"
            },
            {
                "name": "cube",
                "type": "cube",
                "position": [0.0, 1.0, 0.0],
                "rotation": [0.0, 45.0, 0.0],
                "material": "cube_material"
            }
        ]
    }
    
    graphics_handler.save_scene_config("demo_scene", scene_config)
    
    # Load scene configuration
    print("\n2. Loading scene configuration...")
    loaded_scene = graphics_handler.load_scene_config("demo_scene")
    if loaded_scene:
        print("Loaded scene config:")
        print(f"  Camera position: {loaded_scene['camera']['position']}")
        print(f"  Resolution: {loaded_scene['rendering']['resolution']}")
        print(f"  Object count: {len(loaded_scene['objects'])}")
    
    # Save material library
    print("\n3. Saving material library...")
    material_library = {
        "ground_material": {
            "name": "ground_material",
            "diffuse_color": [0.3, 0.3, 0.3],
            "specular_color": [0.1, 0.1, 0.1],
            "ambient_color": [0.1, 0.1, 0.1],
            "shininess": 16.0,
            "texture_file": "ground_diffuse.png",
            "normal_map": "ground_normal.png"
        },
        "cube_material": {
            "name": "cube_material",
            "diffuse_color": [1.0, 0.5, 0.2],
            "specular_color": [1.0, 1.0, 1.0],
            "ambient_color": [0.2, 0.1, 0.05],
            "shininess": 32.0,
            "texture_file": "cube_diffuse.png",
            "metallic": 0.8,
            "roughness": 0.2
        },
        "metal_material": {
            "name": "metal_material",
            "diffuse_color": [0.8, 0.8, 0.8],
            "specular_color": [1.0, 1.0, 1.0],
            "ambient_color": [0.2, 0.2, 0.2],
            "shininess": 128.0,
            "metallic": 1.0,
            "roughness": 0.1
        }
    }
    
    graphics_handler.save_material_library("demo_materials", material_library)
    
    # Load material library
    print("\n4. Loading material library...")
    loaded_materials = graphics_handler.load_material_library("demo_materials")
    if loaded_materials:
        print("Loaded materials:")
        for material_name, material_data in loaded_materials.items():
            print(f"  {material_name}: {material_data['diffuse_color']}")
    
    # Save animation data
    print("\n5. Saving animation data...")
    animation_data = {
        "name": "cube_rotation",
        "duration": 5.0,
        "keyframes": [
            {
                "time": 0.0,
                "rotation": [0.0, 0.0, 0.0]
            },
            {
                "time": 2.5,
                "rotation": [0.0, 180.0, 0.0]
            },
            {
                "time": 5.0,
                "rotation": [0.0, 360.0, 0.0]
            }
        ],
        "interpolation": "linear",
        "target_object": "cube"
    }
    
    graphics_handler.save_animation_data("cube_rotation", animation_data)
    
    # Load animation data
    print("\n6. Loading animation data...")
    loaded_animation = graphics_handler.load_animation_data("cube_rotation")
    if loaded_animation:
        print("Loaded animation:")
        print(f"  Name: {loaded_animation['name']}")
        print(f"  Duration: {loaded_animation['duration']}s")
        print(f"  Keyframes: {len(loaded_animation['keyframes'])}")
    
    print()

def demonstrate_json_validation():
    """Demonstrate JSON schema validation"""
    print("=== JSON Schema Validation Demo ===\n")
    
    # Create validator
    validator = JSONSchemaValidator()
    
    # Test valid scene config
    print("1. Testing valid scene configuration...")
    valid_scene = {
        "camera": {
            "position": [0.0, 5.0, -10.0],
            "target": [0.0, 0.0, 0.0]
        },
        "lighting": {
            "ambient": [0.1, 0.1, 0.1]
        },
        "rendering": {
            "resolution": [1920, 1080]
        }
    }
    
    result = validator.validate_scene_config(valid_scene)
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    
    # Test invalid scene config
    print("\n2. Testing invalid scene configuration...")
    invalid_scene = {
        "camera": "invalid_camera",  # Should be dict
        "lighting": {
            "ambient": [0.1, 0.1, 0.1]
        }
        # Missing rendering
    }
    
    result = validator.validate_scene_config(invalid_scene)
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    
    # Test valid material
    print("\n3. Testing valid material...")
    valid_material = {
        "name": "test_material",
        "diffuse_color": [1.0, 0.5, 0.2],
        "specular_color": [1.0, 1.0, 1.0],
        "shininess": 32.0
    }
    
    result = validator.validate_material(valid_material)
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    
    # Test invalid material
    print("\n4. Testing invalid material...")
    invalid_material = {
        "diffuse_color": "invalid_color",  # Should be list
        "shininess": -5.0  # Should be positive
    }
    
    result = validator.validate_material(invalid_material)
    print(f"Valid: {result['valid']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    
    print()

def demonstrate_json_utilities():
    """Demonstrate JSON utility functions"""
    print("=== JSON Utilities Demo ===\n")
    
    # Create JSON handler
    json_handler = JSONHandler("json_utilities_demo")
    
    # Create complex nested data
    complex_data = {
        "project": {
            "name": "3D Graphics Demo",
            "version": "1.0.0",
            "author": "Demo Author",
            "created": datetime.now().isoformat(),
            "description": "A comprehensive 3D graphics demo project"
        },
        "scenes": [
            {
                "name": "main_scene",
                "objects": [
                    {
                        "type": "mesh",
                        "name": "cube",
                        "transform": {
                            "position": [0.0, 0.0, 0.0],
                            "rotation": [0.0, 0.0, 0.0],
                            "scale": [1.0, 1.0, 1.0]
                        },
                        "geometry": {
                            "type": "cube",
                            "size": 1.0
                        },
                        "material": {
                            "name": "default_material",
                            "diffuse": [1.0, 0.5, 0.2],
                            "specular": [1.0, 1.0, 1.0],
                            "shininess": 32.0
                        }
                    },
                    {
                        "type": "light",
                        "name": "main_light",
                        "transform": {
                            "position": [5.0, 5.0, 5.0]
                        },
                        "light": {
                            "type": "point",
                            "color": [1.0, 1.0, 1.0],
                            "intensity": 1.0,
                            "range": 10.0
                        }
                    }
                ],
                "camera": {
                    "position": [0.0, 3.0, -5.0],
                    "target": [0.0, 0.0, 0.0],
                    "fov": 60.0
                }
            }
        ],
        "materials": {
            "default_material": {
                "diffuse": [1.0, 0.5, 0.2],
                "specular": [1.0, 1.0, 1.0],
                "shininess": 32.0
            },
            "metal_material": {
                "diffuse": [0.8, 0.8, 0.8],
                "specular": [1.0, 1.0, 1.0],
                "shininess": 128.0,
                "metallic": 1.0
            }
        },
        "settings": {
            "rendering": {
                "resolution": [1920, 1080],
                "antialiasing": True,
                "shadows": True,
                "fog": False
            },
            "performance": {
                "max_fps": 60,
                "vsync": True,
                "multisampling": 4
            }
        }
    }
    
    # Save complex data
    print("1. Saving complex nested JSON data...")
    json_handler.save_json("complex_project.json", complex_data, indent=4)
    
    # Load and analyze complex data
    print("\n2. Loading and analyzing complex data...")
    loaded_complex = json_handler.load_json("complex_project.json")
    if loaded_complex:
        print("Project analysis:")
        print(f"  Project: {loaded_complex['project']['name']}")
        print(f"  Version: {loaded_complex['project']['version']}")
        print(f"  Scenes: {len(loaded_complex['scenes'])}")
        print(f"  Materials: {len(loaded_complex['materials'])}")
        
        # Analyze first scene
        first_scene = loaded_complex['scenes'][0]
        print(f"  First scene objects: {len(first_scene['objects'])}")
        
        # Count object types
        object_types = {}
        for obj in first_scene['objects']:
            obj_type = obj['type']
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        print(f"  Object types: {object_types}")
    
    # Pretty print the complex data
    print("\n3. Pretty printing complex data...")
    json_handler.pretty_print_json("complex_project.json")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate JSON handling"""
    print("=== JSON Data Handling Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_json_operations()
    demonstrate_graphics_json_handling()
    demonstrate_json_validation()
    demonstrate_json_utilities()
    
    print("="*60)
    print("JSON Data Handling demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic JSON operations: Save, load, validate, pretty print")
    print("✓ 3D graphics data: Scene configs, materials, animations")
    print("✓ Schema validation: Data structure validation")
    print("✓ Metadata handling: Creation timestamps, versions")
    print("✓ Complex nested data: Project structures")
    print("✓ Error handling: JSON decode errors and validation")

if __name__ == "__main__":
    main()
