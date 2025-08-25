#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
CSV Data Processing Example

Demonstrates CSV file handling for data analysis, export, and processing
in 3D graphics applications.
"""

import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import math
import statistics

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics CSV Processor"
__description__ = "CSV data processing for 3D graphics applications"

# ============================================================================
# CSV HANDLER CLASS
# ============================================================================

class CSVHandler:
    """Class for handling CSV file operations"""
    
    def __init__(self, base_directory: str = "csv_data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.encoding = "utf-8"
    
    def write_csv(self, filename: str, data: List[Dict[str, Any]], 
                  fieldnames: Optional[List[str]] = None, 
                  delimiter: str = ',', 
                  quotechar: str = '"') -> bool:
        """Write data to CSV file"""
        try:
            file_path = self.base_directory / filename
            
            # Determine fieldnames if not provided
            if fieldnames is None and data:
                fieldnames = list(data[0].keys())
            
            with open(file_path, 'w', newline='', encoding=self.encoding) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, 
                                      delimiter=delimiter, quotechar=quotechar)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                writer.writerows(data)
            
            print(f"Successfully wrote CSV file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error writing CSV file {filename}: {e}")
            return False
    
    def read_csv(self, filename: str, delimiter: str = ',', 
                 quotechar: str = '"') -> Optional[List[Dict[str, Any]]]:
        """Read data from CSV file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"CSV file not found: {file_path}")
                return None
            
            data = []
            with open(file_path, 'r', encoding=self.encoding) as file:
                reader = csv.DictReader(file, delimiter=delimiter, quotechar=quotechar)
                
                for row in reader:
                    # Convert numeric strings to appropriate types
                    converted_row = {}
                    for key, value in row.items():
                        converted_row[key] = self._convert_value(value)
                    data.append(converted_row)
            
            print(f"Successfully read CSV file: {file_path}")
            return data
            
        except Exception as e:
            print(f"Error reading CSV file {filename}: {e}")
            return None
    
    def append_csv(self, filename: str, data: List[Dict[str, Any]], 
                   fieldnames: Optional[List[str]] = None,
                   delimiter: str = ',', 
                   quotechar: str = '"') -> bool:
        """Append data to existing CSV file"""
        try:
            file_path = self.base_directory / filename
            
            # Determine fieldnames if not provided
            if fieldnames is None and data:
                fieldnames = list(data[0].keys())
            
            # Check if file exists to determine if we need to write header
            write_header = not file_path.exists()
            
            with open(file_path, 'a', newline='', encoding=self.encoding) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, 
                                      delimiter=delimiter, quotechar=quotechar)
                
                # Write header only if file is new
                if write_header:
                    writer.writeheader()
                
                # Write data rows
                writer.writerows(data)
            
            print(f"Successfully appended to CSV file: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error appending to CSV file {filename}: {e}")
            return False
    
    def get_csv_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get information about CSV file"""
        try:
            file_path = self.base_directory / filename
            
            if not file_path.exists():
                print(f"CSV file not found: {file_path}")
                return None
            
            data = self.read_csv(filename)
            if not data:
                return None
            
            info = {
                "filename": filename,
                "file_size": file_path.stat().st_size,
                "row_count": len(data),
                "column_count": len(data[0]) if data else 0,
                "columns": list(data[0].keys()) if data else [],
                "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting CSV info for {filename}: {e}")
            return None
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert string value to appropriate type"""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        try:
            # Try to convert to int
            return int(value)
        except ValueError:
            try:
                # Try to convert to float
                return float(value)
            except ValueError:
                # Return as string
                return value

# ============================================================================
# 3D GRAPHICS CSV PROCESSORS
# ============================================================================

class GraphicsCSVProcessor:
    """Specialized CSV processor for 3D graphics applications"""
    
    def __init__(self, base_directory: str = "graphics_csv"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.csv_handler = CSVHandler(str(self.base_directory))
    
    def export_vertex_data(self, mesh_name: str, vertices: List[List[float]], 
                          normals: Optional[List[List[float]]] = None,
                          tex_coords: Optional[List[List[float]]] = None) -> bool:
        """Export vertex data to CSV"""
        try:
            data = []
            
            for i, vertex in enumerate(vertices):
                row = {
                    "vertex_id": i,
                    "x": vertex[0],
                    "y": vertex[1],
                    "z": vertex[2]
                }
                
                # Add normals if provided
                if normals and i < len(normals):
                    normal = normals[i]
                    row.update({
                        "nx": normal[0],
                        "ny": normal[1],
                        "nz": normal[2]
                    })
                
                # Add texture coordinates if provided
                if tex_coords and i < len(tex_coords):
                    tex_coord = tex_coords[i]
                    row.update({
                        "u": tex_coord[0],
                        "v": tex_coord[1]
                    })
                
                data.append(row)
            
            filename = f"{mesh_name}_vertices.csv"
            return self.csv_handler.write_csv(filename, data)
            
        except Exception as e:
            print(f"Error exporting vertex data: {e}")
            return False
    
    def export_performance_data(self, performance_data: List[Dict[str, Any]]) -> bool:
        """Export performance data to CSV"""
        try:
            filename = "performance_data.csv"
            return self.csv_handler.write_csv(filename, performance_data)
            
        except Exception as e:
            print(f"Error exporting performance data: {e}")
            return False
    
    def export_scene_analysis(self, scene_name: str, analysis_data: Dict[str, Any]) -> bool:
        """Export scene analysis data to CSV"""
        try:
            # Convert analysis data to list format
            data = []
            
            # Object statistics
            if "objects" in analysis_data:
                for obj_type, count in analysis_data["objects"].items():
                    data.append({
                        "category": "object_type",
                        "name": obj_type,
                        "count": count,
                        "scene": scene_name
                    })
            
            # Material statistics
            if "materials" in analysis_data:
                for material_name, properties in analysis_data["materials"].items():
                    data.append({
                        "category": "material",
                        "name": material_name,
                        "diffuse_r": properties.get("diffuse_color", [0, 0, 0])[0],
                        "diffuse_g": properties.get("diffuse_color", [0, 0, 0])[1],
                        "diffuse_b": properties.get("diffuse_color", [0, 0, 0])[2],
                        "shininess": properties.get("shininess", 0),
                        "scene": scene_name
                    })
            
            # Lighting statistics
            if "lighting" in analysis_data:
                for light_type, count in analysis_data["lighting"].items():
                    data.append({
                        "category": "lighting",
                        "name": light_type,
                        "count": count,
                        "scene": scene_name
                    })
            
            filename = f"{scene_name}_analysis.csv"
            return self.csv_handler.write_csv(filename, data)
            
        except Exception as e:
            print(f"Error exporting scene analysis: {e}")
            return False
    
    def export_animation_data(self, animation_name: str, keyframes: List[Dict[str, Any]]) -> bool:
        """Export animation keyframe data to CSV"""
        try:
            data = []
            
            for i, keyframe in enumerate(keyframes):
                row = {
                    "keyframe_id": i,
                    "time": keyframe.get("time", 0.0),
                    "animation": animation_name
                }
                
                # Add position data
                if "position" in keyframe:
                    pos = keyframe["position"]
                    row.update({
                        "pos_x": pos[0],
                        "pos_y": pos[1],
                        "pos_z": pos[2]
                    })
                
                # Add rotation data
                if "rotation" in keyframe:
                    rot = keyframe["rotation"]
                    row.update({
                        "rot_x": rot[0],
                        "rot_y": rot[1],
                        "rot_z": rot[2]
                    })
                
                # Add scale data
                if "scale" in keyframe:
                    scale = keyframe["scale"]
                    row.update({
                        "scale_x": scale[0],
                        "scale_y": scale[1],
                        "scale_z": scale[2]
                    })
                
                data.append(row)
            
            filename = f"{animation_name}_keyframes.csv"
            return self.csv_handler.write_csv(filename, data)
            
        except Exception as e:
            print(f"Error exporting animation data: {e}")
            return False

# ============================================================================
# CSV DATA ANALYZER
# ============================================================================

class CSVDataAnalyzer:
    """Class for analyzing CSV data"""
    
    @staticmethod
    def analyze_numeric_column(data: List[Dict[str, Any]], column_name: str) -> Dict[str, Any]:
        """Analyze a numeric column in CSV data"""
        try:
            values = []
            for row in data:
                if column_name in row and isinstance(row[column_name], (int, float)):
                    values.append(row[column_name])
            
            if not values:
                return {"error": f"No numeric data found in column '{column_name}'"}
            
            analysis = {
                "column": column_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "sum": sum(values)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing column '{column_name}': {e}"}
    
    @staticmethod
    def analyze_categorical_column(data: List[Dict[str, Any]], column_name: str) -> Dict[str, Any]:
        """Analyze a categorical column in CSV data"""
        try:
            values = []
            for row in data:
                if column_name in row:
                    values.append(str(row[column_name]))
            
            if not values:
                return {"error": f"No data found in column '{column_name}'"}
            
            # Count occurrences
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1
            
            # Sort by count
            sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
            
            analysis = {
                "column": column_name,
                "count": len(values),
                "unique_values": len(value_counts),
                "most_common": sorted_counts[0] if sorted_counts else None,
                "value_counts": dict(sorted_counts)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing column '{column_name}': {e}"}
    
    @staticmethod
    def filter_data(data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter CSV data based on conditions"""
        try:
            filtered_data = []
            
            for row in data:
                match = True
                for column, condition in filters.items():
                    if column not in row:
                        match = False
                        break
                    
                    value = row[column]
                    
                    if isinstance(condition, dict):
                        # Complex condition
                        if "min" in condition and value < condition["min"]:
                            match = False
                        if "max" in condition and value > condition["max"]:
                            match = False
                        if "equals" in condition and value != condition["equals"]:
                            match = False
                        if "contains" in condition and str(condition["contains"]) not in str(value):
                            match = False
                    else:
                        # Simple equality
                        if value != condition:
                            match = False
                
                if match:
                    filtered_data.append(row)
            
            return filtered_data
            
        except Exception as e:
            print(f"Error filtering data: {e}")
            return []
    
    @staticmethod
    def group_data(data: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group CSV data by a column"""
        try:
            grouped = {}
            
            for row in data:
                if group_by in row:
                    key = str(row[group_by])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(row)
            
            return grouped
            
        except Exception as e:
            print(f"Error grouping data: {e}")
            return {}

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_csv_operations():
    """Demonstrate basic CSV operations"""
    print("=== Basic CSV Operations Demo ===\n")
    
    # Create CSV handler
    csv_handler = CSVHandler("csv_demo")
    
    # Create sample data
    sample_data = [
        {"name": "cube", "x": 1.0, "y": 2.0, "z": 3.0, "visible": True},
        {"name": "sphere", "x": -1.0, "y": 0.0, "z": 1.0, "visible": True},
        {"name": "cylinder", "x": 0.0, "y": 1.0, "z": -2.0, "visible": False},
        {"name": "plane", "x": 0.0, "y": 0.0, "z": 0.0, "visible": True}
    ]
    
    # Write CSV file
    print("1. Writing CSV file...")
    csv_handler.write_csv("objects.csv", sample_data)
    
    # Read CSV file
    print("\n2. Reading CSV file...")
    loaded_data = csv_handler.read_csv("objects.csv")
    if loaded_data:
        print("Loaded data:")
        for row in loaded_data:
            print(f"  {row['name']}: ({row['x']}, {row['y']}, {row['z']}) - Visible: {row['visible']}")
    
    # Get CSV info
    print("\n3. Getting CSV file info...")
    info = csv_handler.get_csv_info("objects.csv")
    if info:
        print("File info:")
        print(f"  Rows: {info['row_count']}")
        print(f"  Columns: {info['column_count']}")
        print(f"  Column names: {info['columns']}")
        print(f"  File size: {info['file_size']} bytes")
    
    # Append to CSV
    print("\n4. Appending to CSV file...")
    additional_data = [
        {"name": "pyramid", "x": 2.0, "y": 3.0, "z": 1.0, "visible": True},
        {"name": "torus", "x": -2.0, "y": 1.0, "z": 0.0, "visible": False}
    ]
    csv_handler.append_csv("objects.csv", additional_data)
    
    print()

def demonstrate_graphics_csv_processing():
    """Demonstrate 3D graphics CSV processing"""
    print("=== 3D Graphics CSV Processing Demo ===\n")
    
    # Create graphics CSV processor
    graphics_processor = GraphicsCSVProcessor("graphics_csv_demo")
    
    # Export vertex data
    print("1. Exporting vertex data...")
    vertices = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ]
    
    normals = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ]
    
    tex_coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ]
    
    graphics_processor.export_vertex_data("quad", vertices, normals, tex_coords)
    
    # Export performance data
    print("\n2. Exporting performance data...")
    performance_data = [
        {"frame": 1, "fps": 60.0, "render_time": 16.67, "memory_usage": 512.0},
        {"frame": 2, "fps": 59.8, "render_time": 16.72, "memory_usage": 515.2},
        {"frame": 3, "fps": 60.2, "render_time": 16.61, "memory_usage": 518.1},
        {"frame": 4, "fps": 59.9, "render_time": 16.69, "memory_usage": 520.8},
        {"frame": 5, "fps": 60.1, "render_time": 16.64, "memory_usage": 523.5}
    ]
    
    graphics_processor.export_performance_data(performance_data)
    
    # Export scene analysis
    print("\n3. Exporting scene analysis...")
    scene_analysis = {
        "objects": {
            "cube": 5,
            "sphere": 3,
            "cylinder": 2,
            "plane": 1
        },
        "materials": {
            "metal": {
                "diffuse_color": [0.8, 0.8, 0.8],
                "shininess": 128.0
            },
            "plastic": {
                "diffuse_color": [0.2, 0.2, 0.2],
                "shininess": 32.0
            },
            "wood": {
                "diffuse_color": [0.6, 0.4, 0.2],
                "shininess": 16.0
            }
        },
        "lighting": {
            "directional": 2,
            "point": 3,
            "spot": 1
        }
    }
    
    graphics_processor.export_scene_analysis("demo_scene", scene_analysis)
    
    # Export animation data
    print("\n4. Exporting animation data...")
    keyframes = [
        {"time": 0.0, "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        {"time": 1.0, "position": [1.0, 0.0, 0.0], "rotation": [0.0, 45.0, 0.0], "scale": [1.2, 1.0, 1.0]},
        {"time": 2.0, "position": [2.0, 1.0, 0.0], "rotation": [0.0, 90.0, 0.0], "scale": [1.0, 1.2, 1.0]},
        {"time": 3.0, "position": [1.0, 2.0, 0.0], "rotation": [0.0, 135.0, 0.0], "scale": [0.8, 1.0, 1.0]},
        {"time": 4.0, "position": [0.0, 1.0, 0.0], "rotation": [0.0, 180.0, 0.0], "scale": [1.0, 0.8, 1.0]}
    ]
    
    graphics_processor.export_animation_data("cube_animation", keyframes)
    
    print()

def demonstrate_csv_analysis():
    """Demonstrate CSV data analysis"""
    print("=== CSV Data Analysis Demo ===\n")
    
    # Create CSV handler and analyzer
    csv_handler = CSVHandler("analysis_demo")
    analyzer = CSVDataAnalyzer()
    
    # Create sample data for analysis
    analysis_data = [
        {"object_id": 1, "type": "cube", "x": 1.0, "y": 2.0, "z": 3.0, "scale": 1.0, "visible": True},
        {"object_id": 2, "type": "sphere", "x": -1.0, "y": 0.0, "z": 1.0, "scale": 1.5, "visible": True},
        {"object_id": 3, "type": "cube", "x": 0.0, "y": 1.0, "z": -2.0, "scale": 0.8, "visible": False},
        {"object_id": 4, "type": "cylinder", "x": 2.0, "y": -1.0, "z": 0.0, "scale": 2.0, "visible": True},
        {"object_id": 5, "type": "sphere", "x": -2.0, "y": 1.0, "z": 2.0, "scale": 1.2, "visible": True},
        {"object_id": 6, "type": "cube", "x": 1.5, "y": 0.5, "z": -1.0, "scale": 0.9, "visible": False},
        {"object_id": 7, "type": "cylinder", "x": 0.0, "y": 0.0, "z": 0.0, "scale": 1.0, "visible": True}
    ]
    
    # Write analysis data
    csv_handler.write_csv("analysis_data.csv", analysis_data)
    
    # Read data for analysis
    data = csv_handler.read_csv("analysis_data.csv")
    if not data:
        return
    
    # Analyze numeric columns
    print("1. Analyzing numeric columns...")
    
    # X coordinate analysis
    x_analysis = analyzer.analyze_numeric_column(data, "x")
    if "error" not in x_analysis:
        print("X coordinate analysis:")
        print(f"  Count: {x_analysis['count']}")
        print(f"  Min: {x_analysis['min']}")
        print(f"  Max: {x_analysis['max']}")
        print(f"  Mean: {x_analysis['mean']:.2f}")
        print(f"  Std Dev: {x_analysis['std_dev']:.2f}")
    
    # Scale analysis
    scale_analysis = analyzer.analyze_numeric_column(data, "scale")
    if "error" not in scale_analysis:
        print("\nScale analysis:")
        print(f"  Count: {scale_analysis['count']}")
        print(f"  Min: {scale_analysis['min']}")
        print(f"  Max: {scale_analysis['max']}")
        print(f"  Mean: {scale_analysis['mean']:.2f}")
        print(f"  Median: {scale_analysis['median']:.2f}")
    
    # Analyze categorical columns
    print("\n2. Analyzing categorical columns...")
    
    # Object type analysis
    type_analysis = analyzer.analyze_categorical_column(data, "type")
    if "error" not in type_analysis:
        print("Object type analysis:")
        print(f"  Total objects: {type_analysis['count']}")
        print(f"  Unique types: {type_analysis['unique_values']}")
        print(f"  Most common: {type_analysis['most_common']}")
        print("  Type distribution:")
        for obj_type, count in type_analysis['value_counts'].items():
            print(f"    {obj_type}: {count}")
    
    # Visibility analysis
    visibility_analysis = analyzer.analyze_categorical_column(data, "visible")
    if "error" not in visibility_analysis:
        print("\nVisibility analysis:")
        print(f"  Total objects: {visibility_analysis['count']}")
        print(f"  Visible: {visibility_analysis['value_counts'].get('True', 0)}")
        print(f"  Hidden: {visibility_analysis['value_counts'].get('False', 0)}")
    
    # Filter data
    print("\n3. Filtering data...")
    
    # Filter visible objects
    visible_objects = analyzer.filter_data(data, {"visible": True})
    print(f"Visible objects: {len(visible_objects)}")
    
    # Filter objects with scale > 1.0
    large_objects = analyzer.filter_data(data, {"scale": {"min": 1.0}})
    print(f"Large objects (scale > 1.0): {len(large_objects)}")
    
    # Filter cubes
    cubes = analyzer.filter_data(data, {"type": "cube"})
    print(f"Cubes: {len(cubes)}")
    
    # Group data
    print("\n4. Grouping data...")
    
    # Group by object type
    grouped_by_type = analyzer.group_data(data, "type")
    print("Objects grouped by type:")
    for obj_type, objects in grouped_by_type.items():
        print(f"  {obj_type}: {len(objects)} objects")
    
    # Group by visibility
    grouped_by_visibility = analyzer.group_data(data, "visible")
    print("\nObjects grouped by visibility:")
    for visible, objects in grouped_by_visibility.items():
        print(f"  {visible}: {len(objects)} objects")
    
    print()

def demonstrate_csv_utilities():
    """Demonstrate CSV utility functions"""
    print("=== CSV Utilities Demo ===\n")
    
    # Create CSV handler
    csv_handler = CSVHandler("utilities_demo")
    
    # Create complex data for utilities demonstration
    complex_data = [
        {"timestamp": "2024-01-01 10:00:00", "event": "scene_load", "duration": 1.2, "memory": 256.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:01", "event": "texture_load", "duration": 0.8, "memory": 512.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:02", "event": "shader_compile", "duration": 2.1, "memory": 768.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:03", "event": "mesh_upload", "duration": 0.5, "memory": 1024.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:04", "event": "render_frame", "duration": 16.7, "memory": 1280.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:05", "event": "render_frame", "duration": 16.8, "memory": 1280.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:06", "event": "render_frame", "duration": 17.2, "memory": 1280.0, "status": "warning"},
        {"timestamp": "2024-01-01 10:00:07", "event": "render_frame", "duration": 16.9, "memory": 1280.0, "status": "success"},
        {"timestamp": "2024-01-01 10:00:08", "event": "texture_load", "duration": 1.5, "memory": 1536.0, "status": "error"},
        {"timestamp": "2024-01-01 10:00:09", "event": "render_frame", "duration": 18.1, "memory": 1536.0, "status": "warning"}
    ]
    
    # Write complex data
    print("1. Writing complex performance data...")
    csv_handler.write_csv("performance_log.csv", complex_data)
    
    # Read and analyze the data
    data = csv_handler.read_csv("performance_log.csv")
    if data:
        analyzer = CSVDataAnalyzer()
        
        print("\n2. Analyzing performance data...")
        
        # Duration analysis
        duration_analysis = analyzer.analyze_numeric_column(data, "duration")
        if "error" not in duration_analysis:
            print("Duration analysis:")
            print(f"  Average duration: {duration_analysis['mean']:.2f}ms")
            print(f"  Min duration: {duration_analysis['min']:.2f}ms")
            print(f"  Max duration: {duration_analysis['max']:.2f}ms")
            print(f"  Standard deviation: {duration_analysis['std_dev']:.2f}ms")
        
        # Memory analysis
        memory_analysis = analyzer.analyze_numeric_column(data, "memory")
        if "error" not in memory_analysis:
            print("\nMemory analysis:")
            print(f"  Average memory: {memory_analysis['mean']:.1f}MB")
            print(f"  Memory range: {memory_analysis['min']:.1f} - {memory_analysis['max']:.1f}MB")
        
        # Event analysis
        event_analysis = analyzer.analyze_categorical_column(data, "event")
        if "error" not in event_analysis:
            print("\nEvent analysis:")
            for event, count in event_analysis['value_counts'].items():
                print(f"  {event}: {count} times")
        
        # Status analysis
        status_analysis = analyzer.analyze_categorical_column(data, "status")
        if "error" not in status_analysis:
            print("\nStatus analysis:")
            for status, count in status_analysis['value_counts'].items():
                print(f"  {status}: {count} times")
        
        # Filter successful events
        print("\n3. Filtering successful events...")
        successful_events = analyzer.filter_data(data, {"status": "success"})
        print(f"Successful events: {len(successful_events)}")
        
        # Filter slow events (duration > 17ms)
        print("\n4. Filtering slow events...")
        slow_events = analyzer.filter_data(data, {"duration": {"min": 17.0}})
        print(f"Slow events (>17ms): {len(slow_events)}")
        
        # Group by event type
        print("\n5. Grouping by event type...")
        grouped_events = analyzer.group_data(data, "event")
        for event_type, events in grouped_events.items():
            avg_duration = sum(e["duration"] for e in events) / len(events)
            print(f"  {event_type}: {len(events)} events, avg duration: {avg_duration:.2f}ms")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate CSV processing"""
    print("=== CSV Data Processing Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_csv_operations()
    demonstrate_graphics_csv_processing()
    demonstrate_csv_analysis()
    demonstrate_csv_utilities()
    
    print("="*60)
    print("CSV Data Processing demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic CSV operations: Read, write, append, info")
    print("✓ 3D graphics data: Vertex data, performance, scene analysis")
    print("✓ Data analysis: Numeric and categorical column analysis")
    print("✓ Data filtering: Complex filtering conditions")
    print("✓ Data grouping: Group by categories")
    print("✓ Type conversion: Automatic data type detection")

if __name__ == "__main__":
    main()
