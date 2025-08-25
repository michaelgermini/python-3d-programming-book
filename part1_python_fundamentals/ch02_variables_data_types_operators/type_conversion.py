#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
Type Conversion Example

This example demonstrates type conversion in Python, focusing on
applications in 3D graphics programming and data processing.
"""

def demonstrate_basic_conversions():
    """Demonstrate basic type conversions"""
    print("=== Basic Type Conversions ===\n")
    
    # 1. String to number conversions
    print("1. String to Number Conversions:")
    
    # String to integer
    position_str = "100"
    position_int = int(position_str)
    print(f"   String: '{position_str}' -> Integer: {position_int}")
    
    # String to float
    scale_str = "2.5"
    scale_float = float(scale_str)
    print(f"   String: '{scale_str}' -> Float: {scale_float}")
    
    # String to integer (base conversion)
    hex_str = "1A"  # Hexadecimal
    hex_int = int(hex_str, 16)
    print(f"   Hex string: '{hex_str}' -> Integer: {hex_int}")
    
    # 2. Number to string conversions
    print("\n2. Number to String Conversions:")
    
    # Integer to string
    object_id = 42
    object_id_str = str(object_id)
    print(f"   Integer: {object_id} -> String: '{object_id_str}'")
    
    # Float to string with formatting
    rotation_angle = 45.6789
    angle_str = str(rotation_angle)
    angle_str_formatted = f"{rotation_angle:.2f}"
    print(f"   Float: {rotation_angle} -> String: '{angle_str}'")
    print(f"   Float: {rotation_angle} -> Formatted: '{angle_str_formatted}'")
    
    # 3. Boolean conversions
    print("\n3. Boolean Conversions:")
    
    # Numbers to boolean
    zero = 0
    non_zero = 42
    print(f"   int(0) -> bool: {bool(zero)}")
    print(f"   int(42) -> bool: {bool(non_zero)}")
    
    # Strings to boolean
    empty_str = ""
    non_empty_str = "hello"
    print(f"   str('') -> bool: {bool(empty_str)}")
    print(f"   str('hello') -> bool: {bool(non_empty_str)}")
    
    # Lists to boolean
    empty_list = []
    non_empty_list = [1, 2, 3]
    print(f"   [] -> bool: {bool(empty_list)}")
    print(f"   [1,2,3] -> bool: {bool(non_empty_list)}")

def demonstrate_3d_conversions():
    """Demonstrate type conversions in 3D context"""
    print("\n=== 3D Graphics Type Conversions ===\n")
    
    # 1. Position data conversions
    print("1. Position Data Conversions:")
    
    # String coordinates to numbers
    position_str = "10.5,20.3,15.7"
    coords = position_str.split(',')
    x = float(coords[0])
    y = float(coords[1])
    z = float(coords[2])
    position = [x, y, z]
    
    print(f"   String position: '{position_str}'")
    print(f"   Parsed position: {position}")
    
    # List to string for storage
    position_back_to_str = ','.join([str(coord) for coord in position])
    print(f"   Back to string: '{position_back_to_str}'")
    
    # 2. Color conversions
    print("\n2. Color Conversions:")
    
    # RGB values as strings to integers
    color_str = "255,128,64"
    color_parts = color_str.split(',')
    r = int(color_parts[0])
    g = int(color_parts[1])
    b = int(color_parts[2])
    color = (r, g, b)
    
    print(f"   Color string: '{color_str}'")
    print(f"   RGB tuple: {color}")
    
    # Hex color to RGB
    hex_color = "#FF8040"
    # Remove '#' and convert to RGB
    hex_value = hex_color[1:]  # Remove '#'
    r_hex = int(hex_value[0:2], 16)
    g_hex = int(hex_value[2:4], 16)
    b_hex = int(hex_value[4:6], 16)
    rgb_from_hex = (r_hex, g_hex, b_hex)
    
    print(f"   Hex color: '{hex_color}'")
    print(f"   RGB from hex: {rgb_from_hex}")
    
    # 3. Object properties conversions
    print("\n3. Object Properties Conversions:")
    
    # Object data from string
    object_data_str = "cube,10.5,20.0,15.0,true,1.5"
    parts = object_data_str.split(',')
    
    object_type = parts[0]
    pos_x = float(parts[1])
    pos_y = float(parts[2])
    pos_z = float(parts[3])
    visible = parts[4].lower() == 'true'
    scale = float(parts[5])
    
    object_data = {
        'type': object_type,
        'position': [pos_x, pos_y, pos_z],
        'visible': visible,
        'scale': scale
    }
    
    print(f"   Object string: '{object_data_str}'")
    print(f"   Parsed object: {object_data}")

def demonstrate_safe_conversions():
    """Demonstrate safe type conversion with error handling"""
    print("\n=== Safe Type Conversions ===\n")
    
    def safe_int_conversion(value, default=0):
        """Safely convert value to integer"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def safe_float_conversion(value, default=0.0):
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # 1. Safe number conversions
    print("1. Safe Number Conversions:")
    
    test_values = ["42", "3.14", "invalid", "10.5", ""]
    
    for value in test_values:
        int_result = safe_int_conversion(value)
        float_result = safe_float_conversion(value)
        print(f"   '{value}' -> int: {int_result}, float: {float_result}")
    
    # 2. Safe list conversions
    print("\n2. Safe List Conversions:")
    
    def safe_list_conversion(value, item_type=str):
        """Safely convert string to list of specified type"""
        if not isinstance(value, str):
            return []
        
        try:
            # Remove brackets and split by comma
            clean_value = value.strip('[]')
            if not clean_value:
                return []
            
            items = [item.strip() for item in clean_value.split(',')]
            return [item_type(item) for item in items]
        except (ValueError, TypeError):
            return []
    
    # Test list conversions
    list_strings = [
        "[1,2,3,4,5]",
        "[10.5,20.3,15.7]",
        "[true,false,true]",
        "invalid",
        "[]"
    ]
    
    for list_str in list_strings:
        int_list = safe_list_conversion(list_str, int)
        float_list = safe_list_conversion(list_str, float)
        print(f"   '{list_str}' -> ints: {int_list}, floats: {float_list}")

def demonstrate_advanced_conversions():
    """Demonstrate advanced type conversion techniques"""
    print("\n=== Advanced Type Conversions ===\n")
    
    # 1. Dictionary conversions
    print("1. Dictionary Conversions:")
    
    # String representation to dictionary
    config_str = "width:800,height:600,fullscreen:false,fps:60"
    config_parts = config_str.split(',')
    config_dict = {}
    
    for part in config_parts:
        key, value = part.split(':')
        # Try to convert value to appropriate type
        if value.lower() in ['true', 'false']:
            config_dict[key] = value.lower() == 'true'
        elif value.isdigit():
            config_dict[key] = int(value)
        else:
            config_dict[key] = value
    
    print(f"   Config string: '{config_str}'")
    print(f"   Config dict: {config_dict}")
    
    # 2. Custom object conversions
    print("\n2. Custom Object Conversions:")
    
    class Vector3D:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        
        def __str__(self):
            return f"Vector3D({self.x}, {self.y}, {self.z})"
        
        @classmethod
        def from_string(cls, string):
            """Create Vector3D from string like '10.5,20.3,15.7'"""
            coords = string.split(',')
            return cls(float(coords[0]), float(coords[1]), float(coords[2]))
        
        def to_string(self):
            """Convert Vector3D to string"""
            return f"{self.x},{self.y},{self.z}"
    
    # Test Vector3D conversions
    vector_str = "10.5,20.3,15.7"
    vector = Vector3D.from_string(vector_str)
    vector_back_to_str = vector.to_string()
    
    print(f"   Vector string: '{vector_str}'")
    print(f"   Vector object: {vector}")
    print(f"   Back to string: '{vector_back_to_str}'")
    
    # 3. Type checking and conversion
    print("\n3. Type Checking and Conversion:")
    
    def convert_to_3d_data(value):
        """Convert various input types to 3D data format"""
        if isinstance(value, str):
            # Try to parse as position string
            try:
                coords = value.split(',')
                if len(coords) == 3:
                    return [float(coord) for coord in coords]
            except ValueError:
                pass
            return None
        
        elif isinstance(value, (list, tuple)):
            if len(value) == 3:
                return [float(coord) for coord in value]
            return None
        
        elif isinstance(value, dict):
            if 'x' in value and 'y' in value and 'z' in value:
                return [float(value['x']), float(value['y']), float(value['z'])]
            return None
        
        return None
    
    # Test various input types
    test_inputs = [
        "10.5,20.3,15.7",
        [1, 2, 3],
        (5.5, 10.2, 8.9),
        {'x': 100, 'y': 200, 'z': 300},
        "invalid",
        [1, 2],
        "not,enough,parts,here"
    ]
    
    for test_input in test_inputs:
        result = convert_to_3d_data(test_input)
        print(f"   {test_input} -> {result}")

def demonstrate_practical_examples():
    """Demonstrate practical type conversion examples"""
    print("\n=== Practical Type Conversion Examples ===\n")
    
    # 1. Configuration file parsing
    print("1. Configuration File Parsing:")
    
    # Simulate reading from a config file
    config_lines = [
        "window_width=1920",
        "window_height=1080",
        "fullscreen=true",
        "vsync=true",
        "fps_limit=60",
        "mouse_sensitivity=1.5"
    ]
    
    config = {}
    for line in config_lines:
        if '=' in line:
            key, value = line.split('=', 1)
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                config[key] = value.lower() == 'true'
            elif value.isdigit():
                config[key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                config[key] = float(value)
            else:
                config[key] = value
    
    print("   Parsed configuration:")
    for key, value in config.items():
        print(f"     {key}: {value} ({type(value).__name__})")
    
    # 2. 3D model data conversion
    print("\n2. 3D Model Data Conversion:")
    
    # Simulate vertex data from file
    vertex_data_str = [
        "v 1.0 2.0 3.0",
        "v 4.0 5.0 6.0",
        "v 7.0 8.0 9.0"
    ]
    
    vertices = []
    for line in vertex_data_str:
        parts = line.split()
        if parts[0] == 'v':  # Vertex line
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            vertices.append([x, y, z])
    
    print("   Parsed vertices:")
    for i, vertex in enumerate(vertices):
        print(f"     Vertex {i+1}: {vertex}")
    
    # 3. User input validation and conversion
    print("\n3. User Input Validation:")
    
    def validate_and_convert_position(input_str):
        """Validate and convert user input to position"""
        try:
            coords = input_str.split(',')
            if len(coords) != 3:
                return None, "Position must have exactly 3 coordinates (x,y,z)"
            
            x = float(coords[0])
            y = float(coords[1])
            z = float(coords[2])
            
            # Validate ranges
            if not (-1000 <= x <= 1000 and -1000 <= y <= 1000 and -1000 <= z <= 1000):
                return None, "Coordinates must be between -1000 and 1000"
            
            return [x, y, z], None
            
        except ValueError:
            return None, "Invalid number format"
    
    # Test user inputs
    test_inputs = [
        "10.5,20.3,15.7",
        "1,2,3,4",
        "abc,def,ghi",
        "10000,0,0",
        "0,0,0"
    ]
    
    for user_input in test_inputs:
        position, error = validate_and_convert_position(user_input)
        if position:
            print(f"   '{user_input}' -> Valid position: {position}")
        else:
            print(f"   '{user_input}' -> Error: {error}")

def main():
    """Main function to run all type conversion demonstrations"""
    print("=== Python Type Conversions for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_conversions()
    demonstrate_3d_conversions()
    demonstrate_safe_conversions()
    demonstrate_advanced_conversions()
    demonstrate_practical_examples()
    
    print("\n=== Summary ===")
    print("This chapter covered type conversions:")
    print("✓ Basic conversions (str, int, float, bool)")
    print("✓ 3D-specific conversions (positions, colors, properties)")
    print("✓ Safe conversion with error handling")
    print("✓ Advanced techniques (custom objects, dictionaries)")
    print("✓ Practical applications (config files, model data)")
    
    print("\nType conversions are essential for:")
    print("- Parsing data from files and user input")
    print("- Converting between different data formats")
    print("- Ensuring data type safety in applications")
    print("- Interfacing with external libraries and APIs")
    print("- Data validation and error handling")

if __name__ == "__main__":
    main()
