#!/usr/bin/env python3
"""
Chapter 2: Variables, Data Types, and Operators
Type Conversion Example

This example demonstrates type conversion and casting in Python, focusing on
applications in 3D graphics programming and data processing.
"""

def demonstrate_basic_type_conversion():
    """Demonstrate basic type conversion operations"""
    print("=== Basic Type Conversion ===\n")
    
    # 1. Converting between numbers
    print("1. Number Conversions:")
    
    # Integer to float
    integer_value = 42
    float_value = float(integer_value)
    print(f"   Integer: {integer_value} (type: {type(integer_value)})")
    print(f"   Converted to float: {float_value} (type: {type(float_value)})")
    
    # Float to integer (truncation)
    float_value = 3.7
    integer_value = int(float_value)
    print(f"   Float: {float_value} (type: {type(float_value)})")
    print(f"   Converted to int: {integer_value} (type: {type(integer_value)})")
    
    # String to number
    string_number = "123.45"
    float_from_string = float(string_number)
    int_from_string = int(float(string_number))
    print(f"   String: '{string_number}' (type: {type(string_number)})")
    print(f"   Converted to float: {float_from_string} (type: {type(float_from_string)})")
    print(f"   Converted to int: {int_from_string} (type: {type(int_from_string)})")
    
    # 2. Converting to strings
    print("\n2. String Conversions:")
    
    # Numbers to strings
    number = 42
    float_num = 3.14159
    string_from_number = str(number)
    string_from_float = str(float_num)
    print(f"   Number: {number} (type: {type(number)})")
    print(f"   Converted to string: '{string_from_number}' (type: {type(string_from_number)})")
    print(f"   Float: {float_num} (type: {type(float_num)})")
    print(f"   Converted to string: '{string_from_float}' (type: {type(string_from_float)})")
    
    # Boolean to string
    boolean_value = True
    string_from_bool = str(boolean_value)
    print(f"   Boolean: {boolean_value} (type: {type(boolean_value)})")
    print(f"   Converted to string: '{string_from_bool}' (type: {type(string_from_bool)})")

def demonstrate_3d_data_conversion():
    """Demonstrate type conversion in 3D data context"""
    print("\n=== 3D Data Type Conversion ===\n")
    
    # 1. Converting coordinate data
    print("1. Coordinate Data Conversion:")
    
    # String coordinates to numbers
    x_str = "10.5"
    y_str = "20.0"
    z_str = "15.75"
    
    x_float = float(x_str)
    y_float = float(y_str)
    z_float = float(z_str)
    
    position = [x_float, y_float, z_float]
    print(f"   String coordinates: x='{x_str}', y='{y_str}', z='{z_str}'")
    print(f"   Converted to float: {position}")
    print(f"   Position type: {type(position)}")
    
    # 2. Converting color values
    print("\n2. Color Value Conversion:")
    
    # RGB values as strings
    red_str = "255"
    green_str = "128"
    blue_str = "64"
    
    # Convert to integers
    red = int(red_str)
    green = int(green_str)
    blue = int(blue_str)
    
    color = (red, green, blue)
    print(f"   String RGB: r='{red_str}', g='{green_str}', b='{blue_str}'")
    print(f"   Converted to int: {color}")
    print(f"   Color type: {type(color)}")
    
    # 3. Converting scale factors
    print("\n3. Scale Factor Conversion:")
    
    # Scale as string
    scale_str = "2.5"
    scale_float = float(scale_str)
    
    # Apply scale to dimensions
    width = 10
    height = 8
    depth = 6
    
    scaled_dimensions = [
        width * scale_float,
        height * scale_float,
        depth * scale_float
    ]
    
    print(f"   Original scale string: '{scale_str}'")
    print(f"   Converted to float: {scale_float}")
    print(f"   Original dimensions: [{width}, {height}, {depth}]")
    print(f"   Scaled dimensions: {scaled_dimensions}")

def demonstrate_list_and_tuple_conversion():
    """Demonstrate converting between lists and tuples"""
    print("\n=== List and Tuple Conversion ===\n")
    
    # 1. Converting between lists and tuples
    print("1. List ↔ Tuple Conversion:")
    
    # List to tuple
    position_list = [10, 20, 30]
    position_tuple = tuple(position_list)
    print(f"   List: {position_list} (type: {type(position_list)})")
    print(f"   Converted to tuple: {position_tuple} (type: {type(position_tuple)})")
    
    # Tuple to list
    rotation_tuple = (45, 0, 90)
    rotation_list = list(rotation_tuple)
    print(f"   Tuple: {rotation_tuple} (type: {type(rotation_tuple)})")
    print(f"   Converted to list: {rotation_list} (type: {type(rotation_list)})")
    
    # 2. Converting string representations
    print("\n2. String Representation Conversion:")
    
    # String representation of list
    list_string = "[1, 2, 3, 4, 5]"
    print(f"   String representation: '{list_string}'")
    print(f"   Note: eval() can convert this, but it's not safe for user input")
    
    # Safe conversion using ast.literal_eval (demonstration)
    import ast
    try:
        converted_list = ast.literal_eval(list_string)
        print(f"   Safely converted: {converted_list} (type: {type(converted_list)})")
    except:
        print("   Safe conversion failed")

def demonstrate_error_handling_in_conversion():
    """Demonstrate handling conversion errors"""
    print("\n=== Error Handling in Type Conversion ===\n")
    
    # 1. Handling invalid conversions
    print("1. Invalid Conversion Handling:")
    
    def safe_float_conversion(value):
        """Safely convert a value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def safe_int_conversion(value):
        """Safely convert a value to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    # Test various values
    test_values = ["123", "45.67", "abc", "12.34.56", "", None]
    
    print("   Testing float conversion:")
    for value in test_values:
        result = safe_float_conversion(value)
        print(f"     '{value}' → {result}")
    
    print("\n   Testing int conversion:")
    for value in test_values:
        result = safe_int_conversion(value)
        print(f"     '{value}' → {result}")
    
    # 2. Validating 3D coordinates
    print("\n2. 3D Coordinate Validation:")
    
    def validate_coordinate(value):
        """Validate and convert coordinate value"""
        try:
            coord = float(value)
            if -1000 <= coord <= 1000:  # Reasonable range for 3D coordinates
                return coord
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    coordinate_inputs = ["10.5", "1001", "-500", "abc", "0"]
    
    print("   Validating coordinate inputs:")
    for coord_input in coordinate_inputs:
        result = validate_coordinate(coord_input)
        if result is not None:
            print(f"     '{coord_input}' → {result} (valid)")
        else:
            print(f"     '{coord_input}' → invalid")

def demonstrate_practical_conversion_examples():
    """Demonstrate practical conversion examples in 3D graphics"""
    print("\n=== Practical 3D Graphics Conversion Examples ===\n")
    
    # 1. Converting user input for object creation
    print("1. User Input Conversion:")
    
    # Simulate user input (normally from input() function)
    user_inputs = {
        'position_x': '10.5',
        'position_y': '20.0',
        'position_z': '15.75',
        'scale': '2.0',
        'rotation': '45'
    }
    
    # Convert user inputs to appropriate types
    try:
        position = [
            float(user_inputs['position_x']),
            float(user_inputs['position_y']),
            float(user_inputs['position_z'])
        ]
        scale = float(user_inputs['scale'])
        rotation = float(user_inputs['rotation'])
        
        print(f"   User inputs: {user_inputs}")
        print(f"   Converted position: {position}")
        print(f"   Converted scale: {scale}")
        print(f"   Converted rotation: {rotation}")
        
    except ValueError as e:
        print(f"   Error converting user input: {e}")
    
    # 2. Converting data from different sources
    print("\n2. Data Source Conversion:")
    
    # Simulate data from different sources
    file_data = "10,20,30"  # CSV format
    json_data = '{"x": 15, "y": 25, "z": 35}'  # JSON format
    binary_data = b'40.0,50.0,60.0'  # Binary format
    
    # Convert file data (CSV)
    file_coords = [float(x) for x in file_data.split(',')]
    print(f"   File data: '{file_data}'")
    print(f"   Converted: {file_coords}")
    
    # Convert JSON data (simplified)
    import json
    try:
        json_coords = json.loads(json_data)
        coords_list = [json_coords['x'], json_coords['y'], json_coords['z']]
        print(f"   JSON data: '{json_data}'")
        print(f"   Converted: {coords_list}")
    except json.JSONDecodeError:
        print("   Error parsing JSON data")
    
    # Convert binary data
    binary_coords = [float(x) for x in binary_data.decode().split(',')]
    print(f"   Binary data: {binary_data}")
    print(f"   Converted: {binary_coords}")
    
    # 3. Converting for different coordinate systems
    print("\n3. Coordinate System Conversion:")
    
    # Convert between different units
    meters_to_feet = 3.28084
    degrees_to_radians = 0.0174533  # π/180
    
    # Object in meters
    position_meters = [10.0, 5.0, 2.0]
    rotation_degrees = [45, 90, 0]
    
    # Convert to feet
    position_feet = [coord * meters_to_feet for coord in position_meters]
    
    # Convert to radians
    rotation_radians = [angle * degrees_to_radians for angle in rotation_degrees]
    
    print(f"   Position (meters): {position_meters}")
    print(f"   Position (feet): {[round(x, 2) for x in position_feet]}")
    print(f"   Rotation (degrees): {rotation_degrees}")
    print(f"   Rotation (radians): {[round(x, 4) for x in rotation_radians]}")

def demonstrate_type_checking():
    """Demonstrate type checking and validation"""
    print("\n=== Type Checking and Validation ===\n")
    
    # 1. Checking types before conversion
    print("1. Type Checking:")
    
    def safe_convert_to_list(value):
        """Safely convert value to list"""
        if isinstance(value, list):
            return value
        elif isinstance(value, tuple):
            return list(value)
        elif isinstance(value, str):
            # Try to parse as list-like string
            try:
                # Remove brackets and split by comma
                cleaned = value.strip('[]()').split(',')
                return [float(x.strip()) for x in cleaned]
            except:
                return None
        else:
            return None
    
    test_values = [
        [1, 2, 3],
        (4, 5, 6),
        "[7, 8, 9]",
        "not a list",
        42
    ]
    
    print("   Testing safe conversion to list:")
    for value in test_values:
        result = safe_convert_to_list(value)
        print(f"     {value} (type: {type(value).__name__}) → {result}")
    
    # 2. Validating 3D object data
    print("\n2. 3D Object Data Validation:")
    
    def validate_3d_object_data(data):
        """Validate 3D object data structure"""
        required_fields = ['position', 'rotation', 'scale']
        
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            
            if not isinstance(data[field], (list, tuple)):
                return False, f"Field {field} must be a list or tuple"
            
            if len(data[field]) != 3:
                return False, f"Field {field} must have exactly 3 values"
            
            # Check if all values are numbers
            for value in data[field]:
                if not isinstance(value, (int, float)):
                    return False, f"Field {field} contains non-numeric value: {value}"
        
        return True, "Data is valid"
    
    # Test various data structures
    test_objects = [
        {
            'position': [0, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1]
        },
        {
            'position': [10, 20],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1]
        },
        {
            'position': [0, 0, 0],
            'rotation': [0, 0, 0]
        },
        "not an object"
    ]
    
    print("   Testing 3D object data validation:")
    for i, obj in enumerate(test_objects):
        is_valid, message = validate_3d_object_data(obj)
        print(f"     Object {i+1}: {is_valid} - {message}")

def main():
    """Main function to run all type conversion demonstrations"""
    print("=== Python Type Conversion for 3D Graphics ===\n")
    
    # Run all demonstrations
    demonstrate_basic_type_conversion()
    demonstrate_3d_data_conversion()
    demonstrate_list_and_tuple_conversion()
    demonstrate_error_handling_in_conversion()
    demonstrate_practical_conversion_examples()
    demonstrate_type_checking()
    
    print("\n=== Summary ===")
    print("This chapter covered type conversion:")
    print("✓ Basic type conversion (int, float, str, bool)")
    print("✓ Converting between lists and tuples")
    print("✓ Error handling in type conversion")
    print("✓ Practical applications in 3D graphics")
    print("✓ Type checking and validation")
    print("✓ Safe conversion practices")
    
    print("\nThese conversion concepts are essential for:")
    print("- Processing user input and file data")
    print("- Converting between different data formats")
    print("- Validating 3D object properties")
    print("- Handling data from external sources")
    print("- Ensuring data type safety in applications")

if __name__ == "__main__":
    main()
