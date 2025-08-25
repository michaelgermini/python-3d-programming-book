#!/usr/bin/env python3
"""
Chapter 1: Introduction to Python
Hello World Example

This is a simple introduction to Python syntax and basic concepts.
"""

# This is a comment - it doesn't execute code
print("Hello, World!")  # This prints text to the console

# Python is case-sensitive
print("Python is case-sensitive")
print("python is different from Python")

# Basic arithmetic
print("\n--- Basic Arithmetic ---")
print("2 + 2 =", 2 + 2)
print("10 - 5 =", 10 - 5)
print("3 * 4 =", 3 * 4)
print("15 / 3 =", 15 / 3)

# String operations
print("\n--- String Operations ---")
name = "Python"
print("Hello, " + name + "!")  # String concatenation
print(f"Welcome to {name} programming!")  # f-string (formatted string)

# Multi-line strings
print("\n--- Multi-line String ---")
message = """
This is a multi-line string.
It can span multiple lines.
Perfect for documentation and long text.
"""
print(message)

# Basic input/output
print("\n--- User Input ---")
user_name = input("What's your name? ")
print(f"Nice to meet you, {user_name}!")

# Python's built-in help
print("\n--- Python Help ---")
print("Type 'help()' in Python console to get help")
print("Type 'dir()' to see available functions")
print("Type 'quit()' to exit help")

print("\n--- End of Chapter 1 Introduction ---")
print("Ready to learn more about Python and 3D programming!")
