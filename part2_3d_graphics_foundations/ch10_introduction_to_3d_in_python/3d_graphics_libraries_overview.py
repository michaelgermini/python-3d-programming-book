#!/usr/bin/env python3
"""
Chapter 10: Introduction to 3D in Python
3D Graphics Libraries Overview

Demonstrates the different 3D graphics libraries available in Python,
their strengths, use cases, and basic setup for 3D development.
"""

import sys
import time
import math
from typing import List, Dict, Any, Optional, Tuple
import platform

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Libraries Overview"
__description__ = "Overview of 3D graphics libraries in Python"

# ============================================================================
# 3D GRAPHICS LIBRARIES OVERVIEW
# ============================================================================

class LibraryInfo:
    """Information about a 3D graphics library"""
    
    def __init__(self, name: str, description: str, strengths: List[str], 
                 weaknesses: List[str], use_cases: List[str], 
                 installation: str, website: str):
        self.name = name
        self.description = description
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.use_cases = use_cases
        self.installation = installation
        self.website = website

# Define the major 3D graphics libraries
LIBRARIES = {
    'vpython': LibraryInfo(
        name="VPython",
        description="Simple 3D visualization library designed for education and prototyping",
        strengths=[
            "Very easy to learn and use",
            "Excellent for beginners",
            "Built-in physics simulation",
            "Real-time 3D visualization",
            "Great for educational purposes",
            "No complex setup required"
        ],
        weaknesses=[
            "Limited advanced features",
            "Not suitable for complex 3D applications",
            "Limited customization options",
            "Performance limitations for large scenes"
        ],
        use_cases=[
            "Educational demonstrations",
            "Physics simulations",
            "Simple 3D visualizations",
            "Prototyping 3D concepts",
            "Scientific visualization for beginners"
        ],
        installation="pip install vpython",
        website="https://vpython.org/"
    ),
    
    'pyopengl': LibraryInfo(
        name="PyOpenGL",
        description="Python bindings for OpenGL, providing low-level graphics programming capabilities",
        strengths=[
            "Full OpenGL functionality",
            "Maximum performance and control",
            "Industry standard graphics API",
            "Cross-platform compatibility",
            "Advanced graphics features",
            "Direct hardware access"
        ],
        weaknesses=[
            "Steep learning curve",
            "Complex setup and configuration",
            "Low-level programming required",
            "More code needed for simple tasks",
            "Requires graphics programming knowledge"
        ],
        use_cases=[
            "High-performance 3D applications",
            "Game engines",
            "Professional graphics software",
            "Custom rendering pipelines",
            "Advanced visual effects"
        ],
        installation="pip install PyOpenGL PyOpenGL_accelerate",
        website="https://pyopengl.sourceforge.net/"
    ),
    
    'panda3d': LibraryInfo(
        name="Panda3D",
        description="3D game engine and graphics framework developed by Disney and Carnegie Mellon",
        strengths=[
            "Complete game engine",
            "Built-in physics engine",
            "Advanced rendering features",
            "Good documentation",
            "Active community",
            "Cross-platform support"
        ],
        weaknesses=[
            "Larger learning curve than VPython",
            "More complex than simple visualization libraries",
            "May be overkill for simple applications"
        ],
        use_cases=[
            "3D games and simulations",
            "Interactive 3D applications",
            "Virtual reality applications",
            "Educational games",
            "3D visualization tools"
        ],
        installation="pip install panda3d",
        website="https://www.panda3d.org/"
    ),
    
    'matplotlib_3d': LibraryInfo(
        name="Matplotlib 3D",
        description="3D plotting and visualization extension of the popular Matplotlib library",
        strengths=[
            "Excellent for scientific visualization",
            "Integration with NumPy and Pandas",
            "Publication-quality graphics",
            "Familiar interface for data scientists",
            "Extensive customization options",
            "Great documentation"
        ],
        weaknesses=[
            "Not designed for real-time applications",
            "Limited interactive 3D features",
            "Performance limitations for complex scenes",
            "Not suitable for games or simulations"
        ],
        use_cases=[
            "Scientific data visualization",
            "3D plotting and charts",
            "Research presentations",
            "Data analysis visualization",
            "Academic publications"
        ],
        installation="pip install matplotlib",
        website="https://matplotlib.org/"
    ),
    
    'blender_python': LibraryInfo(
        name="Blender Python API",
        description="Python scripting interface for Blender, the open-source 3D modeling software",
        strengths=[
            "Full access to Blender's capabilities",
            "Professional 3D modeling tools",
            "Animation and rigging support",
            "Advanced rendering (Cycles, Eevee)",
            "Large community and resources",
            "Free and open-source"
        ],
        weaknesses=[
            "Requires Blender installation",
            "Learning curve for Blender concepts",
            "Not a standalone Python library",
            "Limited to Blender's workflow"
        ],
        use_cases=[
            "3D modeling automation",
            "Animation scripting",
            "Rendering automation",
            "Asset generation",
            "3D content creation tools"
        ],
        installation="Download Blender from https://www.blender.org/",
        website="https://docs.blender.org/manual/en/latest/advanced/scripting/"
    ),
    
    'pygame_3d': LibraryInfo(
        name="Pygame (3D extensions)",
        description="2D game library with 3D extensions and OpenGL integration",
        strengths=[
            "Easy to learn for beginners",
            "Good for 2D/3D hybrid games",
            "Simple setup and configuration",
            "Active community",
            "Cross-platform support"
        ],
        weaknesses=[
            "Limited 3D features",
            "Not designed for complex 3D applications",
            "Performance limitations",
            "Basic 3D capabilities only"
        ],
        use_cases=[
            "Simple 3D games",
            "2D/3D hybrid applications",
            "Educational 3D projects",
            "Prototyping simple 3D concepts"
        ],
        installation="pip install pygame",
        website="https://www.pygame.org/"
    ),
    
    'kivy_3d': LibraryInfo(
        name="Kivy (3D capabilities)",
        description="Modern Python library for developing applications with 3D graphics support",
        strengths=[
            "Modern and cross-platform",
            "Touch and gesture support",
            "Mobile-friendly",
            "Good for applications with UI",
            "OpenGL integration"
        ],
        weaknesses=[
            "Learning curve for complex 3D",
            "Limited advanced 3D features",
            "More focused on UI than pure 3D"
        ],
        use_cases=[
            "Mobile 3D applications",
            "Touch-based 3D interfaces",
            "Cross-platform 3D apps",
            "Educational applications"
        ],
        installation="pip install kivy",
        website="https://kivy.org/"
    )
}

# ============================================================================
# LIBRARY COMPARISON AND ANALYSIS
# ============================================================================

def compare_libraries() -> Dict[str, Any]:
    """Compare different 3D graphics libraries"""
    comparison = {
        'ease_of_use': {
            'vpython': 5,
            'pygame_3d': 4,
            'matplotlib_3d': 4,
            'kivy_3d': 3,
            'panda3d': 3,
            'pyopengl': 1,
            'blender_python': 2
        },
        'performance': {
            'pyopengl': 5,
            'panda3d': 4,
            'blender_python': 4,
            'kivy_3d': 3,
            'pygame_3d': 2,
            'matplotlib_3d': 2,
            'vpython': 2
        },
        'features': {
            'blender_python': 5,
            'pyopengl': 5,
            'panda3d': 4,
            'kivy_3d': 3,
            'matplotlib_3d': 3,
            'pygame_3d': 2,
            'vpython': 2
        },
        'documentation': {
            'matplotlib_3d': 5,
            'panda3d': 4,
            'vpython': 4,
            'pygame_3d': 4,
            'kivy_3d': 3,
            'pyopengl': 3,
            'blender_python': 4
        },
        'community': {
            'pygame_3d': 5,
            'matplotlib_3d': 5,
            'blender_python': 5,
            'panda3d': 4,
            'vpython': 3,
            'pyopengl': 3,
            'kivy_3d': 3
        }
    }
    return comparison

def get_library_recommendations(use_case: str) -> List[Tuple[str, str]]:
    """Get library recommendations based on use case"""
    recommendations = {
        'beginner': [
            ('vpython', 'Best for learning 3D concepts'),
            ('pygame_3d', 'Good for simple 3D games'),
            ('matplotlib_3d', 'Excellent for data visualization')
        ],
        'education': [
            ('vpython', 'Designed for educational use'),
            ('panda3d', 'Good for educational games'),
            ('matplotlib_3d', 'Great for scientific visualization')
        ],
        'game_development': [
            ('panda3d', 'Complete game engine'),
            ('pygame_3d', 'Simple 3D games'),
            ('pyopengl', 'Maximum control and performance')
        ],
        'scientific_visualization': [
            ('matplotlib_3d', 'Best for data plotting'),
            ('vpython', 'Good for physics simulations'),
            ('pyopengl', 'Custom visualization pipelines')
        ],
        'professional_3d': [
            ('blender_python', 'Professional 3D modeling'),
            ('pyopengl', 'Custom professional applications'),
            ('panda3d', 'Professional game development')
        ],
        'mobile_apps': [
            ('kivy_3d', 'Cross-platform mobile support'),
            ('panda3d', 'Mobile game development'),
            ('pygame_3d', 'Simple mobile games')
        ]
    }
    return recommendations.get(use_case, [])

# ============================================================================
# INSTALLATION AND SETUP GUIDES
# ============================================================================

def get_installation_guide(library_name: str) -> Dict[str, str]:
    """Get detailed installation guide for a library"""
    guides = {
        'vpython': {
            'basic': 'pip install vpython',
            'detailed': '''
1. Install Python 3.7 or later
2. Run: pip install vpython
3. Test installation:
   python -c "import vpython; print('VPython installed successfully')"
4. For Jupyter notebooks: pip install vpython
            ''',
            'troubleshooting': '''
- If you get SSL errors, try: pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org vpython
- For Windows users, ensure you have the latest pip: python -m pip install --upgrade pip
            '''
        },
        'pyopengl': {
            'basic': 'pip install PyOpenGL PyOpenGL_accelerate',
            'detailed': '''
1. Install Python 3.7 or later
2. Install system dependencies:
   - Windows: Install Visual Studio Build Tools
   - macOS: Install Xcode Command Line Tools
   - Linux: sudo apt-get install python3-dev libgl1-mesa-dev
3. Run: pip install PyOpenGL PyOpenGL_accelerate
4. Test installation:
   python -c "import OpenGL; print('PyOpenGL installed successfully')"
            ''',
            'troubleshooting': '''
- If compilation fails, try installing pre-compiled wheels
- For Windows: pip install PyOpenGL-3.1.6-cp39-cp39-win_amd64.whl
- Ensure you have proper graphics drivers installed
            '''
        },
        'panda3d': {
            'basic': 'pip install panda3d',
            'detailed': '''
1. Install Python 3.7 or later
2. Run: pip install panda3d
3. Test installation:
   python -c "import panda3d; print('Panda3D installed successfully')"
4. Optional: Install additional tools:
   pip install panda3d-tools
            ''',
            'troubleshooting': '''
- For Windows: May need Visual Studio Build Tools
- For Linux: May need additional system libraries
- Check Panda3D documentation for platform-specific instructions
            '''
        },
        'matplotlib': {
            'basic': 'pip install matplotlib',
            'detailed': '''
1. Install Python 3.7 or later
2. Run: pip install matplotlib
3. For 3D support, also install: pip install numpy
4. Test installation:
   python -c "import matplotlib.pyplot as plt; print('Matplotlib installed successfully')"
            ''',
            'troubleshooting': '''
- If plotting fails, try: pip install --upgrade matplotlib
- For backend issues, try: pip install tkinter
- Ensure you have proper display drivers
            '''
        }
    }
    return guides.get(library_name, {})

# ============================================================================
# SYSTEM REQUIREMENTS AND COMPATIBILITY
# ============================================================================

def check_system_compatibility() -> Dict[str, Any]:
    """Check system compatibility for 3D graphics libraries"""
    system_info = {
        'platform': platform.system(),
        'python_version': sys.version,
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'compatibility': {}
    }
    
    # Check OpenGL support
    try:
        import OpenGL
        system_info['compatibility']['opengl'] = True
    except ImportError:
        system_info['compatibility']['opengl'] = False
    
    # Check NumPy support
    try:
        import numpy
        system_info['compatibility']['numpy'] = True
    except ImportError:
        system_info['compatibility']['numpy'] = False
    
    # Check VPython support
    try:
        import vpython
        system_info['compatibility']['vpython'] = True
    except ImportError:
        system_info['compatibility']['vpython'] = False
    
    return system_info

def get_minimum_requirements() -> Dict[str, Dict[str, str]]:
    """Get minimum system requirements for each library"""
    requirements = {
        'vpython': {
            'python': '3.7+',
            'memory': '512 MB RAM',
            'graphics': 'Any graphics card',
            'os': 'Windows, macOS, Linux',
            'additional': 'Web browser for display'
        },
        'pyopengl': {
            'python': '3.7+',
            'memory': '1 GB RAM',
            'graphics': 'OpenGL 2.1+ compatible',
            'os': 'Windows, macOS, Linux',
            'additional': 'Graphics drivers'
        },
        'panda3d': {
            'python': '3.7+',
            'memory': '2 GB RAM',
            'graphics': 'OpenGL 3.3+ compatible',
            'os': 'Windows, macOS, Linux',
            'additional': 'Graphics drivers, build tools'
        },
        'matplotlib_3d': {
            'python': '3.7+',
            'memory': '1 GB RAM',
            'graphics': 'Any graphics card',
            'os': 'Windows, macOS, Linux',
            'additional': 'NumPy, display backend'
        }
    }
    return requirements

# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

def get_quick_start_example(library_name: str) -> str:
    """Get a quick start example for a library"""
    examples = {
        'vpython': '''
# VPython Quick Start
from vpython import *

# Create a simple scene
sphere(pos=vector(0,0,0), radius=1, color=color.red)
box(pos=vector(3,0,0), size=vector(1,1,1), color=color.blue)

# Add some animation
while True:
    rate(30)  # 30 frames per second
    sphere.rotate(angle=0.1, axis=vector(0,1,0))
''',
        'pyopengl': '''
# PyOpenGL Quick Start
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import numpy as np

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glLoadIdentity()
    
    # Draw a simple triangle
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glVertex3f(-0.5, -0.5, 0.0)
    gl.glVertex3f(0.5, -0.5, 0.0)
    gl.glVertex3f(0.0, 0.5, 0.0)
    gl.glEnd()
    
    glut.glutSwapBuffers()

glut.glutInit()
glut.glutCreateWindow(b"PyOpenGL Example")
glut.glutDisplayFunc(display)
glut.glutMainLoop()
''',
        'matplotlib_3d': '''
# Matplotlib 3D Quick Start
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create 3D data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()
'''
    }
    return examples.get(library_name, "No quick start example available")

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_library_info(library_name: str):
    """Display detailed information about a library"""
    if library_name not in LIBRARIES:
        print(f"Library '{library_name}' not found.")
        return
    
    lib = LIBRARIES[library_name]
    print(f"\n{'='*60}")
    print(f"LIBRARY: {lib.name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {lib.description}")
    print(f"\nStrengths:")
    for strength in lib.strengths:
        print(f"  ✓ {strength}")
    print(f"\nWeaknesses:")
    for weakness in lib.weaknesses:
        print(f"  ✗ {weakness}")
    print(f"\nUse Cases:")
    for use_case in lib.use_cases:
        print(f"  • {use_case}")
    print(f"\nInstallation: {lib.installation}")
    print(f"Website: {lib.website}")

def display_comparison_table():
    """Display a comparison table of libraries"""
    comparison = compare_libraries()
    
    print(f"\n{'='*80}")
    print("3D GRAPHICS LIBRARIES COMPARISON")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Library':<15} {'Ease':<5} {'Perf':<5} {'Feat':<5} {'Doc':<5} {'Comm':<5}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for lib_name in LIBRARIES.keys():
        if lib_name in comparison['ease_of_use']:
            row = f"{lib_name:<15}"
            row += f"{comparison['ease_of_use'][lib_name]:<5}"
            row += f"{comparison['performance'][lib_name]:<5}"
            row += f"{comparison['features'][lib_name]:<5}"
            row += f"{comparison['documentation'][lib_name]:<5}"
            row += f"{comparison['community'][lib_name]:<5}"
            print(row)
    
    print("\nLegend: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent")
    print("Ease=Ease of Use, Perf=Performance, Feat=Features, Doc=Documentation, Comm=Community")

def display_recommendations():
    """Display library recommendations for different use cases"""
    print(f"\n{'='*60}")
    print("LIBRARY RECOMMENDATIONS BY USE CASE")
    print(f"{'='*60}")
    
    use_cases = ['beginner', 'education', 'game_development', 
                 'scientific_visualization', 'professional_3d', 'mobile_apps']
    
    for use_case in use_cases:
        print(f"\n{use_case.upper().replace('_', ' ')}:")
        recommendations = get_library_recommendations(use_case)
        for lib_name, reason in recommendations:
            print(f"  • {lib_name}: {reason}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate 3D graphics libraries overview"""
    print("=== 3D Graphics Libraries Overview ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Display system information
    print("1. System Information:")
    system_info = check_system_compatibility()
    print(f"   Platform: {system_info['platform']}")
    print(f"   Python Version: {system_info['python_version'].split()[0]}")
    print(f"   Architecture: {system_info['architecture']}")
    print(f"   OpenGL Support: {'Yes' if system_info['compatibility']['opengl'] else 'No'}")
    print(f"   NumPy Support: {'Yes' if system_info['compatibility']['numpy'] else 'No'}")
    print(f"   VPython Support: {'Yes' if system_info['compatibility']['vpython'] else 'No'}")
    print()
    
    # Display comparison table
    print("2. Library Comparison:")
    display_comparison_table()
    print()
    
    # Display recommendations
    print("3. Library Recommendations:")
    display_recommendations()
    print()
    
    # Display detailed information for each library
    print("4. Detailed Library Information:")
    for lib_name in LIBRARIES.keys():
        display_library_info(lib_name)
    
    # Display installation guides
    print("\n5. Installation Guides:")
    for lib_name in ['vpython', 'pyopengl', 'panda3d', 'matplotlib']:
        guide = get_installation_guide(lib_name)
        if guide:
            print(f"\n{lib_name.upper()}:")
            print(f"  Basic: {guide.get('basic', 'Not available')}")
    
    # Display quick start examples
    print("\n6. Quick Start Examples:")
    for lib_name in ['vpython', 'pyopengl', 'matplotlib_3d']:
        print(f"\n{lib_name.upper()} Quick Start:")
        example = get_quick_start_example(lib_name)
        print(example)
    
    print("\n" + "="*60)
    print("3D Graphics Libraries Overview completed!")
    print("\nKey takeaways:")
    print("• VPython: Best for beginners and education")
    print("• PyOpenGL: Maximum control and performance")
    print("• Panda3D: Complete game engine solution")
    print("• Matplotlib 3D: Excellent for scientific visualization")
    print("• Blender Python: Professional 3D modeling and animation")
    print("• Pygame 3D: Simple 3D games and applications")
    print("• Kivy 3D: Modern cross-platform applications")
    
    print("\nNext steps:")
    print("• Choose a library based on your specific needs")
    print("• Install the selected library using the provided guides")
    print("• Start with the quick start examples")
    print("• Explore the library's documentation and tutorials")
    print("• Build your first 3D application!")

if __name__ == "__main__":
    main()
