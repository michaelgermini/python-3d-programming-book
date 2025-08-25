# Appendix A: Environment Setup

## Overview
This appendix provides comprehensive setup instructions for creating a Python 3D development environment. It covers Python installation, essential libraries, development tools, and configuration for various platforms.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Graphics**: OpenGL 3.3+ compatible graphics card
- **Storage**: At least 5GB free space

### Hardware Recommendations
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: Dedicated graphics card with 4GB+ VRAM
- **Display**: 1920x1080 or higher resolution
- **Input**: Mouse with scroll wheel, optional: 3D mouse

## Python Installation

### Windows
1. **Download Python**:
   - Visit [python.org](https://www.python.org/downloads/)
   - Download Python 3.11 or later
   - Run installer with "Add Python to PATH" checked

2. **Verify Installation**:
   ```bash
   python --version
   pip --version
   ```

### macOS
1. **Using Homebrew** (recommended):
   ```bash
   brew install python
   ```

2. **Using Official Installer**:
   - Download from python.org
   - Run installer package

3. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

### Linux (Ubuntu/Debian)
1. **Install Python**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Verify Installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

## Virtual Environment Setup

### Creating Virtual Environment
```bash
# Windows
python -m venv 3d_python_env

# macOS/Linux
python3 -m venv 3d_python_env
```

### Activating Virtual Environment
```bash
# Windows
3d_python_env\Scripts\activate

# macOS/Linux
source 3d_python_env/bin/activate
```

### Deactivating Virtual Environment
```bash
deactivate
```

## Essential Libraries Installation

### Core 3D Libraries
```bash
# NumPy for numerical computations
pip install numpy

# Matplotlib for 2D plotting and visualization
pip install matplotlib

# Pillow for image processing
pip install Pillow

# PyOpenGL for OpenGL bindings
pip install PyOpenGL PyOpenGL_accelerate

# GLFW for window management
pip install glfw

# Pygame for game development
pip install pygame

# VPython for 3D visualization
pip install vpython

# Blender Python API (if using Blender)
pip install bpy
```

### Advanced Libraries
```bash
# SciPy for scientific computing
pip install scipy

# SymPy for symbolic mathematics
pip install sympy

# PyVista for 3D scientific visualization
pip install pyvista

# Plotly for interactive 3D plots
pip install plotly

# Mayavi for 3D scientific data visualization
pip install mayavi

# VTK for visualization toolkit
pip install vtk
```

### Development Tools
```bash
# Jupyter for interactive development
pip install jupyter

# IPython for enhanced Python shell
pip install ipython

# Black for code formatting
pip install black

# Flake8 for linting
pip install flake8

# Pytest for testing
pip install pytest

# Coverage for test coverage
pip install coverage
```

## IDE Setup

### Visual Studio Code (Recommended)
1. **Install VS Code**:
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. **Install Python Extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" and install Microsoft's Python extension

3. **Configure Python Interpreter**:
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose your virtual environment

4. **Recommended Extensions**:
   - Python
   - Python Indent
   - Python Docstring Generator
   - Python Test Explorer
   - GitLens
   - Live Share

### PyCharm Professional
1. **Install PyCharm**:
   - Download from [jetbrains.com](https://www.jetbrains.com/pycharm/)

2. **Configure Project**:
   - Create new project
   - Set Python interpreter to virtual environment
   - Configure project structure

3. **Enable 3D Support**:
   - Install OpenGL plugin
   - Configure graphics debugging

## Graphics Driver Setup

### Windows
1. **Update Graphics Drivers**:
   - NVIDIA: Download from [nvidia.com](https://www.nvidia.com/drivers/)
   - AMD: Download from [amd.com](https://www.amd.com/support)
   - Intel: Download from [intel.com](https://www.intel.com/content/www/us/en/download/785597/intel-graphics-driver-for-windows-15-40.html)

2. **Verify OpenGL Support**:
   ```python
   import OpenGL.GL as gl
   print(gl.glGetString(gl.GL_VERSION))
   ```

### macOS
1. **Update macOS**:
   - Ensure latest macOS version
   - Graphics drivers are included with system updates

2. **Verify Metal Support**:
   ```python
   import Metal
   print(Metal.MTLCopyAllDevices())
   ```

### Linux
1. **Install Graphics Drivers**:
   ```bash
   # Ubuntu/Debian
   sudo ubuntu-drivers autoinstall
   
   # Or manually install NVIDIA drivers
   sudo apt install nvidia-driver-470
   ```

2. **Install OpenGL Libraries**:
   ```bash
   sudo apt install libgl1-mesa-dev libglu1-mesa-dev
   ```

## Configuration Files

### requirements.txt
Create a requirements.txt file with all dependencies:
```txt
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.3.0
PyOpenGL>=3.1.0
PyOpenGL_accelerate>=3.1.0
glfw>=2.0.0
pygame>=2.1.0
vpython>=3.2.0
scipy>=1.7.0
sympy>=1.9.0
pyvista>=0.32.0
plotly>=5.0.0
jupyter>=1.0.0
ipython>=7.0.0
black>=21.0.0
flake8>=3.9.0
pytest>=6.0.0
coverage>=5.5.0
```

### .gitignore
Create a .gitignore file for Python projects:
```gitignore
# Virtual environments
venv/
env/
3d_python_env/

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Logs
*.log

# Test coverage
.coverage
htmlcov/
```

### setup.py
Create a setup.py for package distribution:
```python
from setuptools import setup, find_packages

setup(
    name="python-3d-programming",
    version="1.0.0",
    description="Python 3D Programming Book Examples",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "Pillow>=8.3.0",
        "PyOpenGL>=3.1.0",
        "glfw>=2.0.0",
        "pygame>=2.1.0",
        "vpython>=3.2.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

## Testing Your Setup

### Basic OpenGL Test
Create a test file `test_opengl.py`:
```python
import glfw
import OpenGL.GL as gl

def test_opengl():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return False
    
    # Create window
    window = glfw.create_window(800, 600, "OpenGL Test", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return False
    
    glfw.make_context_current(window)
    
    # Get OpenGL version
    version = gl.glGetString(gl.GL_VERSION)
    print(f"OpenGL Version: {version}")
    
    # Get vendor
    vendor = gl.glGetString(gl.GL_VENDOR)
    print(f"OpenGL Vendor: {vendor}")
    
    # Get renderer
    renderer = gl.glGetString(gl.GL_RENDERER)
    print(f"OpenGL Renderer: {renderer}")
    
    glfw.terminate()
    return True

if __name__ == "__main__":
    test_opengl()
```

### NumPy Test
Create a test file `test_numpy.py`:
```python
import numpy as np

def test_numpy():
    # Create arrays
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10])
    
    # Basic operations
    c = a + b
    d = a * b
    
    print(f"Array a: {a}")
    print(f"Array b: {b}")
    print(f"Sum: {c}")
    print(f"Product: {d}")
    
    # Matrix operations
    matrix = np.array([[1, 2], [3, 4]])
    determinant = np.linalg.det(matrix)
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {determinant}")
    
    return True

if __name__ == "__main__":
    test_numpy()
```

### VPython Test
Create a test file `test_vpython.py`:
```python
from vpython import *

def test_vpython():
    # Create a sphere
    sphere = sphere(pos=vector(0, 0, 0), radius=1, color=color.red)
    
    # Create a box
    box = box(pos=vector(3, 0, 0), size=vector(1, 1, 1), color=color.blue)
    
    # Create an arrow
    arrow = arrow(pos=vector(-3, 0, 0), axis=vector(1, 1, 0), color=color.green)
    
    print("VPython test scene created successfully!")
    print("You should see a red sphere, blue box, and green arrow.")
    
    return True

if __name__ == "__main__":
    test_vpython()
```

## Troubleshooting Common Issues

### OpenGL Issues
**Problem**: "OpenGL not available" error
**Solution**:
1. Update graphics drivers
2. Install OpenGL libraries:
   ```bash
   # Ubuntu/Debian
   sudo apt install libgl1-mesa-dev libglu1-mesa-dev
   
   # macOS
   brew install mesa
   ```

### NumPy Issues
**Problem**: NumPy import errors
**Solution**:
1. Reinstall NumPy:
   ```bash
   pip uninstall numpy
   pip install numpy
   ```
2. Check Python version compatibility

### VPython Issues
**Problem**: VPython window doesn't appear
**Solution**:
1. Install additional dependencies:
   ```bash
   pip install vpython
   ```
2. Run in Jupyter notebook for better compatibility

### GLFW Issues
**Problem**: GLFW window creation fails
**Solution**:
1. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt install libglfw3-dev
   
   # macOS
   brew install glfw
   ```

## Performance Optimization

### Graphics Settings
1. **Enable Hardware Acceleration**:
   - Update graphics drivers
   - Enable GPU acceleration in applications

2. **Optimize Python Performance**:
   ```bash
   # Install performance libraries
   pip install numba
   pip install cython
   ```

### Memory Management
1. **Monitor Memory Usage**:
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
   ```

2. **Use Generators for Large Data**:
   ```python
   def large_data_generator():
       for i in range(1000000):
           yield i
   ```

## Additional Resources

### Documentation
- [Python Official Documentation](https://docs.python.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [OpenGL Documentation](https://www.opengl.org/documentation/)
- [VPython Documentation](https://www.glowscript.org/docs/VPythonDocs/)

### Online Resources
- [Real Python](https://realpython.com/) - Python tutorials
- [OpenGL Tutorial](https://learnopengl.com/) - OpenGL learning
- [3D Graphics Programming](https://www.3dgraphicsprogramming.com/) - 3D concepts

### Books
- "Python Programming: An Introduction to Computer Science" by John Zelle
- "OpenGL Programming Guide" by Dave Shreiner
- "Real-Time Rendering" by Tomas Akenine-Möller

## Conclusion
This environment setup provides a solid foundation for Python 3D programming. The combination of Python, NumPy, OpenGL, and VPython creates a powerful development environment for both learning and professional 3D graphics development.

Remember to:
- Keep your environment updated
- Use virtual environments for project isolation
- Test your setup regularly
- Document any custom configurations
- Back up your environment configuration
