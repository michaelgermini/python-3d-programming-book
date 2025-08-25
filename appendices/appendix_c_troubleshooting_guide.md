# Appendix C: Troubleshooting Guide

## Overview
This appendix provides solutions to common problems encountered during Python 3D development. It covers installation issues, runtime errors, performance problems, and debugging techniques.

## Installation Issues

### Python Installation Problems

#### Problem: "Python is not recognized as an internal or external command"
**Symptoms**: Command prompt shows error when running `python --version`
**Solutions**:
1. **Add Python to PATH**:
   - Windows: Reinstall Python with "Add Python to PATH" checked
   - Or manually add Python directory to system PATH
2. **Use python3 command**:
   ```bash
   python3 --version
   ```
3. **Check installation location**:
   ```bash
   where python
   ```

#### Problem: "pip is not recognized"
**Symptoms**: Cannot install packages with pip
**Solutions**:
1. **Install pip separately**:
   ```bash
   python -m ensurepip --upgrade
   ```
2. **Use pip3**:
   ```bash
   pip3 install package_name
   ```
3. **Reinstall Python** with pip included

### Library Installation Issues

#### Problem: "Microsoft Visual C++ 14.0 is required"
**Symptoms**: Error when installing packages with C extensions
**Solutions**:
1. **Install Visual Studio Build Tools**:
   - Download from Microsoft Visual Studio website
   - Install "C++ build tools" workload
2. **Use pre-compiled wheels**:
   ```bash
   pip install --only-binary=all package_name
   ```
3. **Use conda instead**:
   ```bash
   conda install package_name
   ```

#### Problem: "Permission denied" on Linux/macOS
**Symptoms**: Cannot install packages due to permission errors
**Solutions**:
1. **Use virtual environment** (recommended):
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   pip install package_name
   ```
2. **Use --user flag**:
   ```bash
   pip install --user package_name
   ```
3. **Fix permissions**:
   ```bash
   sudo chown -R $USER:$USER ~/.local
   ```

#### Problem: "SSL Certificate Error"
**Symptoms**: Cannot download packages due to SSL issues
**Solutions**:
1. **Update pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools
   ```
2. **Use trusted host**:
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org package_name
   ```
3. **Configure proxy settings** if behind corporate firewall

## OpenGL Issues

### Problem: "OpenGL not available"
**Symptoms**: OpenGL functions fail or return errors
**Solutions**:
1. **Update graphics drivers**:
   - NVIDIA: Download latest drivers from nvidia.com
   - AMD: Download latest drivers from amd.com
   - Intel: Update through Windows Update or Intel website

2. **Install OpenGL libraries**:
   ```bash
   # Ubuntu/Debian
   sudo apt install libgl1-mesa-dev libglu1-mesa-dev
   
   # macOS
   brew install mesa
   
   # Windows
   # OpenGL should be included with graphics drivers
   ```

3. **Check OpenGL version**:
   ```python
   import OpenGL.GL as gl
   print(gl.glGetString(gl.GL_VERSION))
   ```

### Problem: "GLFW window creation failed"
**Symptoms**: Cannot create GLFW window
**Solutions**:
1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt install libglfw3-dev
   
   # macOS
   brew install glfw
   ```

2. **Check display settings**:
   - Ensure display is not in sleep mode
   - Check if running in headless environment

3. **Use software rendering**:
   ```python
   import os
   os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
   ```

### Problem: "OpenGL context creation failed"
**Symptoms**: Cannot create OpenGL context
**Solutions**:
1. **Check graphics card compatibility**:
   ```python
   import OpenGL.GL as gl
   print(gl.glGetString(gl.GL_VENDOR))
   print(gl.glGetString(gl.GL_RENDERER))
   ```

2. **Use compatibility profile**:
   ```python
   import glfw
   glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
   glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
   glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
   ```

## NumPy Issues

### Problem: "NumPy import error"
**Symptoms**: Cannot import NumPy or NumPy functions fail
**Solutions**:
1. **Reinstall NumPy**:
   ```bash
   pip uninstall numpy
   pip install numpy
   ```

2. **Check Python version compatibility**:
   ```bash
   python --version
   pip show numpy
   ```

3. **Use conda for better compatibility**:
   ```bash
   conda install numpy
   ```

### Problem: "Memory error with large arrays"
**Symptoms**: Out of memory when creating large NumPy arrays
**Solutions**:
1. **Use smaller data types**:
   ```python
   import numpy as np
   # Use float32 instead of float64
   array = np.zeros((1000, 1000), dtype=np.float32)
   ```

2. **Use memory mapping**:
   ```python
   import numpy as np
   array = np.memmap('large_array.dat', dtype=np.float32, mode='w+', shape=(10000, 10000))
   ```

3. **Process data in chunks**:
   ```python
   def process_large_data(filename, chunk_size=1000):
       for chunk in np.load(filename, mmap_mode='r'):
           # Process chunk
           pass
   ```

## VPython Issues

### Problem: "VPython window doesn't appear"
**Symptoms**: VPython code runs but no 3D window shows
**Solutions**:
1. **Run in Jupyter notebook**:
   ```python
   from vpython import *
   # VPython works better in Jupyter
   ```

2. **Check display settings**:
   - Ensure you're not running in headless mode
   - Check if running on remote server

3. **Use web-based VPython**:
   ```python
   # Use glowscript.org for web-based VPython
   ```

### Problem: "VPython performance issues"
**Symptoms**: Slow rendering or laggy animations
**Solutions**:
1. **Reduce object count**:
   ```python
   # Use fewer objects or lower detail
   sphere = sphere(pos=vector(0,0,0), radius=1, color=color.red)
   ```

2. **Use rate limiting**:
   ```python
   from vpython import *
   while True:
       # Update objects
       rate(30)  # Limit to 30 FPS
   ```

3. **Use static objects when possible**:
   ```python
   # Create objects once, then only update positions
   ```

## Performance Issues

### Problem: "Slow rendering performance"
**Symptoms**: Low frame rates or choppy animations
**Solutions**:
1. **Profile your code**:
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   # Your code here
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats()
   ```

2. **Use NumPy vectorization**:
   ```python
   # Instead of loops
   for i in range(1000):
       result[i] = array[i] * 2
   
   # Use vectorized operations
   result = array * 2
   ```

3. **Optimize rendering**:
   ```python
   # Use display lists or vertex buffer objects
   # Reduce state changes
   # Use frustum culling
   ```

### Problem: "High memory usage"
**Symptoms**: Program uses excessive memory
**Solutions**:
1. **Monitor memory usage**:
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
   ```

2. **Use generators for large data**:
   ```python
   def large_data_generator():
       for i in range(1000000):
           yield i
   ```

3. **Clear unused objects**:
   ```python
   import gc
   gc.collect()  # Force garbage collection
   ```

## Debugging Techniques

### Problem: "Code doesn't work as expected"
**Symptoms**: Program runs but produces incorrect results
**Solutions**:
1. **Add debug prints**:
   ```python
   print(f"Debug: value = {value}")
   print(f"Debug: position = {position}")
   ```

2. **Use Python debugger**:
   ```python
   import pdb
   pdb.set_trace()  # Set breakpoint
   ```

3. **Use logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   logger.debug("Debug message")
   ```

### Problem: "OpenGL errors"
**Symptoms**: OpenGL functions return errors
**Solutions**:
1. **Enable OpenGL error checking**:
   ```python
   import OpenGL.GL as gl
   import OpenGL.GL.debug as gl_debug
   
   gl_debug.glEnable(gl.GL_DEBUG_OUTPUT)
   gl_debug.glDebugMessageCallback(debug_callback, None)
   ```

2. **Check for OpenGL errors**:
   ```python
   def check_gl_error():
       error = gl.glGetError()
       if error != gl.GL_NO_ERROR:
           print(f"OpenGL Error: {error}")
   ```

3. **Use OpenGL debug context**:
   ```python
   import glfw
   glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, glfw.TRUE)
   ```

## Platform-Specific Issues

### Windows Issues

#### Problem: "DLL load failed"
**Symptoms**: Cannot load DLL files for libraries
**Solutions**:
1. **Install Visual C++ Redistributable**:
   - Download from Microsoft website
   - Install both x86 and x64 versions

2. **Check PATH environment**:
   - Ensure library directories are in PATH
   - Restart command prompt after changes

3. **Use conda environment**:
   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   conda install package_name
   ```

#### Problem: "Windows Defender blocking execution"
**Symptoms**: Antivirus blocks Python scripts
**Solutions**:
1. **Add exception to Windows Defender**:
   - Add Python and project directories to exclusions
2. **Sign your code** (for distribution)
3. **Use virtual environment**

### macOS Issues

#### Problem: "Gatekeeper blocking execution"
**Symptoms**: macOS prevents running Python scripts
**Solutions**:
1. **Allow execution in System Preferences**:
   - Go to Security & Privacy
   - Click "Allow Anyway" for blocked applications

2. **Use Homebrew Python**:
   ```bash
   brew install python
   ```

3. **Remove quarantine attribute**:
   ```bash
   xattr -d com.apple.quarantine script.py
   ```

#### Problem: "OpenGL deprecated on macOS"
**Symptoms**: OpenGL warnings or errors
**Solutions**:
1. **Use Metal instead of OpenGL**:
   ```python
   # Use Metal-based libraries
   import Metal
   ```

2. **Use compatibility mode**:
   ```python
   import os
   os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
   ```

### Linux Issues

#### Problem: "Missing system libraries"
**Symptoms**: Import errors for system libraries
**Solutions**:
1. **Install development packages**:
   ```bash
   sudo apt install python3-dev libgl1-mesa-dev libglu1-mesa-dev
   ```

2. **Use package manager**:
   ```bash
   sudo apt install python3-numpy python3-opengl
   ```

3. **Check library paths**:
   ```bash
   ldconfig -p | grep library_name
   ```

#### Problem: "Display issues on remote server"
**Symptoms**: Cannot create windows on headless server
**Solutions**:
1. **Use X11 forwarding**:
   ```bash
   ssh -X user@server
   ```

2. **Use virtual display**:
   ```bash
   export DISPLAY=:0
   Xvfb :0 -screen 0 1024x768x24 &
   ```

3. **Use software rendering**:
   ```python
   import os
   os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
   ```

## Common Error Messages and Solutions

### Import Errors
```
ImportError: No module named 'module_name'
```
**Solution**: Install missing module with pip

### Attribute Errors
```
AttributeError: 'object' has no attribute 'method'
```
**Solution**: Check object type and available methods

### Type Errors
```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```
**Solution**: Convert types or check data types

### Index Errors
```
IndexError: list index out of range
```
**Solution**: Check array bounds and indices

### Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce array size or use memory mapping

### OpenGL Errors
```
OpenGL.error.GLError: (1280, b'invalid enum')
```
**Solution**: Check OpenGL function parameters and state

## Best Practices for Troubleshooting

### 1. Start Simple
- Begin with minimal working examples
- Add complexity gradually
- Test each component separately

### 2. Use Version Control
- Commit working code frequently
- Use branches for experiments
- Document changes and fixes

### 3. Create Test Cases
- Write unit tests for functions
- Create regression tests
- Use automated testing

### 4. Document Issues
- Keep a log of problems and solutions
- Document workarounds
- Share solutions with team

### 5. Use Debugging Tools
- Use IDE debuggers
- Profile performance
- Monitor system resources

## Getting Help

### Online Resources
- [Stack Overflow](https://stackoverflow.com/) - Programming Q&A
- [Python Documentation](https://docs.python.org/) - Official docs
- [OpenGL Documentation](https://www.opengl.org/documentation/) - OpenGL reference
- [GitHub Issues](https://github.com/) - Library-specific issues

### Community Forums
- Python Discord
- Reddit r/Python
- OpenGL forums
- Library-specific mailing lists

### Professional Support
- Commercial support for libraries
- Consulting services
- Training courses

## Conclusion
This troubleshooting guide covers the most common issues encountered in Python 3D development. Remember to:

- Start with simple examples
- Use version control
- Document problems and solutions
- Test thoroughly
- Keep your environment updated
- Ask for help when needed

Most issues can be resolved by following systematic debugging approaches and using the resources available in the Python and 3D graphics communities.
