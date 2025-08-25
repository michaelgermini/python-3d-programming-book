#!/usr/bin/env python3
"""
Chapter 9: Advanced Python Concepts
External Libraries Example

Demonstrates external libraries including NumPy, Pillow, Matplotlib, Pandas,
and other popular libraries for 3D graphics and scientific computing.
"""

import sys
import time
import math
import random
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics External Libraries"
__description__ = "External libraries for 3D graphics and scientific computing"

# ============================================================================
# NUMPY EXAMPLES
# ============================================================================

def demonstrate_numpy():
    """Demonstrate NumPy for numerical computing"""
    print("=== NumPy Examples ===\n")
    
    try:
        import numpy as np
        
        print("1. Basic NumPy arrays:")
        # Create arrays
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        arr3 = np.zeros((3, 3))
        arr4 = np.ones((2, 4))
        arr5 = np.random.rand(3, 3)
        
        print(f"   1D array: {arr1}")
        print(f"   2D array:\n{arr2}")
        print(f"   Zeros array:\n{arr3}")
        print(f"   Ones array:\n{arr4}")
        print(f"   Random array:\n{arr5}")
        
        print("\n2. Array operations:")
        # Mathematical operations
        result1 = arr1 * 2
        result2 = arr1 + arr1
        result3 = np.sqrt(arr1)
        result4 = np.sin(arr1)
        
        print(f"   Array * 2: {result1}")
        print(f"   Array + Array: {result2}")
        print(f"   Square root: {result3}")
        print(f"   Sine: {result4}")
        
        print("\n3. 3D graphics with NumPy:")
        # 3D vector operations
        vertices = np.array([
            [0, 0, 0],  # Bottom front left
            [1, 0, 0],  # Bottom front right
            [1, 1, 0],  # Top front right
            [0, 1, 0],  # Top front left
            [0, 0, 1],  # Bottom back left
            [1, 0, 1],  # Bottom back right
            [1, 1, 1],  # Top back right
            [0, 1, 1]   # Top back left
        ])
        
        # Translation matrix
        translation = np.array([0.5, 0.5, 0.5])
        translated_vertices = vertices + translation
        
        # Rotation matrix (around Z-axis)
        angle = np.pi / 4  # 45 degrees
        rotation_z = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_vertices = np.dot(vertices, rotation_z.T)
        
        print(f"   Original vertices:\n{vertices}")
        print(f"   Translated vertices:\n{translated_vertices}")
        print(f"   Rotated vertices:\n{rotated_vertices}")
        
        print("\n4. Advanced NumPy features:")
        # Broadcasting
        arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
        arr_1d = np.array([10, 20, 30])
        broadcasted = arr_2d + arr_1d
        
        # Reshaping
        flat_array = np.arange(12)
        reshaped = flat_array.reshape(3, 4)
        
        # Indexing and slicing
        sub_array = arr2[1:, :2]
        
        print(f"   Broadcasting result:\n{broadcasted}")
        print(f"   Reshaped array:\n{reshaped}")
        print(f"   Sliced array:\n{sub_array}")
        
    except ImportError:
        print("❌ NumPy not installed. Install with: pip install numpy")
    
    print()

# ============================================================================
# PILLOW (PIL) EXAMPLES
# ============================================================================

def demonstrate_pillow():
    """Demonstrate Pillow for image processing"""
    print("=== Pillow (PIL) Examples ===\n")
    
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
        
        print("1. Creating images:")
        # Create a new image
        width, height = 400, 300
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw shapes
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
        draw.ellipse([200, 50, 300, 150], fill='blue', outline='black')
        draw.polygon([(100, 200), (150, 250), (200, 200)], fill='green')
        
        # Save the image
        img.save('demo_image.png')
        print("   Created demo_image.png with shapes")
        
        print("\n2. Image processing:")
        # Apply filters
        blurred = img.filter(ImageFilter.BLUR)
        blurred.save('blurred_image.png')
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        brightened = enhancer.enhance(1.5)
        brightened.save('brightened_image.png')
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(img)
        contrasted = contrast_enhancer.enhance(2.0)
        contrasted.save('contrasted_image.png')
        
        print("   Applied blur, brightness, and contrast effects")
        
        print("\n3. Image information:")
        print(f"   Image size: {img.size}")
        print(f"   Image mode: {img.mode}")
        print(f"   Image format: {img.format}")
        
        # Get pixel data
        pixel_data = list(img.getdata())
        print(f"   Number of pixels: {len(pixel_data)}")
        
    except ImportError:
        print("❌ Pillow not installed. Install with: pip install Pillow")
    
    print()

# ============================================================================
# MATPLOTLIB EXAMPLES
# ============================================================================

def demonstrate_matplotlib():
    """Demonstrate Matplotlib for plotting and visualization"""
    print("=== Matplotlib Examples ===\n")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("1. Basic plotting:")
        # Create data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Line plot
        ax1.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
        ax1.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
        ax1.set_title('Trigonometric Functions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True)
        
        # Scatter plot
        x_scatter = np.random.randn(50)
        y_scatter = np.random.randn(50)
        colors = np.random.rand(50)
        ax2.scatter(x_scatter, y_scatter, c=colors, alpha=0.6)
        ax2.set_title('Scatter Plot')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # Bar plot
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 56, 78, 32]
        ax3.bar(categories, values, color=['red', 'blue', 'green', 'yellow', 'purple'])
        ax3.set_title('Bar Chart')
        ax3.set_xlabel('Categories')
        ax3.set_ylabel('Values')
        
        # 3D plot
        ax4.remove()  # Remove the 4th subplot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        x_3d = np.linspace(-5, 5, 50)
        y_3d = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x_3d, y_3d)
        Z = np.sin(np.sqrt(X**2 + Y**2))
        ax3d.plot_surface(X, Y, Z, cmap='viridis')
        ax3d.set_title('3D Surface Plot')
        
        plt.tight_layout()
        plt.savefig('matplotlib_demo.png', dpi=300, bbox_inches='tight')
        print("   Created matplotlib_demo.png with multiple plots")
        
        print("\n2. 3D graphics visualization:")
        # Create a 3D cube visualization
        fig_3d = plt.figure(figsize=(10, 8))
        ax_cube = fig_3d.add_subplot(111, projection='3d')
        
        # Define cube vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        # Define cube edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            ax_cube.plot3D(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                'b-', linewidth=2
            )
        
        # Plot vertices
        ax_cube.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c='red', s=100)
        
        ax_cube.set_xlabel('X')
        ax_cube.set_ylabel('Y')
        ax_cube.set_zlabel('Z')
        ax_cube.set_title('3D Cube Visualization')
        
        plt.savefig('3d_cube.png', dpi=300, bbox_inches='tight')
        print("   Created 3d_cube.png with 3D cube visualization")
        
    except ImportError:
        print("❌ Matplotlib not installed. Install with: pip install matplotlib")
    
    print()

# ============================================================================
# PANDAS EXAMPLES
# ============================================================================

def demonstrate_pandas():
    """Demonstrate Pandas for data manipulation"""
    print("=== Pandas Examples ===\n")
    
    try:
        import pandas as pd
        import numpy as np
        
        print("1. Creating DataFrames:")
        # Create sample data
        data = {
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000],
            'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales'],
            'experience': [2, 5, 8, 3, 6]
        }
        
        df = pd.DataFrame(data)
        print(f"   DataFrame:\n{df}")
        
        print("\n2. Data analysis:")
        # Basic statistics
        print(f"   Age statistics:\n{df['age'].describe()}")
        print(f"   Salary statistics:\n{df['salary'].describe()}")
        
        # Grouping
        dept_stats = df.groupby('department').agg({
            'age': 'mean',
            'salary': 'mean',
            'experience': 'mean'
        }).round(2)
        print(f"   Department statistics:\n{dept_stats}")
        
        print("\n3. Data filtering and manipulation:")
        # Filtering
        high_salary = df[df['salary'] > 60000]
        engineering_dept = df[df['department'] == 'Engineering']
        
        print(f"   High salary employees:\n{high_salary}")
        print(f"   Engineering department:\n{engineering_dept}")
        
        # Sorting
        sorted_by_salary = df.sort_values('salary', ascending=False)
        print(f"   Sorted by salary:\n{sorted_by_salary}")
        
        print("\n4. 3D graphics data analysis:")
        # Create 3D graphics performance data
        performance_data = {
            'frame_rate': np.random.normal(60, 10, 100),
            'memory_usage': np.random.normal(512, 100, 100),
            'cpu_usage': np.random.normal(50, 15, 100),
            'gpu_usage': np.random.normal(70, 20, 100),
            'scene_complexity': np.random.uniform(0, 1, 100)
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Performance analysis
        print(f"   Performance statistics:\n{perf_df.describe()}")
        
        # Correlation analysis
        correlation = perf_df.corr()
        print(f"   Correlation matrix:\n{correlation}")
        
        # Performance by scene complexity
        complexity_bins = pd.cut(perf_df['scene_complexity'], bins=5)
        complexity_stats = perf_df.groupby(complexity_bins).agg({
            'frame_rate': 'mean',
            'memory_usage': 'mean',
            'cpu_usage': 'mean',
            'gpu_usage': 'mean'
        }).round(2)
        print(f"   Performance by scene complexity:\n{complexity_stats}")
        
    except ImportError:
        print("❌ Pandas not installed. Install with: pip install pandas")
    
    print()

# ============================================================================
# SCIENTIFIC COMPUTING EXAMPLES
# ============================================================================

def demonstrate_scipy():
    """Demonstrate SciPy for scientific computing"""
    print("=== SciPy Examples ===\n")
    
    try:
        import scipy as sp
        from scipy import optimize, interpolate, signal
        import numpy as np
        
        print("1. Optimization:")
        # Function optimization
        def objective_function(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2
        
        # Find minimum
        result = optimize.minimize(objective_function, [0, 0, 0])
        print(f"   Optimization result: {result.x}")
        print(f"   Minimum value: {result.fun}")
        
        print("\n2. Interpolation:")
        # Create sample data
        x_data = np.linspace(0, 10, 20)
        y_data = np.sin(x_data) + np.random.normal(0, 0.1, 20)
        
        # Interpolate
        f_interp = interpolate.interp1d(x_data, y_data, kind='cubic')
        x_new = np.linspace(0, 10, 100)
        y_new = f_interp(x_new)
        
        print(f"   Original data points: {len(x_data)}")
        print(f"   Interpolated points: {len(x_new)}")
        
        print("\n3. Signal processing:")
        # Create a signal with noise
        t = np.linspace(0, 1, 1000)
        signal_clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        signal_noisy = signal_clean + 0.5 * np.random.randn(1000)
        
        # Apply low-pass filter
        b, a = signal.butter(4, 0.1, 'low')
        signal_filtered = signal.filtfilt(b, a, signal_noisy)
        
        print(f"   Signal length: {len(t)}")
        print(f"   Filter applied: Butterworth low-pass")
        
    except ImportError:
        print("❌ SciPy not installed. Install with: pip install scipy")
    
    print()

# ============================================================================
# WEB FRAMEWORKS EXAMPLES
# ============================================================================

def demonstrate_web_frameworks():
    """Demonstrate web frameworks for 3D graphics applications"""
    print("=== Web Frameworks Examples ===\n")
    
    print("1. Flask example (simulated):")
    flask_code = '''
from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/3d-data')
def get_3d_data():
    data = {
        'vertices': [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        'faces': [[0, 1, 2]],
        'colors': ['red', 'green', 'blue']
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
'''
    print("   Flask application structure:")
    print("   - Web server for 3D graphics")
    print("   - REST API for 3D data")
    print("   - Template rendering")
    
    print("\n2. Streamlit example (simulated):")
    streamlit_code = '''
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title('3D Graphics Dashboard')

# Create 3D scatter plot
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
st.plotly_chart(fig)
'''
    print("   Streamlit application structure:")
    print("   - Interactive web dashboard")
    print("   - Real-time 3D visualization")
    print("   - Data exploration tools")
    
    print()

# ============================================================================
# GAME DEVELOPMENT EXAMPLES
# ============================================================================

def demonstrate_game_libraries():
    """Demonstrate game development libraries"""
    print("=== Game Development Libraries ===\n")
    
    print("1. Pygame example (simulated):")
    pygame_code = '''
import pygame
import sys

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear screen
    screen.fill((0, 0, 0))
    
    # Draw 3D-like objects (2D projection)
    pygame.draw.circle(screen, (255, 0, 0), (400, 300), 50)
    pygame.draw.rect(screen, (0, 255, 0), (350, 250, 100, 100))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
'''
    print("   Pygame features:")
    print("   - 2D graphics and sound")
    print("   - Event handling")
    print("   - Game loop management")
    print("   - Input processing")
    
    print("\n2. PyOpenGL example (simulated):")
    opengl_code = '''
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Initialize Pygame with OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Set up OpenGL
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    glRotatef(1, 3, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw cube
    glBegin(GL_QUADS)
    # ... cube vertices
    glEnd()
    
    pygame.display.flip()
    pygame.time.wait(10)
'''
    print("   PyOpenGL features:")
    print("   - Hardware-accelerated 3D graphics")
    print("   - OpenGL bindings")
    print("   - 3D rendering pipeline")
    print("   - Shader support")
    
    print()

# ============================================================================
# MACHINE LEARNING EXAMPLES
# ============================================================================

def demonstrate_ml_libraries():
    """Demonstrate machine learning libraries"""
    print("=== Machine Learning Libraries ===\n")
    
    print("1. Scikit-learn example (simulated):")
    sklearn_code = '''
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# 3D point cloud clustering
points = np.random.rand(1000, 3)
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(points)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
points_2d = pca.fit_transform(points)
'''
    print("   Scikit-learn features:")
    print("   - 3D point cloud clustering")
    print("   - Dimensionality reduction")
    print("   - Pattern recognition")
    print("   - Data preprocessing")
    
    print("\n2. TensorFlow/PyTorch example (simulated):")
    tensorflow_code = '''
import tensorflow as tf
import numpy as np

# Neural network for 3D object classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Training data
X_train = np.random.rand(1000, 3)  # 3D coordinates
y_train = np.random.randint(0, 10, 1000)  # Object classes

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
'''
    print("   TensorFlow/PyTorch features:")
    print("   - Deep learning for 3D data")
    print("   - Neural network architectures")
    print("   - GPU acceleration")
    print("   - Model training and inference")
    
    print()

# ============================================================================
# UTILITY LIBRARIES
# ============================================================================

def demonstrate_utility_libraries():
    """Demonstrate utility libraries"""
    print("=== Utility Libraries ===\n")
    
    print("1. Requests (HTTP library):")
    requests_code = '''
import requests
import json

# Fetch 3D model data from API
response = requests.get('https://api.example.com/3d-models')
models = response.json()

# Download 3D model file
model_url = 'https://example.com/model.obj'
response = requests.get(model_url)
with open('model.obj', 'wb') as f:
    f.write(response.content)
'''
    print("   Requests features:")
    print("   - HTTP requests for 3D data")
    print("   - API integration")
    print("   - File downloading")
    print("   - JSON data handling")
    
    print("\n2. Beautiful Soup (Web scraping):")
    bs_code = '''
from bs4 import BeautifulSoup
import requests

# Scrape 3D model information
url = 'https://example.com/3d-models'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract model links
model_links = soup.find_all('a', href=True)
for link in model_links:
    if link['href'].endswith('.obj'):
        print(f"Found 3D model: {link['href']}")
'''
    print("   Beautiful Soup features:")
    print("   - Web scraping for 3D resources")
    print("   - HTML parsing")
    print("   - Data extraction")
    print("   - Link discovery")
    
    print("\n3. Pillow (Image processing):")
    pillow_code = '''
from PIL import Image, ImageFilter, ImageEnhance

# Process texture images
image = Image.open('texture.png')
resized = image.resize((512, 512))
blurred = resized.filter(ImageFilter.BLUR)
enhanced = ImageEnhance.Contrast(blurred).enhance(1.5)
enhanced.save('processed_texture.png')
'''
    print("   Pillow features:")
    print("   - Texture image processing")
    print("   - Image resizing and filtering")
    print("   - Color enhancement")
    print("   - Format conversion")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate external libraries"""
    print("=== External Libraries Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all libraries
    demonstrate_numpy()
    demonstrate_pillow()
    demonstrate_matplotlib()
    demonstrate_pandas()
    demonstrate_scipy()
    demonstrate_web_frameworks()
    demonstrate_game_libraries()
    demonstrate_ml_libraries()
    demonstrate_utility_libraries()
    
    print("="*60)
    print("External Libraries demo completed successfully!")
    print("\nKey libraries demonstrated:")
    print("✓ NumPy: Numerical computing and array operations")
    print("✓ Pillow: Image processing and manipulation")
    print("✓ Matplotlib: Data visualization and plotting")
    print("✓ Pandas: Data manipulation and analysis")
    print("✓ SciPy: Scientific computing and optimization")
    print("✓ Web Frameworks: Flask and Streamlit for web applications")
    print("✓ Game Libraries: Pygame and PyOpenGL for game development")
    print("✓ ML Libraries: Scikit-learn and TensorFlow for machine learning")
    print("✓ Utility Libraries: Requests, Beautiful Soup, and more")
    
    print("\nInstallation commands:")
    print("pip install numpy pillow matplotlib pandas scipy")
    print("pip install flask streamlit pygame PyOpenGL")
    print("pip install scikit-learn tensorflow requests beautifulsoup4")

if __name__ == "__main__":
    main()
