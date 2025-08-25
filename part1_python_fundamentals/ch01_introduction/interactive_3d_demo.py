#!/usr/bin/env python3
"""
Chapter 1: Introduction to Python
Interactive 3D Demo

This example demonstrates Python's ability to create interactive 3D visualizations
using matplotlib, showing practical applications in 3D graphics and data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def create_3d_scene():
    """Create a simple 3D scene with various objects"""
    print("Creating 3D scene...")
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the scene
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Python 3D Scene Demo')
    
    # Create a cube
    def create_cube(center, size, color='blue'):
        """Create a cube at the specified center with given size"""
        x, y, z = center
        s = size / 2
        
        # Define the 8 vertices of the cube
        vertices = np.array([
            [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],
            [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]
        ])
        
        # Define the 6 faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
        ]
        
        # Plot each face
        for face in faces:
            face = np.array(face)
            ax.plot_trisurf(face[:, 0], face[:, 1], face[:, 2], 
                          color=color, alpha=0.7)
    
    # Create a sphere
    def create_sphere(center, radius, color='red'):
        """Create a sphere at the specified center with given radius"""
        x, y, z = center
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        sphere_x = x + radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = y + radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                       color=color, alpha=0.7)
    
    # Create a cylinder
    def create_cylinder(center, radius, height, color='green'):
        """Create a cylinder at the specified center"""
        x, y, z = center
        
        # Create cylinder surface
        theta = np.linspace(0, 2 * np.pi, 20)
        h = np.linspace(0, height, 20)
        
        theta_grid, h_grid = np.meshgrid(theta, h)
        
        cylinder_x = x + radius * np.cos(theta_grid)
        cylinder_y = y + radius * np.sin(theta_grid)
        cylinder_z = z + h_grid
        
        ax.plot_surface(cylinder_x, cylinder_y, cylinder_z, 
                       color=color, alpha=0.7)
    
    # Add objects to the scene
    print("Adding objects to scene...")
    
    # Add a cube at the origin
    create_cube([0, 0, 0], 2, 'blue')
    print("✓ Added blue cube at origin")
    
    # Add a sphere
    create_sphere([3, 0, 0], 1, 'red')
    print("✓ Added red sphere")
    
    # Add a cylinder
    create_cylinder([0, 3, 0], 0.8, 2, 'green')
    print("✓ Added green cylinder")
    
    # Add some smaller objects
    create_cube([-3, 0, 0], 1, 'purple')
    create_sphere([0, -3, 0], 0.5, 'orange')
    create_cylinder([3, 3, 0], 0.5, 1.5, 'brown')
    
    # Set view limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    
    return fig, ax

def demonstrate_interactivity():
    """Demonstrate interactive features"""
    print("\n=== Interactive 3D Demo ===")
    print("This demo shows Python's ability to create interactive 3D visualizations.")
    print("Features demonstrated:")
    print("✓ 3D object creation (cubes, spheres, cylinders)")
    print("✓ Interactive rotation and zoom")
    print("✓ Multiple object types in one scene")
    print("✓ Color and transparency effects")
    print("✓ Real-time rendering")
    
    # Create the scene
    fig, ax = create_3d_scene()
    
    # Add some interactive text
    ax.text2D(0.02, 0.98, "Python 3D Scene\nUse mouse to rotate and zoom", 
              transform=ax.transAxes, fontsize=10, 
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print("\nScene created! You can:")
    print("- Rotate the view by clicking and dragging")
    print("- Zoom with the scroll wheel")
    print("- Pan by right-clicking and dragging")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    print("\nDemo completed!")

def create_animated_scene():
    """Create an animated 3D scene"""
    print("\n=== Creating Animated 3D Scene ===")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Animated Python 3D Scene')
    
    # Create animated objects
    angles = np.linspace(0, 2*np.pi, 50)
    
    for i, angle in enumerate(angles):
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title(f'Animated Scene - Frame {i+1}')
        
        # Create rotating objects
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        
        # Sphere that moves in a circle
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = x + 0.5 * np.outer(np.cos(u), np.sin(v))
        sphere_y = y + 0.5 * np.outer(np.sin(u), np.sin(v))
        sphere_z = 0 + 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='red', alpha=0.7)
        
        # Cube that rotates around the origin
        cube_angle = angle * 2
        cube_x = 2 * np.cos(cube_angle)
        cube_y = 2 * np.sin(cube_angle)
        
        # Simple cube representation
        ax.scatter([cube_x], [cube_y], [0], c='blue', s=100, marker='s')
        
        plt.pause(0.1)
    
    plt.show()
    print("Animation completed!")

def demonstrate_data_visualization():
    """Demonstrate 3D data visualization capabilities"""
    print("\n=== 3D Data Visualization Demo ===")
    
    # Generate sample 3D data
    np.random.seed(42)
    n_points = 100
    
    # Create clustered data
    cluster1 = np.random.normal([2, 2, 2], 0.5, (n_points//3, 3))
    cluster2 = np.random.normal([-2, -2, -2], 0.5, (n_points//3, 3))
    cluster3 = np.random.normal([0, 0, 0], 0.8, (n_points//3, 3))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data points
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                        c=range(len(data)), cmap='viridis', s=50)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Data Visualization with Python')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ 3D scatter plot created")
    print("✓ Color-coded data points")
    print("✓ Interactive 3D visualization")

def main():
    """Main function to run the interactive 3D demo"""
    print("=== Python Interactive 3D Demo ===\n")
    
    try:
        # Check if matplotlib is available
        import matplotlib
        print("✓ Matplotlib is available")
        
        # Run the demonstrations
        demonstrate_interactivity()
        
        # Ask user if they want to see animation
        response = input("\nWould you like to see an animated scene? (y/n): ")
        if response.lower() in ['y', 'yes']:
            create_animated_scene()
        
        # Ask user if they want to see data visualization
        response = input("\nWould you like to see 3D data visualization? (y/n): ")
        if response.lower() in ['y', 'yes']:
            demonstrate_data_visualization()
        
    except ImportError:
        print("❌ Matplotlib is not installed. Install it with: pip install matplotlib")
        print("This demo requires matplotlib for 3D visualization.")
    
    print("\n=== Demo Summary ===")
    print("This demo demonstrated Python's capabilities for:")
    print("✓ Creating 3D scenes and objects")
    print("✓ Interactive 3D visualizations")
    print("✓ Real-time rendering and animation")
    print("✓ Data visualization in 3D space")
    print("✓ User interaction and control")
    
    print("\nPython is excellent for 3D graphics because it provides:")
    print("- Easy-to-use libraries (matplotlib, plotly, etc.)")
    print("- High-level abstractions for complex operations")
    print("- Interactive development and visualization")
    print("- Integration with scientific computing tools")
    print("- Cross-platform compatibility")

if __name__ == "__main__":
    main()
