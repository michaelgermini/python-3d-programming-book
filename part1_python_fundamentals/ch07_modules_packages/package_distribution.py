#!/usr/bin/env python3
"""
Chapter 7: Modules and Packages
Package Distribution Example

Demonstrates how to create distributable Python packages with setup tools,
configuration files, and installation procedures for 3D graphics libraries.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics Package Distribution"
__description__ = "Package distribution tools for 3D graphics libraries"

# ============================================================================
# PACKAGE CONFIGURATION CLASSES
# ============================================================================

@dataclass
class PackageInfo:
    """Package information for distribution"""
    name: str
    version: str
    description: str
    author: str
    author_email: str
    url: str
    license: str
    classifiers: List[str]
    keywords: List[str]
    python_requires: str
    install_requires: List[str]
    extras_require: Dict[str, List[str]]
    packages: List[str]
    package_data: Dict[str, List[str]]
    entry_points: Dict[str, List[str]]
    
    def to_setup_dict(self) -> Dict[str, Any]:
        """Convert to setup.py compatible dictionary"""
        return asdict(self)

@dataclass
class PackageStructure:
    """Package directory structure"""
    root_dir: str
    package_dir: str
    docs_dir: str
    tests_dir: str
    examples_dir: str
    scripts_dir: str
    
    def create_structure(self):
        """Create the package directory structure"""
        directories = [
            self.root_dir,
            self.package_dir,
            self.docs_dir,
            self.tests_dir,
            self.examples_dir,
            self.scripts_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")

# ============================================================================
# SETUP.PY GENERATOR
# ============================================================================

class SetupGenerator:
    """Generate setup.py files for package distribution"""
    
    def __init__(self, package_info: PackageInfo):
        self.package_info = package_info
    
    def generate_setup_py(self, output_file: str = "setup.py"):
        """Generate setup.py file"""
        setup_content = f'''#!/usr/bin/env python3
"""
Setup script for {self.package_info.name}
Generated automatically by Package Distribution Tools
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "{self.package_info.description}"

setup(
    name="{self.package_info.name}",
    version="{self.package_info.version}",
    description="{self.package_info.description}",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="{self.package_info.author}",
    author_email="{self.package_info.author_email}",
    url="{self.package_info.url}",
    license="{self.package_info.license}",
    classifiers={self.package_info.classifiers},
    keywords={self.package_info.keywords},
    python_requires="{self.package_info.python_requires}",
    packages=find_packages(),
    install_requires={self.package_info.install_requires},
    extras_require={self.package_info.extras_require},
    package_data={self.package_info.package_data},
    entry_points={self.package_info.entry_points},
    include_package_data=True,
    zip_safe=False,
)
'''
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(setup_content)
        
        print(f"Generated setup.py: {output_file}")
        return setup_content

# ============================================================================
# MANIFEST.IN GENERATOR
# ============================================================================

class ManifestGenerator:
    """Generate MANIFEST.in files for package distribution"""
    
    def __init__(self):
        self.include_patterns = [
            "*.py",
            "*.md",
            "*.txt",
            "*.rst",
            "*.cfg",
            "*.ini",
            "*.json",
            "*.yaml",
            "*.yml"
        ]
        self.exclude_patterns = [
            "*.pyc",
            "__pycache__",
            "*.pyo",
            "*.pyd",
            ".git*",
            ".DS_Store",
            "*.egg-info",
            "build",
            "dist"
        ]
    
    def generate_manifest_in(self, output_file: str = "MANIFEST.in"):
        """Generate MANIFEST.in file"""
        manifest_content = "# MANIFEST.in - Package distribution manifest\n"
        manifest_content += "# Generated automatically by Package Distribution Tools\n\n"
        
        # Include patterns
        manifest_content += "# Include common file types\n"
        for pattern in self.include_patterns:
            manifest_content += f"include {pattern}\n"
        
        manifest_content += "\n# Include documentation\n"
        manifest_content += "include README.md\n"
        manifest_content += "include LICENSE\n"
        manifest_content += "include CHANGELOG.md\n"
        manifest_content += "recursive-include docs *\n"
        
        manifest_content += "\n# Include examples\n"
        manifest_content += "recursive-include examples *\n"
        
        manifest_content += "\n# Include tests\n"
        manifest_content += "recursive-include tests *\n"
        
        manifest_content += "\n# Exclude patterns\n"
        for pattern in self.exclude_patterns:
            manifest_content += f"exclude {pattern}\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(manifest_content)
        
        print(f"Generated MANIFEST.in: {output_file}")
        return manifest_content

# ============================================================================
# PACKAGE BUILDER
# ============================================================================

class PackageBuilder:
    """Build distributable packages"""
    
    def __init__(self, package_info: PackageInfo, structure: PackageStructure):
        self.package_info = package_info
        self.structure = structure
        self.setup_generator = SetupGenerator(package_info)
        self.manifest_generator = ManifestGenerator()
    
    def build_package(self):
        """Build the complete package"""
        print("=== Building Package ===\n")
        
        # Create directory structure
        print("1. Creating directory structure...")
        self.structure.create_structure()
        
        # Generate setup.py
        print("\n2. Generating setup.py...")
        setup_path = os.path.join(self.structure.root_dir, "setup.py")
        self.setup_generator.generate_setup_py(setup_path)
        
        # Generate MANIFEST.in
        print("\n3. Generating MANIFEST.in...")
        manifest_path = os.path.join(self.structure.root_dir, "MANIFEST.in")
        self.manifest_generator.generate_manifest_in(manifest_path)
        
        # Generate README.md
        print("\n4. Generating README.md...")
        readme_path = os.path.join(self.structure.root_dir, "README.md")
        self._generate_readme(readme_path)
        
        # Generate __init__.py files
        print("\n5. Generating __init__.py files...")
        self._generate_init_files()
        
        # Generate requirements.txt
        print("\n6. Generating requirements.txt...")
        requirements_path = os.path.join(self.structure.root_dir, "requirements.txt")
        self._generate_requirements(requirements_path)
        
        print("\nPackage build completed successfully!")
        return True
    
    def _generate_readme(self, output_file: str):
        """Generate README.md file"""
        readme_content = f"""# {self.package_info.name}

{self.package_info.description}

## Installation

```bash
pip install {self.package_info.name}
```

## Quick Start

```python
import {self.package_info.name}

# Your code here
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Documentation

See the `docs/` directory for detailed documentation.

## Examples

See the `examples/` directory for usage examples.

## License

{self.package_info.license}

## Author

{self.package_info.author} ({self.package_info.author_email})
"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print(f"Generated README.md: {output_file}")
    
    def _generate_init_files(self):
        """Generate __init__.py files for packages"""
        # Main package __init__.py
        main_init_path = os.path.join(self.structure.package_dir, "__init__.py")
        main_init_content = f'''"""
{self.package_info.name} - {self.package_info.description}

Version: {self.package_info.version}
Author: {self.package_info.author}
"""

__version__ = "{self.package_info.version}"
__author__ = "{self.package_info.author}"
__email__ = "{self.package_info.author_email}"

# Import main components
# from .core import *
# from .utils import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
'''
        
        with open(main_init_path, "w", encoding="utf-8") as f:
            f.write(main_init_content)
        
        print(f"Generated __init__.py: {main_init_path}")
    
    def _generate_requirements(self, output_file: str):
        """Generate requirements.txt file"""
        requirements_content = "# Requirements for {}\n".format(self.package_info.name)
        requirements_content += "# Generated automatically\n\n"
        
        for requirement in self.package_info.install_requires:
            requirements_content += f"{requirement}\n"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(requirements_content)
        
        print(f"Generated requirements.txt: {output_file}")

# ============================================================================
# PACKAGE INSTALLER
# ============================================================================

class PackageInstaller:
    """Install packages in development mode"""
    
    def __init__(self, package_root: str):
        self.package_root = package_root
        self.setup_py_path = os.path.join(package_root, "setup.py")
    
    def install_development(self):
        """Install package in development mode"""
        if not os.path.exists(self.setup_py_path):
            print(f"Error: setup.py not found at {self.setup_py_path}")
            return False
        
        print(f"Installing package in development mode from: {self.package_root}")
        print("Command: pip install -e .")
        print("(This would install the package in development mode)")
        return True
    
    def install_production(self):
        """Install package in production mode"""
        if not os.path.exists(self.setup_py_path):
            print(f"Error: setup.py not found at {self.setup_py_path}")
            return False
        
        print(f"Installing package in production mode from: {self.package_root}")
        print("Command: pip install .")
        print("(This would install the package in production mode)")
        return True
    
    def build_distribution(self):
        """Build distribution packages"""
        if not os.path.exists(self.setup_py_path):
            print(f"Error: setup.py not found at {self.setup_py_path}")
            return False
        
        print(f"Building distribution packages from: {self.package_root}")
        print("Command: python setup.py sdist bdist_wheel")
        print("(This would create source and wheel distributions)")
        return True

# ============================================================================
# PACKAGE VALIDATOR
# ============================================================================

class PackageValidator:
    """Validate package structure and configuration"""
    
    def __init__(self, package_root: str):
        self.package_root = package_root
    
    def validate_structure(self) -> Dict[str, bool]:
        """Validate package directory structure"""
        required_files = [
            "setup.py",
            "README.md",
            "MANIFEST.in",
            "requirements.txt"
        ]
        
        required_dirs = [
            "src",
            "docs",
            "tests",
            "examples"
        ]
        
        results = {}
        
        # Check required files
        for file_name in required_files:
            file_path = os.path.join(self.package_root, file_name)
            results[f"file_{file_name}"] = os.path.exists(file_path)
        
        # Check required directories
        for dir_name in required_dirs:
            dir_path = os.path.join(self.package_root, dir_name)
            results[f"dir_{dir_name}"] = os.path.exists(dir_path)
        
        return results
    
    def validate_setup_py(self) -> Dict[str, Any]:
        """Validate setup.py configuration"""
        setup_py_path = os.path.join(self.package_root, "setup.py")
        
        if not os.path.exists(setup_py_path):
            return {"error": "setup.py not found"}
        
        # Read and parse setup.py (simplified validation)
        try:
            with open(setup_py_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            validation = {
                "has_setup_call": "setup(" in content,
                "has_name": "name=" in content,
                "has_version": "version=" in content,
                "has_description": "description=" in content,
                "has_author": "author=" in content,
                "has_packages": "packages=" in content,
                "has_install_requires": "install_requires=" in content
            }
            
            return validation
        except Exception as e:
            return {"error": str(e)}
    
    def validate_manifest(self) -> Dict[str, Any]:
        """Validate MANIFEST.in configuration"""
        manifest_path = os.path.join(self.package_root, "MANIFEST.in")
        
        if not os.path.exists(manifest_path):
            return {"error": "MANIFEST.in not found"}
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            validation = {
                "has_include": "include" in content,
                "has_exclude": "exclude" in content,
                "has_recursive_include": "recursive-include" in content,
                "has_readme": "README.md" in content,
                "has_license": "LICENSE" in content
            }
            
            return validation
        except Exception as e:
            return {"error": str(e)}

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_package_configuration():
    """Demonstrate package configuration"""
    print("=== Package Configuration Demo ===\n")
    
    # Create package information
    package_info = PackageInfo(
        name="graphics3d",
        version="1.0.0",
        description="A comprehensive 3D graphics library for Python",
        author="3D Graphics Team",
        author_email="team@graphics3d.org",
        url="https://github.com/graphics3d/python-graphics3d",
        license="MIT",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Multimedia :: Graphics :: 3D Rendering",
            "Topic :: Scientific/Engineering :: Visualization"
        ],
        keywords=["3d", "graphics", "rendering", "opengl", "visualization"],
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.20.0",
            "matplotlib>=3.3.0",
            "pillow>=8.0.0"
        ],
        extras_require={
            "dev": [
                "pytest>=6.0.0",
                "black>=21.0.0",
                "flake8>=3.8.0"
            ],
            "docs": [
                "sphinx>=4.0.0",
                "sphinx-rtd-theme>=1.0.0"
            ],
            "opengl": [
                "pyopengl>=3.1.0",
                "pyopengl-accelerate>=3.1.0"
            ]
        },
        packages=["graphics3d", "graphics3d.math", "graphics3d.rendering"],
        package_data={
            "graphics3d": ["*.glsl", "*.vert", "*.frag", "shaders/*"],
            "graphics3d.examples": ["*.obj", "*.mtl", "textures/*"]
        },
        entry_points={
            "console_scripts": [
                "graphics3d-demo=graphics3d.cli:main",
                "graphics3d-viewer=graphics3d.viewer:main"
            ]
        }
    )
    
    print("Package Information:")
    print(f"  Name: {package_info.name}")
    print(f"  Version: {package_info.version}")
    print(f"  Description: {package_info.description}")
    print(f"  Author: {package_info.author}")
    print(f"  License: {package_info.license}")
    print(f"  Python Requires: {package_info.python_requires}")
    print(f"  Dependencies: {len(package_info.install_requires)} packages")
    print(f"  Extras: {list(package_info.extras_require.keys())}")
    print()

def demonstrate_package_structure():
    """Demonstrate package directory structure"""
    print("=== Package Structure Demo ===\n")
    
    # Create package structure
    structure = PackageStructure(
        root_dir="graphics3d_package",
        package_dir="graphics3d_package/src/graphics3d",
        docs_dir="graphics3d_package/docs",
        tests_dir="graphics3d_package/tests",
        examples_dir="graphics3d_package/examples",
        scripts_dir="graphics3d_package/scripts"
    )
    
    print("Package Structure:")
    print(f"  Root: {structure.root_dir}")
    print(f"  Package: {structure.package_dir}")
    print(f"  Docs: {structure.docs_dir}")
    print(f"  Tests: {structure.tests_dir}")
    print(f"  Examples: {structure.examples_dir}")
    print(f"  Scripts: {structure.scripts_dir}")
    print()
    
    # Create structure (commented out to avoid creating actual directories)
    # structure.create_structure()
    print("(Directory structure creation would happen here)")
    print()

def demonstrate_setup_generation():
    """Demonstrate setup.py generation"""
    print("=== Setup.py Generation Demo ===\n")
    
    # Create package info
    package_info = PackageInfo(
        name="demo3d",
        version="0.1.0",
        description="Demo 3D graphics package",
        author="Demo Author",
        author_email="demo@example.com",
        url="https://github.com/demo/demo3d",
        license="MIT",
        classifiers=["Development Status :: 3 - Alpha"],
        keywords=["demo", "3d"],
        python_requires=">=3.7",
        install_requires=["numpy"],
        extras_require={},
        packages=["demo3d"],
        package_data={},
        entry_points={}
    )
    
    # Generate setup.py
    generator = SetupGenerator(package_info)
    setup_content = generator.generate_setup_py("demo_setup.py")
    
    print("Generated setup.py content preview:")
    print(setup_content[:500] + "...")
    print()

def demonstrate_manifest_generation():
    """Demonstrate MANIFEST.in generation"""
    print("=== MANIFEST.in Generation Demo ===\n")
    
    # Generate MANIFEST.in
    generator = ManifestGenerator()
    manifest_content = generator.generate_manifest_in("demo_manifest.in")
    
    print("Generated MANIFEST.in content:")
    print(manifest_content)
    print()

def demonstrate_package_building():
    """Demonstrate complete package building"""
    print("=== Package Building Demo ===\n")
    
    # Create package info
    package_info = PackageInfo(
        name="simple3d",
        version="0.1.0",
        description="Simple 3D graphics package",
        author="Simple Author",
        author_email="simple@example.com",
        url="https://github.com/simple/simple3d",
        license="MIT",
        classifiers=["Development Status :: 3 - Alpha"],
        keywords=["simple", "3d"],
        python_requires=">=3.7",
        install_requires=["numpy"],
        extras_require={},
        packages=["simple3d"],
        package_data={},
        entry_points={}
    )
    
    # Create package structure
    structure = PackageStructure(
        root_dir="simple3d_package",
        package_dir="simple3d_package/src/simple3d",
        docs_dir="simple3d_package/docs",
        tests_dir="simple3d_package/tests",
        examples_dir="simple3d_package/examples",
        scripts_dir="simple3d_package/scripts"
    )
    
    # Build package
    builder = PackageBuilder(package_info, structure)
    
    print("Building package...")
    print("(This would create the complete package structure)")
    print("1. Directory structure")
    print("2. setup.py")
    print("3. MANIFEST.in")
    print("4. README.md")
    print("5. __init__.py files")
    print("6. requirements.txt")
    print()

def demonstrate_package_installation():
    """Demonstrate package installation"""
    print("=== Package Installation Demo ===\n")
    
    # Create installer
    installer = PackageInstaller("simple3d_package")
    
    print("Package Installation Options:")
    print()
    
    # Development installation
    installer.install_development()
    print()
    
    # Production installation
    installer.install_production()
    print()
    
    # Build distribution
    installer.build_distribution()
    print()

def demonstrate_package_validation():
    """Demonstrate package validation"""
    print("=== Package Validation Demo ===\n")
    
    # Create validator
    validator = PackageValidator("simple3d_package")
    
    print("Package Validation:")
    print()
    
    # Validate structure
    print("1. Structure Validation:")
    structure_results = validator.validate_structure()
    for item, exists in structure_results.items():
        status = "✓" if exists else "✗"
        print(f"   {status} {item}")
    print()
    
    # Validate setup.py
    print("2. Setup.py Validation:")
    setup_results = validator.validate_setup_py()
    if "error" in setup_results:
        print(f"   ✗ {setup_results['error']}")
    else:
        for item, valid in setup_results.items():
            status = "✓" if valid else "✗"
            print(f"   {status} {item}")
    print()
    
    # Validate MANIFEST.in
    print("3. MANIFEST.in Validation:")
    manifest_results = validator.validate_manifest()
    if "error" in manifest_results:
        print(f"   ✗ {manifest_results['error']}")
    else:
        for item, valid in manifest_results.items():
            status = "✓" if valid else "✗"
            print(f"   {status} {item}")
    print()

def demonstrate_complete_workflow():
    """Demonstrate complete package distribution workflow"""
    print("=== Complete Package Distribution Workflow ===\n")
    
    print("Complete workflow for creating a distributable package:")
    print()
    print("1. Define package information")
    print("   - Name, version, description")
    print("   - Dependencies and requirements")
    print("   - Entry points and classifiers")
    print()
    print("2. Create package structure")
    print("   - Source code directory")
    print("   - Documentation directory")
    print("   - Tests directory")
    print("   - Examples directory")
    print()
    print("3. Generate distribution files")
    print("   - setup.py (package configuration)")
    print("   - MANIFEST.in (file inclusion rules)")
    print("   - README.md (documentation)")
    print("   - requirements.txt (dependencies)")
    print()
    print("4. Build and distribute")
    print("   - Development installation: pip install -e .")
    print("   - Production installation: pip install .")
    print("   - Distribution build: python setup.py sdist bdist_wheel")
    print("   - Upload to PyPI: twine upload dist/*")
    print()
    print("5. Validate package")
    print("   - Check directory structure")
    print("   - Validate configuration files")
    print("   - Test installation")
    print("   - Verify functionality")
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate package distribution"""
    print("=== Package Distribution Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_package_configuration()
    demonstrate_package_structure()
    demonstrate_setup_generation()
    demonstrate_manifest_generation()
    demonstrate_package_building()
    demonstrate_package_installation()
    demonstrate_package_validation()
    demonstrate_complete_workflow()
    
    print("="*60)
    print("Package Distribution demo completed successfully!")
    print("\nKey features:")
    print("✓ Package configuration: Metadata and dependencies")
    print("✓ Directory structure: Organized package layout")
    print("✓ Setup generation: Automated setup.py creation")
    print("✓ Manifest generation: File inclusion rules")
    print("✓ Package building: Complete package creation")
    print("✓ Installation tools: Development and production installs")
    print("✓ Package validation: Structure and configuration checks")
    print("✓ Distribution workflow: End-to-end package distribution")

if __name__ == "__main__":
    main()
