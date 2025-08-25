#!/usr/bin/env python3
"""
Chapter 8: File I/O and Data Processing
File System Operations Example

Demonstrates file system operations for directory management, file organization,
and file system utilities in 3D graphics applications.
"""

import os
import shutil
import glob
import fnmatch
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import math
import hashlib

# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "1.0.0"
__author__ = "3D Graphics File System Handler"
__description__ = "File system operations for 3D graphics applications"

# ============================================================================
# FILE SYSTEM MANAGER CLASS
# ============================================================================

class FileSystemManager:
    """Class for managing file system operations"""
    
    def __init__(self, base_directory: str = "file_system_data"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
    
    def create_directory(self, directory_path: str) -> bool:
        """Create a directory and its parents if needed"""
        try:
            full_path = self.base_directory / directory_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Successfully created directory: {full_path}")
            return True
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    
    def list_directory(self, directory_path: str = "", 
                      include_files: bool = True, 
                      include_dirs: bool = True,
                      pattern: str = "*") -> List[Dict[str, Any]]:
        """List contents of a directory"""
        try:
            full_path = self.base_directory / directory_path
            
            if not full_path.exists():
                print(f"Directory not found: {full_path}")
                return []
            
            contents = []
            
            for item in full_path.iterdir():
                # Apply pattern filter
                if not fnmatch.fnmatch(item.name, pattern):
                    continue
                
                # Check if we should include this item
                if item.is_file() and not include_files:
                    continue
                if item.is_dir() and not include_dirs:
                    continue
                
                stat = item.stat()
                item_info = {
                    "name": item.name,
                    "path": str(item.relative_to(self.base_directory)),
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": stat.st_size if item.is_file() else 0,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:]
                }
                contents.append(item_info)
            
            # Sort by name
            contents.sort(key=lambda x: x["name"])
            return contents
            
        except Exception as e:
            print(f"Error listing directory {directory_path}: {e}")
            return []
    
    def copy_file(self, source_path: str, destination_path: str) -> bool:
        """Copy a file from source to destination"""
        try:
            source_full = self.base_directory / source_path
            dest_full = self.base_directory / destination_path
            
            if not source_full.exists():
                print(f"Source file not found: {source_full}")
                return False
            
            # Create destination directory if needed
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_full, dest_full)
            print(f"Successfully copied: {source_full} -> {dest_full}")
            return True
            
        except Exception as e:
            print(f"Error copying file {source_path} -> {destination_path}: {e}")
            return False
    
    def move_file(self, source_path: str, destination_path: str) -> bool:
        """Move a file from source to destination"""
        try:
            source_full = self.base_directory / source_path
            dest_full = self.base_directory / destination_path
            
            if not source_full.exists():
                print(f"Source file not found: {source_full}")
                return False
            
            # Create destination directory if needed
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_full), str(dest_full))
            print(f"Successfully moved: {source_full} -> {dest_full}")
            return True
            
        except Exception as e:
            print(f"Error moving file {source_path} -> {destination_path}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            full_path = self.base_directory / file_path
            
            if not full_path.exists():
                print(f"File not found: {full_path}")
                return False
            
            full_path.unlink()
            print(f"Successfully deleted file: {full_path}")
            return True
            
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False
    
    def search_files(self, pattern: str, directory: str = "", 
                    recursive: bool = True) -> List[str]:
        """Search for files matching a pattern"""
        try:
            search_path = self.base_directory / directory
            
            if not search_path.exists():
                print(f"Search directory not found: {search_path}")
                return []
            
            if recursive:
                # Use glob for recursive search
                search_pattern = str(search_path / "**" / pattern)
                matches = glob.glob(search_pattern, recursive=True)
            else:
                # Use glob for non-recursive search
                search_pattern = str(search_path / pattern)
                matches = glob.glob(search_pattern)
            
            # Convert to relative paths
            relative_matches = []
            for match in matches:
                match_path = Path(match)
                if match_path.is_relative_to(self.base_directory):
                    relative_matches.append(str(match_path.relative_to(self.base_directory)))
            
            return sorted(relative_matches)
            
        except Exception as e:
            print(f"Error searching files with pattern {pattern}: {e}")
            return []

# ============================================================================
# 3D GRAPHICS FILE ORGANIZER
# ============================================================================

class GraphicsFileOrganizer:
    """Specialized file organizer for 3D graphics applications"""
    
    def __init__(self, base_directory: str = "graphics_files"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        self.fs_manager = FileSystemManager(str(self.base_directory))
        
        # Define standard directory structure
        self.directories = {
            "models": "models",
            "textures": "textures",
            "materials": "materials",
            "scenes": "scenes",
            "animations": "animations",
            "scripts": "scripts",
            "configs": "configs",
            "exports": "exports",
            "temp": "temp",
            "backups": "backups"
        }
        
        # Define file extensions for different types
        self.file_extensions = {
            "models": [".obj", ".fbx", ".dae", ".blend", ".3ds", ".max"],
            "textures": [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".hdr"],
            "materials": [".mat", ".mtl", ".material"],
            "scenes": [".scene", ".blend", ".max", ".ma", ".mb"],
            "animations": [".anim", ".bvh", ".fbx", ".dae"],
            "scripts": [".py", ".js", ".lua", ".mel"],
            "configs": [".json", ".xml", ".yaml", ".ini", ".cfg"],
            "exports": [".obj", ".fbx", ".dae", ".gltf", ".glb"]
        }
    
    def setup_project_structure(self) -> bool:
        """Set up the standard project directory structure"""
        try:
            print("Setting up 3D graphics project structure...")
            
            for category, dir_name in self.directories.items():
                self.fs_manager.create_directory(dir_name)
                print(f"  Created: {dir_name}/")
            
            # Create subdirectories for better organization
            subdirs = {
                "models": ["characters", "props", "environments"],
                "textures": ["diffuse", "normal", "specular"],
                "materials": ["characters", "props", "environments"],
                "scenes": ["levels", "cutscenes", "tests"],
                "animations": ["characters", "props", "camera"],
                "scripts": ["tools", "plugins", "utilities"],
                "configs": ["rendering", "physics", "audio"],
                "exports": ["models", "textures", "scenes"]
            }
            
            for parent_dir, subdir_list in subdirs.items():
                for subdir in subdir_list:
                    self.fs_manager.create_directory(f"{parent_dir}/{subdir}")
                    print(f"  Created: {parent_dir}/{subdir}/")
            
            print("Project structure setup completed!")
            return True
            
        except Exception as e:
            print(f"Error setting up project structure: {e}")
            return False
    
    def organize_file(self, file_path: str, category: str = None) -> bool:
        """Organize a file into the appropriate directory based on its type"""
        try:
            if not self.fs_manager.base_directory.joinpath(file_path).exists():
                print(f"File not found: {file_path}")
                return False
            
            # Determine category if not provided
            if category is None:
                category = self._determine_file_category(file_path)
            
            if category not in self.directories:
                print(f"Unknown category: {category}")
                return False
            
            # Create destination path
            dest_path = f"{self.directories[category]}/{Path(file_path).name}"
            
            # Move file to appropriate directory
            return self.fs_manager.move_file(file_path, dest_path)
            
        except Exception as e:
            print(f"Error organizing file {file_path}: {e}")
            return False
    
    def organize_directory(self, source_directory: str) -> Dict[str, int]:
        """Organize all files in a directory into appropriate categories"""
        try:
            results = {"organized": 0, "skipped": 0, "errors": 0}
            
            # List all files in the source directory
            files = self.fs_manager.list_directory(source_directory, include_dirs=False)
            
            for file_info in files:
                file_path = file_info["path"]
                
                # Skip if it's already in a category directory
                if any(file_path.startswith(cat + "/") for cat in self.directories.values()):
                    results["skipped"] += 1
                    continue
                
                # Organize the file
                if self.organize_file(file_path):
                    results["organized"] += 1
                else:
                    results["errors"] += 1
            
            return results
            
        except Exception as e:
            print(f"Error organizing directory {source_directory}: {e}")
            return {"organized": 0, "skipped": 0, "errors": 1}
    
    def find_assets_by_type(self, asset_type: str, recursive: bool = True) -> List[str]:
        """Find assets of a specific type"""
        try:
            if asset_type not in self.file_extensions:
                print(f"Unknown asset type: {asset_type}")
                return []
            
            extensions = self.file_extensions[asset_type]
            found_files = []
            
            for ext in extensions:
                pattern = f"*{ext}"
                files = self.fs_manager.search_files(pattern, recursive=recursive)
                found_files.extend(files)
            
            return sorted(list(set(found_files)))  # Remove duplicates
            
        except Exception as e:
            print(f"Error finding assets by type {asset_type}: {e}")
            return []
    
    def _determine_file_category(self, file_path: str) -> str:
        """Determine the category of a file based on its extension"""
        ext = Path(file_path).suffix.lower()
        
        for category, extensions in self.file_extensions.items():
            if ext in extensions:
                return category
        
        # Default category for unknown extensions
        return "temp"

# ============================================================================
# FILE SYSTEM UTILITIES
# ============================================================================

class FileSystemUtilities:
    """Utility functions for file system operations"""
    
    @staticmethod
    def calculate_directory_size(directory_path: Path) -> int:
        """Calculate the total size of a directory"""
        total_size = 0
        try:
            for item in directory_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            print(f"Error calculating directory size: {e}")
        return total_size
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def find_duplicate_files(directory_path: Path) -> Dict[str, List[str]]:
        """Find duplicate files based on content hash"""
        hash_dict = {}
        
        try:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    # Calculate file hash
                    hash_md5 = hashlib.md5()
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    file_hash = hash_md5.hexdigest()
                    
                    if file_hash not in hash_dict:
                        hash_dict[file_hash] = []
                    hash_dict[file_hash].append(str(file_path))
        except Exception as e:
            print(f"Error finding duplicate files: {e}")
        
        # Return only duplicates
        return {k: v for k, v in hash_dict.items() if len(v) > 1}

# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_basic_file_system_operations():
    """Demonstrate basic file system operations"""
    print("=== Basic File System Operations Demo ===\n")
    
    # Create file system manager
    fs_manager = FileSystemManager("fs_demo")
    
    # Create directories
    print("1. Creating directories...")
    fs_manager.create_directory("models")
    fs_manager.create_directory("textures")
    fs_manager.create_directory("scenes")
    
    # Create some sample files
    print("\n2. Creating sample files...")
    sample_files = [
        ("models/cube.obj", "This is a cube model file"),
        ("textures/diffuse.png", "This is a texture file"),
        ("scenes/level1.scene", "This is a scene file"),
        ("config.json", "This is a config file")
    ]
    
    for file_path, content in sample_files:
        full_path = fs_manager.base_directory / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"  Created: {file_path}")
    
    # List directory contents
    print("\n3. Listing directory contents...")
    contents = fs_manager.list_directory()
    for item in contents:
        print(f"  {item['name']} ({'file' if item['is_file'] else 'dir'}) - {FileSystemUtilities.format_file_size(item['size'])}")
    
    # Search for files
    print("\n4. Searching for files...")
    obj_files = fs_manager.search_files("*.obj")
    print(f"  Found {len(obj_files)} .obj files: {obj_files}")
    
    # Copy and move files
    print("\n5. Copying and moving files...")
    fs_manager.copy_file("config.json", "backup/config_backup.json")
    fs_manager.move_file("config.json", "scenes/config.json")
    
    print()

def demonstrate_graphics_file_organization():
    """Demonstrate 3D graphics file organization"""
    print("=== 3D Graphics File Organization Demo ===\n")
    
    # Create graphics file organizer
    organizer = GraphicsFileOrganizer("graphics_org_demo")
    
    # Set up project structure
    print("1. Setting up project structure...")
    organizer.setup_project_structure()
    
    # Create sample assets
    print("\n2. Creating sample assets...")
    sample_assets = [
        ("temp/character_model.obj", "Character model data"),
        ("temp/texture_diffuse.png", "Diffuse texture data"),
        ("temp/material_metal.mat", "Metal material data"),
        ("temp/animation_walk.anim", "Walking animation data"),
        ("temp/script_tool.py", "Python script data"),
        ("temp/config_render.json", "Rendering config data")
    ]
    
    for asset_path, content in sample_assets:
        full_path = organizer.base_directory / asset_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"  Created: {asset_path}")
    
    # Organize files
    print("\n3. Organizing files...")
    results = organizer.organize_directory("temp")
    print(f"  Organized: {results['organized']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Errors: {results['errors']}")
    
    # Find assets by type
    print("\n4. Finding assets by type...")
    model_files = organizer.find_assets_by_type("models")
    texture_files = organizer.find_assets_by_type("textures")
    print(f"  Model files: {model_files}")
    print(f"  Texture files: {texture_files}")
    
    print()

def demonstrate_file_system_utilities():
    """Demonstrate file system utilities"""
    print("=== File System Utilities Demo ===\n")
    
    # Create a test directory structure
    test_dir = Path("utilities_demo")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample files
    print("1. Creating test directory structure...")
    test_files = [
        ("models/cube.obj", "Cube model data" * 1000),  # Large file
        ("textures/diffuse.png", "Texture data"),
        ("duplicate1.txt", "Same content"),
        ("duplicate2.txt", "Same content"),  # Duplicate
        ("unique.txt", "Different content")
    ]
    
    for file_path, content in test_files:
        full_path = test_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"  Created: {file_path}")
    
    # Calculate directory size
    print("\n2. Calculating directory size...")
    total_size = FileSystemUtilities.calculate_directory_size(test_dir)
    print(f"  Total size: {FileSystemUtilities.format_file_size(total_size)}")
    
    # Find duplicate files
    print("\n3. Finding duplicate files...")
    duplicates = FileSystemUtilities.find_duplicate_files(test_dir)
    if duplicates:
        print("  Found duplicate files:")
        for file_hash, file_list in duplicates.items():
            print(f"    Hash {file_hash[:8]}...:")
            for file_path in file_list:
                print(f"      {file_path}")
    else:
        print("  No duplicate files found")
    
    # Clean up
    shutil.rmtree(test_dir)
    print("\n4. Cleaned up test directory")
    
    print()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate file system operations"""
    print("=== File System Operations Demo ===\n")
    
    print(f"Library: {__description__}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    
    # Demonstrate all components
    demonstrate_basic_file_system_operations()
    demonstrate_graphics_file_organization()
    demonstrate_file_system_utilities()
    
    print("="*60)
    print("File System Operations demo completed successfully!")
    print("\nKey features:")
    print("✓ Basic file operations: Create, delete, copy, move, list")
    print("✓ Directory management: Create, delete, navigate")
    print("✓ File organization: Automatic categorization and sorting")
    print("✓ Search capabilities: Pattern matching and recursive search")
    print("✓ Asset management: 3D graphics specific organization")
    print("✓ File utilities: Size calculation, duplicates, tree view")

if __name__ == "__main__":
    main()
