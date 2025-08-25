# Chapter 9: Functional Programming

## Overview

This chapter introduces functional programming concepts in Python, applied specifically to 3D graphics and mathematical operations. Functional programming emphasizes pure functions, immutability, and higher-order functions, which are particularly valuable in 3D graphics for creating predictable, testable, and composable code.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand and implement pure functions with no side effects
- Use immutable data structures for 3D graphics operations
- Apply higher-order functions to process 3D data
- Compose functions to create complex transformations
- Implement functional programming patterns in 3D graphics applications
- Use lambda expressions and function factories
- Apply functional concepts to mathematical and geometric operations

## Key Concepts

### 1. Pure Functions
- **Definition**: Functions that always return the same output for the same input and have no side effects
- **Benefits**: Predictable, testable, and cacheable
- **Applications**: Mathematical calculations, geometric transformations, color operations

### 2. Immutability
- **Definition**: Data structures that cannot be modified after creation
- **Benefits**: Thread safety, predictable behavior, easier debugging
- **Applications**: 3D vectors, matrices, colors, scene objects

### 3. Higher-Order Functions
- **Definition**: Functions that take other functions as arguments or return functions
- **Benefits**: Code reusability, abstraction, composition
- **Applications**: Transformation pipelines, filtering, mapping operations

### 4. Function Composition
- **Definition**: Combining multiple functions to create new functions
- **Benefits**: Modular design, complex operations from simple parts
- **Applications**: 3D transformation chains, rendering pipelines

### 5. Lambda Expressions
- **Definition**: Anonymous functions for simple operations
- **Benefits**: Concise syntax, inline function creation
- **Applications**: Simple transformations, predicates, callbacks

## Files in This Chapter

### 1. `functional_basics.py`
**Focus**: Core functional programming concepts with 3D applications

**Key Topics**:
- Pure functions for 3D mathematics
- Immutable Vector3D class
- Function composition
- Higher-order functions (map, filter, reduce)
- Lambda expressions

**Learning Outcomes**:
- Understand pure function principles
- Implement immutable 3D data structures
- Use functional composition for transformations
- Apply functional patterns to 3D operations

### 2. `higher_order_functions.py`
**Focus**: Advanced higher-order function techniques

**Key Topics**:
- Functions that take functions as arguments
- Functions that return functions
- Currying and partial application
- Function factories
- Decorators as higher-order functions
- Memoization

**Learning Outcomes**:
- Create function factories for 3D transformations
- Implement curried functions for mathematical operations
- Use decorators for validation and timing
- Apply memoization for performance optimization

### 3. `3d_transformations.py`
**Focus**: Functional programming applied to 3D transformations

**Key Topics**:
- Pure transformation functions
- Immutable matrix operations
- Transformation pipelines
- Camera and projection matrices
- Functional scene processing

**Learning Outcomes**:
- Implement pure 3D transformation functions
- Create functional transformation pipelines
- Apply functional concepts to camera operations
- Process 3D scenes using functional patterns

### 4. `pure_functions.py`
**Focus**: Pure functions for 3D graphics and mathematics

**Key Topics**:
- Mathematical pure functions
- Geometric calculations
- Physics simulations
- Color operations
- Function purity testing

**Learning Outcomes**:
- Implement pure mathematical functions
- Create pure geometric operations
- Apply pure functions to physics calculations
- Test function purity

## Practical Applications

### 1. 3D Graphics Processing
- **Scene transformation pipelines**
- **Camera and projection operations**
- **Geometric calculations**
- **Color and lighting operations**

### 2. Mathematical Operations
- **Vector and matrix operations**
- **Geometric transformations**
- **Interpolation and blending**
- **Physics simulations**

### 3. Data Processing
- **Point cloud processing**
- **Mesh operations**
- **Animation systems**
- **Rendering pipelines**

## Code Examples

### Pure Function Example
```python
def calculate_distance(v1: Vector3D, v2: Vector3D) -> float:
    """Pure function: calculates distance between two 3D points."""
    diff = v1 - v2
    return diff.magnitude()
```

### Higher-Order Function Example
```python
def create_translation(offset: Vector3D) -> Callable[[Vector3D], Vector3D]:
    """Function factory: creates a translation function."""
    def translate(point: Vector3D) -> Vector3D:
        return point + offset
    return translate
```

### Function Composition Example
```python
def compose(*functions: Callable) -> Callable:
    """Composes multiple functions."""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner
```

## Best Practices

### 1. Function Design
- **Keep functions pure** - no side effects
- **Use descriptive names** - clear intent
- **Limit function size** - single responsibility
- **Use type hints** - improve clarity

### 2. Data Structures
- **Prefer immutable objects** - use frozen dataclasses
- **Use value objects** - Vector3D, Color, Matrix4x4
- **Avoid mutable state** - create new objects instead

### 3. Composition
- **Compose simple functions** - build complexity from simplicity
- **Use pipelines** - chain transformations
- **Separate concerns** - transformation vs. calculation

### 4. Performance
- **Use memoization** - cache expensive calculations
- **Lazy evaluation** - defer computation until needed
- **Optimize hot paths** - profile and optimize critical sections

## Common Patterns

### 1. Transformation Pipeline
```python
pipeline = (TransformationPipeline()
           .translate(offset)
           .scale(factor)
           .rotate(angle))
result = pipeline.apply_to_points(points)
```

### 2. Function Factory
```python
def create_filter(predicate: Callable) -> Callable:
    return lambda items: [item for item in items if predicate(item)]
```

### 3. Curried Function
```python
def distance_from_point(reference: Vector3D) -> Callable[[Vector3D], float]:
    return lambda point: calculate_distance(point, reference)
```

## Exercises and Projects

### 1. Basic Exercises
- Implement pure functions for basic 3D operations
- Create function factories for common transformations
- Build simple transformation pipelines

### 2. Intermediate Projects
- Create a functional 3D scene processor
- Implement a pure physics simulation system
- Build a functional animation framework

### 3. Advanced Challenges
- Design a functional rendering pipeline
- Create a pure procedural generation system
- Implement a functional game engine architecture

## Related Chapters

- **Chapter 10**: Iterators and Generators (functional data processing)
- **Chapter 11**: Decorators and Context Managers (higher-order functions)
- **Chapter 16**: 3D Math Foundations (mathematical applications)
- **Chapter 18**: Transformations (practical applications)

## Summary

Functional programming provides powerful tools for creating clean, predictable, and composable 3D graphics code. By emphasizing pure functions, immutability, and higher-order functions, you can build robust systems that are easier to test, debug, and maintain. These concepts are particularly valuable in 3D graphics where mathematical operations and transformations are fundamental to the domain.

The examples in this chapter demonstrate how functional programming principles can be applied to real-world 3D graphics problems, from basic vector operations to complex transformation pipelines. Understanding these concepts will help you write better, more maintainable 3D graphics code.
