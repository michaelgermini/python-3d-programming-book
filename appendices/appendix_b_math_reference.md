# Appendix B: Math Reference

## Overview
This appendix provides a comprehensive reference for mathematical concepts, formulas, and algorithms used in 3D graphics programming. It serves as a quick reference for vector mathematics, matrix operations, geometric calculations, and other essential mathematical tools.

## Vector Mathematics

### Vector Operations
A vector in 3D space is represented as **v** = (x, y, z)

#### Basic Operations
- **Addition**: **a** + **b** = (aₓ + bₓ, aᵧ + bᵧ, aᵤ + bᵤ)
- **Subtraction**: **a** - **b** = (aₓ - bₓ, aᵧ - bᵧ, aᵤ - bᵤ)
- **Scalar Multiplication**: s**a** = (saₓ, saᵧ, saᵤ)
- **Dot Product**: **a** · **b** = aₓbₓ + aᵧbᵧ + aᵤbᵤ
- **Cross Product**: **a** × **b** = (aᵧbᵤ - aᵤbᵧ, aᵤbₓ - aₓbᵤ, aₓbᵧ - aᵧbₓ)

#### Vector Properties
- **Magnitude**: |**v**| = √(x² + y² + z²)
- **Normalization**: **v̂** = **v** / |**v**|
- **Distance**: d(**a**, **b**) = |**a** - **b**|
- **Angle**: cos θ = (**a** · **b**) / (|**a**| |**b**|)

#### Vector Identities
- **a** · (**b** × **c**) = **b** · (**c** × **a**) = **c** · (**a** × **b**) (Scalar Triple Product)
- **a** × (**b** × **c**) = **b**(**a** · **c**) - **c**(**a** · **b**) (Vector Triple Product)
- |**a** × **b**| = |**a**| |**b**| sin θ

### Vector Types
- **Position Vector**: Points to a location in space
- **Direction Vector**: Represents direction (usually normalized)
- **Normal Vector**: Perpendicular to a surface
- **Tangent Vector**: Tangent to a curve or surface

## Matrix Mathematics

### Matrix Operations
A 4×4 matrix M is represented as:
```
M = [m₀₀  m₀₁  m₀₂  m₀₃]
    [m₁₀  m₁₁  m₁₂  m₁₃]
    [m₂₀  m₂₁  m₂₂  m₂₃]
    [m₃₀  m₃₁  m₃₂  m₃₃]
```

#### Basic Operations
- **Addition**: (A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ
- **Multiplication**: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
- **Transpose**: (Aᵀ)ᵢⱼ = Aⱼᵢ
- **Determinant**: |M| = Σᵢⱼₖₗ εᵢⱼₖₗ m₀ᵢ m₁ⱼ m₂ₖ m₃ₗ

#### Special Matrices
- **Identity Matrix**: I = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
- **Zero Matrix**: 0 = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]

### Transformation Matrices

#### Translation Matrix
```
T(tₓ, tᵧ, tᵤ) = [1  0  0  tₓ]
                [0  1  0  tᵧ]
                [0  0  1  tᵤ]
                [0  0  0   1]
```

#### Scaling Matrix
```
S(sₓ, sᵧ, sᵤ) = [sₓ  0   0   0]
                [0   sᵧ  0   0]
                [0   0   sᵤ  0]
                [0   0   0   1]
```

#### Rotation Matrices
**Around X-axis**:
```
Rₓ(θ) = [1    0       0       0]
        [0   cos θ   -sin θ    0]
        [0   sin θ    cos θ    0]
        [0    0       0       1]
```

**Around Y-axis**:
```
Rᵧ(θ) = [ cos θ    0   sin θ    0]
        [   0      1     0       0]
        [-sin θ    0   cos θ     0]
        [   0      0     0       1]
```

**Around Z-axis**:
```
Rᵤ(θ) = [cos θ   -sin θ    0    0]
        [sin θ    cos θ     0    0]
        [  0       0        1    0]
        [  0       0        0    1]
```

#### Look-At Matrix
```
LookAt(eye, target, up) = [rightₓ  upₓ  -forwardₓ  0]
                          [rightᵧ  upᵧ  -forwardᵧ  0]
                          [rightᵤ  upᵤ  -forwardᵤ  0]
                          [-eye·right  -eye·up  eye·forward  1]
```

#### Perspective Projection Matrix
```
Perspective(fov, aspect, near, far) = [f/aspect  0    0                   0]
                                      [0         f    0                   0]
                                      [0         0   (far+near)/(near-far)  (2*far*near)/(near-far)]
                                      [0         0   -1                   0]
where f = 1/tan(fov/2)
```

#### Orthographic Projection Matrix
```
Ortho(left, right, bottom, top, near, far) = [2/(right-left)  0               0               -(right+left)/(right-left)]
                                              [0               2/(top-bottom)  0               -(top+bottom)/(top-bottom)]
                                              [0               0               -2/(far-near)   -(far+near)/(far-near)]
                                              [0               0               0               1]
```

## Quaternions

### Quaternion Representation
A quaternion q = (w, x, y, z) = w + xi + yj + zk

#### Basic Operations
- **Addition**: q₁ + q₂ = (w₁ + w₂, x₁ + x₂, y₁ + y₂, z₁ + z₂)
- **Multiplication**: q₁q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂,
                              w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂,
                              w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂,
                              w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)
- **Conjugate**: q* = (w, -x, -y, -z)
- **Norm**: |q| = √(w² + x² + y² + z²)
- **Inverse**: q⁻¹ = q* / |q|²

#### Rotation Quaternion
For rotation by angle θ around axis **n**:
q = (cos(θ/2), nₓ sin(θ/2), nᵧ sin(θ/2), nᵤ sin(θ/2))

#### Quaternion to Matrix Conversion
```
M = [1-2y²-2z²    2xy-2wz    2xz+2wy    0]
    [2xy+2wz    1-2x²-2z²   2yz-2wx    0]
    [2xz-2wy     2yz+2wx   1-2x²-2y²   0]
    [    0          0         0        1]
```

#### Spherical Linear Interpolation (SLERP)
```
slerp(q₁, q₂, t) = q₁(q₁⁻¹q₂)ᵗ
```

## Geometric Calculations

### Line and Plane Equations

#### Line Equation
**Parametric**: **p**(t) = **p₀** + t**d**
**Symmetric**: (x - x₀)/dₓ = (y - y₀)/dᵧ = (z - z₀)/dᵤ

#### Plane Equation
**Point-Normal**: **n** · (**p** - **p₀**) = 0
**General**: ax + by + cz + d = 0

#### Distance Calculations
- **Point to Line**: d = |(**p** - **p₀**) × **d**| / |**d**|
- **Point to Plane**: d = |**n** · (**p** - **p₀**)| / |**n**|
- **Line to Line**: d = |(**p₂** - **p₁**) · (**d₁** × **d₂**)| / |**d₁** × **d₂**|

### Intersection Tests

#### Ray-Sphere Intersection
```
t = -b ± √(b² - 4ac) / 2a
where a = |d|², b = 2d·(o-c), c = |o-c|² - r²
```

#### Ray-Triangle Intersection (Möller-Trumbore)
```
t = (q·e₂) / (d·e₁)
u = (p·e₁) / (d·e₁)
v = (q·d) / (d·e₁)
where e₁ = v₁ - v₀, e₂ = v₂ - v₀, p = d × e₂, q = (o - v₀) × e₁
```

#### AABB Intersection
```
intersect = (minₓ ≤ maxₓ) && (minᵧ ≤ maxᵧ) && (minᵤ ≤ maxᵤ)
where minᵢ = max(a_minᵢ, b_minᵢ), maxᵢ = min(a_maxᵢ, b_maxᵢ)
```

### Bounding Volumes

#### Axis-Aligned Bounding Box (AABB)
```
center = (min + max) / 2
size = max - min
radius = |size| / 2
```

#### Oriented Bounding Box (OBB)
```
center = (min + max) / 2
half_extents = (max - min) / 2
```

#### Bounding Sphere
```
center = average of all points
radius = max distance from center to any point
```

## Interpolation and Curves

### Linear Interpolation
```
lerp(a, b, t) = a + t(b - a)
```

### Bilinear Interpolation
```
bilinear(p₀₀, p₁₀, p₀₁, p₁₁, s, t) = (1-s)(1-t)p₀₀ + s(1-t)p₁₀ + (1-s)tp₀₁ + stp₁₁
```

### Bézier Curves

#### Quadratic Bézier
```
B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
```

#### Cubic Bézier
```
B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
```

#### General Bézier
```
B(t) = Σᵢ₌₀ⁿ C(n,i) (1-t)ⁿ⁻ⁱ tⁱ Pᵢ
where C(n,i) = n! / (i!(n-i)!)
```

### B-Splines
```
B(t) = Σᵢ₌₀ⁿ Nᵢ,ₖ(t) Pᵢ
where Nᵢ,ₖ(t) = (t-tᵢ)/(tᵢ₊ₖ₋₁-tᵢ) Nᵢ,ₖ₋₁(t) + (tᵢ₊ₖ-t)/(tᵢ₊ₖ-tᵢ₊₁) Nᵢ₊₁,ₖ₋₁(t)
```

## Numerical Methods

### Root Finding

#### Newton's Method
```
xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ)
```

#### Bisection Method
```
if f(a)f(b) < 0, then root is in [a, b]
c = (a + b) / 2
if f(a)f(c) < 0, then root is in [a, c], else in [c, b]
```

### Integration

#### Trapezoidal Rule
```
∫ᵇₐ f(x)dx ≈ h/2 [f(a) + 2Σᵢ₌₁ⁿ⁻¹ f(xᵢ) + f(b)]
where h = (b-a)/n, xᵢ = a + ih
```

#### Simpson's Rule
```
∫ᵇₐ f(x)dx ≈ h/3 [f(a) + 4Σᵢ₌₁,₃,₅,...ⁿ⁻¹ f(xᵢ) + 2Σᵢ₌₂,₄,₆,...ⁿ⁻² f(xᵢ) + f(b)]
```

### Optimization

#### Gradient Descent
```
xₙ₊₁ = xₙ - α∇f(xₙ)
```

#### Conjugate Gradient
```
xₙ₊₁ = xₙ + αₙpₙ
pₙ₊₁ = -∇f(xₙ₊₁) + βₙpₙ
where βₙ = ∇f(xₙ₊₁)² / ∇f(xₙ)²
```

## Probability and Statistics

### Probability Distributions

#### Normal Distribution
```
f(x) = (1/σ√(2π)) e^(-(x-μ)²/(2σ²))
```

#### Uniform Distribution
```
f(x) = 1/(b-a) for x ∈ [a, b]
```

### Random Number Generation

#### Linear Congruential Generator
```
xₙ₊₁ = (axₙ + c) mod m
```

#### Box-Muller Transform
```
Z₀ = √(-2 ln U₁) cos(2πU₂)
Z₁ = √(-2 ln U₁) sin(2πU₂)
where U₁, U₂ are uniform random numbers
```

## Computational Geometry

### Convex Hull (Graham Scan)
1. Find point with lowest y-coordinate
2. Sort points by polar angle
3. Build hull using stack

### Triangulation (Delaunay)
- No point is inside the circumcircle of any triangle
- Maximizes minimum angle

### Voronoi Diagram
- Each cell contains points closest to a given site
- Dual to Delaunay triangulation

## Optimization Techniques

### Memory Optimization
- **Structure of Arrays (SoA)**: Store components separately
- **Array of Structures (AoS)**: Store complete objects together
- **Cache-friendly access patterns**: Sequential memory access

### Performance Optimization
- **SIMD operations**: Vectorized computations
- **Branch prediction**: Minimize conditional branches
- **Loop unrolling**: Reduce loop overhead
- **Function inlining**: Reduce function call overhead

## Mathematical Constants

### Common Constants
- **π** ≈ 3.14159265359
- **e** ≈ 2.71828182846
- **φ** ≈ 1.61803398875 (Golden ratio)
- **√2** ≈ 1.41421356237
- **√3** ≈ 1.73205080757

### Conversion Factors
- **Degrees to Radians**: θ_rad = θ_deg × π/180
- **Radians to Degrees**: θ_deg = θ_rad × 180/π

## Error Analysis

### Floating Point Errors
- **Machine Epsilon**: ε = 2⁻⁵² ≈ 2.22×10⁻¹⁶ (double precision)
- **Relative Error**: |x - x̃| / |x|
- **Absolute Error**: |x - x̃|

### Numerical Stability
- **Condition Number**: κ(A) = |A| |A⁻¹|
- **Well-conditioned**: κ(A) ≈ 1
- **Ill-conditioned**: κ(A) >> 1

## Conclusion
This mathematical reference provides the essential formulas and concepts needed for 3D graphics programming. Understanding these mathematical foundations is crucial for implementing efficient and accurate 3D graphics algorithms.

Remember to:
- Use appropriate precision for your application
- Consider numerical stability in algorithms
- Optimize mathematical operations for performance
- Validate results with known test cases
- Document mathematical assumptions and limitations
