"""
Ejemplo de uso del método de Euler para sistemas de EDOs
"""

from metodos import euler_sistema

# Ejemplo: Oscilador armónico
print("Sistema: dy1/dx = y2, dy2/dx = -y1")
print("Solución analítica: y1(x) = cos(x), y2(x) = -sin(x)")
print()

# El sistema es:
#   dy1/dx = y2
#   dy2/dx = -y1
# donde Y = [y1, y2] es un vector con ambas variables

# Primera ecuación: dy1/dx = y2
# lambda x, Y: Y[1] significa:
#   - Recibe dos parámetros: x y Y
#   - Retorna Y[1] (que es y2)
f1 = lambda x, Y: Y[1]

# Segunda ecuación: dy2/dx = -y1
# lambda x, Y: -Y[0] significa:
#   - Recibe dos parámetros: x y Y
#   - Retorna -Y[0] (que es -y1)
f2 = lambda x, Y: -Y[0]

# Resolver con Euler
X, Y = euler_sistema(
    funciones=[f1, f2],
    x0=0,
    y0=[1.0, 0.0],
    xf=6.28,
    n=50
)

print("\nComparación con solución analítica:")
print(f"{'x':>8} {'y1(euler)':>12} {'y1(exacta)':>12} {'Error':>12}")
print("-" * 48)

import math
for i in range(0, len(X), 10):
    x = X[i]
    y1_euler = Y[i][0]
    y1_exacta = math.cos(x)
    error = abs(y1_euler - y1_exacta)
    print(f"{x:8.4f} {y1_euler:12.6f} {y1_exacta:12.6f} {error:12.6f}")
