"""
Ejemplo de uso de los métodos para EDOs de Orden Superior

Este archivo muestra cómo resolver ecuaciones diferenciales de orden m >= 2
usando los métodos implementados.
"""

from metodos import (
    euler_orden_superior,
    heun_orden_superior,
    punto_medio_orden_superior,
    runge_kutta4_orden_superior
)
import math

print("="*70)
print("TUTORIAL: Resolver EDOs de Orden Superior")
print("="*70)
print()

print("CONCEPTO CLAVE:")
print("-" * 70)
print("Una EDO de orden m se convierte en un sistema de m ecuaciones de")
print("primer orden, donde cada variable yi representa la derivada i-1:")
print()
print("  y1 = y      (la función original)")
print("  y2 = y'     (primera derivada)")
print("  y3 = y''    (segunda derivada)")
print("  y4 = y'''   (tercera derivada)")
print("  ...")
print("  ym = y^(m-1) (derivada m-1)")
print()

print("="*70)
print("EJEMPLO 1: EDO de Orden 2 (más común)")
print("="*70)
print()
print("Problema: Oscilador armónico simple")
print("  Ecuación: y'' = -y")
print("  Condiciones iniciales: y(0) = 1, y'(0) = 0")
print()
print("PASO 1: Definir la función f(x, y, y') que retorna y''")
print()

# La ecuación es y'' = -y
# Entonces f(x, y, y') = -y
f_ejemplo1 = lambda x, y, yp: -y

print("Código:")
print("  f = lambda x, y, yp: -y")
print()
print("  Nota: x no se usa en este caso, pero debe estar en la firma")
print("        y es la función (y1)")
print("        yp es la primera derivada (y2)")
print()

print("PASO 2: Definir condiciones iniciales [y(0), y'(0)]")
print()
print("Código:")
print("  y0 = [1.0, 0.0]  # y(0) = 1, y'(0) = 0")
print()

print("PASO 3: Resolver con el método deseado")
print()
print("Código:")
print("  X, Y = runge_kutta4_orden_superior(")
print("      f, x0=0, y0=[1.0, 0.0], xf=6.28, n=50, orden=2")
print("  )")
print()

# Resolver
X1, Y1 = runge_kutta4_orden_superior(
    f_ejemplo1, 
    x0=0, 
    y0=[1.0, 0.0], 
    xf=2*math.pi, 
    n=50, 
    orden=2,
    verbose=False
)

print("PASO 4: Interpretar los resultados")
print()
print("  Y[i][0] contiene y en el paso i")
print("  Y[i][1] contiene y' en el paso i")
print()
print("Resultados (primeros 5 puntos):")
print(f"{'i':>3} {'x':>10} {'y':>12} {'y\'':>12}")
print("-" * 40)
for i in range(5):
    print(f"{i:3d} {X1[i]:10.4f} {Y1[i][0]:12.6f} {Y1[i][1]:12.6f}")

print()
print("="*70)
print("EJEMPLO 2: EDO de Orden 3")
print("="*70)
print()
print("Problema: y''' = -2*y'' - 3*y' - y")
print("Condiciones iniciales: y(0) = 1, y'(0) = 0, y''(0) = 0")
print()
print("PASO 1: Definir f(x, y, y', y'') que retorna y'''")
print()

# y''' = -2*y'' - 3*y' - y
f_ejemplo2 = lambda x, y, yp, ypp: -2*ypp - 3*yp - y

print("Código:")
print("  f = lambda x, y, yp, ypp: -2*ypp - 3*yp - y")
print()
print("  Nota: ypp es la segunda derivada (y3)")
print()

print("PASO 2: Definir condiciones iniciales [y(0), y'(0), y''(0)]")
print()
print("Código:")
print("  y0 = [1.0, 0.0, 0.0]  # y(0) = 1, y'(0) = 0, y''(0) = 0")
print()

print("PASO 3: Resolver (especificar orden=3)")
print()

X2, Y2 = runge_kutta4_orden_superior(
    f_ejemplo2, 
    x0=0, 
    y0=[1.0, 0.0, 0.0], 
    xf=3, 
    n=30, 
    orden=3,
    verbose=False
)

print("Resultados:")
print(f"{'i':>3} {'x':>8} {'y':>12} {'y\'':>12} {'y\'\'':>12}")
print("-" * 52)
for i in range(0, len(X2), 6):
    print(f"{i:3d} {X2[i]:8.4f} {Y2[i][0]:12.6f} {Y2[i][1]:12.6f} {Y2[i][2]:12.6f}")

print()
print("="*70)
print("EJEMPLO 3: Comparar diferentes métodos")
print("="*70)
print()
print("Problema: y'' + y = 0, y(0) = 1, y'(0) = 0")
print("Solución exacta: y = cos(x)")
print()

f_comp = lambda x, y, yp: -y
y0_comp = [1.0, 0.0]

X_euler, Y_euler = euler_orden_superior(f_comp, 0, y0_comp, math.pi, 20, 2, verbose=False)
X_heun, Y_heun = heun_orden_superior(f_comp, 0, y0_comp, math.pi, 20, 2, verbose=False)
X_pm, Y_pm = punto_medio_orden_superior(f_comp, 0, y0_comp, math.pi, 20, 2, verbose=False)
X_rk4, Y_rk4 = runge_kutta4_orden_superior(f_comp, 0, y0_comp, math.pi, 20, 2, verbose=False)

print("Comparación de métodos en x = π:")
print()
print(f"{'Método':>20} {'y(π) aproximado':>18} {'Error':>15}")
print("-" * 55)

y_exacto = math.cos(math.pi)
print(f"{'Solución exacta':>20} {y_exacto:18.10f} {'--':>15}")
print(f"{'Euler':>20} {Y_euler[-1][0]:18.10f} {abs(Y_euler[-1][0] - y_exacto):15.6e}")
print(f"{'Heun':>20} {Y_heun[-1][0]:18.10f} {abs(Y_heun[-1][0] - y_exacto):15.6e}")
print(f"{'Punto Medio':>20} {Y_pm[-1][0]:18.10f} {abs(Y_pm[-1][0] - y_exacto):15.6e}")
print(f"{'Runge-Kutta 4':>20} {Y_rk4[-1][0]:18.10f} {abs(Y_rk4[-1][0] - y_exacto):15.6e}")

print()
print("="*70)
print("RESUMEN DE USO")
print("="*70)
print()
print("Para una EDO de orden m: y^(m) = f(x, y, y', ..., y^(m-1))")
print()
print("1. Definir la función:")
print("   f = lambda x, y, yp, ...: <expresión que retorna y^(m)>")
print()
print("2. Definir condiciones iniciales:")
print("   y0 = [y(x0), y'(x0), y''(x0), ..., y^(m-1)(x0)]")
print()
print("3. Llamar al método:")
print("   X, Y = metodo_orden_superior(f, x0, y0, xf, n, orden=m)")
print()
print("4. Acceder a resultados:")
print("   Y[i][0] = y en el paso i")
print("   Y[i][1] = y' en el paso i")
print("   Y[i][2] = y'' en el paso i")
print("   ...")
print("   Y[i][m-1] = y^(m-1) en el paso i")
print()
print("MÉTODOS DISPONIBLES:")
print("  - euler_orden_superior          (orden 1, menos preciso)")
print("  - heun_orden_superior           (orden 2)")
print("  - punto_medio_orden_superior    (orden 2)")
print("  - runge_kutta4_orden_superior   (orden 4, más preciso)")
