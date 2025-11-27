"""
Ejemplo completo de uso de todos los métodos para sistemas de EDOs
"""

from metodos import (
    euler_sistema,
    heun_sistema,
    punto_medio_sistema,
    runge_kutta4_sistema
)
import math

print("="*70)
print("COMPARACIÓN DE MÉTODOS PARA SISTEMAS DE EDOs")
print("="*70)
print()

# Sistema: Oscilador armónico simple
# dy1/dx = y2
# dy2/dx = -y1
# Solución exacta: y1(x) = cos(x), y2(x) = -sin(x)

print("Sistema: dy1/dx = y2, dy2/dx = -y1")
print("Condiciones iniciales: y1(0) = 1, y2(0) = 0")
print("Solución exacta: y1(x) = cos(x), y2(x) = -sin(x)")
print()

# Definir las funciones del sistema
f1 = lambda x, Y: Y[1]   # dy1/dx = y2
f2 = lambda x, Y: -Y[0]  # dy2/dx = -y1

funciones = [f1, f2]
y0 = [1.0, 0.0]
xf = 2 * math.pi  # 2π
n = 20  # número de pasos

print("="*70)
print("1. MÉTODO DE EULER (Orden 1)")
print("="*70)
X_euler, Y_euler = euler_sistema(funciones, 0, y0, xf, n)

print("\n" + "="*70)
print("2. MÉTODO DE HEUN (Orden 2)")
print("="*70)
X_heun, Y_heun = heun_sistema(funciones, 0, y0, xf, n)

print("\n" + "="*70)
print("3. MÉTODO DEL PUNTO MEDIO (Orden 2)")
print("="*70)
X_pm, Y_pm = punto_medio_sistema(funciones, 0, y0, xf, n)

print("\n" + "="*70)
print("4. MÉTODO DE RUNGE-KUTTA 4 (Orden 4)")
print("="*70)
X_rk4, Y_rk4 = runge_kutta4_sistema(funciones, 0, y0, xf, n)

# Tabla comparativa
print("\n" + "="*70)
print("TABLA COMPARATIVA DE RESULTADOS")
print("="*70)
print()
print(f"{'x':>8} {'Exacto':>12} {'Euler':>12} {'Heun':>12} {'Pto.Medio':>12} {'RK4':>12}")
print("-" * 80)

for i in range(0, len(X_euler), 5):
    x = X_euler[i]
    y_exacto = math.cos(x)
    y_euler = Y_euler[i][0]
    y_heun = Y_heun[i][0]
    y_pm = Y_pm[i][0]
    y_rk4 = Y_rk4[i][0]
    
    print(f"{x:8.4f} {y_exacto:12.6f} {y_euler:12.6f} {y_heun:12.6f} {y_pm:12.6f} {y_rk4:12.6f}")

# Errores finales
print("\n" + "="*70)
print("ERRORES ABSOLUTOS EN x = 2π")
print("="*70)
print()

y_exacto_final = math.cos(xf)
error_euler = abs(Y_euler[-1][0] - y_exacto_final)
error_heun = abs(Y_heun[-1][0] - y_exacto_final)
error_pm = abs(Y_pm[-1][0] - y_exacto_final)
error_rk4 = abs(Y_rk4[-1][0] - y_exacto_final)

print(f"{'Método':>20} {'Aproximación':>15} {'Error Absoluto':>18}")
print("-" * 55)
print(f"{'Solución Exacta':>20} {y_exacto_final:15.10f} {0.0:18.10e}")
print(f"{'Euler':>20} {Y_euler[-1][0]:15.10f} {error_euler:18.10e}")
print(f"{'Heun':>20} {Y_heun[-1][0]:15.10f} {error_heun:18.10e}")
print(f"{'Punto Medio':>20} {Y_pm[-1][0]:15.10f} {error_pm:18.10e}")
print(f"{'Runge-Kutta 4':>20} {Y_rk4[-1][0]:15.10f} {error_rk4:18.10e}")

print("\n" + "="*70)
print("CONCLUSIONES")
print("="*70)
print()
print("1. RK4 es el método MÁS PRECISO (error más pequeño)")
print("2. Heun y Punto Medio tienen precisión similar (ambos orden 2)")
print("3. Euler es el MENOS PRECISO pero más simple")
print("4. RK4 requiere más cálculos por paso pero vale la pena por su precisión")
print()
print("RECOMENDACIÓN:")
print("- Use RK4 cuando necesite alta precisión")
print("- Use Heun o Punto Medio para balance precisión/velocidad")
print("- Use Euler solo para pruebas rápidas o sistemas simples")
