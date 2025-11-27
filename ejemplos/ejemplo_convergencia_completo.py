"""
Ejemplo completo: Análisis de convergencia para todos los métodos de EDO
Compara los factores de convergencia de Euler, Heun, Punto Medio y RK4
"""

import matplotlib.pyplot as plt
from metodos import (
    calcular_factor_convergencia_euler,
    calcular_factor_convergencia_heun,
    calcular_factor_convergencia_punto_medio,
    calcular_factor_convergencia_rk4
)

# Problema: dy/dx = y, y(0) = 1, solución exacta: y = e^x
f = lambda x, y: y

print("=" * 70)
print("ANÁLISIS DE CONVERGENCIA - MÉTODOS DE RESOLUCIÓN DE EDOs")
print("=" * 70)
print("\nProblema: dy/dx = y, y(0) = 1")
print("Solución exacta: y(x) = e^x")
print("Intervalo: [0, 1], n = 20 pasos\n")

# Calcular factores de convergencia para cada método
print("Calculando factores de convergencia...\n")

X1, factores1, promedio1 = calcular_factor_convergencia_euler(f, 0, 1, 1, n=20)
print()
X2, factores2, promedio2 = calcular_factor_convergencia_heun(f, 0, 1, 1, n=20)
print()
X3, factores3, promedio3 = calcular_factor_convergencia_punto_medio(f, 0, 1, 1, n=20)
print()
X4, factores4, promedio4 = calcular_factor_convergencia_rk4(f, 0, 1, 1, n=20)

# Resumen de resultados
print(f"\n{'='*70}")
print("RESUMEN DE RESULTADOS:")
print(f"{'='*70}")
print(f"{'Método':<20} {'Orden teórico':<15} {'Factor promedio':<18} {'Confirmación'}")
print(f"{'-'*70}")
print(f"{'Euler':<20} {'1':<15} {promedio1:<18.4f} {'✅' if abs(promedio1 - 1.0) < 0.3 else '⚠️'}")
print(f"{'Heun':<20} {'2':<15} {promedio2:<18.4f} {'✅' if abs(promedio2 - 2.0) < 0.3 else '⚠️'}")
print(f"{'Punto Medio':<20} {'2':<15} {promedio3:<18.4f} {'✅' if abs(promedio3 - 2.0) < 0.3 else '⚠️'}")
print(f"{'Runge-Kutta 4':<20} {'4':<15} {promedio4:<18.4f} {'✅' if abs(promedio4 - 4.0) < 0.5 else '⚠️'}")
print(f"{'='*70}")

# Crear visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de Convergencia - Métodos de Resolución de EDOs\ndy/dx = y, y(0) = 1', 
             fontsize=16, fontweight='bold')

# Euler
axes[0, 0].plot(X1, factores1, 'o-', linewidth=2, markersize=5, label='Factor calculado')
axes[0, 0].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Orden teórico = 1')
axes[0, 0].axhline(y=promedio1, color='g', linestyle=':', linewidth=2, label=f'Promedio = {promedio1:.3f}')
axes[0, 0].set_xlabel('x', fontsize=11)
axes[0, 0].set_ylabel('Factor de convergencia', fontsize=11)
axes[0, 0].set_title('Método de Euler (orden 1)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Heun
axes[0, 1].plot(X2, factores2, 'o-', linewidth=2, markersize=5, label='Factor calculado', color='orange')
axes[0, 1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Orden teórico = 2')
axes[0, 1].axhline(y=promedio2, color='g', linestyle=':', linewidth=2, label=f'Promedio = {promedio2:.3f}')
axes[0, 1].set_xlabel('x', fontsize=11)
axes[0, 1].set_ylabel('Factor de convergencia', fontsize=11)
axes[0, 1].set_title('Método de Heun (orden 2)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Punto Medio
axes[1, 0].plot(X3, factores3, 'o-', linewidth=2, markersize=5, label='Factor calculado', color='green')
axes[1, 0].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Orden teórico = 2')
axes[1, 0].axhline(y=promedio3, color='g', linestyle=':', linewidth=2, label=f'Promedio = {promedio3:.3f}')
axes[1, 0].set_xlabel('x', fontsize=11)
axes[1, 0].set_ylabel('Factor de convergencia', fontsize=11)
axes[1, 0].set_title('Método del Punto Medio (orden 2)', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Runge-Kutta 4
axes[1, 1].plot(X4, factores4, 'o-', linewidth=2, markersize=5, label='Factor calculado', color='red')
axes[1, 1].axhline(y=4.0, color='r', linestyle='--', linewidth=2, label='Orden teórico = 4')
axes[1, 1].axhline(y=promedio4, color='g', linestyle=':', linewidth=2, label=f'Promedio = {promedio4:.3f}')
axes[1, 1].set_xlabel('x', fontsize=11)
axes[1, 1].set_ylabel('Factor de convergencia', fontsize=11)
axes[1, 1].set_title('Método de Runge-Kutta 4 (orden 4)', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

print(f"\n{'='*70}")
print("Mostrando gráfico comparativo...")
print(f"{'='*70}")
plt.show()
