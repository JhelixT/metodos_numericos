"""
Guía 9 - Problema 2
EDO: dy/dt = y^(1/3), y(0) = 0 tiene dos soluciones:
    y₁(t) = [(2/3)t]^(3/2) (no trivial) y y₂(t) = 0 (trivial)

Objetivo (salida simplificada): usar Euler y mostrar qué solución
se reproduce según la condición inicial: y(0)=0 vs y(0)=10^(-16).
"""

import math
from metodos import euler


def f(t, y):
    """dy/dt = y^(1/3)"""
    if y < 0:
        return 0  # Evitar raíz de número negativo
    return y**(1/3)


def solucion1(t):
    """y₁(t) = [(2/3)t]^(3/2) = (2/3)^(3/2) * t^(3/2)"""
    return ((2/3) * t)**(3/2)


def solucion2(t):
    """y₂(t) = 0"""
    return 0

def resolver_con_euler():
    """Resolver con Euler usando diferentes condiciones iniciales"""
    
    print("\nGuía 9 - Problema 2: Euler en y' = y^(1/3), y(0)=0")
    print("Método: Euler, h=0.01, intervalo [0, 2]\n")
    
    t0, tf = 0, 2
    h = 0.01
    n = int((tf - t0) / h)
    
    # Caso 1: y(0) = 0 (exactamente cero)
    print(f"\nCaso 1: y(0) = 0 (condición inicial exacta)")
    print("-"*70)
    y0_caso1 = 0
    T1, Y1 = euler(f, t0, y0_caso1, tf, n, verbose=False)
    
    print(f"Resultado: y permanece en 0 para todo t")
    print(f"y(0.5) = {Y1[int(0.5/h)]:.10f}")
    print(f"y(1.0) = {Y1[int(1.0/h)]:.10f}")
    print(f"y(2.0) = {Y1[-1]:.10f}")
    print(f"\n→ Reproduce la solución y₂(t) = 0")
    
    # Caso 2: y(0) = 10^(-16) (perturbación pequeña)
    print(f"\nCaso 2: y(0) = 10^(-16) (perturbación pequeña)")
    print("-"*70)
    y0_caso2 = 1e-16
    T2, Y2 = euler(f, t0, y0_caso2, tf, n, verbose=False)
    
    print(f"{'t':<10} {'Euler':<20} {'y₁(t)=[(2/3)t]^(3/2)':<25} {'Diferencia':<15}")
    print("-"*70)
    
    tiempos_muestra = [0.5, 1.0, 1.5, 2.0]
    for t_muestra in tiempos_muestra:
        idx = int(t_muestra / h)
        y_euler = Y2[idx]
        y_exacta = solucion1(t_muestra)
        diff = abs(y_euler - y_exacta)
        print(f"{t_muestra:<10.1f} {y_euler:<20.10f} {y_exacta:<20.10f} {diff:<15.2e}")
    
    print(f"\n→ Reproduce la solución y₁(t) = [(2/3)t]^(3/2)")
    
    # Resumen breve
    print("\nResumen:")
    print("- y(0) = 0           → y₂(t) = 0")
    print("- y(0) = 10^(-16)    → y₁(t) = [(2/3)t]^(3/2)\n")


if __name__ == "__main__":
    # Salida simplificada: solo resolución con Euler y conclusión
    resolver_con_euler()
