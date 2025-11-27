"""
Guía 9 - Problema 1.1
Resolver: dy/dt = t² - y con y(0) = 1, t ∈ [0, 3]
Solución exacta: y(t) = -e^(-t) + t² - 2t + 2

Método de Euler con h = 0.2 y h = 0.1
Calcular error global para cada caso
"""

import math
from metodos import euler


def f(t, y):
    """dy/dt = t² - y"""
    return t**2 - y


def solucion_exacta(t):
    """y(t) = -e^(-t) + t² - 2t + 2"""
    return -math.exp(-t) + t**2 - 2*t + 2


if __name__ == "__main__":
    # Condiciones iniciales
    t0, y0 = 0, 1
    tf = 3
    
    print("\n" + "="*70)
    print("  Problema 1.1: dy/dt = t² - y,  y(0) = 1")
    print("="*70)
    
    # Resolver con h = 0.2
    n1 = int((tf - t0) / 0.2)
    T1, Y1 = euler(f, t0, y0, tf, n1, verbose=False)
    
    # Calcular errores globales para h = 0.2
    errores1 = [abs(solucion_exacta(T1[i]) - Y1[i]) for i in range(len(T1))]
    error_global1 = max(errores1)
    
    # Resolver con h = 0.1
    n2 = int((tf - t0) / 0.1)
    T2, Y2 = euler(f, t0, y0, tf, n2, verbose=False)
    
    # Calcular errores globales para h = 0.1
    errores2 = [abs(solucion_exacta(T2[i]) - Y2[i]) for i in range(len(T2))]
    error_global2 = max(errores2)
    
    # Mostrar resultados
    print(f"\nMétodo de Euler con h = 0.2 (n = {n1} pasos)")
    print(f"Error global: eT = {error_global1:.6f}\n")
    
    print(f"Método de Euler con h = 0.1 (n = {n2} pasos)")
    print(f"Error global: eT = {error_global2:.6f}\n")
    
    # Tabla comparativa (algunos puntos)
    print(f"{'t':<8} {'y(t) exacta':<15} {'ȳ(h=0.2)':<15} {'Error h=0.2':<15} {'ȳ(h=0.1)':<15} {'Error h=0.1':<15}")
    print("-"*85)
    
    for i in range(0, len(T1), 1):
        t = T1[i]
        y_exacta = solucion_exacta(t)
        y_euler1 = Y1[i]
        error1 = abs(y_exacta - y_euler1)
        
        # Encontrar el valor correspondiente en T2
        idx2 = min(range(len(T2)), key=lambda j: abs(T2[j] - t))
        y_euler2 = Y2[idx2]
        error2 = abs(y_exacta - y_euler2)
        
        print(f"{t:<8.1f} {y_exacta:<15.6f} {y_euler1:<15.6f} {error1:<15.6f} {y_euler2:<15.6f} {error2:<15.6f}")
    
    print("="*70 + "\n")
