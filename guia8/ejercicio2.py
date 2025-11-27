"""
Guía 8 - Ejercicio 2
Cuadratura de Gauss vs Trapecio y Simpson

Resuelve: ∫₀^π sin(2φ)e^(-φ) dφ
"""

import math
from metodos import gauss_legendre, trapecio, simpson


def f(phi):
    """Función a integrar: sin(2φ) * e^(-φ)"""
    return math.sin(2 * phi) * math.exp(-phi)


def solucion_exacta():
    """Solución analítica: (e^(-π) + 2)/5"""
    return (math.exp(-math.pi) + 2) / 5


if __name__ == "__main__":
    a, b = 0, math.pi
    exacto = solucion_exacta()
    
    print("\n" + "="*60)
    print("  ∫₀^π sin(2φ)e^(-φ) dφ")
    print("="*60)
    print(f"Solución exacta: {exacto:.8f}\n")
    
    # Cuadratura de Gauss
    gauss_2 = gauss_legendre(f, a, b, n_puntos=2, verbose=False)
    gauss_3 = gauss_legendre(f, a, b, n_puntos=3, verbose=False)
    
    # Métodos compuestos
    trap = trapecio(f=f, a=a, b=b, n=10, verbose=False)
    simp = simpson(f=f, a=a, b=b, n=10, verbose=False)
    
    # Resultados
    print(f"{'Método':<25} {'Resultado':<15} {'Error':<12}")
    print("-"*60)
    print(f"{'Gauss (2 puntos)':<25} {gauss_2:<15.8f} {abs(gauss_2-exacto):<12.2e}")
    print(f"{'Gauss (3 puntos)':<25} {gauss_3:<15.8f} {abs(gauss_3-exacto):<12.2e}")
    print(f"{'Trapecio (n=10)':<25} {trap:<15.8f} {abs(trap-exacto):<12.2e}")
    print(f"{'Simpson (n=10)':<25} {simp:<15.8f} {abs(simp-exacto):<12.2e}")
    print("="*60 + "\n")
