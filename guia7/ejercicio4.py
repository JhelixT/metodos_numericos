from metodos import trapecio, simpson
import sympy as sp
import math

if __name__ == "__main__":
    # Definir variable simbólica
    x = sp.Symbol('x')
    
    # Definir funciones como expresiones simbólicas
    f1 = x**3
    f2 = sp.sin(x)
    
    # Derivar simbólicamente
    f1p = sp.diff(f1, x)  # 3*x**2
    f2p = sp.diff(f2, x)  # cos(x)
    
    # Convertir a funciones callable (lambdify)
    f1p_callable = sp.lambdify(x, f1p, 'math')
    f2p_callable = sp.lambdify(x, f2p, 'math')

    # Definicion de funciones de curva: L = integral de sqrt(1 + [f'(x)]^2) dx
    curva_f1 = lambda t: math.sqrt(1 + (f1p_callable(t))**2)
    curva_f2 = lambda t: math.sqrt(1 + (f2p_callable(t))**2)

    # Lista de funciones con sus parámetros
    funciones = [
        (curva_f1, "f(x) = x^3", "f'(x) = 3x^2", 0, 1),
        (curva_f2, "f(x) = sin(x)", "f'(x) = cos(x)", 0, math.pi/4)
    ]

    print("=" * 90)
    print("CALCULO DE LONGITUD DE CURVA")
    print("=" * 90)
    
    for i, (func, nombre, derivada, a, b) in enumerate(funciones, 1):
        # Calcular longitudes
        long_trapecio = trapecio(func, a, b, n=10, verbose=False)
        long_simpson = simpson(func, a, b, n=10, verbose=False)
        diferencia = abs(long_trapecio - long_simpson)
        
        print(f"\n{i}. Funcion: {nombre}")
        print(f"   Derivada: {derivada}")
        print(f"   Intervalo: [{a}, {b:.6f}]" if b > 2 else f"   Intervalo: [{a}, {b}]")
        print("-" * 90)
        print(f"   Longitud de curva (Trapecio n=10):  {long_trapecio:.12f}")
        print(f"   Longitud de curva (Simpson n=10):   {long_simpson:.12f}")
        print(f"   Diferencia absoluta:                 {diferencia:.12e}")
        print("-" * 90)
    
    print("\n" + "=" * 90)
    print("Calculo completado")
    print("=" * 90)