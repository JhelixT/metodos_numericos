import math
import matplotlib.pyplot as plt
import os
from typing import Callable, Tuple, List

yexacta = lambda x: (x+1)**2 - 0.5*math.exp(x)

def main():
    X, Y = runge_kutta4_alternativo(f=lambda x, y: y-x**2+1, x0=0, y0=0.5, xf=2, n=20, verbose=False)
    
    print(f"Xi= {X}")
    print(f"Yi= {Y}")
    print(f"El valor de y(1.5) es : {Y[X.index(1.5000000000000002)]:.10f}")
    print(f"El valor de y(2) es : {Y[X.index(2.0000000000000004)]:.10f}")
    print(f"El error para y(1.5) es: {abs(yexacta(1.5)-Y[X.index(1.5000000000000002)]):.10f}") #Error de python al calcular flotantes
    print(f"El error para y(2) es: {abs(yexacta(2)-Y[X.index(2.0000000000000004)]):.10f}")

    guardar_plot(X, Y)

def guardar_plot(Xrk,Yrk, nombre_archivo: str = "ComparativoRK4vsExacta.png"):
    # Malla fina para exacta
    xs = [i / 400 for i in range(0, 800)]  # 0..2 con paso 0.0025
    ys = [yexacta(x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, 'k-', label='Exacta (x+1)^2 - (e^x)/2')
    plt.plot(Xrk, Yrk, 'o-', label='Método propuesto ')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, nombre_archivo)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFigura guardada en: {out_path}")



def runge_kutta4_alternativo(f, x0, y0, xf, n, verbose=True):
    """
    Resuelve una EDO de primer orden usando el método de Runge-Kutta 4 clásico.
    
    Resuelve dy/dx = f(x,y) con condición inicial y(x0) = y0
    usando el método de Runge-Kutta de orden 4 (RK4 clásico).
    
    Este es el método más utilizado por su excelente balance entre
    precisión (orden 4) y costo computacional.
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de valores aproximados de y
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> # Resolver dy/dx = x + y, y(0) = 1
        >>> f = lambda x, y: x + y
        >>> X, Y = runge_kutta4(f, x0=0, y0=1, xf=1, n=10)
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable de la forma f(x, y)")
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    
    if verbose:
        print(f"Método de Runge-Kutta 4 (RK4 clásico)")
        print(f"EDO: dy/dx = f(x,y), y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
        print(f"Precisión: O(h⁴)")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [0] * (n + 1)
    
    X[0] = x0
    Y[0] = y0
    
    # Método de Runge-Kutta 4 alternativo
    for i in range(n):
        X[i+1] = X[i] + h
        
        # Calcular las 4 pendientes
        k1 = f(X[i], Y[i])
        k2 = f(X[i] + h/3, Y[i] + h*k1/3)
        k3 = f(X[i] + 2*h/3, Y[i] - h*k1/3 + h*k2)
        k4 = f(X[i] + h, Y[i] + h*k1 - h*k2 + h*k3)
        
        # Combinar las pendientes con pesos 1-3-3-1
        Y[i+1] = Y[i] + (h / 8) * (k1 + 3*k2 + 3*k3 + k4)
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y
main()