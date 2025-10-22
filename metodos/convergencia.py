"""
Análisis de convergencia para métodos de resolución de EDOs

Calcula el factor de convergencia para los métodos de Euler, Heun, 
Punto Medio y Runge-Kutta 4 punto a punto.
"""

import math
from .edo1 import euler, heun, punto_medio, runge_kutta4


def _calcular_factor_convergencia_generico(metodo, nombre_metodo, f, x0, y0, xf, n, verbose=True):
    """
    Función genérica para calcular el factor de convergencia de cualquier método EDO.
    
    Ejecuta el método 3 veces con n, 2*n y 4*n pasos, y calcula el factor de
    convergencia para cada punto xi usando:
    
    factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
    
    Args:
        metodo (callable): Función del método EDO (euler, heun, punto_medio, runge_kutta4)
        nombre_metodo (str): Nombre del método para mostrar en verbose
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos base
        verbose (bool, optional): Si True, imprime resumen. Por defecto True.
        
    Returns:
        tuple: (X, factores, factor_promedio) donde:
            - X: lista de puntos x donde se calculó el factor
            - factores: lista con los factores de convergencia en cada punto
            - factor_promedio: promedio de los factores válidos
    """
    # Ejecutar el método con n, 2n y 4n pasos
    X1, Y1 = metodo(f, x0, y0, xf, n, verbose=False)
    X2, Y2 = metodo(f, x0, y0, xf, 2*n, verbose=False)
    X3, Y3 = metodo(f, x0, y0, xf, 4*n, verbose=False)
    
    h = (xf - x0) / n
    
    # Calcular factores para cada punto
    X_factores = []
    factores = []
    
    for i in range(len(X1)):
        xi = X1[i]
        y1_i = Y1[i]
        y2_i = Y2[2*i]  # Punto correspondiente en X2
        y3_i = Y3[4*i]  # Punto correspondiente en X3
        
        # Calcular diferencias
        diff_12 = abs(y1_i - y2_i)
        diff_23 = abs(y2_i - y3_i)
        
        # Calcular factor si las diferencias no son cero
        if diff_23 > 1e-15 and diff_12 > 1e-15:
            factor = math.log(diff_12 / diff_23) / math.log(2)
            X_factores.append(xi)
            factores.append(factor)
    
    # Calcular promedio
    factor_promedio = sum(factores) / len(factores) if factores else float('nan')
    
    if verbose:
        print(f"Factor de convergencia de {nombre_metodo}:")
        print(f"  Intervalo: [{x0}, {xf}], n={n} (h={h:.6f})")
        print(f"  Puntos analizados: {len(factores)}")
        print(f"  Factor promedio: {factor_promedio:.6f}")
        print(f"  Rango: [{min(factores):.6f}, {max(factores):.6f}]")
    
    return X_factores, factores, factor_promedio


def calcular_factor_convergencia_euler(f, x0, y0, xf, n, verbose=True):
    """
    Calcula el factor de convergencia del método de Euler punto a punto.
    
    Ejecuta Euler 3 veces con n, 2*n y 4*n pasos, y calcula el factor de
    convergencia para cada punto xi usando:
    
    factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos base
        verbose (bool, optional): Si True, imprime resumen. Por defecto True.
        
    Returns:
        tuple: (X, factores, factor_promedio) donde:
            - X: lista de puntos x donde se calculó el factor
            - factores: lista con los factores de convergencia en cada punto
            - factor_promedio: promedio de los factores válidos
            
    Examples:
        >>> f = lambda x, y: y
        >>> X, factores, promedio = calcular_factor_convergencia_euler(f, 0, 1, 1, n=10)
        >>> # Graficar: plt.plot(X, factores)
    """
    return _calcular_factor_convergencia_generico(euler, "Euler", f, x0, y0, xf, n, verbose)


def calcular_factor_convergencia_heun(f, x0, y0, xf, n, verbose=True):
    """
    Calcula el factor de convergencia del método de Heun punto a punto.
    
    Ejecuta Heun 3 veces con n, 2*n y 4*n pasos, y calcula el factor de
    convergencia para cada punto xi usando:
    
    factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos base
        verbose (bool, optional): Si True, imprime resumen. Por defecto True.
        
    Returns:
        tuple: (X, factores, factor_promedio) donde:
            - X: lista de puntos x donde se calculó el factor
            - factores: lista con los factores de convergencia en cada punto
            - factor_promedio: promedio de los factores válidos
            
    Examples:
        >>> f = lambda x, y: y
        >>> X, factores, promedio = calcular_factor_convergencia_heun(f, 0, 1, 1, n=10)
        >>> # Graficar: plt.plot(X, factores)
    """
    return _calcular_factor_convergencia_generico(heun, "Heun", f, x0, y0, xf, n, verbose)


def calcular_factor_convergencia_punto_medio(f, x0, y0, xf, n, verbose=True):
    """
    Calcula el factor de convergencia del método del Punto Medio punto a punto.
    
    Ejecuta Punto Medio 3 veces con n, 2*n y 4*n pasos, y calcula el factor de
    convergencia para cada punto xi usando:
    
    factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos base
        verbose (bool, optional): Si True, imprime resumen. Por defecto True.
        
    Returns:
        tuple: (X, factores, factor_promedio) donde:
            - X: lista de puntos x donde se calculó el factor
            - factores: lista con los factores de convergencia en cada punto
            - factor_promedio: promedio de los factores válidos
            
    Examples:
        >>> f = lambda x, y: y
        >>> X, factores, promedio = calcular_factor_convergencia_punto_medio(f, 0, 1, 1, n=10)
        >>> # Graficar: plt.plot(X, factores)
    """
    return _calcular_factor_convergencia_generico(punto_medio, "Punto Medio", f, x0, y0, xf, n, verbose)


def calcular_factor_convergencia_rk4(f, x0, y0, xf, n, verbose=True):
    """
    Calcula el factor de convergencia del método de Runge-Kutta 4 punto a punto.
    
    Ejecuta RK4 3 veces con n, 2*n y 4*n pasos, y calcula el factor de
    convergencia para cada punto xi usando:
    
    factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos base
        verbose (bool, optional): Si True, imprime resumen. Por defecto True.
        
    Returns:
        tuple: (X, factores, factor_promedio) donde:
            - X: lista de puntos x donde se calculó el factor
            - factores: lista con los factores de convergencia en cada punto
            - factor_promedio: promedio de los factores válidos
            
    Examples:
        >>> f = lambda x, y: y
        >>> X, factores, promedio = calcular_factor_convergencia_rk4(f, 0, 1, 1, n=10)
        >>> # Graficar: plt.plot(X, factores)
    """
    return _calcular_factor_convergencia_generico(runge_kutta4, "Runge-Kutta 4", f, x0, y0, xf, n, verbose)

