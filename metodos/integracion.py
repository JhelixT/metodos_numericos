"""
Métodos numéricos para integración.

Este módulo contiene implementaciones de métodos de integración numérica
para calcular aproximaciones de integrales definidas.
"""

import numpy as np
from .aproximacion import curvas_spline, evaluar_spline


def trapecio(f=None, a=None, b=None, n=None, X=None, Y=None, verbose=True):
    """
    Calcula la integral numérica usando la regla del trapecio.
    
    Soporta dos modos de operación:
    
    1. Función continua: Requiere f, a, b, n
       - Divide [a,b] en n subintervalos equiespaciados
       - Aplica la regla del trapecio en cada subintervalo
       
    2. Datos tabulados: Requiere X, Y
       - Si los datos están equiespaciados: aplica regla del trapecio directamente
       - Si los datos NO están equiespaciados: construye splines cúbicos y aplica trapecio
    
    Args:
        f (callable, optional): Función a integrar
        a (float, optional): Límite inferior de integración
        b (float, optional): Límite superior de integración
        n (int, optional): Número de subintervalos para función continua
        X (list[float], optional): Coordenadas x de datos tabulados
        Y (list[float], optional): Coordenadas y de datos tabulados
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.
        
    Returns:
        float: Aproximación de la integral
        
    Raises:
        ValueError: Si los parámetros proporcionados no son válidos
        
    Examples:
        >>> # Modo 1: Función continua
        >>> def f(x): return x**2
        >>> resultado = trapecio(f=f, a=0, b=3, n=100)
        
        >>> # Modo 2: Datos tabulados equiespaciados
        >>> X = [0, 1, 2, 3]
        >>> Y = [0, 1, 4, 9]
        >>> resultado = trapecio(X=X, Y=Y)
        
        >>> # Modo 2: Datos tabulados NO equiespaciados (usa splines)
        >>> X = [0, 0.5, 2, 3]
        >>> Y = [0, 0.25, 4, 9]
        >>> resultado = trapecio(X=X, Y=Y)
    """
    
    # Modo 1: Función continua
    if f is not None and a is not None and b is not None and n is not None:
        if verbose:
            print(f"Modo: Función continua f(x) en [{a}, {b}] con {n} subintervalos")
        
        h = (b - a) / n
        suma = (f(a) + f(b)) / 2
        
        for i in range(1, n):
            x_i = a + i * h
            suma += f(x_i)
        
        integral = h * suma
        
        if verbose:
            print(f"Resultado: ∫f(x)dx ≈ {integral}")
        
        return integral
    
    # Modo 2: Datos tabulados
    elif X is not None and Y is not None:
        if len(X) != len(Y):
            raise ValueError("Los arrays X e Y deben tener la misma longitud")
        
        if len(X) < 2:
            raise ValueError("Se necesitan al menos 2 puntos para integrar")
        
        X = [float(x) for x in X]
        Y = [float(y) for y in Y]
        
        # Verificar si los datos están equiespaciados
        diferencias = [X[i+1] - X[i] for i in range(len(X)-1)]
        h = diferencias[0]
        equiespaciado = all(abs(diff - h) < 1e-10 for diff in diferencias)
        
        if equiespaciado:
            # Caso simple: datos equiespaciados
            if verbose:
                print(f"Modo: Datos tabulados equiespaciados ({len(X)} puntos, h={h})")
            
            suma = (Y[0] + Y[-1]) / 2
            for i in range(1, len(Y)-1):
                suma += Y[i]
            
            integral = h * suma
            
            if verbose:
                print(f"Resultado: ∫f(x)dx ≈ {integral}")
            
            return integral
        
        else:
            # Caso complejo: datos NO equiespaciados → usar splines
            if verbose:
                print(f"Modo: Datos tabulados NO equiespaciados ({len(X)} puntos)")
                print("Se construirán splines cúbicos para la integración...")
            
            funciones_spline, _, X_ordenado, _ = curvas_spline(X=X, Y=Y, verbose=False)
            
            num_subintervalos = len(X) - 1
            
            integral_total = 0.0
            
            for k in range(num_subintervalos):
                a_k = X_ordenado[k]
                b_k = X_ordenado[k+1]
                
                n_puntos = 100
                h_spline = (b_k - a_k) / n_puntos
                
                suma = (evaluar_spline(a_k, funciones_spline, X_ordenado) + 
                       evaluar_spline(b_k, funciones_spline, X_ordenado)) / 2
                
                for i in range(1, n_puntos):
                    x_i = a_k + i * h_spline
                    suma += evaluar_spline(x_i, funciones_spline, X_ordenado)
                
                integral_k = h_spline * suma
                integral_total += integral_k
            
            if verbose:
                print(f"Resultado: ∫f(x)dx ≈ {integral_total}")
            
            return integral_total
    
    else:
        raise ValueError(
            "Parámetros inválidos. Use uno de los siguientes modos:\n"
            "  1. Función continua: f, a, b, n\n"
            "  2. Datos tabulados: X, Y"
        )
