"""
Métodos numéricos para diferenciación.

Este módulo contiene implementaciones de métodos de diferenciación numérica
para calcular aproximaciones de derivadas.
"""


def diferenciacion(f, a, b, n, verbose=True):
    """
    Calcula la derivada numérica de una función en puntos equiespaciados.
    
    Utiliza diferencias finitas con precisión O(h²):
    - Puntos extremos: fórmulas de 3 puntos hacia adelante/atrás
    - Puntos interiores: fórmula centrada de 3 puntos
    
    Args:
        f (callable): Función a derivar
        a (float): Límite inferior del intervalo
        b (float): Límite superior del intervalo
        n (int): Número de subintervalos (debe ser >= 2)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, fp) donde:
            - X: lista de puntos donde se evalúa la derivada
            - fp: lista de valores de la derivada en cada punto
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> # Derivada de f(x) = x²
        >>> f = lambda x: x**2
        >>> X, fp = diferenciacion(f, a=0, b=2, n=10)
        >>> # fp[i] ≈ 2*X[i] (derivada analítica)
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable")
    
    if a >= b:
        raise ValueError(f"El límite inferior a={a} debe ser menor que b={b}")
    
    if not isinstance(n, int) or n < 2:
        raise ValueError(f"n debe ser un entero >= 2, recibido: {n}")
    
    if verbose:
        print(f"Diferenciación numérica en [{a}, {b}] con {n} subintervalos ({n+1} puntos)")
        print(f"Precisión: O(h²) con h = {(b-a)/n:.6f}")
    
    h = (b - a) / n
    
    # Inicializar arrays
    X = [0] * (n + 1)
    fp = [0] * (n + 1)
    
    # Generar puntos
    for i in range(n + 1):
        X[i] = a + i * h
    
    # Fórmula de 3 puntos hacia adelante para el punto inicial (precisión O(h²))
    fp[0] = (-3*f(X[0]) + 4*f(X[1]) - f(X[2])) / (2*h)
    
    # Fórmula centrada para puntos interiores (precisión O(h²))
    for i in range(1, n):
        fp[i] = (f(X[i+1]) - f(X[i-1])) / (2*h)
    
    # Fórmula de 3 puntos hacia atrás para el punto final (precisión O(h²))
    fp[n] = (f(X[n-2]) - 4*f(X[n-1]) + 3*f(X[n])) / (2*h)
    
    if verbose:
        print(f"Derivada calculada en {n+1} puntos")
        print(f"Rango de valores: f'(x) ∈ [{min(fp):.6f}, {max(fp):.6f}]")
    
    return X, fp


