"""
Métodos numéricos para resolución de Ecuaciones Diferenciales de Orden 1

Este módulo contiene implementaciones de métodos de resolución de EDOs de orden 1 de la forma 
dy/dx = f(x,y) con dato inicial y(x0) = y0
"""


def euler(f, x0, y0, xf, n, verbose=True):
    """
    Resuelve una EDO de primer orden usando el método de Euler.
    
    Resuelve dy/dx = f(x,y) con condición inicial y(x0) = y0
    usando el método de Euler (explícito, orden 1).
    
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
        >>> X, Y = euler(f, x0=0, y0=1, xf=1, n=10)
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
        print(f"Método de Euler")
        print(f"EDO: dy/dx = f(x,y), y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [0] * (n + 1)
    
    X[0] = x0
    Y[0] = y0
    
    # Método de Euler: y_{i+1} = y_i + h*f(x_i, y_i)
    for i in range(n):
        X[i+1] = X[0] + (i+1) * h
        Y[i+1] = Y[i] + h * f(X[i], Y[i])
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y


def heun(f, x0, y0, xf, n, verbose=True):
    """
    Resuelve una EDO de primer orden usando el método de Heun.
    
    Resuelve dy/dx = f(x,y) con condición inicial y(x0) = y0
    usando el método de Heun (predictor-corrector, orden 2).
    
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
        >>> X, Y = heun(f, x0=0, y0=1, xf=1, n=10)
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
        print(f"Método de Heun (Predictor-Corrector)")
        print(f"EDO: dy/dx = f(x,y), y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [0] * (n + 1)
    
    X[0] = x0
    Y[0] = y0
    
    # Método de Heun
    for i in range(n):
        X[i+1] = X[0] + (i+1) * h
        
        # Predictor (Euler)
        Yp = Y[i] + h * f(X[i], Y[i])
        
        # Corrector (promedio de pendientes)
        Y[i+1] = Y[i] + (h / 2) * (f(X[i], Y[i]) + f(X[i+1], Yp))
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y


def punto_medio(f, x0, y0, xf, n, verbose=True):
    """
    Resuelve una EDO de primer orden usando el método del punto medio.
    
    Resuelve dy/dx = f(x,y) con condición inicial y(x0) = y0
    usando el método del punto medio (Runge-Kutta de orden 2).
    
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
        >>> X, Y = punto_medio(f, x0=0, y0=1, xf=1, n=10)
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
        print(f"Método del Punto Medio (RK2)")
        print(f"EDO: dy/dx = f(x,y), y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [0] * (n + 1)
    
    X[0] = x0
    Y[0] = y0
    
    # Método del punto medio
    for i in range(n):
        X[i+1] = X[0] + (i+1) * h
        
        # Evaluar en el punto medio
        Xm = X[i] + h / 2
        Ym = Y[i] + (h / 2) * f(X[i], Y[i])
        
        # Usar la pendiente en el punto medio
        Y[i+1] = Y[i] + h * f(Xm, Ym)
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y


def runge_kutta4(f, x0, y0, xf, n, verbose=True):
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
        
    Notes:
        El método RK4 usa 4 evaluaciones de f por paso:
        k1 = f(x_i, y_i)
        k2 = f(x_i + h/2, y_i + h*k1/2)
        k3 = f(x_i + h/2, y_i + h*k2/2)
        k4 = f(x_i + h, y_i + h*k3)
        y_{i+1} = y_i + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
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
    
    # Método de Runge-Kutta 4
    for i in range(n):
        X[i+1] = X[0] + (i+1) * h
        
        # Calcular las 4 pendientes
        k1 = f(X[i], Y[i])
        k2 = f(X[i] + h/2, Y[i] + h*k1/2)
        k3 = f(X[i] + h/2, Y[i] + h*k2/2)
        k4 = f(X[i] + h, Y[i] + h*k3)
        
        # Combinar las pendientes con pesos 1-2-2-1
        Y[i+1] = Y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y