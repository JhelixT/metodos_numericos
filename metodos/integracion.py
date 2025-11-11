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
       
    2. Datos tabulados: Requiere X, Y (opcionalmente n)
       - Construye splines cúbicos para aproximar los datos
       - Equiespacía el dominio [X[0], X[-1]] con n intervalos
       - Evalúa los splines en los puntos equiespaciados
       - Aplica la regla del trapecio a los datos equiespaciados
    
    Args:
        f (callable, optional): Función a integrar
        a (float, optional): Límite inferior de integración
        b (float, optional): Límite superior de integración
        n (int, optional): Número de subintervalos. Si no se especifica con datos 
                          tabulados, se usa n = len(X) - 1
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
        
        >>> # Modo 2: Datos tabulados (n por defecto)
        >>> X = [0, 1, 2, 3]
        >>> Y = [0, 1, 4, 9]
        >>> resultado = trapecio(X=X, Y=Y)
        
        >>> # Modo 2: Datos tabulados (n especificado)
        >>> X = [0, 0.5, 2, 3]
        >>> Y = [0, 0.25, 4, 9]
        >>> resultado = trapecio(X=X, Y=Y, n=500)
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
        
        if n is None:
            n = len(X) - 1  # Valor por defecto: número de intervalos originales
        
        X_original = [float(x) for x in X]
        Y_original = [float(y) for y in Y]
        
        if verbose:
            print(f"Modo: Datos tabulados ({len(X_original)} puntos)")
            print(f"Se construirán splines cúbicos y se equiespaciarán con n={n} intervalos...")
        
        # Calcular funciones splines para aproximar los datos originales
        funciones_spline, _, X_ordenado, _ = curvas_spline(X=X_original, Y=Y_original, verbose=False)
        
        # Definir extremos del intervalo
        a = X_ordenado[0]   # X[0]: extremo inicial
        b = X_ordenado[-1]  # X[-1]: extremo final
        h = (b - a) / n
        
        if verbose:
            print(f"Equiespaciando: [{a}, {b}] con n={n}, h={h:.6f}")
        
        # Crear nuevo arreglo X equiespaciado: X[i] = X[0] + i*h
        X_eq = [a + i * h for i in range(n + 1)]
        
        # Crear arreglo Y evaluando splines: Y[i] = evaluar_spline(X[i], funciones_spline)
        Y_eq = [evaluar_spline(X_eq[i], funciones_spline, X_ordenado) for i in range(n + 1)]
        
        # Aplicar método del trapecio con los arreglos equiespaciados
        suma = (Y_eq[0] + Y_eq[-1]) / 2
        for i in range(1, n):
            suma += Y_eq[i]
        
        integral = h * suma
        
        if verbose:
            print(f"Resultado: ∫f(x)dx ≈ {integral}")
        
        return integral
    
    else:
        raise ValueError(
            "Parámetros inválidos. Use uno de los siguientes modos:\n"
            "  1. Función continua: f, a, b, n\n"
            "  2. Datos tabulados: X, Y"
        )
def simpson(f=None, a=None, b=None, n=None, X=None, Y=None, verbose=True):
    """
    Calcula la integral numérica usando la regla de Simpson 1/3.
    
    Soporta dos modos de operación:
    
    1. Función continua: Requiere f, a, b, n (n debe ser PAR)
       - Divide [a,b] en n subintervalos equiespaciados
       - Aplica la regla de Simpson 1/3
       
    2. Datos tabulados: Requiere X, Y (opcionalmente n)
       - Construye splines cúbicos para aproximar los datos
       - Equiespacía el dominio [X[0], X[-1]] con n intervalos (asegura que n sea PAR)
       - Evalúa los splines en los puntos equiespaciados
       - Aplica la regla de Simpson 1/3 a los datos equiespaciados
    
    Args:
        f (callable, optional): Función a integrar
        a (float, optional): Límite inferior de integración
        b (float, optional): Límite superior de integración
        n (int, optional): Número de subintervalos (debe ser PAR). Si no se especifica
                          con datos tabulados, se usa n = len(X) si len(X) es par, 
                          o n = len(X) - 1 si len(X) es impar
        X (list[float], optional): Coordenadas x de datos tabulados
        Y (list[float], optional): Coordenadas y de datos tabulados
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        float: Aproximación de la integral
        
    Raises:
        ValueError: Si n no es par o los parámetros son inválidos
        
    Examples:
        >>> # Modo 1: Función continua
        >>> def f(x): return x**2
        >>> resultado = simpson(f=f, a=0, b=3, n=100)
        
        >>> # Modo 2: Datos tabulados (n por defecto)
        >>> X = [0, 1, 2, 3]
        >>> Y = [0, 1, 4, 9]
        >>> resultado = simpson(X=X, Y=Y)
        
        >>> # Modo 2: Datos tabulados (n especificado)
        >>> X = [0, 0.5, 2, 3]
        >>> Y = [0, 0.25, 4, 9]
        >>> resultado = simpson(X=X, Y=Y, n=500)
    """
    # Modo 1: Función continua
    if f is not None and a is not None and b is not None and n is not None:
        if n % 2 != 0:
            raise ValueError("El número de intervalos debe ser par para Simpson 1/3")
        
        if verbose:
            print(f"Modo: Función continua f(x) en [{a}, {b}] con n={n}")
        
        h = (b - a) / n
        suma = f(a) + f(b)
        
        for i in range(1, n):
            xi = a + i * h
            coef = 4 if i % 2 == 1 else 2
            suma += coef * f(xi)
        
        integral = (h / 3) * suma
        
        if verbose:
            print(f"Resultado: ∫f(x)dx ≈ {integral}")
        
        return integral
    
    # Modo 2: Datos tabulados
    elif X is not None and Y is not None:
        if len(X) != len(Y):
            raise ValueError("Los arrays X e Y deben tener la misma longitud")
        
        if len(X) < 3:
            raise ValueError("Se necesitan al menos 3 puntos para Simpson 1/3")
        
        if n is None:
            # Si la cantidad de puntos es par, usar len(X)
            # Si es impar, usar len(X) - 1
            n = len(X) if len(X) % 2 == 0 else len(X) - 1
        
        # Asegurar que n sea PAR
        if n % 2 != 0:
            n += 1
        
        X_original = [float(x) for x in X]
        Y_original = [float(y) for y in Y]
        
        if verbose:
            print(f"Modo: Datos tabulados ({len(X_original)} puntos)")
            print(f"Se construirán splines cúbicos y se equiespaciarán con n={n} intervalos...")
        
        # Calcular funciones splines para aproximar los datos originales
        funciones_spline, _, X_ordenado, _ = curvas_spline(X=X_original, Y=Y_original, verbose=False)
        
        # Definir extremos del intervalo
        a = X_ordenado[0]   # X[0]: extremo inicial
        b = X_ordenado[-1]  # X[-1]: extremo final
        h = (b - a) / n
        
        if verbose:
            print(f"Equiespaciando: [{a}, {b}] con n={n} (PAR), h={h:.6f}")
        
        # Crear nuevo arreglo X equiespaciado: X[i] = X[0] + i*h
        X_eq = [a + i * h for i in range(n + 1)]
        
        # Crear arreglo Y evaluando splines: Y[i] = evaluar_spline(X[i], funciones_spline)
        Y_eq = [evaluar_spline(X_eq[i], funciones_spline, X_ordenado) for i in range(n + 1)]
        
        # Aplicar método de Simpson 1/3 con los arreglos equiespaciados
        suma = Y_eq[0] + Y_eq[-1]
        
        for i in range(1, n):
            coef = 4 if i % 2 == 1 else 2
            suma += coef * Y_eq[i]
        
        integral = (h / 3) * suma
        
        if verbose:
            print(f"Resultado: ∫f(x)dx ≈ {integral}")
        
        return integral
    
    else:
        raise ValueError(
            "Parámetros inválidos. Use uno de los siguientes modos:\n"
            "  1. Función continua: f, a, b, n (n debe ser PAR)\n"
            "  2. Datos tabulados: X, Y (opcionalmente n)"
        )


def gauss_legendre(f, a, b, n_puntos=2, verbose=True):
    """
    Calcula la integral numérica usando cuadratura de Gauss-Legendre.
    
    La cuadratura de Gauss-Legendre aproxima la integral usando puntos de cuadratura
    óptimamente ubicados y sus factores de ponderación correspondientes.
    
    Transforma la integral ∫[a,b] f(x)dx a ∫[-1,1] g(x')dx' mediante el cambio:
        x = ((b-a)x' + (b+a))/2
        dx = (b-a)/2 dx'
    
    Args:
        f (callable): Función a integrar
        a (float): Límite inferior de integración
        b (float): Límite superior de integración
        n_puntos (int): Número de puntos de cuadratura (2, 3, 4, 5 o 6).
                       Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        float: Aproximación de la integral
        
    Raises:
        ValueError: Si n_puntos no está en el rango [2, 6]
        
    Examples:
        >>> def f(x): return x**2
        >>> # Usando 2 puntos
        >>> resultado = gauss_legendre(f, 0, 3, n_puntos=2)
        >>> # Usando 4 puntos (mayor precisión)
        >>> resultado = gauss_legendre(f, 0, 3, n_puntos=4)
        
    Notes:
        - Mayor número de puntos → mayor precisión
        - Exacto para polinomios de grado ≤ 2n-1 (n = número de puntos)
        - 2 puntos: exacto hasta grado 3
        - 3 puntos: exacto hasta grado 5
        - 4 puntos: exacto hasta grado 7
        - 5 puntos: exacto hasta grado 9
        - 6 puntos: exacto hasta grado 11
    """
    
    # Tabla 22.1: Factores de ponderación y argumentos de la función
    # Fuente: Chapra & Canale, "Numerical Methods for Engineers"
    tablas_gauss = {
        2: {
            'c': [1.0000000, 1.0000000],
            'x': [-0.5773502692, 0.5773502692]
        },
        3: {
            'c': [0.5555556, 0.8888889, 0.5555556],
            'x': [-0.7745966692, 0.0, 0.7745966692]
        },
        4: {
            'c': [0.3478548, 0.6521452, 0.6521452, 0.3478548],
            'x': [-0.8611363116, -0.3399810435, 0.3399810435, 0.8611363116]
        },
        5: {
            'c': [0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269],
            'x': [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
        },
        6: {
            'c': [0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245],
            'x': [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142]
        }
    }
    
    if n_puntos not in tablas_gauss:
        raise ValueError(f"n_puntos debe ser 2, 3, 4, 5 o 6. Recibido: {n_puntos}")
    
    if verbose:
        print(f"Cuadratura de Gauss-Legendre con {n_puntos} puntos")
        print(f"Intervalo: [{a}, {b}]")
    
    # Obtener coeficientes y argumentos para n_puntos
    coeficientes = tablas_gauss[n_puntos]['c']
    argumentos = tablas_gauss[n_puntos]['x']
    
    # Cambio de variable: [a,b] → [-1,1]
    # x = ((b-a)x' + (b+a))/2
    # dx = (b-a)/2 dx'
    
    suma = 0.0
    if verbose:
        print(f"\n{'i':<5} {'c_i':<15} {'x_i':<15} {'x (transf.)':<15} {'f(x)':<15}")
        print("-" * 70)
    
    for i in range(n_puntos):
        ci = coeficientes[i]
        x_prima = argumentos[i]  # x' en [-1, 1] (dominio canónico)
        
        # Transformar x' → x en [a, b]
        xi = ((b - a) * x_prima + (b + a)) / 2
        
        # Evaluar f(x)
        fi = f(xi)
        
        if verbose:
            print(f"{i:<5} {ci:<15.10f} {x_prima:<15.10f} {xi:<15.10f} {fi:<15.10f}")
        
        suma += ci * fi
    
    # Aplicar factor de escala
    integral = ((b - a) / 2) * suma
    
    if verbose:
        print("-" * 70)
        print(f"\nFactor de escala: (b-a)/2 = ({b}-{a})/2 = {(b-a)/2}")
        print(f"Resultado: ∫f(x)dx ≈ {integral}")
    
    return integral
        