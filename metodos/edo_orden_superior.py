"""
Métodos numéricos para resolución de Ecuaciones Diferenciales de Orden Superior

Este módulo resuelve EDOs de orden m >= 2 convirtiéndolas en sistemas de ecuaciones
de primer orden. Aprovecha los métodos ya implementados en sistemas_edo.py.

Una EDO de orden m:
    y^(m) = f(x, y, y', y'', ..., y^(m-1))

Se convierte en un sistema de m ecuaciones de primer orden:
    y1 = y
    y2 = y' 
    y3 = y''
    ...
    ym = y^(m-1)
    
    dy1/dx = y2
    dy2/dx = y3
    ...
    dy(m-1)/dx = ym
    dym/dx = f(x, y1, y2, ..., ym)
"""

try:
    # Importación relativa (cuando se usa como módulo)
    from .sistemas_edo import (
        euler_sistema,
        heun_sistema,
        punto_medio_sistema,
        runge_kutta4_sistema
    )
except ImportError:
    # Importación absoluta (cuando se ejecuta directamente)
    from sistemas_edo import (
        euler_sistema,
        heun_sistema,
        punto_medio_sistema,
        runge_kutta4_sistema
    )


def _convertir_a_sistema(f, orden):
    """
    Convierte una EDO de orden superior en un sistema de ecuaciones de primer orden.
    
    Args:
        f: Función f(x, y, y', y'', ..., y^(m-1)) que define la EDO
        orden: Orden de la ecuación diferencial
        
    Returns:
        Lista de funciones para el sistema de primer orden
    """
    funciones = []
    
    # Las primeras (orden-1) ecuaciones son: dyi/dx = y(i+1)
    for i in range(orden - 1):
        funciones.append(lambda x, Y, idx=i: Y[idx + 1])
    
    # La última ecuación es: dym/dx = f(x, y1, y2, ..., ym)
    funciones.append(lambda x, Y: f(x, *Y))
    
    return funciones


def euler_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True):
    """
    Resuelve una EDO de orden superior usando el método de Euler.
    
    Convierte la EDO de orden m en un sistema de m ecuaciones de primer orden
    y luego aplica el método de Euler para sistemas.
    
    Args:
        f (callable): Función que define la derivada de orden superior.
                     Para orden m, f(x, y, y', ..., y^(m-1)) retorna y^(m)
        x0 (float): Valor inicial de x
        y0 (list): Lista de condiciones iniciales [y(x0), y'(x0), ..., y^(m-1)(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        orden (int, optional): Orden de la EDO. Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de listas, donde Y[i] = [y, y', y'', ..., y^(m-1)] en el paso i
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> # EDO de orden 2: y'' = -y (oscilador armónico)
        >>> f = lambda x, y, yp: -y
        >>> X, Y = euler_orden_superior(f, x0=0, y0=[1, 0], xf=6.28, n=100, orden=2)
        >>> # Y[i][0] contiene y, Y[i][1] contiene y'
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if orden < 2:
        raise ValueError(f"orden debe ser >= 2, recibido: {orden}")
    
    if len(y0) != orden:
        raise ValueError(
            f"y0 debe tener {orden} elementos para una EDO de orden {orden}, "
            f"recibido: {len(y0)} elementos"
        )
    
    if verbose:
        print(f"Método de Euler para EDO de Orden {orden}")
        print(f"EDO: y^({orden}) = f(x, y, y', ..., y^({orden-1}))")
        print(f"Condiciones iniciales: y^(i)({x0}) = {y0}")
        print(f"Convirtiendo a sistema de {orden} ecuaciones de primer orden...")
    
    # Convertir a sistema de primer orden
    funciones = _convertir_a_sistema(f, orden)
    
    # Resolver usando euler_sistema
    X, Y = euler_sistema(funciones, x0, y0, xf, n, verbose=False)
    
    if verbose:
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {(xf-x0)/n:.6f})")
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y^(i)({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def heun_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True):
    """
    Resuelve una EDO de orden superior usando el método de Heun.
    
    Args:
        f (callable): Función que define la derivada de orden superior
        x0 (float): Valor inicial de x
        y0 (list): Lista de condiciones iniciales [y(x0), y'(x0), ..., y^(m-1)(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        orden (int, optional): Orden de la EDO. Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) con las soluciones
        
    Examples:
        >>> # EDO de orden 2: y'' = -y
        >>> f = lambda x, y, yp: -y
        >>> X, Y = heun_orden_superior(f, x0=0, y0=[1, 0], xf=6.28, n=100, orden=2)
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if orden < 2:
        raise ValueError(f"orden debe ser >= 2, recibido: {orden}")
    
    if len(y0) != orden:
        raise ValueError(
            f"y0 debe tener {orden} elementos para una EDO de orden {orden}, "
            f"recibido: {len(y0)} elementos"
        )
    
    if verbose:
        print(f"Método de Heun para EDO de Orden {orden} (Predictor-Corrector)")
        print(f"EDO: y^({orden}) = f(x, y, y', ..., y^({orden-1}))")
        print(f"Condiciones iniciales: y^(i)({x0}) = {y0}")
        print(f"Convirtiendo a sistema de {orden} ecuaciones de primer orden...")
    
    # Convertir a sistema de primer orden
    funciones = _convertir_a_sistema(f, orden)
    
    # Resolver usando heun_sistema
    X, Y = heun_sistema(funciones, x0, y0, xf, n, verbose=False)
    
    if verbose:
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {(xf-x0)/n:.6f})")
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y^(i)({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def punto_medio_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True):
    """
    Resuelve una EDO de orden superior usando el método del punto medio.
    
    Args:
        f (callable): Función que define la derivada de orden superior
        x0 (float): Valor inicial de x
        y0 (list): Lista de condiciones iniciales [y(x0), y'(x0), ..., y^(m-1)(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        orden (int, optional): Orden de la EDO. Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) con las soluciones
        
    Examples:
        >>> # EDO de orden 2: y'' = -y
        >>> f = lambda x, y, yp: -y
        >>> X, Y = punto_medio_orden_superior(f, x0=0, y0=[1, 0], xf=6.28, n=100, orden=2)
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if orden < 2:
        raise ValueError(f"orden debe ser >= 2, recibido: {orden}")
    
    if len(y0) != orden:
        raise ValueError(
            f"y0 debe tener {orden} elementos para una EDO de orden {orden}, "
            f"recibido: {len(y0)} elementos"
        )
    
    if verbose:
        print(f"Método del Punto Medio para EDO de Orden {orden} (RK2)")
        print(f"EDO: y^({orden}) = f(x, y, y', ..., y^({orden-1}))")
        print(f"Condiciones iniciales: y^(i)({x0}) = {y0}")
        print(f"Convirtiendo a sistema de {orden} ecuaciones de primer orden...")
    
    # Convertir a sistema de primer orden
    funciones = _convertir_a_sistema(f, orden)
    
    # Resolver usando punto_medio_sistema
    X, Y = punto_medio_sistema(funciones, x0, y0, xf, n, verbose=False)
    
    if verbose:
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {(xf-x0)/n:.6f})")
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y^(i)({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def runge_kutta4_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True):
    """
    Resuelve una EDO de orden superior usando el método de Runge-Kutta 4.
    
    Args:
        f (callable): Función que define la derivada de orden superior
        x0 (float): Valor inicial de x
        y0 (list): Lista de condiciones iniciales [y(x0), y'(x0), ..., y^(m-1)(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        orden (int, optional): Orden de la EDO. Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) con las soluciones
        
    Examples:
        >>> # EDO de orden 2: y'' = -y
        >>> f = lambda x, y, yp: -y
        >>> X, Y = runge_kutta4_orden_superior(f, x0=0, y0=[1, 0], xf=6.28, n=50, orden=2)
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if orden < 2:
        raise ValueError(f"orden debe ser >= 2, recibido: {orden}")
    
    if len(y0) != orden:
        raise ValueError(
            f"y0 debe tener {orden} elementos para una EDO de orden {orden}, "
            f"recibido: {len(y0)} elementos"
        )
    
    if verbose:
        print(f"Método de Runge-Kutta 4 para EDO de Orden {orden} (RK4 clásico)")
        print(f"EDO: y^({orden}) = f(x, y, y', ..., y^({orden-1}))")
        print(f"Condiciones iniciales: y^(i)({x0}) = {y0}")
        print(f"Convirtiendo a sistema de {orden} ecuaciones de primer orden...")
        print(f"Precisión: O(h⁴)")
    
    # Convertir a sistema de primer orden
    funciones = _convertir_a_sistema(f, orden)
    
    # Resolver usando runge_kutta4_sistema
    X, Y = runge_kutta4_sistema(funciones, x0, y0, xf, n, verbose=False)
    
    if verbose:
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {(xf-x0)/n:.6f})")
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y^(i)({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


# Ejemplos de uso
if __name__ == "__main__":
    import math
    
    print("="*70)
    print("Ejemplo 1: EDO de Orden 2 - Oscilador Armónico Simple")
    print("Ecuación: y'' = -y")
    print("Condiciones iniciales: y(0) = 1, y'(0) = 0")
    print("Solución exacta: y(x) = cos(x)")
    print("="*70)
    print()
    
    # Definir la EDO de orden 2: y'' = -y
    # f(x, y, y') = -y
    f_osc = lambda x, y, yp: -y
    
    # Resolver con RK4
    X, Y = runge_kutta4_orden_superior(
        f_osc, 
        x0=0, 
        y0=[1.0, 0.0],  # y(0) = 1, y'(0) = 0
        xf=2*math.pi, 
        n=20, 
        orden=2
    )
    
    print("\nComparación con solución exacta:")
    print(f"{'x':>8} {'y (RK4)':>12} {'y (exacta)':>12} {'Error':>12}")
    print("-" * 48)
    
    for i in range(0, len(X), 5):
        x = X[i]
        y_rk4 = Y[i][0]  # Y[i][0] es y, Y[i][1] es y'
        y_exacto = math.cos(x)
        error = abs(y_rk4 - y_exacto)
        print(f"{x:8.4f} {y_rk4:12.8f} {y_exacto:12.8f} {error:12.8f}")
    
    print("\n" + "="*70)
    print("Ejemplo 2: EDO de Orden 2 - Masa-Resorte con Amortiguamiento")
    print("Ecuación: y'' + 0.5*y' + 2*y = 0")
    print("Reescrito: y'' = -0.5*y' - 2*y")
    print("Condiciones iniciales: y(0) = 1, y'(0) = 0")
    print("="*70)
    print()
    
    # y'' = -0.5*y' - 2*y
    f_amort = lambda x, y, yp: -0.5*yp - 2*y
    
    X2, Y2 = runge_kutta4_orden_superior(
        f_amort, 
        x0=0, 
        y0=[1.0, 0.0], 
        xf=10, 
        n=100, 
        orden=2,
        verbose=False
    )
    
    print("Solución (sistema amortiguado):")
    print(f"{'x':>8} {'y':>12} {'y\'':>12}")
    print("-" * 34)
    
    for i in range(0, len(X2), 20):
        x = X2[i]
        y = Y2[i][0]
        yp = Y2[i][1]
        print(f"{x:8.4f} {y:12.6f} {yp:12.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 3: EDO de Orden 3")
    print("Ecuación: y''' = -y'' - y' - y")
    print("Condiciones iniciales: y(0) = 1, y'(0) = 0, y''(0) = 0")
    print("="*70)
    print()
    
    # y''' = -y'' - y' - y
    f_orden3 = lambda x, y, yp, ypp: -ypp - yp - y
    
    X3, Y3 = runge_kutta4_orden_superior(
        f_orden3, 
        x0=0, 
        y0=[1.0, 0.0, 0.0],  # y(0), y'(0), y''(0)
        xf=5, 
        n=50, 
        orden=3
    )
    
    print("\nTabla de resultados:")
    print(f"{'x':>8} {'y':>12} {'y\'':>12} {'y\'\'':>12}")
    print("-" * 48)
    
    for i in range(0, len(X3), 10):
        x = X3[i]
        y = Y3[i][0]
        yp = Y3[i][1]
        ypp = Y3[i][2]
        print(f"{x:8.4f} {y:12.6f} {yp:12.6f} {ypp:12.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 4: Comparación de Métodos para EDO de Orden 2")
    print("="*70)
    print()
    
    f_comp = lambda x, y, yp: -y
    y0_comp = [1.0, 0.0]
    xf_comp = math.pi
    n_comp = 10
    
    X_e, Y_e = euler_orden_superior(f_comp, 0, y0_comp, xf_comp, n_comp, orden=2, verbose=False)
    X_h, Y_h = heun_orden_superior(f_comp, 0, y0_comp, xf_comp, n_comp, orden=2, verbose=False)
    X_pm, Y_pm = punto_medio_orden_superior(f_comp, 0, y0_comp, xf_comp, n_comp, orden=2, verbose=False)
    X_rk4, Y_rk4 = runge_kutta4_orden_superior(f_comp, 0, y0_comp, xf_comp, n_comp, orden=2, verbose=False)
    
    y_exacto = math.cos(xf_comp)
    
    print(f"Comparación en x = π (valor exacto: {y_exacto:.10f}):")
    print(f"{'Método':>20} {'Aproximación':>15} {'Error':>15}")
    print("-" * 52)
    print(f"{'Euler':>20} {Y_e[-1][0]:15.10f} {abs(Y_e[-1][0] - y_exacto):15.10e}")
    print(f"{'Heun':>20} {Y_h[-1][0]:15.10f} {abs(Y_h[-1][0] - y_exacto):15.10e}")
    print(f"{'Punto Medio':>20} {Y_pm[-1][0]:15.10f} {abs(Y_pm[-1][0] - y_exacto):15.10e}")
    print(f"{'Runge-Kutta 4':>20} {Y_rk4[-1][0]:15.10f} {abs(Y_rk4[-1][0] - y_exacto):15.10e}")
