"""
Métodos numéricos para resolución de Sistemas de Ecuaciones Diferenciales de Orden 1

Este módulo contiene implementaciones del método de Euler para sistemas de EDOs de la forma:
    dy1/dx = f1(x, y1, y2, ..., yn)
    dy2/dx = f2(x, y1, y2, ..., yn)
    ...
    dyn/dx = fn(x, y1, y2, ..., yn)
con datos iniciales y1(x0), y2(x0), ..., yn(x0)

Es una extensión directa del módulo edo1.py para sistemas de n ecuaciones.
"""


def euler_sistema(funciones, x0, y0, xf, n, verbose=True):
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Euler.
    
    Resuelve el sistema:
        dy1/dx = f1(x, y1, y2, ..., yn)
        dy2/dx = f2(x, y1, y2, ..., yn)
        ...
        dyn/dx = fn(x, y1, y2, ..., yn)
    
    con condiciones iniciales y1(x0), y2(x0), ..., yn(x0)
    usando el método de Euler (explícito, orden 1).
    
    Args:
        funciones (list): Lista de funciones [f1, f2, ..., fn] donde cada fi(x, Y)
                         define dyi/dx = fi(x, Y) y Y = [y1, y2, ..., yn]
        x0 (float): Valor inicial de x
        y0 (list): Lista de valores iniciales [y1(x0), y2(x0), ..., yn(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de listas, donde Y[i] = [y1_i, y2_i, ..., yn_i]
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> # Sistema: dy1/dx = y2, dy2/dx = -y1
        >>> f1 = lambda x, Y: Y[1]
        >>> f2 = lambda x, Y: -Y[0]
        >>> funciones = [f1, f2]
        >>> X, Y = euler_sistema(funciones, x0=0, y0=[1, 0], xf=6.28, n=100)
    """
    # Validaciones
    if not isinstance(funciones, list) or len(funciones) == 0:
        raise ValueError("funciones debe ser una lista no vacía de funciones")
    
    for i, f in enumerate(funciones):
        if not callable(f):
            raise ValueError(f"funciones[{i}] debe ser callable de la forma f(x, Y)")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if len(funciones) != len(y0):
        raise ValueError(
            f"El número de funciones ({len(funciones)}) debe coincidir "
            f"con el número de condiciones iniciales ({len(y0)})"
        )
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    num_ecuaciones = len(funciones)
    
    if verbose:
        print(f"Método de Euler para Sistemas de EDOs")
        print(f"Sistema de {num_ecuaciones} ecuaciones: dyi/dx = fi(x, y1, y2, ..., yn)")
        print(f"Condiciones iniciales: y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [[0] * num_ecuaciones for _ in range(n + 1)]
    
    X[0] = x0
    Y[0] = list(y0)  # Copiar condiciones iniciales
    
    # Método de Euler para sistemas:
    # Y_{i+1} = Y_i + h * F(x_i, Y_i)
    # donde F(x, Y) = [f1(x,Y), f2(x,Y), ..., fn(x,Y)]
    for i in range(n):
        X[i+1] = X[i] + h
        
        # Calcular cada componente del sistema
        for j in range(num_ecuaciones):
            Y[i+1][j] = Y[i][j] + h * funciones[j](X[i], Y[i])
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def heun_sistema(funciones, x0, y0, xf, n, verbose=True):
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Heun.
    
    Resuelve el sistema usando el método de Heun (predictor-corrector, orden 2).
    Es una extensión del método heun() de edo1.py para sistemas de n ecuaciones.
    
    Args:
        funciones (list): Lista de funciones [f1, f2, ..., fn] donde cada fi(x, Y)
                         define dyi/dx = fi(x, Y) y Y = [y1, y2, ..., yn]
        x0 (float): Valor inicial de x
        y0 (list): Lista de valores iniciales [y1(x0), y2(x0), ..., yn(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de listas, donde Y[i] = [y1_i, y2_i, ..., yn_i]
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> f1 = lambda x, Y: Y[1]
        >>> f2 = lambda x, Y: -Y[0]
        >>> X, Y = heun_sistema([f1, f2], x0=0, y0=[1, 0], xf=6.28, n=100)
    """
    # Validaciones
    if not isinstance(funciones, list) or len(funciones) == 0:
        raise ValueError("funciones debe ser una lista no vacía de funciones")
    
    for i, f in enumerate(funciones):
        if not callable(f):
            raise ValueError(f"funciones[{i}] debe ser callable de la forma f(x, Y)")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if len(funciones) != len(y0):
        raise ValueError(
            f"El número de funciones ({len(funciones)}) debe coincidir "
            f"con el número de condiciones iniciales ({len(y0)})"
        )
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    num_ecuaciones = len(funciones)
    
    if verbose:
        print(f"Método de Heun para Sistemas de EDOs (Predictor-Corrector)")
        print(f"Sistema de {num_ecuaciones} ecuaciones: dyi/dx = fi(x, y1, y2, ..., yn)")
        print(f"Condiciones iniciales: y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [[0] * num_ecuaciones for _ in range(n + 1)]
    
    X[0] = x0
    Y[0] = list(y0)
    
    # Método de Heun para sistemas
    for i in range(n):
        X[i+1] = X[i] + h
        
        # Predictor (Euler): Yp = Y_i + h * F(x_i, Y_i)
        Yp = [0] * num_ecuaciones
        for j in range(num_ecuaciones):
            Yp[j] = Y[i][j] + h * funciones[j](X[i], Y[i])
        
        # Corrector (promedio de pendientes)
        for j in range(num_ecuaciones):
            Y[i+1][j] = Y[i][j] + (h / 2) * (funciones[j](X[i], Y[i]) + funciones[j](X[i+1], Yp))
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def punto_medio_sistema(funciones, x0, y0, xf, n, verbose=True):
    """
    Resuelve un sistema de EDOs de primer orden usando el método del punto medio.
    
    Resuelve el sistema usando el método del punto medio (Runge-Kutta de orden 2).
    Es una extensión del método punto_medio() de edo1.py para sistemas de n ecuaciones.
    
    Args:
        funciones (list): Lista de funciones [f1, f2, ..., fn] donde cada fi(x, Y)
                         define dyi/dx = fi(x, Y) y Y = [y1, y2, ..., yn]
        x0 (float): Valor inicial de x
        y0 (list): Lista de valores iniciales [y1(x0), y2(x0), ..., yn(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de listas, donde Y[i] = [y1_i, y2_i, ..., yn_i]
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> f1 = lambda x, Y: Y[1]
        >>> f2 = lambda x, Y: -Y[0]
        >>> X, Y = punto_medio_sistema([f1, f2], x0=0, y0=[1, 0], xf=6.28, n=100)
    """
    # Validaciones
    if not isinstance(funciones, list) or len(funciones) == 0:
        raise ValueError("funciones debe ser una lista no vacía de funciones")
    
    for i, f in enumerate(funciones):
        if not callable(f):
            raise ValueError(f"funciones[{i}] debe ser callable de la forma f(x, Y)")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if len(funciones) != len(y0):
        raise ValueError(
            f"El número de funciones ({len(funciones)}) debe coincidir "
            f"con el número de condiciones iniciales ({len(y0)})"
        )
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    num_ecuaciones = len(funciones)
    
    if verbose:
        print(f"Método del Punto Medio para Sistemas de EDOs (RK2)")
        print(f"Sistema de {num_ecuaciones} ecuaciones: dyi/dx = fi(x, y1, y2, ..., yn)")
        print(f"Condiciones iniciales: y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [[0] * num_ecuaciones for _ in range(n + 1)]
    
    X[0] = x0
    Y[0] = list(y0)
    
    # Método del punto medio para sistemas
    for i in range(n):
        X[i+1] = X[i] + h
        
        # Calcular Ym en el punto medio
        Xm = X[i] + h / 2
        Ym = [0] * num_ecuaciones
        for j in range(num_ecuaciones):
            Ym[j] = Y[i][j] + (h / 2) * funciones[j](X[i], Y[i])
        
        # Usar la pendiente en el punto medio
        for j in range(num_ecuaciones):
            Y[i+1][j] = Y[i][j] + h * funciones[j](Xm, Ym)
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


def runge_kutta4_sistema(funciones, x0, y0, xf, n, verbose=True):
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Runge-Kutta 4.
    
    Resuelve el sistema usando el método de Runge-Kutta de orden 4 (RK4 clásico).
    Es una extensión del método runge_kutta4() de edo1.py para sistemas de n ecuaciones.
    
    Este es el método más utilizado por su excelente balance entre
    precisión (orden 4) y costo computacional.
    
    Args:
        funciones (list): Lista de funciones [f1, f2, ..., fn] donde cada fi(x, Y)
                         define dyi/dx = fi(x, Y) y Y = [y1, y2, ..., yn]
        x0 (float): Valor inicial de x
        y0 (list): Lista de valores iniciales [y1(x0), y2(x0), ..., yn(x0)]
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de listas, donde Y[i] = [y1_i, y2_i, ..., yn_i]
            
    Raises:
        ValueError: Si los parámetros son inválidos
        
    Examples:
        >>> f1 = lambda x, Y: Y[1]
        >>> f2 = lambda x, Y: -Y[0]
        >>> X, Y = runge_kutta4_sistema([f1, f2], x0=0, y0=[1, 0], xf=6.28, n=50)
        
    Notes:
        El método RK4 calcula 4 pendientes (k1, k2, k3, k4) para cada variable
        y las combina con pesos 1-2-2-1 para obtener una aproximación de orden 4.
    """
    # Validaciones
    if not isinstance(funciones, list) or len(funciones) == 0:
        raise ValueError("funciones debe ser una lista no vacía de funciones")
    
    for i, f in enumerate(funciones):
        if not callable(f):
            raise ValueError(f"funciones[{i}] debe ser callable de la forma f(x, Y)")
    
    if not isinstance(y0, list) or len(y0) == 0:
        raise ValueError("y0 debe ser una lista no vacía de condiciones iniciales")
    
    if len(funciones) != len(y0):
        raise ValueError(
            f"El número de funciones ({len(funciones)}) debe coincidir "
            f"con el número de condiciones iniciales ({len(y0)})"
        )
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    num_ecuaciones = len(funciones)
    
    if verbose:
        print(f"Método de Runge-Kutta 4 para Sistemas de EDOs (RK4 clásico)")
        print(f"Sistema de {num_ecuaciones} ecuaciones: dyi/dx = fi(x, y1, y2, ..., yn)")
        print(f"Condiciones iniciales: y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
        print(f"Precisión: O(h⁴)")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [[0] * num_ecuaciones for _ in range(n + 1)]
    
    X[0] = x0
    Y[0] = list(y0)
    
    # Método de Runge-Kutta 4 para sistemas
    for i in range(n):
        X[i+1] = X[i] + h
        
        # Calcular k1, k2, k3, k4 para cada ecuación
        k1 = [0] * num_ecuaciones
        k2 = [0] * num_ecuaciones
        k3 = [0] * num_ecuaciones
        k4 = [0] * num_ecuaciones
        
        # k1 = F(x_i, Y_i)
        for j in range(num_ecuaciones):
            k1[j] = funciones[j](X[i], Y[i])
        
        # k2 = F(x_i + h/2, Y_i + h*k1/2)
        Y_temp = [Y[i][j] + h * k1[j] / 2 for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k2[j] = funciones[j](X[i] + h/2, Y_temp)
        
        # k3 = F(x_i + h/2, Y_i + h*k2/2)
        Y_temp = [Y[i][j] + h * k2[j] / 2 for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k3[j] = funciones[j](X[i] + h/2, Y_temp)
        
        # k4 = F(x_i + h, Y_i + h*k3)
        Y_temp = [Y[i][j] + h * k3[j] for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k4[j] = funciones[j](X[i] + h, Y_temp)
        
        # Combinar con pesos 1-2-2-1
        for j in range(num_ecuaciones):
            Y[i+1][j] = Y[i][j] + (h / 6) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valores finales: y({xf}) ≈ {[f'{yi:.6f}' for yi in Y[-1]]}")
    
    return X, Y


# Ejemplo de uso
if __name__ == "__main__":
    print("="*70)
    print("Ejemplo 1: Oscilador armónico simple")
    print("Sistema: dy1/dx = y2, dy2/dx = -y1")
    print("Condiciones iniciales: y1(0) = 1, y2(0) = 0")
    print("="*70)
    
    # Definir las funciones del sistema
    def f1(x, Y):
        """dy1/dx = y2"""
        return Y[1]
    
    def f2(x, Y):
        """dy2/dx = -y1"""
        return -Y[0]
    
    funciones = [f1, f2]
    y0 = [1.0, 0.0]
    
    # Resolver el sistema
    X, Y = euler_sistema(funciones, x0=0, y0=y0, xf=6.28, n=20, verbose=True)
    
    # Mostrar algunos resultados
    print("\nTabla de resultados (cada 5 pasos):")
    print(f"{'x':>10} {'y1':>12} {'y2':>12}")
    print("-" * 36)
    for i in range(0, len(X), 5):
        print(f"{X[i]:10.4f} {Y[i][0]:12.6f} {Y[i][1]:12.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 2: Sistema lineal 2x2")
    print("Sistema: dy1/dx = y1 + 2*y2, dy2/dx = 3*y1 + 2*y2")
    print("Condiciones iniciales: y1(0) = 1, y2(0) = 0")
    print("="*70)
    
    def g1(x, Y):
        """dy1/dx = y1 + 2*y2"""
        return Y[0] + 2*Y[1]
    
    def g2(x, Y):
        """dy2/dx = 3*y1 + 2*y2"""
        return 3*Y[0] + 2*Y[1]
    
    funciones2 = [g1, g2]
    y0_2 = [1.0, 0.0]
    
    X2, Y2 = euler_sistema(funciones2, x0=0, y0=y0_2, xf=1, n=10, verbose=True)
    
    print("\nTabla de resultados:")
    print(f"{'x':>10} {'y1':>12} {'y2':>12}")
    print("-" * 36)
    for i in range(len(X2)):
        print(f"{X2[i]:10.4f} {Y2[i][0]:12.6f} {Y2[i][1]:12.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 3: Sistema de 3 ecuaciones")
    print("Sistema: dy1/dx = -y1 + y2")
    print("         dy2/dx = y1 - y2 + y3")
    print("         dy3/dx = -y2 - y3")
    print("Condiciones iniciales: y1(0) = 1, y2(0) = 0, y3(0) = 0")
    print("="*70)
    
    def h1(x, Y):
        """dy1/dx = -y1 + y2"""
        return -Y[0] + Y[1]
    
    def h2(x, Y):
        """dy2/dx = y1 - y2 + y3"""
        return Y[0] - Y[1] + Y[2]
    
    def h3(x, Y):
        """dy3/dx = -y2 - y3"""
        return -Y[1] - Y[2]
    
    funciones3 = [h1, h2, h3]
    y0_3 = [1.0, 0.0, 0.0]
    
    X3, Y3 = euler_sistema(funciones3, x0=0, y0=y0_3, xf=5, n=25, verbose=True)
    
    print("\nTabla de resultados (cada 5 pasos):")
    print(f"{'x':>10} {'y1':>12} {'y2':>12} {'y3':>12}")
    print("-" * 48)
    for i in range(0, len(X3), 5):
        print(f"{X3[i]:10.4f} {Y3[i][0]:12.6f} {Y3[i][1]:12.6f} {Y3[i][2]:12.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 4: Modelo Lotka-Volterra (depredador-presa)")
    print("Sistema: dx/dt = 1.0*x - 0.5*x*y  (presas)")
    print("         dy/dt = 0.5*x*y - 1.0*y  (depredadores)")
    print("Condiciones iniciales: x(0) = 2, y(0) = 1")
    print("="*70)
    
    # Parámetros del modelo
    alpha = 1.0   # Tasa de crecimiento de presas
    beta = 0.5    # Tasa de depredación
    delta = 0.5   # Eficiencia de conversión
    gamma = 1.0   # Tasa de mortalidad de depredadores
    
    def presas(t, Y):
        """dx/dt = alpha*x - beta*x*y"""
        x, y = Y[0], Y[1]
        return alpha * x - beta * x * y
    
    def depredadores(t, Y):
        """dy/dt = delta*x*y - gamma*y"""
        x, y = Y[0], Y[1]
        return delta * x * y - gamma * y
    
    funciones4 = [presas, depredadores]
    y0_4 = [2.0, 1.0]
    
    X4, Y4 = euler_sistema(funciones4, x0=0, y0=y0_4, xf=20, n=200, verbose=True)
    
    print("\nTabla de resultados (cada 50 pasos):")
    print(f"{'t':>10} {'Presas':>12} {'Depredadores':>15}")
    print("-" * 39)
    for i in range(0, len(X4), 50):
        print(f"{X4[i]:10.4f} {Y4[i][0]:12.6f} {Y4[i][1]:15.6f}")
    
    print("\n" + "="*70)
    print("Ejemplo 5: Comparación de métodos - Oscilador armónico")
    print("Sistema: dy1/dx = y2, dy2/dx = -y1")
    print("Condiciones iniciales: y1(0) = 1, y2(0) = 0")
    print("Solución exacta: y1(x) = cos(x)")
    print("="*70)
    
    # Definir funciones
    f1_comp = lambda x, Y: Y[1]
    f2_comp = lambda x, Y: -Y[0]
    funciones_comp = [f1_comp, f2_comp]
    y0_comp = [1.0, 0.0]
    
    # Resolver con los 4 métodos
    import math
    
    X_euler, Y_euler = euler_sistema(funciones_comp, 0, y0_comp, math.pi, 10, verbose=False)
    X_heun, Y_heun = heun_sistema(funciones_comp, 0, y0_comp, math.pi, 10, verbose=False)
    X_pm, Y_pm = punto_medio_sistema(funciones_comp, 0, y0_comp, math.pi, 10, verbose=False)
    X_rk4, Y_rk4 = runge_kutta4_sistema(funciones_comp, 0, y0_comp, math.pi, 10, verbose=False)
    
    print("\nComparación de errores en x = π (valor exacto: -1.0):")
    print(f"{'Método':>20} {'Aproximación':>15} {'Error':>15}")
    print("-" * 52)
    
    y_exacto = math.cos(math.pi)
    print(f"{'Exacto':>20} {y_exacto:15.10f} {0.0:15.10f}")
    print(f"{'Euler':>20} {Y_euler[-1][0]:15.10f} {abs(Y_euler[-1][0] - y_exacto):15.10f}")
    print(f"{'Heun':>20} {Y_heun[-1][0]:15.10f} {abs(Y_heun[-1][0] - y_exacto):15.10f}")
    print(f"{'Punto Medio':>20} {Y_pm[-1][0]:15.10f} {abs(Y_pm[-1][0] - y_exacto):15.10f}")
    print(f"{'Runge-Kutta 4':>20} {Y_rk4[-1][0]:15.10f} {abs(Y_rk4[-1][0] - y_exacto):15.10f}")
    
    print("\nObservación: RK4 es el más preciso, seguido de Heun y Punto Medio,")
    print("mientras que Euler tiene el mayor error.")
