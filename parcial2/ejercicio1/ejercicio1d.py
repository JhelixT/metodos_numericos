def main():
    Xspline=[0, 0.25, 0.5, 0.75, 1]
    Yspline=[0, 0.3166, 0.7905, 1.4433, 2.2873]
    X = [0.00, 0.20, 0.43, 0.55, 0.85, 0.90, 1.00]
    Y = [0.00, 0.2426, 0.6408, 0.9059, 1.7577, 1.9267, 2.2873]

    integralTabla=simpson(X=X, Y=Y, n=4, verbose=True)
    integralSpline=simpson(X=Xspline, Y=Yspline, verbose=True)
    #print(f"El resultado de la integral con simpson compuesta y datos tabulados es: {integralTabla:.10f}")
    #print(f"El error absoluto es: {abs(0.9093306736-integralTabla):.10f}")

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
def curvas_spline(X=None, Y=None, nombre_archivo=None, verbose=True):
    """
    Construye splines cúbicos naturales para interpolar puntos dados.
    
    Para n puntos tenemos n-1 intervalos, cada uno con un polinomio cúbico:
    S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k para x en [X_k, X_{k+1}]
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y  
        nombre_archivo (str, optional): Ruta al archivo de datos
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.
        
    Returns:
        tuple: (funciones_spline, coeficientes, X, Y)
            - funciones_spline: lista de funciones para cada intervalo
            - coeficientes: matriz de coeficientes [a_k, b_k, c_k, d_k] para cada k
            - X, Y: puntos originales ordenados
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    X, Y = _ordenar_puntos_xy(X, Y, verbose=verbose)
    n = len(X)
    
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para construir splines cúbicos")
    
    num_intervalos = n - 1
    num_coef = 4 * num_intervalos
    
    A = [[0 for i in range(num_coef)] for j in range(num_coef)]
    B = [0 for i in range(num_coef)]
    
    fila = 0
    
    # 1. Ecuaciones de evaluación: 2*(n-1) ecuaciones
    for k in range(num_intervalos):
        for j in range(4):
            A[fila][4*k + j] = X[k]**(3-j)
        B[fila] = Y[k]
        fila += 1
        
        for j in range(4):
            A[fila][4*k + j] = X[k+1]**(3-j)
        B[fila] = Y[k+1]
        fila += 1
    
    # 2. Ecuaciones de continuidad de primera derivada: (n-2) ecuaciones
    for k in range(num_intervalos - 1):
        x_punto = X[k+1]
        
        A[fila][4*k] = 3*x_punto**2
        A[fila][4*k + 1] = 2*x_punto
        A[fila][4*k + 2] = 1
        
        A[fila][4*(k+1)] = -3*x_punto**2
        A[fila][4*(k+1) + 1] = -2*x_punto
        A[fila][4*(k+1) + 2] = -1
        
        B[fila] = 0
        fila += 1
    
    # 3. Ecuaciones de continuidad de segunda derivada: (n-2) ecuaciones
    for k in range(num_intervalos - 1):
        x_punto = X[k+1]
        
        A[fila][4*k] = 6*x_punto
        A[fila][4*k + 1] = 2
        
        A[fila][4*(k+1)] = -6*x_punto
        A[fila][4*(k+1) + 1] = -2
        
        B[fila] = 0
        fila += 1
    
    # 4. Condiciones de frontera naturales: 2 ecuaciones
    A[fila][0] = 6*X[0]
    A[fila][1] = 2
    B[fila] = 0
    fila += 1
    
    last_interval = num_intervalos - 1
    A[fila][4*last_interval] = 6*X[n-1]
    A[fila][4*last_interval + 1] = 2
    B[fila] = 0
    fila += 1
    
    coeficientes_planos = gauss_pivot(A, B, verbose=False)
    
    coeficientes = []
    for k in range(num_intervalos):
        a_k = coeficientes_planos[4*k]
        b_k = coeficientes_planos[4*k + 1]
        c_k = coeficientes_planos[4*k + 2]
        d_k = coeficientes_planos[4*k + 3]
        coeficientes.append([a_k, b_k, c_k, d_k])
    
    funciones_spline = []
    for k in range(num_intervalos):
        a_k, b_k, c_k, d_k = coeficientes[k]
        def crear_spline(a, b, c, d):
            return lambda x: a*x**3 + b*x**2 + c*x + d
        funciones_spline.append(crear_spline(a_k, b_k, c_k, d_k))
    
    return funciones_spline, coeficientes, X, Y


def evaluar_spline(x, funciones_spline, X):
    """
    Evalúa el spline en un punto x, encontrando automáticamente el intervalo correcto.
    
    Args:
        x (float): Punto donde evaluar el spline
        funciones_spline (list[callable]): Lista de funciones spline por intervalo
        X (list[float]): Puntos de interpolación (nodos)
    
    Returns:
        float: Valor del spline en x
        
    Raises:
        ValueError: Si x está fuera del dominio de los splines
    """
    if x < X[0] or x > X[-1]:
        raise ValueError(f"El punto x={x} está fuera del dominio [{X[0]}, {X[-1]}]")
    
    for k in range(len(X) - 1):
        if X[k] <= x <= X[k+1]:
            return funciones_spline[k](x)
    
    return funciones_spline[-1](x)
def _validar_datos_xy(X=None, Y=None, nombre_archivo=None):
    """
    Función auxiliar para validar y preparar datos de entrada X,Y.
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        tuple[list[float], list[float]]: Datos X,Y validados y convertidos a float
        
    Raises:
        ValueError: Si los datos de entrada no son válidos
    """
    if nombre_archivo is not None:
        X, Y = leer_puntos_xy(nombre_archivo)
    elif X is not None and Y is not None:
        if len(X) != len(Y):
            raise ValueError("Las listas X e Y deben tener la misma longitud")
        X = list(map(float, X))
        Y = list(map(float, Y))
    else:
        raise ValueError("Debe proporcionar arrays X e Y, o un nombre de archivo")
    return X, Y


def _ordenar_puntos_xy(X, Y, verbose=True):
    """
    Ordena los puntos (X,Y) según los valores de X de forma ascendente.
    
    Args:
        X (list[float]): Coordenadas x
        Y (list[float]): Coordenadas y
        verbose (bool, optional): Si True, muestra advertencias. Por defecto True.
        
    Returns:
        tuple[list[float], list[float]]: Puntos ordenados (X_ord, Y_ord)
        
    Raises:
        ValueError: Si hay valores duplicados en X
    """
    n = len(X)
    
    indices_ordenados = sorted(range(n), key=lambda i: X[i])
    
    X_ordenado = [X[i] for i in indices_ordenados]
    Y_ordenado = [Y[i] for i in indices_ordenados]
    
    for i in range(n-1):
        if abs(X_ordenado[i+1] - X_ordenado[i]) < 1e-12:
            raise ValueError(f"Valores duplicados en X: X[{i}] = X[{i+1}] = {X_ordenado[i]}. "
                           "Se requieren valores únicos para interpolación.")
    
    if verbose and indices_ordenados != list(range(n)):
        print("⚠️  ADVERTENCIA: Los puntos se reordenaron según X.")
        print(f"   Original: X = {X}")
        print(f"   Ordenado: X = {X_ordenado}")
    
    return X_ordenado, Y_ordenado


def leer_puntos_xy(nombre_archivo):
    """
    Lee puntos (x,y) desde un archivo de texto.
    El archivo debe contener dos columnas de números separados por coma o espacio.

    Args:
        nombre_archivo (str): Ruta al archivo que contiene los puntos

    Returns:
        tuple[list[float], list[float]]: Listas de coordenadas x e y

    Raises:
        ValueError: Si una línea del archivo no contiene exactamente dos valores
    """
    xs, ys = [], []
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.replace(",", " ").split()
            if len(partes) != 2:
                raise ValueError(f"Línea inválida: {linea}")
            x, y = map(float, partes)
            xs.append(x)
            ys.append(y)
    return xs, ys


def triangulacion(A, B, verbose=True):
    """
    Realiza la triangulación superior de una matriz aumentada [A|B] usando eliminación gaussiana con pivoteo.
    
    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.

    Returns:
        None: Modifica A y B in-place

    Note:
        - Modifica las matrices A y B in-place
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - El pivoteo selecciona el elemento más grande en valor absoluto de cada columna
        - Evita problemas con elementos diagonales cercanos a cero
    """
    n = len(A)
    for i in range(n-1):
        p = i
        if abs(A[i][i]) < 1e-2:
            for l in range(i+1, n):
                if abs(A[l][i]) > abs(A[p][i]):
                    p = l

            for m in range(i, n):
                A[p][m], A[i][m] = A[i][m], A[p][m]

            B[p], B[i] = B[i], B[p]

        for j in range(i+1, n):
            factor = -A[j][i]/A[i][i]
            for k in range(i, n):
                A[j][k] = A[i][k]*factor + A[j][k]

            B[j] = B[i]*factor + B[j]
    
    if verbose:
        print()


def determinante(A, verbose=True):
    """
    Calcula el determinante de una matriz triangular superior.
    Para una matriz triangular, el determinante es el producto de los elementos diagonales.

    Args:
        A (list[list[float]]): Matriz triangular superior
        verbose (bool, optional): Si True, imprime el resultado. Por defecto True.

    Returns:
        float: Determinante de la matriz

    Note:
        - Asume que la matriz ya está en forma triangular superior
        - Un determinante cero indica que el sistema no tiene solución única
        - Se usa después de la triangulación en el método de Gauss
    """
    n = len(A)
    prod = 1
    for i in range(n):
        prod *= A[i][i]
    
    if prod == 0:
        if verbose:
            print("Matriz determinante 0\n")
    elif verbose:
        print("Determinante =", prod, "\n")
    
    return prod


def gauss_pivot(A, B, verbose=True):
    """
    Resuelve un sistema de ecuaciones lineales usando eliminación gaussiana con pivoteo.
    El método consiste en tres pasos: triangulación, verificación de determinante y sustitución hacia atrás.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.

    Returns:
        list[float]: Vector solución del sistema

    Note:
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - Verifica si el sistema tiene solución única (determinante ≠ 0)
        - La sustitución hacia atrás resuelve el sistema triangular superior
    """
    n = len(A)
    X = [0] * n
    triangulacion(A, B, verbose=False)
    det = determinante(A, verbose=verbose)
    
    if det == 0:
        if verbose:
            print("Sistema sin solución única")
        return X

    X[n-1] = B[n-1]/A[n-1][n-1]

    for i in range(n-1, -1, -1):
        suma = B[i]
        for j in range(i+1, n):
            suma = suma - A[i][j]*X[j]
        suma = suma/A[i][i]
        X[i] = suma
    
    if verbose:
        print("Las soluciones del sistema son: ")
        for i in range(n):
            print(f"X{i} = {X[i]}")
    
    return X

main()