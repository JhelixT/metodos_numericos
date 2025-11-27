import os
import numpy as np

def main():
    print("=" * 80)
    print("INTERPOLACIÓN POLINÓMICA Y SPLINES CÚBICOS")
    print("=" * 80)
    
    t = [1, 3, 5, 7, 13]
    vel = [800, 2310, 3090, 3940, 4755]
    
    print("DATOS DE ENTRADA:")
    print("  Tiempo (t):     ", [f"{x:4.0f}" for x in t])
    print("  Velocidad (v):  ", [f"{y:4.0f}" for y in vel])
    print(f"  Número de puntos: {len(t)}")
    print("-" * 80)
    
    # Interpolación de Lagrange
    print("INTERPOLACIÓN DE LAGRANGE:")
    val_t4 = interpolacion_lagrange(4, t, vel)
    val_t6 = interpolacion_lagrange(6, t, vel)
    print(f"  v(t=4) = {val_t4:8.2f} cm/s")
    print(f"  v(t=6) = {val_t6:8.2f} cm/s")
    
    print("\nPOLINOMIO DE LAGRANGE:")
    p, coef, t, vel = interpolacion(t, vel)
    polinomio_str = ""
    for i in range(len(coef)):
        if i == 0:
            polinomio_str += f"{coef[i]:8.4f}"
        else:
            signo = "+" if coef[i] >= 0 else ""
            polinomio_str += f" {signo}{coef[i]:8.4f}x^{i}"
    print(f"  P(x) = {polinomio_str}")
    
    print("\n" + "=" * 80)
    print("INTERPOLACIÓN CON SPLINES CÚBICOS NATURALES")
    print("-" * 80)
    
    funciones_spline, coef_spline, t, vel = curvas_spline(t, vel)
    val_spline = evaluar_spline(10, funciones_spline, t)
    print(f"Interpolación para t=10: {val_spline:8.2f} cm/s")
    
    intervalo_spline = buscar_intervalo(10, t)
    print(f"\nFUNCIÓN SPLINE UTILIZADA (Intervalo [{t[intervalo_spline]}, {t[intervalo_spline+1]}]):")
    coef_intervalo = coef_spline[intervalo_spline]
    spline_str = f"  S_{intervalo_spline}(x) = {coef_intervalo[0]:8.6f}x³"
    spline_str += f" + {coef_intervalo[1]:8.6f}x²"
    spline_str += f" + {coef_intervalo[2]:8.6f}x"
    spline_str += f" + {coef_intervalo[3]:8.6f}"
    print(spline_str)
    print("=" * 80)
    

    
def _ordenar_puntos_xy(X, Y):
    """
    Ordena los puntos (X,Y) según los valores de X de forma ascendente.
    
    Args:
        X (list[float]): Coordenadas x
        Y (list[float]): Coordenadas y
        
    Returns:
        tuple[list[float], list[float]]: Puntos ordenados (X_ord, Y_ord)
        
    Raises:
        ValueError: Si hay valores duplicados en X
    """
    n = len(X)
    
    # Crear lista de índices ordenados según X
    indices_ordenados = sorted(range(n), key=lambda i: X[i])
    
    # Aplicar el ordenamiento
    X_ordenado = [X[i] for i in indices_ordenados]
    Y_ordenado = [Y[i] for i in indices_ordenados]
    
    # Verificar puntos duplicados
    for i in range(n-1):
        if abs(X_ordenado[i+1] - X_ordenado[i]) < 1e-12:
            raise ValueError(f"Valores duplicados en X: X[{i}] = X[{i+1}] = {X_ordenado[i]}. "
                           "Se requieren valores únicos para interpolación.")
    
    # Mostrar advertencia si se reordenó
    if indices_ordenados != list(range(n)):
        print("⚠️  NOTA: Los puntos fueron reordenados automáticamente según X.")
    
    return X_ordenado, Y_ordenado

def interpolacion_lagrange(input_x, X=None, Y=None, nombre_archivo=None):
    """
    Calcula el valor interpolado para un punto dado usando el método de Lagrange.
    
    Args:
        input_x (float): Punto donde se desea interpolar
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        float: Valor interpolado en input_x
        
    Raises:
        ValueError: Si los datos de entrada no son válidos
    """
    n = len(X)

    suma = 0
    for i in range(n):
        prod = 1
        for j in range(n):
            if i != j:
                prod = prod * (input_x - X[j])/(X[i]-X[j])
        suma = suma + Y[i]*prod
    return suma

def interpolacion(X=None, Y=None, nombre_archivo=None, ordenar_automatico=True):
    """
    Realiza interpolación polinómica usando el método de Vandermonde.
    Construye un polinomio que pasa exactamente por todos los puntos dados.

    Args:
        X (list[float], optional): Lista de coordenadas x de los puntos.
        Y (list[float], optional): Lista de coordenadas y de los puntos.
        nombre_archivo (str, optional): Archivo con los puntos a interpolar.
            Si se proporciona, los puntos se leerán del archivo ignorando X e Y.

    Returns:
        tuple:
            - callable: Función del polinomio interpolador
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos

    Raises:
        ValueError: Si no se proporcionan ni puntos ni archivo, o si X e Y tienen diferente longitud

    Note:
        - Usa la matriz de Vandermonde: A[i][j] = x_i^j
        - Resuelve el sistema usando eliminación gaussiana
        - El grado del polinomio es n-1, donde n es el número de puntos
        - Puede ser numéricamente inestable para muchos puntos

    Example:
        # Usando arrays directamente
        X = [1, 2, 3]
        Y = [1, 4, 9]
        p, coef, X, Y = interpolacion(X=X, Y=Y)

        # Usando un archivo
        p, coef, X, Y = interpolacion(nombre_archivo='datos.txt')
    """
    
    if ordenar_automatico:
        X, Y = _ordenar_puntos_xy(X, Y)

    n = len(X)

    A = [[0 for j in range(n)] for i in range(n)]
    B = [0 for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = X[i]**j
        B[i] = Y[i]

    coef = gauss_pivot(A, B)  # coef[0] = x^0, coef[1] = x^1, ...

    # función polinómica evaluable
    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    return p, coef, X, Y
def triangulacion(A, B):
    """
    Realiza la triangulación superior de una matriz aumentada [A|B] usando eliminación gaussiana con pivoteo.
    
    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes

    Note:
        - Modifica las matrices A y B in-place
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - El pivoteo selecciona el elemento más grande en valor absoluto de cada columna
        - Evita problemas con elementos diagonales cercanos a cero
    """
    n = len(A)
    for i in range(n-1):
        p=i
        if abs(A[i][i]) < 1e-2:
            for l in range(i+1, n):
                if abs(A[l][i]) > abs(A[p][i]):
                    p=l

            for m in range(i, n):
                A[p][m] , A[i][m] = A[i][m], A[p][m]

            B[p], B[i] = B[i], B[p]

        for j in range (i+1, n):
            factor = -A[j][i]/A[i][i]
            for k in range(i, n):
                A[j][k] = A[i][k]*factor + A[j][k]

            B[j] = B[i]*factor + B[j]

    #pp.pprint(A)
    print()

def determinante(A):
    """
    Calcula el determinante de una matriz triangular superior.
    Para una matriz triangular, el determinante es el producto de los elementos diagonales.

    Args:
        A (list[list[float]]): Matriz triangular superior

    Note:
        - Asume que la matriz ya está en forma triangular superior
        - Un determinante cero indica que el sistema no tiene solución única
        - Se usa después de la triangulación en el método de Gauss
    """
    n = len(A)
    prod = 1
    for i in range(n):
        prod*=A[i][i]
    if prod==0:
        print("Matriz determinante 0\n")
        return
    # print("Determinante = ",prod,"\n")

def gauss_pivot(A, B):
    """
    Resuelve un sistema de ecuaciones lineales usando eliminación gaussiana con pivoteo.
    El método consiste en tres pasos: triangulación, verificación de determinante y sustitución hacia atrás.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes

    Returns:
        list[float]: Vector solución del sistema

    Note:
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - Verifica si el sistema tiene solución única (determinante ≠ 0)
        - La sustitución hacia atrás resuelve el sistema triangular superior
    """
    n = len(A)
    X = [0] * n
    triangulacion(A, B)
    determinante(A)

    X[n-1] = B[n-1]/A[n-1][n-1]

    for i in range(n-1, -1, -1):
        sum = B[i]
        for j in range(i+1, n):
            sum = sum - A[i][j]*X[j]
        sum = sum/A[i][i]

        X[i] = sum
    #print("Las soluciones del sistema son: ")
    #for i in range(n):
        #print(f"X{i} = {X[i]}")
    return X
def limpiar_terminal():
    """
    Limpia la pantalla de la terminal de manera compatible con múltiples sistemas operativos.
    Utiliza el comando 'clear' en sistemas Unix/Linux/Mac y 'cls' en Windows.
    """
    os.system('clear' if os.name == 'posix' else 'cls')

def curvas_spline(X=None, Y=None, nombre_archivo=None):
    """
    Construye splines cúbicos naturales para interpolar puntos dados.
    
    Para n puntos tenemos n-1 intervalos, cada uno con un polinomio cúbico:
    S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k para x en [X_k, X_{k+1}]
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y  
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        tuple: (funciones_spline, coeficientes, X, Y)
            - funciones_spline: lista de funciones para cada intervalo
            - coeficientes: matriz de coeficientes [a_k, b_k, c_k, d_k] para cada k
                donde S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k
            - X, Y: puntos originales
    """
    X, Y = _ordenar_puntos_xy(X, Y)  # Ordenar automáticamente
    n = len(X)
    
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para construir splines cúbicos")
    
    num_intervalos = n - 1
    num_coef = 4 * num_intervalos
    
    A = [[0 for i in range(num_coef)] for j in range(num_coef)]
    B = [0 for i in range(num_coef)]
    
    fila = 0  # Contador de filas para asegurar que no se sobreescriban
    
    # 1. Ecuaciones de evaluación: 2*(n-1) ecuaciones
    # S_k(X_k) = Y_k y S_k(X_{k+1}) = Y_{k+1} para k = 0, 1, ..., n-2
    for k in range(num_intervalos):
        # S_k(X_k) = Y_k
        for j in range(4):
            A[fila][4*k + j] = X[k]**(3-j)  # a_k*X_k³ + b_k*X_k² + c_k*X_k + d_k
        B[fila] = Y[k]
        fila += 1
        
        # S_k(X_{k+1}) = Y_{k+1}
        for j in range(4):
            A[fila][4*k + j] = X[k+1]**(3-j)
        B[fila] = Y[k+1]
        fila += 1
    
    # 2. Ecuaciones de continuidad de primera derivada: (n-2) ecuaciones
    # S'_k(X_{k+1}) = S'_{k+1}(X_{k+1}) para k = 0, 1, ..., n-3
    for k in range(num_intervalos - 1):
        # S'_k(x) = 3*a_k*x² + 2*b_k*x + c_k
        # S'_{k+1}(x) = 3*a_{k+1}*x² + 2*b_{k+1}*x + c_{k+1}
        x_punto = X[k+1]
        
        # Coeficientes de S'_k(X_{k+1})
        A[fila][4*k] = 3*x_punto**2      # 3*a_k*x²
        A[fila][4*k + 1] = 2*x_punto     # 2*b_k*x
        A[fila][4*k + 2] = 1             # c_k
        
        # Coeficientes de -S'_{k+1}(X_{k+1})
        A[fila][4*(k+1)] = -3*x_punto**2     # -3*a_{k+1}*x²
        A[fila][4*(k+1) + 1] = -2*x_punto   # -2*b_{k+1}*x
        A[fila][4*(k+1) + 2] = -1           # -c_{k+1}
        
        B[fila] = 0
        fila += 1
    
    # 3. Ecuaciones de continuidad de segunda derivada: (n-2) ecuaciones
    # S''_k(X_{k+1}) = S''_{k+1}(X_{k+1}) para k = 0, 1, ..., n-3
    for k in range(num_intervalos - 1):
        # S''_k(x) = 6*a_k*x + 2*b_k
        # S''_{k+1}(x) = 6*a_{k+1}*x + 2*b_{k+1}
        x_punto = X[k+1]
        
        # Coeficientes de S''_k(X_{k+1})
        A[fila][4*k] = 6*x_punto      # 6*a_k*x
        A[fila][4*k + 1] = 2          # 2*b_k
        
        # Coeficientes de -S''_{k+1}(X_{k+1})
        A[fila][4*(k+1)] = -6*x_punto     # -6*a_{k+1}*x
        A[fila][4*(k+1) + 1] = -2         # -2*b_{k+1}
        
        B[fila] = 0
        fila += 1
    
    # 4. Condiciones de frontera naturales: 2 ecuaciones
    # S''_0(X_0) = 0 y S''_{n-2}(X_{n-1}) = 0
    
    # S''_0(X_0) = 0: 6*a_0*X_0 + 2*b_0 = 0
    A[fila][0] = 6*X[0]      # 6*a_0*X_0
    A[fila][1] = 2           # 2*b_0
    B[fila] = 0
    fila += 1
    
    # S''_{n-2}(X_{n-1}) = 0: 6*a_{n-2}*X_{n-1} + 2*b_{n-2} = 0
    last_interval = num_intervalos - 1
    A[fila][4*last_interval] = 6*X[n-1]       # 6*a_{n-2}*X_{n-1}
    A[fila][4*last_interval + 1] = 2          # 2*b_{n-2}
    B[fila] = 0
    fila += 1
    
    # Resolver el sistema
    coeficientes_planos = gauss_pivot(A, B)
    
    # Reorganizar coeficientes por intervalo
    coeficientes = []
    for k in range(num_intervalos):
        a_k = coeficientes_planos[4*k]      # coeficiente de x³
        b_k = coeficientes_planos[4*k + 1]  # coeficiente de x²
        c_k = coeficientes_planos[4*k + 2]  # coeficiente de x
        d_k = coeficientes_planos[4*k + 3]  # término constante
        coeficientes.append([a_k, b_k, c_k, d_k])
    
    # Crear funciones evaluables para cada intervalo
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
    # Verificar que x está en el dominio
    if x < X[0] or x > X[-1]:
        raise ValueError(f"El punto x={x} está fuera del dominio [{X[0]}, {X[-1]}]")
    
    # Buscar el intervalo correcto usando búsqueda lineal
    for k in range(len(X) - 1):
        if X[k] <= x <= X[k+1]:
            return funciones_spline[k](x)
    
    # Si no se encuentra (caso edge), usar el último intervalo
    return funciones_spline[-1](x)

def buscar_intervalo(x, X):
    """
    Encuentra el índice del intervalo [X_k, X_{k+1}] que contiene el punto x.
    
    Args:
        x (float): Punto a buscar
        X (list[float]): Puntos ordenados (nodos del spline)
    
    Returns:
        int: Índice k del intervalo [X_k, X_{k+1}] que contiene x
        
    Raises:
        ValueError: Si x está fuera del dominio
    """
    if x < X[0] or x > X[-1]:
        raise ValueError(f"El punto x={x} está fuera del dominio [{X[0]}, {X[-1]}]")
    
    # Método 1: Búsqueda lineal (O(n))
    # Adecuado para pocos puntos o evaluaciones esporádicas
    for k in range(len(X) - 1):
        if X[k] <= x <= X[k+1]:
            return k
    
    # Si x == X[-1] exactamente
    return len(X) - 2


main()