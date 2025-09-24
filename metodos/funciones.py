import math
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

def limpiar_terminal():
    """
    Limpia la pantalla de la terminal de manera compatible con múltiples sistemas operativos.
    Utiliza el comando 'clear' en sistemas Unix/Linux/Mac y 'cls' en Windows.
    """
    os.system('clear' if os.name == 'posix' else 'cls')

def resolverJG(A, B):
    """
    Resuelve un sistema de ecuaciones lineales utilizando métodos iterativos (Jacobi o Gauss-Seidel).
    
    Args:
        A (list[list[float]]): Matriz de coeficientes del sistema
        B (list[float]): Vector de términos independientes
    
    Returns:
        list[float]: Vector solución del sistema

    Note:
        La función verifica si la matriz es diagonalmente dominante antes de proceder.
        Permite elegir entre el método de Jacobi o Gauss-Seidel.
    """
    esDiagDom(A)

    n = len(A)
    Xn = [0] * n  # Vector de soluciones nuevas
    Xv = [0] * n  # Vector de soluciones viejas

    print("1-Resolver con Jacobi")
    print("2-Resolver con Gauss-Seidel")
    opcion = int(input())
    if opcion == 1:
        jacobi(A, B, Xn, Xv)
    elif opcion == 2:
        gauss_seidel(A,B, Xn, Xv)
    
    return Xn
    
def esDiagDom(A):
    """
    Verifica si una matriz es diagonalmente dominante.
    Una matriz es diagonalmente dominante si el valor absoluto de cada elemento diagonal
    es mayor que la suma de los valores absolutos de los demás elementos en su fila.

    Args:
        A (list[list[float]]): Matriz cuadrada a verificar

    Returns:
        None: Imprime un mensaje si la matriz no es diagonalmente dominante
    
    Note:
        Esta propiedad es importante para garantizar la convergencia de métodos iterativos
        como Jacobi y Gauss-Seidel.
    """
    n = len(A)
    suma = 0
    for j in range(1, n):
        suma = suma + abs(A[0][j])
    if abs(A[0][0]) < suma:
        print("La matriz no es Diagonalmente Dominante")
        return
    for i in range(1, n):
        suma = 0
        for j in range (n):
            if i != j:
                suma = suma + abs(A[i][j])
        if abs(A[i][i]) < suma:
            print("La matriz no es Diagonalmente Dominante")
            return
        
def jacobi(A, B, Xn, Xv):
    """
    Implementa el método iterativo de Jacobi para resolver sistemas de ecuaciones lineales.
    El método de Jacobi actualiza cada componente de la solución usando los valores de la iteración anterior.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior

    Note:
        - Requiere que la matriz sea diagonalmente dominante para garantizar convergencia
        - En cada iteración, calcula x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - Se detiene cuando el error es menor que la tolerancia especificada
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
    n = len(A)
    count = 0
    errorV = 1000
    tolerancia = float(input("Ingrese la tolerancia: "))
    while True:
        count+=1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i!=j:
                    suma = suma + A[i][j]*Xv[j]
            Xn[i] = (B[i] - suma)/A[i][i]
        error = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
        error = math.sqrt(error) 
        if error > errorV:
            print("El metodo no converge")
            return
        errorV = error
        for i in range(n):
            Xv[i]=Xn[i]
        if error < tolerancia:
            break;
    print("Las soluciones son:")
    for i in range(n):
        print(f"X{i} = {Xn[i]}")
    print(f"Con un error de {error}")
    print(f"Se resolvio en {count} iteraciones")

def gauss_seidel(A, B, Xn, Xv, omega=1):
    """
    Implementa el método iterativo de Gauss-Seidel con factor de relajación (SOR) para resolver sistemas de ecuaciones lineales.
    A diferencia de Jacobi, utiliza los valores actualizados tan pronto como estén disponibles.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior
        omega (float, optional): Factor de relajación. Defaults to 1.
            - omega < 1: sub-relajación
            - omega = 1: método de Gauss-Seidel estándar
            - omega > 1: sobre-relajación

    Note:
        - Generalmente converge más rápido que Jacobi
        - Usa valores actualizados inmediatamente: x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - El factor de relajación puede acelerar la convergencia
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
    n = len(A)
    count = 0
    errorV = 1000
    tolerancia = float(input("Ingrese la tolerancia: "))
    while True:
        count+=1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    if j < i:
                        suma += A[i][j] * Xn[j]   # ya actualizado
                    else:
                        suma += A[i][j] * Xv[j]   # valor viejo
            Xn[i] = (B[i] - suma) / A[i][i]
    
            Xn[i] = omega*Xn[i] + (1-omega)*Xv[i] #Factor de Relajacion, por default omega = 1

        error = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
        error = math.sqrt(error) 
        if error > errorV:
            print("El metodo no converge")
            return
        errorV = error
        for i in range(n):
            Xv[i]=Xn[i]
        if error < tolerancia:
            break;
    print("Las soluciones son:")
    for i in range(n):
        print(f"X{i} = {Xn[i]}")
    print(f"Con un error de {error}")
    print(f"Se resolvio en {count} iteraciones")


def buscar_raiz(f, a, b, tipo_error, tolerancia, metodo=None):
    """
    Encuentra una raíz de una función en un intervalo dado usando el método de Bisección o Regula Falsi.

    Args:
        f (callable): Función continua de la cual se busca la raíz
        a (float): Límite inferior del intervalo
        b (float): Límite superior del intervalo
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual
        tolerancia (float): Tolerancia deseada para el error
        metodo (int, optional): Método a utilizar
            1: Bisección - divide el intervalo por la mitad
            2: Regula Falsi - usa interpolación lineal
            None: Permite al usuario elegir el método

    Returns:
        tuple[float, list[float]]: Tupla con la raíz encontrada y lista de errores en cada iteración

    Note:
        - Requiere que f(a) y f(b) tengan signos opuestos (teorema del valor intermedio)
        - Bisección: convergencia garantizada pero lenta
        - Regula Falsi: convergencia más rápida pero puede ser lenta en algunos casos
    """
    if f(a)*f(b) > 0:
        print("No hay raiz dentro de este intervalo")
        return None, None
    
    if metodo is None:
        metodo = int(input("Seleccione método:\n1- Bisección\n2- Regula-Falsi\n"))

    anterior = a
    contador = 0
    errores = []  # Lista para almacenar errores
    
    # Convertir la tolerancia a decimal si es porcentual
    tolerancia = tolerancia/100 if tipo_error == 2 else tolerancia
    
    while True:
        c = (a+b)/2 if metodo == 1 else (a*f(b)-b*f(a))/(f(b)-f(a))
        contador += 1

        if f(c) == 0:
            error = 0
            errores.append(error)
            break
        elif f(a)*f(c) > 0:
            a = c
        else:
            b = c

        error = abs(c-anterior)/abs(c) if tipo_error == 2 else abs(c-anterior)  # Sin multiplicar por 100
        errores.append(error*100 if tipo_error == 2 else error)  # Convertir a porcentaje solo para almacenar
        anterior = c

        if error < tolerancia:
            break
    
    print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
    print(f"Le tomó {contador} iteraciones")
    return c, errores

def raiz_punto_fijo(g, g_prime, a, tolerancia, tipo_error):
    """
    Implementa el método del punto fijo para encontrar raíces de una ecuación.
    El método consiste en reescribir f(x) = 0 como x = g(x) y encontrar el punto fijo de g.

    Args:
        g (callable): Función g(x) en la forma x = g(x)
        g_prime (callable): Derivada de g(x)
        a (float): Valor inicial para comenzar la iteración
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual

    Returns:
        float: Punto fijo encontrado (raíz de la ecuación original)

    Note:
        - Requiere que |g'(x)| < 1 en el intervalo de interés para garantizar convergencia
        - La velocidad de convergencia depende de qué tan cercano a 0 es |g'(x)|
        - Es crucial elegir una buena función g(x) para garantizar convergencia
    """
    contador=0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while True:
        contador+=1
        if abs(g_prime(g ,a))>=1:
            print("El metodo no converge")
            return None
        c=g(a)

        error = abs(c-a) if tipo_error == 1 else abs(c-a)/abs(a)

        a=c

        if error<tolerancia:
            break
    print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%'if tipo_error==2 else''}")
    print(f"Le tomo {contador} iteraciones")
    return c

def newton_raphson(f, f_prime, x0, tolerancia, tipo_error):
    """
    Implementa el método de Newton-Raphson para encontrar raíces de una función.
    El método utiliza la derivada de la función para encontrar mejores aproximaciones
    mediante la fórmula: x_{n+1} = x_n - f(x_n)/f'(x_n)

    Args:
        f (callable): Función de la cual se busca la raíz
        f_prime (callable): Derivada de la función f
        x0 (float): Aproximación inicial
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual

    Returns:
        float: Raíz encontrada

    Note:
        - Convergencia cuadrática cuando la aproximación está cerca de la raíz
        - Requiere que f'(x) ≠ 0 cerca de la raíz
        - Puede diverger si la aproximación inicial no es adecuada
        - Se detiene si la derivada es muy cercana a cero (punto crítico)
    """
    contador = 0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while contador<=10000:
        contador+=1
        if abs(f_prime(x0))<1e-4:
            print("Derivada muy pequeña")
            return None
        x1= x0 - f(x0)/f_prime(x0)
        error = abs(x1-x0) if tipo_error==1 else abs(x1-x0)/x0
        x0=x1
        if(error<tolerancia):
            break
    if tipo_error == 2:
        print(f"La raiz es {x1} con un error de {error*100:.2e}%")
    else:
        print(f"La raiz es {x1} con un error de {error:.2e}")

    print(f"Le tomo {contador} iteraciones")
    print(f"La raiz evaluada es {f(x1):.2e}")
    return x1


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
    print("Determinante = ",prod,"\n")

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
    print("Las soluciones del sistema son: ")
    for i in range(n):
        print(f"X{i} = {X[i]}")
    return X


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
        X = list(map(float, X))  # Asegurar que los valores son float
        Y = list(map(float, Y))
    else:
        raise ValueError("Debe proporcionar arrays X e Y, o un nombre de archivo")
    return X, Y

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
        print("⚠️  ADVERTENCIA: Los puntos se reordenaron según X.")
        print(f"   Original: X = {X}")
        print(f"   Ordenado: X = {X_ordenado}")
    
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
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
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
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    
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

    limpiar_terminal()

    # función polinómica evaluable
    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    return p, coef, X, Y

def graficar_interpolacion(p, coef, X, Y, funcion_real=None, a=None, b=None):
    """
    Visualiza el resultado de la interpolación polinómica.

    Args:
        p (callable): Función del polinomio interpolador
        coef (list[float]): Coeficientes del polinomio
        X (list[float]): Coordenadas x de los puntos originales
        Y (list[float]): Coordenadas y de los puntos originales
        funcion_real (callable, optional): Función original si se conoce

    Note:
        - Muestra los puntos originales en rojo
        - Grafica el polinomio interpolador en azul
        - Si se proporciona, muestra la función original en verde punteado
        - Imprime los coeficientes del polinomio ordenados por potencia
    """
    if a is None:
        a = min(X)
    if b is None:
        b = max(X)

    x_vals = np.linspace(a, b, 500)
    y_vals = [p(x) for x in x_vals]

    plt.scatter(X, Y, color="red", label="Puntos datos")
    plt.plot(x_vals, y_vals, label="Polinomio interpolador", color="blue")

    if funcion_real is not None:
        y_real = [funcion_real(x) for x in x_vals]
        plt.plot(x_vals, y_real, label="Función real (posible origen)", linestyle="--", color="green")
    
    print("Coeficientes del polinomio (ordenados por potencia):")
    for i, c in enumerate(coef):
        print(f"x^{i}: {c}")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolación polinómica")
    plt.grid(True)
    plt.show()



def regresion_polinomica (X=None, Y=None, nombre_archivo=None, grado=1):
    """
    Realiza una regresión polinómica de grado especificado sobre los datos.
    
    Args:
        grado (int, optional): Grado del polinomio de regresión. Por defecto 1 (regresión lineal)
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        tuple:
            - callable: Función del polinomio de regresión
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos
            - float: Coeficiente de correlación (r)
            
    Raises:
        ValueError: Si los datos son insuficientes para el grado especificado
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    n = len(X)
    
    if n <= grado:
        raise ValueError(f"Se necesitan al menos {grado + 1} puntos para una regresión de grado {grado}")
    
    # Crear y llenar matrices del sistema
    A = [[0 for j in range(grado + 1)] for i in range(grado + 1)]
    B = [0 for j in range(grado + 1)]

    for l in range(grado + 1):
        sumaxy = sum(y * x**l for x, y in zip(X, Y))
        B[l] = sumaxy
        for m in range(grado + 1):
            sumax = sum(x**(l+m) for x in X)
            A[l][m] = sumax

    coef = gauss_pivot(A, B)
    
    limpiar_terminal()

    # Función polinómica evaluable
    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    # Calcular coeficiente de correlación
    Y_prom = sum(Y) / n
    Sr = sum((p(x) - y)**2 for x, y in zip(X, Y))  # Error Funcional
    St = sum((y - Y_prom)**2 for y in Y)  # Error estadistico
    
    r = math.sqrt((St - Sr)/St) if St > 0 else 0

    return p, coef, X, Y, r

def graficar_regresion(p, coef, X, Y, r):
    """
    Visualiza el resultado de la regresión polinómica junto con los puntos originales
    y muestra el coeficiente de correlación.

    Args:
        p (callable): Función del polinomio de regresión
        coef (list[float]): Coeficientes del polinomio
        X (list[float]): Coordenadas x de los puntos originales
        Y (list[float]): Coordenadas y de los puntos originales
        r (float): Coeficiente de correlación del ajuste

    Note:
        - Muestra los puntos originales como puntos rojos
        - Grafica la función de regresión como una línea azul continua
        - Muestra el coeficiente de correlación en la leyenda
        - Imprime los coeficientes del polinomio ordenados por potencia
    """
    x_vals = np.linspace(min(X), max(X), 500)
    y_vals = [p(x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color="red", label="Puntos datos")
    plt.plot(x_vals, y_vals, label=f"Regresión (r = {r:.4f})", color="blue")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regresión Polinómica")
    
    print("\nCoeficientes del polinomio (ordenados por potencia):")
    for i, c in enumerate(coef):
        print(f"x^{i}: {c}")

    plt.show()

def graficar_funciones(*funciones, nombres=None, x_min=-10, x_max=10, n_puntos=1000):
    """
    Crea un gráfico interactivo y zoomeable de una o más funciones matemáticas.

    Args:
        *funciones (callable): Una o más funciones matemáticas para graficar
        nombres (list[str], optional): Lista con los nombres de las funciones para la leyenda.
            Si no se proporciona, se usarán nombres genéricos (f1, f2, etc.)
        x_min (float, optional): Límite inferior del dominio. Por defecto -10
        x_max (float, optional): Límite superior del dominio. Por defecto 10
        n_puntos (int, optional): Número de puntos para el gráfico. Por defecto 1000

    Example:
        # Graficar una función
        graficar_funciones(lambda x: x**2)

        # Graficar múltiples funciones con nombres personalizados
        graficar_funciones(
            lambda x: x**2,
            lambda x: math.sin(x),
            nombres=['Parábola', 'Seno']
        )
    
    Note:
        - Use las teclas + y - para hacer zoom
        - Use las flechas del teclado para moverse por el gráfico
        - Use el botón de reset para volver a la vista inicial
        - Los colores se asignan automáticamente para cada función
    """
    # Validar entradas
    if not funciones:
        raise ValueError("Debe proporcionar al menos una función")
    
    if nombres is None:
        nombres = [f'f{i+1}' for i in range(len(funciones))]
    elif len(nombres) != len(funciones):
        raise ValueError("La cantidad de nombres debe coincidir con la cantidad de funciones")

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)  # Hacer espacio para el botón

    # Generar los datos
    x = np.linspace(x_min, x_max, n_puntos)
    
    # Graficar cada función
    lines = []
    for f, nombre in zip(funciones, nombres):
        try:
            y = [f(xi) for xi in x]
            line, = ax.plot(x, y, label=nombre)
            lines.append(line)
        except Exception as e:
            print(f"Error al graficar {nombre}: {e}")

    # Configurar el gráfico
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Guardar los límites originales para el botón de reset
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    # Agregar botón de reset
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    reset_button = Button(reset_ax, 'Reset')

    def reset(event):
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)
        plt.draw()

    reset_button.on_clicked(reset)

    # Configurar el zoom con el teclado
    def on_key(event):
        if event.key == '=':  # Zoom in
            ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
            ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        elif event.key == '-':  # Zoom out
            ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
            ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        elif event.key == 'left':
            xlim = ax.get_xlim()
            delta = (xlim[1] - xlim[0]) * 0.1
            ax.set_xlim(xlim[0] - delta, xlim[1] - delta)
        elif event.key == 'right':
            xlim = ax.get_xlim()
            delta = (xlim[1] - xlim[0]) * 0.1
            ax.set_xlim(xlim[0] + delta, xlim[1] + delta)
        elif event.key == 'up':
            ylim = ax.get_ylim()
            delta = (ylim[1] - ylim[0]) * 0.1
            ax.set_ylim(ylim[0] + delta, ylim[1] + delta)
        elif event.key == 'down':
            ylim = ax.get_ylim()
            delta = (ylim[1] - ylim[0]) * 0.1
            ax.set_ylim(ylim[0] - delta, ylim[1] - delta)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

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
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
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
    
    limpiar_terminal()
    
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

def buscar_intervalo_binario(x, X):
    """
    Encuentra el intervalo usando búsqueda binaria (más eficiente para muchos puntos).
    
    Args:
        x (float): Punto a buscar
        X (list[float]): Puntos ordenados (nodos del spline)
    
    Returns:
        int: Índice k del intervalo [X_k, X_{k+1}] que contiene x
    """
    if x < X[0] or x > X[-1]:
        raise ValueError(f"El punto x={x} está fuera del dominio [{X[0]}, {X[-1]}]")
    
    # Búsqueda binaria (O(log n))
    # Más eficiente para muchos puntos o evaluaciones frecuentes
    izq, der = 0, len(X) - 2
    
    while izq <= der:
        medio = (izq + der) // 2
        
        if X[medio] <= x <= X[medio + 1]:
            return medio
        elif x < X[medio]:
            der = medio - 1
        else:
            izq = medio + 1
    
    # Fallback (no debería llegar aquí si x está en el dominio)
    return len(X) - 2

def graficar_splines(funciones_spline, coeficientes, X, Y, funcion_real=None):
    """
    Visualiza las curvas spline junto con los puntos originales.
    
    Args:
        funciones_spline (list[callable]): Lista de funciones para cada intervalo
        coeficientes (list[list[float]]): Coeficientes [a, b, c, d] para cada intervalo
        X (list[float]): Coordenadas x de los puntos originales
        Y (list[float]): Coordenadas y de los puntos originales
        funcion_real (callable, optional): Función original si se conoce
    """
    plt.figure(figsize=(12, 8))
    
    # Graficar los puntos originales
    plt.scatter(X, Y, color="red", s=50, label="Puntos datos", zorder=5)
    
    # Graficar cada spline en su intervalo correspondiente
    colores = plt.cm.tab10(np.linspace(0, 1, len(funciones_spline)))
    
    for k in range(len(funciones_spline)):
        x_intervalo = np.linspace(X[k], X[k+1], 100)
        y_intervalo = [funciones_spline[k](x) for x in x_intervalo]
        
        plt.plot(x_intervalo, y_intervalo, 
                color=colores[k], 
                label=f'Spline {k+1}: [{X[k]:.2f}, {X[k+1]:.2f}]',
                linewidth=2)
    
    # Si se proporciona la función real, mostrarla
    if funcion_real is not None:
        x_continuo = np.linspace(min(X), max(X), 500)
        y_real = [funcion_real(x) for x in x_continuo]
        plt.plot(x_continuo, y_real, 
                label="Función real", 
                linestyle="--", 
                color="green", 
                alpha=0.7,
                linewidth=2)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolación con Splines Cúbicos Naturales")
    
    # Imprimir los coeficientes
    print("\nCoeficientes de los splines cúbicos:")
    print("S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k")
    print("-" * 50)
    for k, (a, b, c, d) in enumerate(coeficientes):
        print(f"Intervalo {k+1} [{X[k]:.2f}, {X[k+1]:.2f}]:")
        print(f"  S_{k+1}(x) = {a:.4f}*x³ + {b:.4f}*x² + {c:.4f}*x + {d:.4f}")
    
    plt.show() 




