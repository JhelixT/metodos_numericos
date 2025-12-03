import math
import matplotlib.pyplot as plt
import numpy as np

from metodos.aproximacion import _ordenar_puntos_xy, _validar_datos_xy
from metodos.aproximacion import graficar_interpolacion
from metodos.raices import biseccion
def main():
    X = [0.5, 0.8, 1.3, 2.0]
    Y = [-0.716, -0.103, 3.419, 52.598]
    n = len(X)
    p, coef, Xp, Yp, r = regresion_polinomica(X, Y, grado=2, verbose=True)
    
    print("Coeficientes:", coef)
    print("Coeficiente de correlación (r):", r)
    graficar_regresion(p, coef, X, Y, r)
    
    pol, coef_int, X_int, Y_int = interpolacion(X, Y)
    print("Coeficientes del polinomio interpolador:", coef_int)
    graficar_interpolacion(pol, coef_int, X_int, Y_int, funcion_real=p)

    raizaprox= biseccion(pol, 0.8, 1.3, tolerancia=1e-5, verbose=True)
    raizreal= biseccion(lambda x: math.exp(x**2) -2, 0.8, 1.3, tolerancia=1e-5, verbose=True)

def interpolacion(X=None, Y=None, nombre_archivo=None, ordenar_automatico=True, verbose=True):
    """
    Realiza interpolación polinómica usando el método de Vandermonde.
    Construye un polinomio que pasa exactamente por todos los puntos dados.

    Args:
        X (list[float], optional): Lista de coordenadas x de los puntos.
        Y (list[float], optional): Lista de coordenadas y de los puntos.
        nombre_archivo (str, optional): Archivo con los puntos a interpolar.
        ordenar_automatico (bool, optional): Si True, ordena los puntos automáticamente. Por defecto True.
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.

    Returns:
        tuple:
            - callable: Función del polinomio interpolador
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos

    Note:
        - Usa la matriz de Vandermonde: A[i][j] = x_i^j
        - Resuelve el sistema usando eliminación gaussiana
        - El grado del polinomio es n-1, donde n es el número de puntos
        - Puede ser numéricamente inestable para muchos puntos
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    
    if ordenar_automatico:
        X, Y = _ordenar_puntos_xy(X, Y, verbose=verbose)

    n = len(X)

    A = [[0 for j in range(n)] for i in range(n)]
    B = [0 for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = X[i]**j
        B[i] = Y[i]

    coef = gauss_pivot(A, B, verbose=False)

    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    return p, coef, X, Y    
def graficar_regresion(p, coef, X, Y, r):
    """
    Visualiza el resultado de la regresión polinómica.
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
    plt.title(f"Regresión Polinómica Funcion {coef[0]:.4f}Exp(x^2) + {coef[1]:.4f}")
    
    plt.show()

def regresion_polinomica(X=None, Y=None, nombre_archivo=None, grado=1, verbose=True):
    """
    Realiza una regresión polinómica de grado especificado sobre los datos.
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        grado (int, optional): Grado del polinomio de regresión. Por defecto 1 (regresión lineal)
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.
        
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
    n = len(X)
    
    if n <= grado:
        raise ValueError(f"Se necesitan al menos {grado + 1} puntos para una regresión de grado {grado}")
    
    A = [[0 for j in range(grado + 1)] for i in range(grado + 1)]
    B = [0 for j in range(grado + 1)]

    A=[[sum(math.exp(2*xi**2) for xi in X), sum(math.exp(xi**2) for xi in X)],
       [sum(math.exp(xi**2) for xi in X), n]]

    B=[sum(yi*math.exp(xi**2) for xi, yi in zip(X, Y)),
       sum(yi for yi in Y)]
    
    coef = gauss_pivot(A, B, verbose=False)

    def p(x):
        return coef[0]*math.exp(x**2) + coef[1]

    Y_prom = sum(Y) / n
    Sr = sum((p(x) - y)**2 for x, y in zip(X, Y))
    St = sum((y - Y_prom)**2 for y in Y)
    
    r = math.sqrt((St - Sr)/St) if St > 0 else 0

    return p, coef, X, Y, r
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