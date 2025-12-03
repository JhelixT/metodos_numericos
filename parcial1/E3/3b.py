import math
import os
import numpy as np

def main():
    print("=" * 70)
    print("REGRESIÓN NO LINEAL - AJUSTE TRIGONOMÉTRICO")
    print("=" * 70)
    
    X = [1, 1.2, 1.5, 2, 2.6, 2.9, 3]
    Y = [-0.236, 0.209, 0.853, 1.746, 2.231, 2.163, 2.110]
    
    print("DATOS DE ENTRADA:")
    print("  X:", [f"{x:.1f}" for x in X])
    print("  Y:", [f"{y:.3f}" for y in Y])
    print(f"  Número de puntos: {len(X)}")
    print("-" * 70)
    
    p, coef, X, Y, r, St, Sr = regresion(X, Y, grado = 1)
    
    print("\nRESULTADOS DE LA REGRESIÓN:")
    print(f"Función de ajuste: f(x) = {coef[0]:8.4f}·sen(x) + {coef[1]:8.4f}·cos(x)")
    print("-" * 70)
    print("ESTADÍSTICAS DEL AJUSTE:")
    print(f"  Error estadístico (St): {St:12.6f}")
    print(f"  Error de ajuste (Sr):   {Sr:12.6f}")
    print(f"  Coeficiente de correlación (r): {r:8.6f}")
    print(f"  Coeficiente de determinación (r²): {r**2:8.6f}")
    print("=" * 70)
def regresion (X=None, Y=None, nombre_archivo=None, grado=1):
    """
    Realiza una regresión de grado especificado sobre los datos.
    
    Args:
        grado (int, optional): Grado del polinomio de regresión. Por defecto 1 (regresión lineal o 2 incognitas)
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
    n = len(X)
    
    if n <= grado:
        raise ValueError(f"Se necesitan al menos {grado + 1} puntos para una regresión de grado {grado}")
    
    # Crear y llenar matrices del sistema
    A = [[sum(np.sin(X)**2), sum(np.cos(X)*np.sin(X))],
         [sum(np.cos(X)*np.sin(X)), sum(np.cos(X)**2)]
         ]
    B = [sum(Y*np.sin(X)), sum(Y*np.cos(X)) ]

    
    coef = gauss_pivot(A, B)
    
    limpiar_terminal()

    # Función polinómica evaluable
    def p(x):
        return coef[0]*np.sin(x) + coef[1]*np.cos(x)

    # Calcular coeficiente de correlación
    Y_prom = sum(Y) / n
    Sr = sum((p(x) - y)**2 for x, y in zip(X, Y))  # Error Funcional
    St = sum((y - Y_prom)**2 for y in Y)  # Error estadistico
    
    r = math.sqrt((St - Sr)/St) if St > 0 else 0

    return p, coef, X, Y, r, St, Sr
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
    print("COEFICIENTES DEL SISTEMA:")
    for i in range(n):
        print(f"  Coeficiente {i+1}: {X[i]:10.6f}")
    return X

def limpiar_terminal():
    """
    Limpia la pantalla de la terminal de manera compatible con múltiples sistemas operativos.
    Utiliza el comando 'clear' en sistemas Unix/Linux/Mac y 'cls' en Windows.
    """
    os.system('clear' if os.name == 'posix' else 'cls')
main()