import matplotlib.pyplot as plt
import numpy as np
from metodos import funciones as func
"""
Problema n°2: Los datos de la tabla corresponden a la misma función del problema 1.

Tabla de datos:
x     y
1.0  -0.148
1.2  -0.040
1.5   0.181
1.75  0.419
2.0   0.700

a) Obtenga el polinomio interpolador con nodos en los datos de la tabla.

b) A partir del polinomio interpolador, calcule el valor aproximado de f(1.6), y 
   determine el error absoluto exacto.

c) Graficar la función error (e(x) = |P(x) - f(x)|) y explique su comportamiento 
   cerca de los nodos. (Adjunte el plot por aula virtual).

d) Ajuste los datos de la tabla por medio de regresión lineal y determine, justificando su respuesta, si es 
   un buen ajuste. Escriba la función lineal obtenida, y a partir de ella aproxime la raíz de f(x). ¿Cuál 
   es el error de esta aproximación? (Tome como valor exacto de la raíz x = 1.2620326547, y adjunte 
   captura de la consola donde figuren los resultados y el coeficiente de correlación).

e) Repita el inciso d), pero usando regresión cuadrática. ¿Cuál ajusta mejor los datos?
"""

def main():
    X = [1, 1.2, 1.5, 1.75, 2.0]
    Y = [-0.148, -0.040, 0.181, 0.419, 0.700]
    print("POLINOMIO INTERPOLADOR")
    p, coef, X, Y = interpolacion(X, Y)
    graficar_interpolacion(p, coef, X, Y)
    comparar_error(p,min(X), max(X))
    f = lambda x: np.log(x**2+1)-np.sin(x)
    print("ERROR PARA f(1.6)")
    error = np.abs(p(1.6)-f(1.6))
    print(f"El error absoluto es: {error}")

    print("REGRESION LINEAL")
    reg, coef, X, Y, r = func.regresion_polinomica(X, Y, grado=1)
    func.graficar_regresion(reg, coef, X, Y, r)
    print(f"La funcion de ajuste es: {coef[1]}x + ({coef[0]})")
    print("Aproximacion de la raiz de f(x)")
    raiz, errores = func.buscar_raiz(reg, 1, 2, 1, 1e-12, 1)
    errorAprox = np.abs(1.2620326547 - raiz)
    print(f"El error de aproximacion de la raiz mediante la regresion es: {errorAprox}")

    print("REGRESION CUADRATICA")
    reg2, coef, X, Y, r = func.regresion_polinomica(X, Y, grado=2)
    func.graficar_regresion(reg2, coef, X, Y, r)
    print(f"La funcion de ajuste es: {coef[2]}x^2 + ({coef[1]})x + ({coef[0]})")
    print("Aproximacion de la raiz de f(x)")
    raiz, errores = func.buscar_raiz(reg2, 1, 2, 1, 1e-12, 1)
    errorAprox = np.abs(1.2620326547 - raiz)
    print(f"El error de aproximacion de la raiz mediante la regresion es: {errorAprox}")


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
    n = len(X)

    A = [[0 for j in range(n)] for i in range(n)]
    B = [0 for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = X[i]**j
        B[i] = Y[i]

    coef = func.gauss_pivot(A, B)  # coef[0] = x^0, coef[1] = x^1, ...

    func.limpiar_terminal()

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

def comparar_error(p, xmin, xmax):
    f = lambda x: np.log(x**2+1)-np.sin(x)

    x_vals = np.linspace(xmin, xmax, 500)
    y_interpolado = p(x_vals)
    y_real = f(x_vals)

    error = np.abs(y_real - y_interpolado)
    errormax = np.max(error)
    ecm = np.mean(error**2)

    #print(f"Error maximo: {errormax}")
    #print(f"Error cuadratico medio: {ecm}")

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(x_vals, y_real, "r--", label = "f(x) = ln(x^2 + 1) - sin(x)")
    plt.plot(x_vals, y_interpolado, "g-", label = "Polinomio P(x)")
    plt.title("Funcion vs Polinomio")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(x_vals, error, "m-")
    plt.title("Error |f(x)-p(x)|")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
main()