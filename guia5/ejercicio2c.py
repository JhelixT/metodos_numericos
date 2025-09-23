import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f
import numpy as np
import matplotlib.pyplot as plt

def main():
    valores_x = [1, 0.5, 2, 3, 1.2, 2.5, 0]
    archivo = os.path.join(os.path.dirname(__file__), "inputs", "datos2c.txt")
    generar_datos(valores_x, archivo, separador=" ")
    p, coef, X, Y = f.interpolacion(nombre_archivo=archivo)
    f.graficar_interpolacion(p, coef, X, Y, funcion_real=lambda x: x**x)
    comparar_error(p, min(X), max(X))


def generar_datos(xs, filename, separador=","):
    """
    xs: lista de valores de x (ejemplo: [0, 0.5, 1, 2])
    filename: nombre del archivo de salida
    separador: separador entre x y f(x) ("," o " ")
    """
    with open(filename, "w") as f:
        for x in xs:
            fx = x**x
            f.write(f"{x:.6f}{separador}{fx:.6f}\n")

    print(f"Datos guardados en {filename}")

def comparar_error(P, xmin, xmax, puntos=500):
    """
    Compara el polinomio P(x) contra f(x) = e^(-x^2) en [xmin, xmax].

    Parámetros:
    - P: función polinómica (llamable, ej. def P(x): return ...)
    - xmin, xmax: intervalo de comparación
    - puntos: número de puntos para muestreo del error

    Imprime el error máximo y el ECM.
    Grafica la función real vs. polinomio y el error.
    """
    # Función real
    f = lambda x: x**x

    # Puntos densos
    x_dense = np.linspace(xmin, xmax, puntos)
    y_real = f(x_dense)
    y_interp = P(x_dense)

    # Error
    error = np.abs(y_real - y_interp)
    error_max = np.max(error)
    error_mse = np.mean(error**2)

    print(f"Error máximo = {error_max:.6e}")
    print(f"Error cuadrático medio = {error_mse:.6e}")

    # Gráfico comparativo
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(x_dense, y_real, "g--", label="f(x) = x^x")
    plt.plot(x_dense, y_interp, "b-", label="Polinomio P(x)")
    plt.title("Función vs Polinomio")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(x_dense, error, "m-")
    plt.title("Error |f(x) - P(x)|")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
main()