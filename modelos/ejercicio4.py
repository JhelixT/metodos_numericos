"""
Modelos - Ejercicio 4 (Problema n° 2)

Se propone un método de resolución de EDOs de la forma dy/dx = f(x,y),
con dato inicial y(x0)=y0, en el que cada nuevo valor se obtiene a partir de
los dos pasos anteriores, dado por:

    y_{i+1} = y_i + h * ( 2 f(x_i, y_i) - f(x_{i-1}, y_{i-1}) )  para i > 0

Para i = 0 se toma el valor inicial, y para i = 1 se usa un método de un paso
(como Euler) para calcular y1.

Problema: En x ∈ [0,1]
    dy/dx = (x e^{x^2}) / y ,   y(0) = 1

Incisos:
 a) Escriba (en papel) el pseudo-código del método e impleméntelo.
 b) Resuelva la EDO de forma exacta.
 c) Aproxime usando el método propuesto con h = 0.1; escriba y(0.5) y y(1) y su
    error absoluto respecto de la solución exacta.
 d) Repita c) usando RK4.
 e) Grafique en un único plot las tres soluciones (exacta, propuesto y RK4) y
    guarde la figura.
"""

from __future__ import annotations
import math
import os
from typing import Callable, Tuple, List

import matplotlib.pyplot as plt

# Usamos RK4 del módulo de métodos existentes
from metodos.edo1 import runge_kutta4


def metodo_propuesto(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    xf: float,
    n: int,
    *,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """Método de 2 pasos: y_{i+1} = y_i + h(2f_i - f_{i-1}).

    - Para i=0: X[0]=x0, Y[0]=y0
    - Para i=1: y1 = y0 + h f(x0,y0)  (Euler, 1 paso)
    - Para i>=1: y_{i+1} = y_i + h(2 f(x_i,y_i) - f(x_{i-1},y_{i-1}))

    Returns:
        (X, Y) arreglos con n+1 puntos en [x0, xf].
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    if x0 >= xf:
        raise ValueError("Se requiere x0 < xf")

    h = (xf - x0) / n
    if verbose:
        print("Método propuesto (2 pasos): y_{i+1} = y_i + h(2f_i - f_{i-1})")
        print(f"Intervalo: [{x0}, {xf}] con n={n} pasos (h={h:.3f})")

    X = [0.0] * (n + 1)
    Y = [0.0] * (n + 1)

    X[0] = x0
    Y[0] = y0

    # Paso 1 por Euler (arranque)
    X[1] = X[0] + h
    Y[1] = Y[0] + h * f(X[0], Y[0])

    # Pasos siguientes
    for i in range(1, n):
        X[i + 1] = X[0] + (i + 1) * h
        fi = f(X[i], Y[i])
        fim1 = f(X[i - 1], Y[i - 1])
        Y[i + 1] = Y[i] + h * (2 * fi - fim1)

    return X, Y


def f_rhs(x: float, y: float) -> float:
    """f(x,y) = (x e^{x^2}) / y"""
    return (x * math.exp(x * x)) / y


def y_exacta(x: float) -> float:
    """Solución exacta: y(x) = e^{x^2 / 2} (positiva por y(0)=1)."""
    return math.exp(0.5 * x * x)


def valores_y_errores(X: List[float], Y: List[float], puntos: List[float]):
    """Devuelve (valor_aprox, valor_exacto, error_abs) en los puntos solicitados."""
    h = X[1] - X[0]
    res = []
    for xp in puntos:
        idx = round((xp - X[0]) / h)
        ya = Y[idx]
        ye = y_exacta(xp)
        res.append((xp, ya, ye, abs(ya - ye)))
    return res


def guardar_plot(Xp, Yp, Xrk, yrk, nombre_archivo: str = "modelos_ej4_plot.png"):
    # Malla fina para exacta
    xs = [i / 400 for i in range(0, 401)]  # 0..1 con paso 0.0025
    ys = [y_exacta(x) for x in xs]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, 'k-', label='Exacta e^{x^2/2}')
    plt.plot(Xp, Yp, 'o-', label='Método propuesto (2 pasos)')
    plt.plot(Xrk, yrk, 's-', label='RK4')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, nombre_archivo)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFigura guardada en: {out_path}")


def main():
    x0, y0, xf, h = 0.0, 1.0, 1.0, 0.1
    n = int((xf - x0) / h)

    # a) (Pseudo-código se documenta en este archivo; implementación abajo)

    # b) Solución exacta: y(x) = exp(x^2/2)
    #   (usada más abajo para comparar)

    # c) Método propuesto
    Xp, Yp = metodo_propuesto(f_rhs, x0, y0, xf, n, verbose=False)
    print("Método propuesto (h=0.1):")
    for xp, ya, ye, err in valores_y_errores(Xp, Yp, [0.5, 1.0]):
        print(f"  y({xp:.1f}) ≈ {ya:.10f}  |  exacta = {ye:.10f}  |  error = {err:.2e}")

    # d) RK4 en la misma malla
    Xrk, yrk = runge_kutta4(f_rhs, x0, y0, xf, n, verbose=False)
    print("\nRK4 (h=0.1):")
    for xp, ya, ye, err in valores_y_errores(Xrk, yrk, [0.5, 1.0]):
        print(f"  y({xp:.1f}) ≈ {ya:.10f}  |  exacta = {ye:.10f}  |  error = {err:.2e}")

    # e) Plot
    guardar_plot(Xp, Yp, Xrk, yrk)


if __name__ == "__main__":
    main()
