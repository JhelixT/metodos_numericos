import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    deformacion = [0.46, 0.34, 0.73, 0.95, 1.02, 1.10, 1.48, 1.51, 1.62, 1.93, 2.09, 2.12]
    tension = [34.5, 34.5, 34.5, 69, 69,  69, 103.5, 103.5, 103.5 , 138, 138, 138]

    p, coef, deformacion, tension, r = f.regresion_polinomica(deformacion, tension)
    f.graficar_regresion(p, coef, deformacion, tension, r)
    print(f"El modulo de la elasticidad segun los valores relevados es de: {abs(coef[1])}")

    # Error Cuadrático Medio: ECM = sqrt((1/N) * sum((f(xk) - yk)^2))
    N = len(deformacion)
    suma_errores_cuadrados = sum([(p(deformacion[i]) - tension[i])**2 for i in range(N)])
    Ecm = ((1/N) * suma_errores_cuadrados)**(1/2)

    print(f"ECM: {Ecm}")

main()