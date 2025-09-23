import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    X =[-1, 0, 1, 2, 3, 4, 5, 6]
    Y =[10, 9, 7, 5, 4, 3, 0, -1] 

    p, coef, X, Y , r = f.regresion_polinomica(X, Y, grado = 1)
    f.graficar_regresion(p, coef, X, Y, r)
    
    # Error Cuadrático Medio: ECM = sqrt((1/N) * sum((f(xk) - yk)^2))
    N = len(X)
    suma_errores_cuadrados = sum([(p(X[i]) - Y[i])**2 for i in range(N)])
    Ecm = ((1/N) * suma_errores_cuadrados)**(1/2)
    
    print(f"El error cuadrático medio es: {Ecm:.4f}")
main()