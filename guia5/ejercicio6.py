import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    h = [0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150]
    ro = [1, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741]

    p, coef, h, ro, r = f.regresion_polinomica(h, ro, grado = 2)
    f.graficar_regresion(p, coef, h, ro, r)
    print(f"La densidad relativa a la altitud h = 10.5 [km] es de: {p(10.5)}")
main()