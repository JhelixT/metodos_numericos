import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    # Usar ruta relativa que funcione en cualquier sistema operativo
    archivo = os.path.join(os.path.dirname(__file__), "inputs", "datos1.txt")
    p, coef, X, Y = f.interpolacion(archivo)

    f.graficar_interpolacion(p, coef, X, Y, funcion_real=lambda x: x + 2/x)

main()