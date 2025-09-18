import metodos_prueba as mp
import math
import os

def f(x):
    return (math.log(x)-(1-1/x))/(math.log(x) + (1-1/x)/(2/3)) - 0.3

if __name__ == "__main__":

    os.system('clear' if os.name == 'posix' else 'cls')

    mp.graficar_funcion(f, 1, 100)
    
    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tipo_error = int(input("Tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia = float(input("Ingrese la tolerancia deseada: "))

    raiz_bi, errores_bi= mp.buscar_raiz(f, a, b, tipo_error, tolerancia, 1)