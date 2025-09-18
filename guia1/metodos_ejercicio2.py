import metodos_prueba as mp
import math
import os

def f(x):
    return x**10 - 1

if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tipo_error = int(input("Tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia = float(input("Ingrese la tolerancia deseada: "))

    raiz_bi, errores_bi = mp.buscar_raiz(f, a, b, tipo_error, tolerancia, 1)
    raiz_rf, errores_rf = mp.buscar_raiz(f, a, b, tipo_error, tolerancia, 2)

    if raiz_bi is not None and raiz_rf is not None:
        mp.grafico_comparativo(errores_bi, errores_rf, a, b)


