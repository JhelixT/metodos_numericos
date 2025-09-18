import metodos_prueba as mp
import os

def f(x):
    return 6*x**3 - 5*x**2 + 7*x - 2

if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tipo_error = int(input("Tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia = float(input("Ingrese la tolerancia deseada: "))

    raiz, errores = mp.buscar_raiz(f, a, b, tipo_error, tolerancia)
    if raiz:
        mp.grafico(f, a, b, raiz)
