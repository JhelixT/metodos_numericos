import math
import matplotlib.pyplot as plt
import numpy as np
import os

def f(x):
    return math.log(x) + math.exp(math.sin(x)) - x 

def buscar_raiz(f, a, b, tipo_error, tolerancia, metodo=None):
    if f(a)*f(b) > 0:
        print("No hay raiz dentro de este intervalo")
        return None, None
    
    if metodo is None:
        metodo = int(input("Seleccione método:\n1- Bisección\n2- Regula-Falsi\n"))

    anterior = a
    contador = 0
    errores = []  # Lista para almacenar errores
    
    # Convertir la tolerancia a decimal si es porcentual
    tolerancia = tolerancia/100 if tipo_error == 2 else tolerancia
    
    while True:
        c = (a+b)/2 if metodo == 1 else (a*f(b)-b*f(a))/(f(b)-f(a))
        contador += 1

        if f(c) == 0:
            error = 0
            errores.append(error)
            break
        elif f(a)*f(c) > 0:
            a = c
        else:
            b = c

        error = abs(c-anterior)/abs(c) if tipo_error == 2 else abs(c-anterior)  # Sin multiplicar por 100
        errores.append(error*100 if tipo_error == 2 else error)  # Convertir a porcentaje solo para almacenar
        anterior = c

        if error < tolerancia:
            break
    
    print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
    print(f"Le tomó {contador} iteraciones")
    return c, errores


def grafico_comparativo(errores_biseccion, errores_regula, a, b):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(errores_biseccion) + 1), errores_biseccion, 'b-o', label='Bisección')
    plt.semilogy(range(1, len(errores_regula) + 1), errores_regula, 'r-o', label='Regula-Falsi')
    
    # Agregar líneas verticales para mostrar el intervalo
    plt.axvline(max(len(errores_biseccion), len(errores_regula)), color='gray', linestyle='--')
    
    plt.grid(True)
    plt.xlabel('Iteración')
    plt.ylabel('Error (escala log)')
    plt.title(f'Comparación de convergencia entre métodos\nIntervalo [{a}, {b}]')
    plt.legend()
    plt.show()


def graficar_funcion(f, a, b, puntos=1000):
    """
    Grafica una función matemática en un intervalo dado
    
    Args:
        f: función a graficar
        a: límite inferior del intervalo
        b: límite superior del intervalo
        puntos: cantidad de puntos para el gráfico
    """
    x = np.linspace(a, b, puntos)
    y = [f(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gráfico de la función en el intervalo [{a}, {b}]')
    plt.legend()
    plt.show()


def grafico(f, a, b, raiz):
    x_vals = np.linspace(a, b, 1000)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="f(x)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(raiz, color='red', linestyle='--', label=f"Raiz: {raiz:.5f}")
    plt.scatter([raiz], [f(raiz)], color='red')

    zoom_radio = 1
    plt.xlim(raiz - zoom_radio, raiz + zoom_radio)
    plt.ylim(f(raiz) - 1, f(raiz) + 1)

    plt.title("Método numérico")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    a = float(input("Ingrese el extremo inferior del intervalo (a): "))
    b = float(input("Ingrese el extremo superior del intervalo (b): "))
    tipo_error = int(input("Tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia = float(input("Ingrese la tolerancia deseada: "))

    raiz = buscar_raiz(f, a, b, tipo_error, tolerancia)
    if raiz:
        grafico(a, b, raiz)





