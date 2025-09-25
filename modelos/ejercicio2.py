import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f
import numpy as np
def main():
    f = lambda x : 3*x + np.sin(x) - np.exp(x)
    f_prime = lambda x: (f(x+0.01)-f(x-0.01))/(2*0.01)
    print("RAIZ CON EL METODO DE NEWTON RAPHSON")
    raiz = newton_raphson(f, f_prime, 0.5, 1e-8, 1)

    print("RAIZ CON METODO DE PUNTO FIJO")
    g = lambda x: (np.exp(x)-np.sin(x))/3
    g_prime = lambda x: (g(x+0.01)-g(x))/0.01
    raiz = raiz_punto_fijo(g, g_prime, 0.5, 1e-8, 1)

def newton_raphson(f, f_prime, x0, tolerancia, tipo_error):
    """
    Implementa el método de Newton-Raphson para encontrar raíces de una función.
    El método utiliza la derivada de la función para encontrar mejores aproximaciones
    mediante la fórmula: x_{n+1} = x_n - f(x_n)/f'(x_n)

    Args:
        f (callable): Función de la cual se busca la raíz
        f_prime (callable): Derivada de la función f
        x0 (float): Aproximación inicial
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual

    Returns:
        float: Raíz encontrada

    Note:
        - Convergencia cuadrática cuando la aproximación está cerca de la raíz
        - Requiere que f'(x) ≠ 0 cerca de la raíz
        - Puede diverger si la aproximación inicial no es adecuada
        - Se detiene si la derivada es muy cercana a cero (punto crítico)
    """
    contador = 0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while contador<=10000:
        contador+=1
        if abs(f_prime(x0))<1e-4:
            print("Derivada muy pequeña")
            return None
        x1= x0 - f(x0)/f_prime(x0)
        error = abs(x1-x0) if tipo_error==1 else abs(x1-x0)/x0
        x0=x1
        if(error<tolerancia):
            break
    if tipo_error == 2:
        print(f"La raiz es {x1} con un error de {error*100:.2e}%")
    else:
        print(f"La raiz es {x1} con un error de {error:.2e}")

    print(f"Le tomo {contador} iteraciones")
    print(f"La raiz evaluada es {f(x1):.2e}")
    return x1

def raiz_punto_fijo(g, g_prime, a, tolerancia, tipo_error):
    """
    Implementa el método del punto fijo para encontrar raíces de una ecuación.
    El método consiste en reescribir f(x) = 0 como x = g(x) y encontrar el punto fijo de g.

    Args:
        g (callable): Función g(x) en la forma x = g(x)
        g_prime (callable): Derivada de g(x)
        a (float): Valor inicial para comenzar la iteración
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual

    Returns:
        float: Punto fijo encontrado (raíz de la ecuación original)

    Note:
        - Requiere que |g'(x)| < 1 en el intervalo de interés para garantizar convergencia
        - La velocidad de convergencia depende de qué tan cercano a 0 es |g'(x)|
        - Es crucial elegir una buena función g(x) para garantizar convergencia
    """
    contador=0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while True:
        contador+=1
        if abs(g_prime(a))>=1:
            print("El metodo no converge")
            return None
        c=g(a)

        error = abs(c-a) if tipo_error == 1 else abs(c-a)/abs(a)

        a=c

        if error<tolerancia:
            break
    print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%'if tipo_error==2 else''}")
    print(f"Le tomo {contador} iteraciones")
    return c

main()