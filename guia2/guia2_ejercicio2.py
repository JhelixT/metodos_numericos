import math
import sympy as sp
import os

def g(x):
    return x**5 - 3*x**3 - 2*x**2 + 2

def gp(h, x):
    return (h(x+0.001)-h(x))/0.001

def raiz_punto_fijo(a, tolerancia, tipo_error):
    contador=0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while True:
        contador+=1
        if abs(gp(g ,a))>=1:
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

if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    a = float(input("Ingrese el valor inicial: "))
    tipo_error=int(input("Ingrese el tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia= float(input("Ingrese la tolerancia: "))
    raiz = raiz_punto_fijo(a, tolerancia, tipo_error)