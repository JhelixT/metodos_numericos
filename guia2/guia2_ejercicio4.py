import math
import sympy as sp
import os

#Creacion de la funcion y su derivada
x = sp.symbols("x")
expr = 4*x**3 - 52*x**2 + 160*x - 100
f = sp.lambdify(x,expr, "numpy")

expr_prime = sp.diff(expr, x)
f_prime = sp.lambdify(x, expr_prime, "numpy")

def newton_raphson(x0, tolerancia, tipo_error):
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
        


if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    a = float(input("Ingrese el valor inicial: "))
    tipo_error=int(input("Ingrese el tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia= float(input("Ingrese la tolerancia: "))
    raiz = newton_raphson(a, tolerancia, tipo_error)