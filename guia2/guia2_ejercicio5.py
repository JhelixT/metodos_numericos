import math
import os

def f(x):
    return x*math.cosh(12/x) - x - 5

def fp(x0, x1):
    return (f(x1)-f(x0))/(x1-x0)

def secante(x0,x1,tolerancia,tipo_error):
    contador = 0
    tolerancia = tolerancia/100 if tipo_error==2 else tolerancia

    while True:
        contador+=1
        
        x2 = x1 - f(x1)/fp(x0,x1)
        
        error = abs(x2-x1) if tipo_error==1 else abs(x2-x1)/abs(x1)

        x0= x1
        x1= x2

        if error<tolerancia:
            break
    
    if tipo_error == 2:
        print(f"La raiz es {x2} con un error de {error*100:.2e}%")
    else:
        print(f"La raiz es {x2} con un error de {error:.2e}")

    print(f"Le tomo {contador} iteraciones")
    return x2

if __name__ == "__main__":
    os.system('clear' if os.name == 'posix' else 'cls')
    x0 = float(input("Ingrese x0: "))
    x1 = float(input("Ingrese el x1: "))
    tipo_error=int(input("Ingrese el tipo de error (1-Absoluto, 2-Porcentual): "))
    tolerancia= float(input("Ingrese la tolerancia: "))
    raiz = secante(x0, x1, tolerancia, tipo_error)