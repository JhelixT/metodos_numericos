import math
import pprint as pp
import numpy as np
import matplotlib as plt
import os

def limpiar_terminal():
    """Función para limpiar la terminal de manera multiplataforma"""
    os.system('clear' if os.name == 'posix' else 'cls')

def resolverJG(A, B):

    esDiagDom(A)

    n = len(A)
    Xn = [0] * n
    Xv = [0] * n

    print("1-Resolver con Jacobi")
    print("2-Resolver con Gauss-Seidel")
    opcion = int(input())
    if opcion == 1:
        jacobi(A, B, Xn, Xv)
    elif opcion == 2:
        gauss_seidel(A,B, Xn, Xv)
    
    return Xn
    
def esDiagDom(A):
    n = len(A)
    suma = 0
    for j in range(1, n):
        suma = suma + abs(A[0][j])
    if abs(A[0][0]) < suma:
        print("La matriz no es Diagonalmente Dominante")
        return
    for i in range(1, n):
        suma = 0
        for j in range (n):
            if i != j:
                suma = suma + abs(A[i][j])
        if abs(A[i][i]) < suma:
            print("La matriz no es Diagonalmente Dominante")
            return
        
def jacobi(A, B, Xn, Xv):
    n = len(A)
    count = 0
    errorV = 1000
    tolerancia = float(input("Ingrese la tolerancia: "))
    while True:
        count+=1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i!=j:
                    suma = suma + A[i][j]*Xv[j]
            Xn[i] = (B[i] - suma)/A[i][i]
        error = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
        error = math.sqrt(error) 
        if error > errorV:
            print("El metodo no converge")
            return
        errorV = error
        for i in range(n):
            Xv[i]=Xn[i]
        if error < tolerancia:
            break;
    print("Las soluciones son:")
    for i in range(n):
        print(f"X{i} = {Xn[i]}")
    print(f"Con un error de {error}")
    print(f"Se resolvio en {count} iteraciones")

def gauss_seidel(A,B, Xn, Xv, omega=1):
    n = len(A)
    count = 0
    errorV = 1000
    tolerancia = float(input("Ingrese la tolerancia: "))
    while True:
        count+=1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    if j < i:
                        suma += A[i][j] * Xn[j]   # ya actualizado
                    else:
                        suma += A[i][j] * Xv[j]   # valor viejo
            Xn[i] = (B[i] - suma) / A[i][i]
    
            Xn[i] = omega*Xn[i] + (1-omega)*Xv[i] #Factor de Relajacion, por default omega = 1

        error = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
        error = math.sqrt(error) 
        if error > errorV:
            print("El metodo no converge")
            return
        errorV = error
        for i in range(n):
            Xv[i]=Xn[i]
        if error < tolerancia:
            break;
    print("Las soluciones son:")
    for i in range(n):
        print(f"X{i} = {Xn[i]}")
    print(f"Con un error de {error}")
    print(f"Se resolvio en {count} iteraciones")


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

def raiz_punto_fijo(g, g_prime, a, tolerancia, tipo_error):
    contador=0
    tolerancia= tolerancia/100 if tipo_error==2 else tolerancia
    while True:
        contador+=1
        if abs(g_prime(g ,a))>=1:
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

def newton_raphson(f, f_prime, x0, tolerancia, tipo_error):
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


def triangulacion(A, B):
    n = len(A)
    for i in range(n-1):
        p=i
        if abs(A[i][i]) < 1e-2:
            for l in range(i+1, n):
                if abs(A[l][i]) > abs(A[p][i]):
                    p=l

            for m in range(i, n):
                A[p][m] , A[i][m] = A[i][m], A[p][m]

            B[p], B[i] = B[i], B[p]

        for j in range (i+1, n):
            factor = -A[j][i]/A[i][i]
            for k in range(i, n):
                A[j][k] = A[i][k]*factor + A[j][k]

            B[j] = B[i]*factor + B[j]

    pp.pprint(A)
    print()

def determinante (A):
    n = len(A)
    prod = 1
    for i in range(n):
        prod*=A[i][i]
    if prod==0:
        print("Matriz determinante 0\n")
        return
    print("Determinante = ",prod,"\n")

def gauss_pivot(A, B):
    n = len(A)
    X = [0] * n
    triangulacion(A, B)
    determinante(A)

    X[n-1] = B[n-1]/A[n-1][n-1]

    for i in range(n-1, -1, -1):
        sum = B[i]
        for j in range(i+1, n):
            sum = sum - A[i][j]*X[j]
        sum = sum/A[i][i]

        X[i] = sum
    print("Las soluciones del sistema son: ")
    for i in range(n):
        print(f"X{i} = {X[i]}")
    return X


def leer_puntos_xy(nombre_archivo):
    xs, ys = [], []
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.replace(",", " ").split()
            if len(partes) != 2:
                raise ValueError(f"Línea inválida: {linea}")
            x, y = map(float, partes)
            xs.append(x)
            ys.append(y)
    return xs, ys

def interpolacion(nombre_archivo):
    X, Y = leer_puntos_xy(nombre_archivo)

    n = len(X)

    A = [[0 for j in range(n)] for i in range(n)]
    B = [0 for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = X[i]**j
        B[i] = Y[i]

    coef = gauss_pivot(A, B)  # coef[0] = x^0, coef[1] = x^1, ...

    # función polinómica evaluable
    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    return p, coef, X, Y

def graficar_interpolacion(p, coef, X, Y, funcion_real=None):
    x_vals = np.linspace(min(X), max(X), 500)
    y_vals = [p(x) for x in x_vals]

    plt.scatter(X, Y, color="red", label="Puntos datos")
    plt.plot(x_vals, y_vals, label="Polinomio interpolador", color="blue")

    if funcion_real is not None:
        y_real = [funcion_real(x) for x in x_vals]
        plt.plot(x_vals, y_real, label="Función real (posible origen)", linestyle="--", color="green")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolación polinómica")
    plt.show()

    print("Coeficientes del polinomio (ordenados por potencia):")
    for i, c in enumerate(coef):
        print(f"x^{i}: {c}")
