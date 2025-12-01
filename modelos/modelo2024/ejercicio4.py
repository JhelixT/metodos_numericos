import math
import matplotlib.pyplot as plt
def main():
    f= lambda x, y: -2*y + math.exp(-x)
    X, Y = heum_orden3(f, x0=0, y0=2, xf=1, n=100, verbose=False) #h=0.01
    Y_exacta = lambda x: math.exp(-2*x) + math.exp(-x)

    # Graficar resultados
    plt.plot(X, Y, label="Heun Orden 3", color='black')
    plt.plot(X, [Y_exacta(x) for x in X], label="Solución Exacta", linestyle='dashed', color='red')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparación Método de Heun Orden 3 vs Solución Exacta")
    plt.show()

    #Error Global
    errores = [abs(Y_exacta(X[i]) - Y[i]) for i in range(len(X))]
    error_global = max(errores)
    print(f"Error global máximo: {error_global:.6e}")

    #Valores en y(0.4) y y(0.8)
    print(f"Valor aproximado en y(0.4): {Y[encontrar_indice(X,0.4)]:.6f}")
    print(f"Valor aproximado en y(0.8): {Y[encontrar_indice(X,0.8)]:.6f}") 

def heum_orden3(f, x0, y0, xf, n, verbose=True):
    """
    Resuelve una EDO de primer orden usando el método de Heum de orden 3
    
    Args:
        f (callable): Función f(x,y) que define la EDO dy/dx = f(x,y)
        x0 (float): Valor inicial de x
        y0 (float): Valor inicial de y (condición inicial)
        xf (float): Valor final de x
        n (int): Número de pasos (debe ser >= 1)
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        tuple: (X, Y) donde:
            - X: lista de valores de x
            - Y: lista de valores aproximados de y
            
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    # Validaciones
    if not callable(f):
        raise ValueError("f debe ser una función callable de la forma f(x, y)")
    
    if x0 >= xf:
        raise ValueError(f"x0={x0} debe ser menor que xf={xf}")
    
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, recibido: {n}")
    
    h = (xf - x0) / n
    
    if verbose:
        print(f"Método de Heum Orden 3")
        print(f"EDO: dy/dx = f(x,y), y({x0}) = {y0}")
        print(f"Intervalo: [{x0}, {xf}] con {n} pasos (h = {h:.6f})")
    
    # Inicializar arrays
    X = [0] * (n + 1)
    Y = [0] * (n + 1)
    
    X[0] = x0
    Y[0] = y0
    
    for i in range(n):
        X[i+1] = X[0] + (i+1) * h

        k1 = f(X[i], Y[i])
        k2 = f(X[i] + h/3, Y[i] + h*k1/3)
        k3 = f(X[i] + 2*h/3, Y[i] + 2*h*k2/3)
        
        Y[i+1] = Y[i] + h*(k1/4 + 3*k3/4)
    
    if verbose:
        print(f"Solución calculada en {n+1} puntos")
        print(f"Valor final: y({xf}) ≈ {Y[-1]:.6f}")
    
    return X, Y

def encontrar_indice(lista, valor, tolerancia=1e-9):
    for i, x in enumerate(lista):
        if abs(x - valor) < tolerancia:  # Compara con tolerancia
            return i
    raise ValueError(f"Valor {valor} no encontrado")

main()