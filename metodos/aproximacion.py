"""
Métodos numéricos para aproximación de datos.

Este módulo contiene implementaciones de métodos de interpolación, regresión
y construcción de splines cúbicos para aproximar funciones a partir de datos.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from .sistemas_lineales import gauss_pivot


def leer_puntos_xy(nombre_archivo):
    """
    Lee puntos (x,y) desde un archivo de texto.
    El archivo debe contener dos columnas de números separados por coma o espacio.

    Args:
        nombre_archivo (str): Ruta al archivo que contiene los puntos

    Returns:
        tuple[list[float], list[float]]: Listas de coordenadas x e y

    Raises:
        ValueError: Si una línea del archivo no contiene exactamente dos valores
    """
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


def _validar_datos_xy(X=None, Y=None, nombre_archivo=None):
    """
    Función auxiliar para validar y preparar datos de entrada X,Y.
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        tuple[list[float], list[float]]: Datos X,Y validados y convertidos a float
        
    Raises:
        ValueError: Si los datos de entrada no son válidos
    """
    if nombre_archivo is not None:
        X, Y = leer_puntos_xy(nombre_archivo)
    elif X is not None and Y is not None:
        if len(X) != len(Y):
            raise ValueError("Las listas X e Y deben tener la misma longitud")
        X = list(map(float, X))
        Y = list(map(float, Y))
    else:
        raise ValueError("Debe proporcionar arrays X e Y, o un nombre de archivo")
    return X, Y


def _ordenar_puntos_xy(X, Y, verbose=True):
    """
    Ordena los puntos (X,Y) según los valores de X de forma ascendente.
    
    Args:
        X (list[float]): Coordenadas x
        Y (list[float]): Coordenadas y
        verbose (bool, optional): Si True, muestra advertencias. Por defecto True.
        
    Returns:
        tuple[list[float], list[float]]: Puntos ordenados (X_ord, Y_ord)
        
    Raises:
        ValueError: Si hay valores duplicados en X
    """
    n = len(X)
    
    indices_ordenados = sorted(range(n), key=lambda i: X[i])
    
    X_ordenado = [X[i] for i in indices_ordenados]
    Y_ordenado = [Y[i] for i in indices_ordenados]
    
    for i in range(n-1):
        if abs(X_ordenado[i+1] - X_ordenado[i]) < 1e-12:
            raise ValueError(f"Valores duplicados en X: X[{i}] = X[{i+1}] = {X_ordenado[i]}. "
                           "Se requieren valores únicos para interpolación.")
    
    if verbose and indices_ordenados != list(range(n)):
        print("⚠️  ADVERTENCIA: Los puntos se reordenaron según X.")
        print(f"   Original: X = {X}")
        print(f"   Ordenado: X = {X_ordenado}")
    
    return X_ordenado, Y_ordenado


def interpolacion_lagrange(input_x, X=None, Y=None, nombre_archivo=None):
    """
    Calcula el valor interpolado para un punto dado usando el método de Lagrange.
    
    Args:
        input_x (float): Punto donde se desea interpolar
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        
    Returns:
        float: Valor interpolado en input_x
        
    Raises:
        ValueError: Si los datos de entrada no son válidos
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    n = len(X)

    suma = 0
    for i in range(n):
        prod = 1
        for j in range(n):
            if i != j:
                prod = prod * (input_x - X[j])/(X[i]-X[j])
        suma = suma + Y[i]*prod
    return suma


def interpolacion(X=None, Y=None, nombre_archivo=None, ordenar_automatico=True, verbose=True):
    """
    Realiza interpolación polinómica usando el método de Vandermonde.
    Construye un polinomio que pasa exactamente por todos los puntos dados.

    Args:
        X (list[float], optional): Lista de coordenadas x de los puntos.
        Y (list[float], optional): Lista de coordenadas y de los puntos.
        nombre_archivo (str, optional): Archivo con los puntos a interpolar.
        ordenar_automatico (bool, optional): Si True, ordena los puntos automáticamente. Por defecto True.
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.

    Returns:
        tuple:
            - callable: Función del polinomio interpolador
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos

    Note:
        - Usa la matriz de Vandermonde: A[i][j] = x_i^j
        - Resuelve el sistema usando eliminación gaussiana
        - El grado del polinomio es n-1, donde n es el número de puntos
        - Puede ser numéricamente inestable para muchos puntos
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    
    if ordenar_automatico:
        X, Y = _ordenar_puntos_xy(X, Y, verbose=verbose)

    n = len(X)

    A = [[0 for j in range(n)] for i in range(n)]
    B = [0 for j in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = X[i]**j
        B[i] = Y[i]

    coef = gauss_pivot(A, B, verbose=False)

    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    return p, coef, X, Y


def regresion_polinomica(X=None, Y=None, nombre_archivo=None, grado=1, verbose=True):
    """
    Realiza una regresión polinómica de grado especificado sobre los datos.
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y
        nombre_archivo (str, optional): Ruta al archivo de datos
        grado (int, optional): Grado del polinomio de regresión. Por defecto 1 (regresión lineal)
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.
        
    Returns:
        tuple:
            - callable: Función del polinomio de regresión
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos
            - float: Coeficiente de correlación (r)
            
    Raises:
        ValueError: Si los datos son insuficientes para el grado especificado
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    n = len(X)
    
    if n <= grado:
        raise ValueError(f"Se necesitan al menos {grado + 1} puntos para una regresión de grado {grado}")
    
    A = [[0 for j in range(grado + 1)] for i in range(grado + 1)]
    B = [0 for j in range(grado + 1)]

    for l in range(grado + 1):
        sumaxy = sum(y * x**l for x, y in zip(X, Y))
        B[l] = sumaxy
        for m in range(grado + 1):
            sumax = sum(x**(l+m) for x in X)
            A[l][m] = sumax

    coef = gauss_pivot(A, B, verbose=False)

    def p(x):
        return sum(c * (x**i) for i, c in enumerate(coef))

    Y_prom = sum(Y) / n
    Sr = sum((p(x) - y)**2 for x, y in zip(X, Y))
    St = sum((y - Y_prom)**2 for y in Y)
    
    r = math.sqrt((St - Sr)/St) if St > 0 else 0

    return p, coef, X, Y, r


def curvas_spline(X=None, Y=None, nombre_archivo=None, verbose=True):
    """
    Construye splines cúbicos naturales para interpolar puntos dados.
    
    Para n puntos tenemos n-1 intervalos, cada uno con un polinomio cúbico:
    S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k para x en [X_k, X_{k+1}]
    
    Args:
        X (list[float], optional): Lista de coordenadas x
        Y (list[float], optional): Lista de coordenadas y  
        nombre_archivo (str, optional): Ruta al archivo de datos
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.
        
    Returns:
        tuple: (funciones_spline, coeficientes, X, Y)
            - funciones_spline: lista de funciones para cada intervalo
            - coeficientes: matriz de coeficientes [a_k, b_k, c_k, d_k] para cada k
            - X, Y: puntos originales ordenados
    """
    X, Y = _validar_datos_xy(X, Y, nombre_archivo)
    X, Y = _ordenar_puntos_xy(X, Y, verbose=verbose)
    n = len(X)
    
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para construir splines cúbicos")
    
    num_intervalos = n - 1
    num_coef = 4 * num_intervalos
    
    A = [[0 for i in range(num_coef)] for j in range(num_coef)]
    B = [0 for i in range(num_coef)]
    
    fila = 0
    
    # 1. Ecuaciones de evaluación: 2*(n-1) ecuaciones
    for k in range(num_intervalos):
        for j in range(4):
            A[fila][4*k + j] = X[k]**(3-j)
        B[fila] = Y[k]
        fila += 1
        
        for j in range(4):
            A[fila][4*k + j] = X[k+1]**(3-j)
        B[fila] = Y[k+1]
        fila += 1
    
    # 2. Ecuaciones de continuidad de primera derivada: (n-2) ecuaciones
    for k in range(num_intervalos - 1):
        x_punto = X[k+1]
        
        A[fila][4*k] = 3*x_punto**2
        A[fila][4*k + 1] = 2*x_punto
        A[fila][4*k + 2] = 1
        
        A[fila][4*(k+1)] = -3*x_punto**2
        A[fila][4*(k+1) + 1] = -2*x_punto
        A[fila][4*(k+1) + 2] = -1
        
        B[fila] = 0
        fila += 1
    
    # 3. Ecuaciones de continuidad de segunda derivada: (n-2) ecuaciones
    for k in range(num_intervalos - 1):
        x_punto = X[k+1]
        
        A[fila][4*k] = 6*x_punto
        A[fila][4*k + 1] = 2
        
        A[fila][4*(k+1)] = -6*x_punto
        A[fila][4*(k+1) + 1] = -2
        
        B[fila] = 0
        fila += 1
    
    # 4. Condiciones de frontera naturales: 2 ecuaciones
    A[fila][0] = 6*X[0]
    A[fila][1] = 2
    B[fila] = 0
    fila += 1
    
    last_interval = num_intervalos - 1
    A[fila][4*last_interval] = 6*X[n-1]
    A[fila][4*last_interval + 1] = 2
    B[fila] = 0
    fila += 1
    
    coeficientes_planos = gauss_pivot(A, B, verbose=False)
    
    coeficientes = []
    for k in range(num_intervalos):
        a_k = coeficientes_planos[4*k]
        b_k = coeficientes_planos[4*k + 1]
        c_k = coeficientes_planos[4*k + 2]
        d_k = coeficientes_planos[4*k + 3]
        coeficientes.append([a_k, b_k, c_k, d_k])
    
    funciones_spline = []
    for k in range(num_intervalos):
        a_k, b_k, c_k, d_k = coeficientes[k]
        def crear_spline(a, b, c, d):
            return lambda x: a*x**3 + b*x**2 + c*x + d
        funciones_spline.append(crear_spline(a_k, b_k, c_k, d_k))
    
    return funciones_spline, coeficientes, X, Y


def evaluar_spline(x, funciones_spline, X):
    """
    Evalúa el spline en un punto x, encontrando automáticamente el intervalo correcto.
    
    Args:
        x (float): Punto donde evaluar el spline
        funciones_spline (list[callable]): Lista de funciones spline por intervalo
        X (list[float]): Puntos de interpolación (nodos)
    
    Returns:
        float: Valor del spline en x
        
    Raises:
        ValueError: Si x está fuera del dominio de los splines
    """
    if x < X[0] or x > X[-1]:
        raise ValueError(f"El punto x={x} está fuera del dominio [{X[0]}, {X[-1]}]")
    
    for k in range(len(X) - 1):
        if X[k] <= x <= X[k+1]:
            return funciones_spline[k](x)
    
    return funciones_spline[-1](x)


# Funciones de graficación

def graficar_interpolacion(p, coef, X, Y, funcion_real=None, a=None, b=None):
    """
    Visualiza el resultado de la interpolación polinómica.
    """
    if a is None:
        a = min(X)
    if b is None:
        b = max(X)

    x_vals = np.linspace(a, b, 500)
    y_vals = [p(x) for x in x_vals]

    plt.scatter(X, Y, color="red", label="Puntos datos")
    plt.plot(x_vals, y_vals, label="Polinomio interpolador", color="blue")

    if funcion_real is not None:
        y_real = [funcion_real(x) for x in x_vals]
        plt.plot(x_vals, y_real, label="Función real (posible origen)", linestyle="--", color="green")
    
    print("Coeficientes del polinomio (ordenados por potencia):")
    for i, c in enumerate(coef):
        print(f"x^{i}: {c}")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolación polinómica")
    plt.grid(True)
    plt.show()


def graficar_regresion(p, coef, X, Y, r):
    """
    Visualiza el resultado de la regresión polinómica.
    """
    x_vals = np.linspace(min(X), max(X), 500)
    y_vals = [p(x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color="red", label="Puntos datos")
    plt.plot(x_vals, y_vals, label=f"Regresión (r = {r:.4f})", color="blue")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regresión Polinómica")
    
    print("\nCoeficientes del polinomio (ordenados por potencia):")
    for i, c in enumerate(coef):
        print(f"x^{i}: {c}")

    plt.show()


def visualizar_polinomio(polinomio, coef, X, Y, x_eval, titulo="Polinomio Interpolador", verbose=True):
    """
    Visualiza el polinomio interpolador y marca el punto específico de evaluación.
    
    Args:
        polinomio (callable): Función del polinomio interpolador
        coef (list[float]): Coeficientes del polinomio
        X (list[float]): Coordenadas x de los puntos de interpolación
        Y (list[float]): Coordenadas y de los puntos de interpolación
        x_eval (float): Punto donde se evalúa el polinomio
        titulo (str, optional): Título del gráfico. Por defecto "Polinomio Interpolador"
        verbose (bool, optional): Si False, solo imprime la función en una línea. Por defecto True.
    """
    # Calcular el valor en el punto de evaluación
    y_eval = polinomio(x_eval)
    
    # Crear rango para graficar el polinomio
    x_min = min(min(X), x_eval) - 0.5
    x_max = max(max(X), x_eval) + 0.5
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = [polinomio(x) for x in x_vals]
    
    # Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Graficar el polinomio
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Polinomio interpolador')
    
    # Graficar los puntos de datos originales
    plt.scatter(X, Y, color='red', s=100, zorder=5, label='Puntos de datos', marker='o')
    
    # Marcar el punto de evaluación
    plt.scatter([x_eval], [y_eval], color='green', s=150, zorder=6, 
                marker='*', edgecolors='black', linewidth=1.5,
                label=f'Evaluación en x={x_eval:.2f}')
    
    # Línea vertical en el punto de evaluación
    plt.axvline(x=x_eval, color='green', linestyle='--', alpha=0.3)
    
    # Línea horizontal en el valor de evaluación
    plt.axhline(y=y_eval, color='green', linestyle='--', alpha=0.3)
    
    # Configurar el gráfico
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    
    # Mostrar información en el gráfico
    texto = f'P({x_eval:.2f}) = {y_eval:.6f}'
    plt.text(x_eval, y_eval, f'  {texto}', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Imprimir información del polinomio
    if verbose:
        print(f"\n{'='*60}")
        print(f"POLINOMIO INTERPOLADOR (grado {len(coef)-1})")
        print(f"{'='*60}")
        print("\nForma general: P(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ\n")
        print("Coeficientes:")
        for i, c in enumerate(coef):
            signo = '+' if c >= 0 else ''
            print(f"  c{i} (coef. de x^{i}): {signo}{c:.8f}")
        
        print(f"\nForma explícita:")
        terminos = []
        for i, c in enumerate(coef):
            if abs(c) < 1e-10:
                continue
            if i == 0:
                terminos.append(f"{c:.6f}")
            elif i == 1:
                terminos.append(f"{c:+.6f}x")
            else:
                terminos.append(f"{c:+.6f}x^{i}")
        print(f"  P(x) = {' '.join(terminos)}")
        
        print(f"\nEvaluación en x = {x_eval}:")
        print(f"  P({x_eval}) = {y_eval:.8f}")
        print(f"{'='*60}\n")
    else:
        # Formato simplificado: una línea
        terminos = []
        for i, c in enumerate(coef):
            if abs(c) < 1e-10:
                continue
            if i == 0:
                terminos.append(f"{c:.6f}")
            elif i == 1:
                terminos.append(f"{c:+.6f}x")
            else:
                terminos.append(f"{c:+.6f}x^{i}")
        print(f"P(x) = {' '.join(terminos)}")
    
    plt.tight_layout()
    plt.show()


def visualizar_spline(funciones_spline, coeficientes, X, Y, x_eval, titulo="Spline Cúbico", verbose=True):
    """
    Visualiza el spline cúbico y marca el punto específico de evaluación.
    
    Args:
        funciones_spline (list[callable]): Lista de funciones spline por intervalo
        coeficientes (list[list[float]]): Coeficientes [a,b,c,d] de cada spline
        X (list[float]): Coordenadas x de los puntos de interpolación
        Y (list[float]): Coordenadas y de los puntos de interpolación
        x_eval (float): Punto donde se evalúa el spline
        titulo (str, optional): Título del gráfico. Por defecto "Spline Cúbico"
        verbose (bool, optional): Si False, solo imprime la función en una línea. Por defecto True.
    """
    # Calcular el valor en el punto de evaluación
    y_eval = evaluar_spline(x_eval, funciones_spline, X)
    
    # Encontrar el intervalo donde está x_eval
    intervalo_eval = None
    for k in range(len(X) - 1):
        if X[k] <= x_eval <= X[k+1]:
            intervalo_eval = k
            break
    
    # Crear la figura
    plt.figure(figsize=(12, 7))
    
    # Graficar cada segmento del spline
    colores = plt.cm.tab10(np.linspace(0, 1, len(funciones_spline)))
    
    for k in range(len(funciones_spline)):
        x_intervalo = np.linspace(X[k], X[k+1], 100)
        y_intervalo = [funciones_spline[k](x) for x in x_intervalo]
        
        # Resaltar el intervalo donde se evalúa
        if k == intervalo_eval:
            plt.plot(x_intervalo, y_intervalo, 
                    color='green', linewidth=3, 
                    label=f'Spline activo: [{X[k]:.2f}, {X[k+1]:.2f}]',
                    zorder=4)
        else:
            plt.plot(x_intervalo, y_intervalo, 
                    color=colores[k], linewidth=2, alpha=0.6,
                    label=f'Spline {k+1}: [{X[k]:.2f}, {X[k+1]:.2f}]')
    
    # Graficar los puntos de datos originales
    plt.scatter(X, Y, color='red', s=100, zorder=5, label='Puntos de datos', marker='o')
    
    # Marcar el punto de evaluación
    plt.scatter([x_eval], [y_eval], color='green', s=200, zorder=6, 
                marker='*', edgecolors='black', linewidth=2,
                label=f'Evaluación en x={x_eval:.2f}')
    
    # Línea vertical en el punto de evaluación
    plt.axvline(x=x_eval, color='green', linestyle='--', alpha=0.3)
    
    # Línea horizontal en el valor de evaluación
    plt.axhline(y=y_eval, color='green', linestyle='--', alpha=0.3)
    
    # Configurar el gráfico
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    
    # Mostrar información en el gráfico
    texto = f'S({x_eval:.2f}) = {y_eval:.6f}'
    plt.text(x_eval, y_eval, f'  {texto}', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Imprimir información del spline activo
    if intervalo_eval is not None:
        a, b, c, d = coeficientes[intervalo_eval]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SPLINE CÚBICO ACTIVO EN x = {x_eval}")
            print(f"{'='*70}")
            print(f"\nIntervalo: [{X[intervalo_eval]:.4f}, {X[intervalo_eval+1]:.4f}]")
            print(f"Spline número: {intervalo_eval + 1} de {len(funciones_spline)}")
            print(f"\nForma general: S(x) = ax³ + bx² + cx + d")
            print(f"\nCoeficientes:")
            print(f"  a (coef. de x³): {a:.8f}")
            print(f"  b (coef. de x²): {b:.8f}")
            print(f"  c (coef. de x):  {c:.8f}")
            print(f"  d (término indep.): {d:.8f}")
            print(f"\nForma explícita:")
            print(f"  S(x) = {a:.6f}x³ {b:+.6f}x² {c:+.6f}x {d:+.6f}")
            print(f"\nEvaluación:")
            print(f"  S({x_eval}) = {a:.6f}·({x_eval})³ {b:+.6f}·({x_eval})² {c:+.6f}·({x_eval}) {d:+.6f}")
            print(f"  S({x_eval}) = {y_eval:.8f}")
            print(f"{'='*70}\n")
        else:
            # Formato simplificado: una línea
            print(f"S(x) = {a:.6f}x³ {b:+.6f}x² {c:+.6f}x {d:+.6f}  [Intervalo {intervalo_eval+1}: [{X[intervalo_eval]:.2f}, {X[intervalo_eval+1]:.2f}]]")
    
    plt.tight_layout()
    plt.show()


def graficar_splines(funciones_spline, coeficientes, X, Y, funcion_real=None):
    """
    Visualiza las curvas spline junto con los puntos originales.
    """
    plt.figure(figsize=(12, 8))
    
    plt.scatter(X, Y, color="red", s=50, label="Puntos datos", zorder=5)
    
    colores = plt.cm.tab10(np.linspace(0, 1, len(funciones_spline)))
    
    for k in range(len(funciones_spline)):
        x_intervalo = np.linspace(X[k], X[k+1], 100)
        y_intervalo = [funciones_spline[k](x) for x in x_intervalo]
        
        plt.plot(x_intervalo, y_intervalo, 
                color=colores[k], 
                label=f'Spline {k+1}: [{X[k]:.2f}, {X[k+1]:.2f}]',
                linewidth=2)
    
    if funcion_real is not None:
        x_continuo = np.linspace(min(X), max(X), 500)
        y_real = [funcion_real(x) for x in x_continuo]
        plt.plot(x_continuo, y_real, 
                label="Función real", 
                linestyle="--", 
                color="green", 
                alpha=0.7,
                linewidth=2)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolación con Splines Cúbicos Naturales")
    
    print("\nCoeficientes de los splines cúbicos:")
    print("S_k(x) = a_k*x³ + b_k*x² + c_k*x + d_k")
    print("-" * 50)
    for k, (a, b, c, d) in enumerate(coeficientes):
        print(f"Intervalo {k+1} [{X[k]:.2f}, {X[k+1]:.2f}]:")
        print(f"  S_{k+1}(x) = {a:.4f}*x³ + {b:.4f}*x² + {c:.4f}*x + {d:.4f}")
    
    plt.show()
