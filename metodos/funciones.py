import math
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

def limpiar_terminal():
    """
    Limpia la pantalla de la terminal de manera compatible con múltiples sistemas operativos.
    Utiliza el comando 'clear' en sistemas Unix/Linux/Mac y 'cls' en Windows.
    """
    os.system('clear' if os.name == 'posix' else 'cls')

def resolverJG(A, B):
    """
    Resuelve un sistema de ecuaciones lineales utilizando métodos iterativos (Jacobi o Gauss-Seidel).
    
    Args:
        A (list[list[float]]): Matriz de coeficientes del sistema
        B (list[float]): Vector de términos independientes
    
    Returns:
        list[float]: Vector solución del sistema

    Note:
        La función verifica si la matriz es diagonalmente dominante antes de proceder.
        Permite elegir entre el método de Jacobi o Gauss-Seidel.
    """
    esDiagDom(A)

    n = len(A)
    Xn = [0] * n  # Vector de soluciones nuevas
    Xv = [0] * n  # Vector de soluciones viejas

    print("1-Resolver con Jacobi")
    print("2-Resolver con Gauss-Seidel")
    opcion = int(input())
    if opcion == 1:
        jacobi(A, B, Xn, Xv)
    elif opcion == 2:
        gauss_seidel(A,B, Xn, Xv)
    
    return Xn
    
def esDiagDom(A):
    """
    Verifica si una matriz es diagonalmente dominante.
    Una matriz es diagonalmente dominante si el valor absoluto de cada elemento diagonal
    es mayor que la suma de los valores absolutos de los demás elementos en su fila.

    Args:
        A (list[list[float]]): Matriz cuadrada a verificar

    Returns:
        None: Imprime un mensaje si la matriz no es diagonalmente dominante
    
    Note:
        Esta propiedad es importante para garantizar la convergencia de métodos iterativos
        como Jacobi y Gauss-Seidel.
    """
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
    """
    Implementa el método iterativo de Jacobi para resolver sistemas de ecuaciones lineales.
    El método de Jacobi actualiza cada componente de la solución usando los valores de la iteración anterior.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior

    Note:
        - Requiere que la matriz sea diagonalmente dominante para garantizar convergencia
        - En cada iteración, calcula x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - Se detiene cuando el error es menor que la tolerancia especificada
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
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

def gauss_seidel(A, B, Xn, Xv, omega=1):
    """
    Implementa el método iterativo de Gauss-Seidel con factor de relajación (SOR) para resolver sistemas de ecuaciones lineales.
    A diferencia de Jacobi, utiliza los valores actualizados tan pronto como estén disponibles.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior
        omega (float, optional): Factor de relajación. Defaults to 1.
            - omega < 1: sub-relajación
            - omega = 1: método de Gauss-Seidel estándar
            - omega > 1: sobre-relajación

    Note:
        - Generalmente converge más rápido que Jacobi
        - Usa valores actualizados inmediatamente: x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - El factor de relajación puede acelerar la convergencia
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
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
    """
    Encuentra una raíz de una función en un intervalo dado usando el método de Bisección o Regula Falsi.

    Args:
        f (callable): Función continua de la cual se busca la raíz
        a (float): Límite inferior del intervalo
        b (float): Límite superior del intervalo
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual
        tolerancia (float): Tolerancia deseada para el error
        metodo (int, optional): Método a utilizar
            1: Bisección - divide el intervalo por la mitad
            2: Regula Falsi - usa interpolación lineal
            None: Permite al usuario elegir el método

    Returns:
        tuple[float, list[float]]: Tupla con la raíz encontrada y lista de errores en cada iteración

    Note:
        - Requiere que f(a) y f(b) tengan signos opuestos (teorema del valor intermedio)
        - Bisección: convergencia garantizada pero lenta
        - Regula Falsi: convergencia más rápida pero puede ser lenta en algunos casos
    """
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


def triangulacion(A, B):
    """
    Realiza la triangulación superior de una matriz aumentada [A|B] usando eliminación gaussiana con pivoteo.
    
    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes

    Note:
        - Modifica las matrices A y B in-place
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - El pivoteo selecciona el elemento más grande en valor absoluto de cada columna
        - Evita problemas con elementos diagonales cercanos a cero
    """
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

def determinante(A):
    """
    Calcula el determinante de una matriz triangular superior.
    Para una matriz triangular, el determinante es el producto de los elementos diagonales.

    Args:
        A (list[list[float]]): Matriz triangular superior

    Note:
        - Asume que la matriz ya está en forma triangular superior
        - Un determinante cero indica que el sistema no tiene solución única
        - Se usa después de la triangulación en el método de Gauss
    """
    n = len(A)
    prod = 1
    for i in range(n):
        prod*=A[i][i]
    if prod==0:
        print("Matriz determinante 0\n")
        return
    print("Determinante = ",prod,"\n")

def gauss_pivot(A, B):
    """
    Resuelve un sistema de ecuaciones lineales usando eliminación gaussiana con pivoteo.
    El método consiste en tres pasos: triangulación, verificación de determinante y sustitución hacia atrás.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes

    Returns:
        list[float]: Vector solución del sistema

    Note:
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - Verifica si el sistema tiene solución única (determinante ≠ 0)
        - La sustitución hacia atrás resuelve el sistema triangular superior
    """
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

def interpolacion(X=None, Y=None, nombre_archivo=None):
    """
    Realiza interpolación polinómica usando el método de Vandermonde.
    Construye un polinomio que pasa exactamente por todos los puntos dados.

    Args:
        X (list[float], optional): Lista de coordenadas x de los puntos.
        Y (list[float], optional): Lista de coordenadas y de los puntos.
        nombre_archivo (str, optional): Archivo con los puntos a interpolar.
            Si se proporciona, los puntos se leerán del archivo ignorando X e Y.

    Returns:
        tuple:
            - callable: Función del polinomio interpolador
            - list[float]: Coeficientes del polinomio
            - list[float]: Coordenadas x de los puntos
            - list[float]: Coordenadas y de los puntos

    Raises:
        ValueError: Si no se proporcionan ni puntos ni archivo, o si X e Y tienen diferente longitud

    Note:
        - Usa la matriz de Vandermonde: A[i][j] = x_i^j
        - Resuelve el sistema usando eliminación gaussiana
        - El grado del polinomio es n-1, donde n es el número de puntos
        - Puede ser numéricamente inestable para muchos puntos

    Example:
        # Usando arrays directamente
        X = [1, 2, 3]
        Y = [1, 4, 9]
        p, coef, X, Y = interpolacion(X=X, Y=Y)

        # Usando un archivo
        p, coef, X, Y = interpolacion(nombre_archivo='datos.txt')
    """
    if nombre_archivo is not None:
        X, Y = leer_puntos_xy(nombre_archivo)
    elif X is not None and Y is not None:
        if len(X) != len(Y):
            raise ValueError("Las listas X e Y deben tener la misma longitud")
        X = list(map(float, X))  # Asegurar que los valores son float
        Y = list(map(float, Y))
    else:
        raise ValueError("Debe proporcionar arrays X e Y, o un nombre de archivo")

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
    """
    Visualiza el resultado de la interpolación polinómica.

    Args:
        p (callable): Función del polinomio interpolador
        coef (list[float]): Coeficientes del polinomio
        X (list[float]): Coordenadas x de los puntos originales
        Y (list[float]): Coordenadas y de los puntos originales
        funcion_real (callable, optional): Función original si se conoce

    Note:
        - Muestra los puntos originales en rojo
        - Grafica el polinomio interpolador en azul
        - Si se proporciona, muestra la función original en verde punteado
        - Imprime los coeficientes del polinomio ordenados por potencia
    """
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

def graficar_funciones(*funciones, nombres=None, x_min=-10, x_max=10, n_puntos=1000):
    """
    Crea un gráfico interactivo y zoomeable de una o más funciones matemáticas.

    Args:
        *funciones (callable): Una o más funciones matemáticas para graficar
        nombres (list[str], optional): Lista con los nombres de las funciones para la leyenda.
            Si no se proporciona, se usarán nombres genéricos (f1, f2, etc.)
        x_min (float, optional): Límite inferior del dominio. Por defecto -10
        x_max (float, optional): Límite superior del dominio. Por defecto 10
        n_puntos (int, optional): Número de puntos para el gráfico. Por defecto 1000

    Example:
        # Graficar una función
        graficar_funciones(lambda x: x**2)

        # Graficar múltiples funciones con nombres personalizados
        graficar_funciones(
            lambda x: x**2,
            lambda x: math.sin(x),
            nombres=['Parábola', 'Seno']
        )
    
    Note:
        - Use las teclas + y - para hacer zoom
        - Use las flechas del teclado para moverse por el gráfico
        - Use el botón de reset para volver a la vista inicial
        - Los colores se asignan automáticamente para cada función
    """
    # Validar entradas
    if not funciones:
        raise ValueError("Debe proporcionar al menos una función")
    
    if nombres is None:
        nombres = [f'f{i+1}' for i in range(len(funciones))]
    elif len(nombres) != len(funciones):
        raise ValueError("La cantidad de nombres debe coincidir con la cantidad de funciones")

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)  # Hacer espacio para el botón

    # Generar los datos
    x = np.linspace(x_min, x_max, n_puntos)
    
    # Graficar cada función
    lines = []
    for f, nombre in zip(funciones, nombres):
        try:
            y = [f(xi) for xi in x]
            line, = ax.plot(x, y, label=nombre)
            lines.append(line)
        except Exception as e:
            print(f"Error al graficar {nombre}: {e}")

    # Configurar el gráfico
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Guardar los límites originales para el botón de reset
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    # Agregar botón de reset
    reset_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
    reset_button = Button(reset_ax, 'Reset')

    def reset(event):
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)
        plt.draw()

    reset_button.on_clicked(reset)

    # Configurar el zoom con el teclado
    def on_key(event):
        if event.key == '=':  # Zoom in
            ax.set_xlim(ax.get_xlim()[0] * 0.9, ax.get_xlim()[1] * 0.9)
            ax.set_ylim(ax.get_ylim()[0] * 0.9, ax.get_ylim()[1] * 0.9)
        elif event.key == '-':  # Zoom out
            ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
            ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        elif event.key == 'left':
            xlim = ax.get_xlim()
            delta = (xlim[1] - xlim[0]) * 0.1
            ax.set_xlim(xlim[0] - delta, xlim[1] - delta)
        elif event.key == 'right':
            xlim = ax.get_xlim()
            delta = (xlim[1] - xlim[0]) * 0.1
            ax.set_xlim(xlim[0] + delta, xlim[1] + delta)
        elif event.key == 'up':
            ylim = ax.get_ylim()
            delta = (ylim[1] - ylim[0]) * 0.1
            ax.set_ylim(ylim[0] + delta, ylim[1] + delta)
        elif event.key == 'down':
            ylim = ax.get_ylim()
            delta = (ylim[1] - ylim[0]) * 0.1
            ax.set_ylim(ylim[0] - delta, ylim[1] - delta)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
