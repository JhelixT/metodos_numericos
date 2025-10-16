"""
Métodos numéricos para localización de raíces de funciones.

Este módulo contiene implementaciones de diferentes métodos iterativos para
encontrar raíces (ceros) de funciones continuas.
"""

def buscar_raiz(f, a, b, tipo_error, tolerancia, metodo=None, verbose=True):
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
            None: Permite al usuario elegir el método (requiere verbose=True)
        verbose (bool, optional): Si True, imprime resultados y permite input del usuario.
            Por defecto True.

    Returns:
        tuple[float, list[float], int]: Tupla con:
            - raíz encontrada
            - lista de errores en cada iteración
            - número de iteraciones realizadas

    Raises:
        ValueError: Si f(a) y f(b) tienen el mismo signo o si metodo es None y verbose=False

    Note:
        - Requiere que f(a) y f(b) tengan signos opuestos (teorema del valor intermedio)
        - Bisección: convergencia garantizada pero lenta
        - Regula Falsi: convergencia más rápida pero puede ser lenta en algunos casos
    """
    if f(a)*f(b) > 0:
        if verbose:
            print("No hay raiz dentro de este intervalo")
        return None, None, 0
    
    if metodo is None:
        if not verbose:
            raise ValueError("Debe especificar el método cuando verbose=False")
        metodo = int(input("Seleccione método:\n1- Bisección\n2- Regula-Falsi\n"))

    anterior = a
    contador = 0
    errores = []
    
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

        error = abs(c-anterior)/abs(c) if tipo_error == 2 else abs(c-anterior)
        errores.append(error*100 if tipo_error == 2 else error)
        anterior = c

        if error < tolerancia:
            break
    
    if verbose:
        print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
        print(f"Le tomó {contador} iteraciones")
    
    return c, errores, contador


def raiz_punto_fijo(g, g_prime, a, tolerancia, tipo_error, verbose=True):
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
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.

    Returns:
        tuple[float, float, int]: Tupla con:
            - punto fijo encontrado (raíz de la ecuación original)
            - error final
            - número de iteraciones realizadas

    Note:
        - Requiere que |g'(x)| < 1 en el intervalo de interés para garantizar convergencia
        - La velocidad de convergencia depende de qué tan cercano a 0 es |g'(x)|
        - Es crucial elegir una buena función g(x) para garantizar convergencia
    """
    contador = 0
    tolerancia = tolerancia/100 if tipo_error==2 else tolerancia
    
    while True:
        contador += 1
        if abs(g_prime(a)) >= 1:
            if verbose:
                print("El metodo no converge")
            return None, None, contador
        c = g(a)

        error = abs(c-a) if tipo_error == 1 else abs(c-a)/abs(a)

        a = c

        if error < tolerancia:
            break
    
    if verbose:
        print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%'if tipo_error==2 else''}")
        print(f"Le tomo {contador} iteraciones")
    
    return c, error, contador


def newton_raphson(f, f_prime, x0, tolerancia, tipo_error, verbose=True, max_iter=10000):
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
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.
        max_iter (int, optional): Número máximo de iteraciones. Por defecto 10000.

    Returns:
        tuple[float, float, int]: Tupla con:
            - raíz encontrada
            - error final
            - número de iteraciones realizadas

    Note:
        - Convergencia cuadrática cuando la aproximación está cerca de la raíz
        - Requiere que f'(x) ≠ 0 cerca de la raíz
        - Puede diverger si la aproximación inicial no es adecuada
        - Se detiene si la derivada es muy cercana a cero (punto crítico)
    """
    contador = 0
    tolerancia = tolerancia/100 if tipo_error==2 else tolerancia
    
    while contador <= max_iter:
        contador += 1
        if abs(f_prime(x0)) < 1e-4:
            if verbose:
                print("Derivada muy pequeña")
            return None, None, contador
        x1 = x0 - f(x0)/f_prime(x0)
        error = abs(x1-x0) if tipo_error==1 else abs(x1-x0)/abs(x0) if abs(x0) > 1e-12 else abs(x1-x0)
        x0 = x1
        if error < tolerancia:
            break
    
    if verbose:
        if tipo_error == 2:
            print(f"La raiz es {x1} con un error de {error*100:.2e}%")
        else:
            print(f"La raiz es {x1} con un error de {error:.2e}")
        print(f"Le tomo {contador} iteraciones")
        print(f"La raiz evaluada es {f(x1):.2e}")
    
    return x1, error, contador


def metodo_secante(f, x0, x1, tolerancia, tipo_error, verbose=True, max_iter=10000):
    """
    Implementa el método de la secante para encontrar raíces de una función.
    El método utiliza dos puntos iniciales y aproxima la derivada usando la pendiente
    entre los dos puntos más recientes: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Args:
        f (callable): Función de la cual se busca la raíz
        x0 (float): Primera aproximación inicial
        x1 (float): Segunda aproximación inicial
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.
        max_iter (int, optional): Número máximo de iteraciones. Por defecto 10000.

    Returns:
        tuple[float, float, int]: Tupla con:
            - raíz encontrada
            - error final
            - número de iteraciones realizadas

    Note:
        - Convergencia superlineal (más rápida que bisección, más lenta que Newton-Raphson)
        - No requiere calcular la derivada de la función
        - Puede fallar si f(x_n) - f(x_{n-1}) se aproxima a cero
        - Requiere dos aproximaciones iniciales
    """
    contador = 0
    if tipo_error == 2: 
        tolerancia = tolerancia/100
    
    while contador <= max_iter:
        contador += 1
        
        if abs(f(x1) - f(x0)) < 1e-12:
            if verbose:
                print("Diferencia de funciones muy pequeña")
            return None, None, contador
            
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        
        error = abs(x2 - x1) if tipo_error == 1 else abs(x2 - x1) / abs(x2) if abs(x2) > 1e-12 else abs(x2 - x1)
        
        x0, x1 = x1, x2
        
        if error < tolerancia:
            break
    
    if verbose:
        if tipo_error == 2:
            print(f"La raiz es {x2} con un error de {error*100:.2e}%")
        else:
            print(f"La raiz es {x2} con un error de {error:.2e}")
        print(f"Le tomo {contador} iteraciones")
        print(f"La raiz evaluada es {f(x2):.2e}")
    
    return x2, error, contador
