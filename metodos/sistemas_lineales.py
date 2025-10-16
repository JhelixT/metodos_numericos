"""
Métodos numéricos para resolución de sistemas de ecuaciones lineales.

Este módulo contiene implementaciones de métodos directos e iterativos para
resolver sistemas de ecuaciones lineales Ax = B.
"""

import math


def esDiagDom(A, verbose=True):
    """
    Verifica si una matriz es diagonalmente dominante.
    Una matriz es diagonalmente dominante si el valor absoluto de cada elemento diagonal
    es mayor que la suma de los valores absolutos de los demás elementos en su fila.

    Args:
        A (list[list[float]]): Matriz cuadrada a verificar
        verbose (bool, optional): Si True, imprime mensajes. Por defecto True.

    Returns:
        bool: True si la matriz es diagonalmente dominante, False en caso contrario
    
    Note:
        Esta propiedad es importante para garantizar la convergencia de métodos iterativos
        como Jacobi y Gauss-Seidel.
    """
    n = len(A)
    suma = 0
    for j in range(1, n):
        suma = suma + abs(A[0][j])
    if abs(A[0][0]) < suma:
        if verbose:
            print("La matriz no es Diagonalmente Dominante")
        return False
    
    for i in range(1, n):
        suma = 0
        for j in range(n):
            if i != j:
                suma = suma + abs(A[i][j])
        if abs(A[i][i]) < suma:
            if verbose:
                print("La matriz no es Diagonalmente Dominante")
            return False
    
    return True


def jacobi(A, B, Xn, Xv, tolerancia, tipo_error=1, verbose=True, max_iter=10000):
    """
    Implementa el método iterativo de Jacobi para resolver sistemas de ecuaciones lineales.
    El método de Jacobi actualiza cada componente de la solución usando los valores de la iteración anterior.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior
        tolerancia (float): Tolerancia deseada para el error
        tipo_error (int, optional): 1=absoluto, 2=porcentual. Por defecto 1.
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.
        max_iter (int, optional): Número máximo de iteraciones. Por defecto 10000.

    Returns:
        tuple[list[float], float, int, bool]: Tupla con:
            - vector solución
            - error final
            - número de iteraciones
            - convergió (True) o no (False)

    Note:
        - Requiere que la matriz sea diagonalmente dominante para garantizar convergencia
        - En cada iteración, calcula x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - Se detiene cuando el error es menor que la tolerancia especificada
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
    n = len(A)
    count = 0
    errorV = 1000
    if tipo_error == 2: 
        tolerancia = tolerancia/100
    
    while count < max_iter:
        count += 1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    suma = suma + A[i][j]*Xv[j]
            Xn[i] = (B[i] - suma)/A[i][i]
        
        error = 0
        norma = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
            if tipo_error == 2: 
                norma += Xn[i]**2
        error = math.sqrt(error)
        if tipo_error == 2 and norma > 0: 
            error = error / math.sqrt(norma)
        
        if error > errorV:
            if verbose:
                print("El metodo no converge")
            return Xn, error, count, False
        
        errorV = error
        for i in range(n):
            Xv[i] = Xn[i]
        
        if error < tolerancia:
            break
    
    if verbose:
        print("Las soluciones son:")
        for i in range(n):
            print(f"X{i} = {Xn[i]}")
        print(f"Con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
        print(f"Se resolvio en {count} iteraciones")
    
    return Xn, error, count, True


def gauss_seidel(A, B, Xn, Xv, tolerancia, omega=1, tipo_error=1, verbose=True, max_iter=10000):
    """
    Implementa el método iterativo de Gauss-Seidel con factor de relajación (SOR) para resolver sistemas de ecuaciones lineales.
    A diferencia de Jacobi, utiliza los valores actualizados tan pronto como estén disponibles.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        Xn (list[float]): Vector para almacenar la solución nueva
        Xv (list[float]): Vector para almacenar la solución anterior
        tolerancia (float): Tolerancia deseada para el error
        omega (float, optional): Factor de relajación. Por defecto 1.
            - omega < 1: sub-relajación
            - omega = 1: método de Gauss-Seidel estándar
            - omega > 1: sobre-relajación
        tipo_error (int, optional): 1=absoluto, 2=porcentual. Por defecto 1.
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.
        max_iter (int, optional): Número máximo de iteraciones. Por defecto 10000.

    Returns:
        tuple[list[float], float, int, bool]: Tupla con:
            - vector solución
            - error final
            - número de iteraciones
            - convergió (True) o no (False)

    Note:
        - Generalmente converge más rápido que Jacobi
        - Usa valores actualizados inmediatamente: x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        - El factor de relajación puede acelerar la convergencia
        - El error se calcula como la norma euclidiana de la diferencia entre iteraciones
    """
    n = len(A)
    count = 0
    errorV = 1000
    if tipo_error == 2: 
        tolerancia = tolerancia/100
    
    while count < max_iter:
        count += 1
        for i in range(n):
            suma = 0
            for j in range(n):
                if i != j:
                    if j < i:
                        suma += A[i][j] * Xn[j]
                    else:
                        suma += A[i][j] * Xv[j]
            Xn[i] = (B[i] - suma) / A[i][i]
            Xn[i] = omega*Xn[i] + (1-omega)*Xv[i]

        error = 0
        norma = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
            if tipo_error == 2: 
                norma += Xn[i]**2
        error = math.sqrt(error)
        if tipo_error == 2 and norma > 0: 
            error = error / math.sqrt(norma)
        
        if error > errorV:
            if verbose:
                print("El metodo no converge")
            return Xn, error, count, False
        
        errorV = error
        for i in range(n):
            Xv[i] = Xn[i]
        
        if error < tolerancia:
            break
    
    if verbose:
        print("Las soluciones son:")
        for i in range(n):
            print(f"X{i} = {Xn[i]}")
        print(f"Con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
        print(f"Se resolvio en {count} iteraciones")
    
    return Xn, error, count, True


def resolverJG(A, B, tolerancia=None, tipo_error=1, metodo=None, omega=1, verbose=True):
    """
    Resuelve un sistema de ecuaciones lineales utilizando métodos iterativos (Jacobi o Gauss-Seidel).
    
    Args:
        A (list[list[float]]): Matriz de coeficientes del sistema
        B (list[float]): Vector de términos independientes
        tolerancia (float, optional): Tolerancia para el error. Si es None y verbose=True, se solicita al usuario.
        tipo_error (int, optional): 1=absoluto, 2=porcentual. Por defecto 1.
        metodo (int, optional): 1=Jacobi, 2=Gauss-Seidel. Si es None y verbose=True, se solicita al usuario.
        omega (float, optional): Factor de relajación para Gauss-Seidel. Por defecto 1.
        verbose (bool, optional): Si True, imprime resultados y permite input del usuario. Por defecto True.
    
    Returns:
        tuple[list[float], float, int, bool]: Tupla con:
            - vector solución
            - error final
            - número de iteraciones
            - convergió (True) o no (False)

    Note:
        La función verifica si la matriz es diagonalmente dominante antes de proceder.
        Permite elegir entre el método de Jacobi o Gauss-Seidel.
    """
    if not esDiagDom(A, verbose=verbose):
        n = len(A)
        return [0]*n, None, 0, False

    n = len(A)
    Xn = [0] * n
    Xv = [0] * n

    if metodo is None:
        if not verbose:
            raise ValueError("Debe especificar el método cuando verbose=False")
        print("1-Resolver con Jacobi")
        print("2-Resolver con Gauss-Seidel")
        metodo = int(input())
    
    if tolerancia is None:
        if not verbose:
            raise ValueError("Debe especificar la tolerancia cuando verbose=False")
        if verbose:
            tipo_error = int(input("Tipo de error (1=absoluto, 2=porcentual): ") or "1")
        tolerancia = float(input("Ingrese la tolerancia: "))
    
    if metodo == 1:
        return jacobi(A, B, Xn, Xv, tolerancia, tipo_error, verbose)
    elif metodo == 2:
        return gauss_seidel(A, B, Xn, Xv, tolerancia, omega, tipo_error, verbose)
    else:
        raise ValueError("Método inválido. Use 1 para Jacobi o 2 para Gauss-Seidel")


def triangulacion(A, B, verbose=True):
    """
    Realiza la triangulación superior de una matriz aumentada [A|B] usando eliminación gaussiana con pivoteo.
    
    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        verbose (bool, optional): Si True, imprime información adicional. Por defecto True.

    Returns:
        None: Modifica A y B in-place

    Note:
        - Modifica las matrices A y B in-place
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - El pivoteo selecciona el elemento más grande en valor absoluto de cada columna
        - Evita problemas con elementos diagonales cercanos a cero
    """
    n = len(A)
    for i in range(n-1):
        p = i
        if abs(A[i][i]) < 1e-2:
            for l in range(i+1, n):
                if abs(A[l][i]) > abs(A[p][i]):
                    p = l

            for m in range(i, n):
                A[p][m], A[i][m] = A[i][m], A[p][m]

            B[p], B[i] = B[i], B[p]

        for j in range(i+1, n):
            factor = -A[j][i]/A[i][i]
            for k in range(i, n):
                A[j][k] = A[i][k]*factor + A[j][k]

            B[j] = B[i]*factor + B[j]
    
    if verbose:
        print()


def determinante(A, verbose=True):
    """
    Calcula el determinante de una matriz triangular superior.
    Para una matriz triangular, el determinante es el producto de los elementos diagonales.

    Args:
        A (list[list[float]]): Matriz triangular superior
        verbose (bool, optional): Si True, imprime el resultado. Por defecto True.

    Returns:
        float: Determinante de la matriz

    Note:
        - Asume que la matriz ya está en forma triangular superior
        - Un determinante cero indica que el sistema no tiene solución única
        - Se usa después de la triangulación en el método de Gauss
    """
    n = len(A)
    prod = 1
    for i in range(n):
        prod *= A[i][i]
    
    if prod == 0:
        if verbose:
            print("Matriz determinante 0\n")
    elif verbose:
        print("Determinante =", prod, "\n")
    
    return prod


def gauss_pivot(A, B, verbose=True):
    """
    Resuelve un sistema de ecuaciones lineales usando eliminación gaussiana con pivoteo.
    El método consiste en tres pasos: triangulación, verificación de determinante y sustitución hacia atrás.

    Args:
        A (list[list[float]]): Matriz de coeficientes
        B (list[float]): Vector de términos independientes
        verbose (bool, optional): Si True, imprime resultados. Por defecto True.

    Returns:
        list[float]: Vector solución del sistema

    Note:
        - Implementa pivoteo parcial para mejorar la estabilidad numérica
        - Verifica si el sistema tiene solución única (determinante ≠ 0)
        - La sustitución hacia atrás resuelve el sistema triangular superior
    """
    n = len(A)
    X = [0] * n
    triangulacion(A, B, verbose=False)
    det = determinante(A, verbose=verbose)
    
    if det == 0:
        if verbose:
            print("Sistema sin solución única")
        return X

    X[n-1] = B[n-1]/A[n-1][n-1]

    for i in range(n-1, -1, -1):
        suma = B[i]
        for j in range(i+1, n):
            suma = suma - A[i][j]*X[j]
        suma = suma/A[i][i]
        X[i] = suma
    
    if verbose:
        print("Las soluciones del sistema son: ")
        for i in range(n):
            print(f"X{i} = {X[i]}")
    
    return X
