import math
def main():
    print("=" * 70)
    print("RESOLUCIÓN DE SISTEMA LINEAL - MÉTODO ITERATIVO GAUSS-SEIDEL")
    print("=" * 70)
    
    A = [[4, 1, 0, 0, 0, -1],
         [2, 10, -1, 0, -2, 0],
         [0, 3, 8, 0, -1, -3],
         [0, -1, 0, 4, 0, 1],
         [1, -2, 0, 0, 5, 0],
         [0, 0, 3, -2, 0, 6]
         ]
    B = [0, 9, 7, 20, 22, 37]
    
    print("MATRIZ DE COEFICIENTES (A):")
    for i, fila in enumerate(A):
        print(f"  [{' '.join(f'{val:6.2f}' for val in fila)}]")
    
    print(f"\nVECTOR DE TÉRMINOS INDEPENDIENTES (B):")
    print(f"  [{'  '.join(f'{val:6.2f}' for val in B)}]")
    print("-" * 70)
    
    resolverGS(A, B)
    print("=" * 70)
def resolverGS(A, B):
    """
    Resuelve un sistema de ecuaciones lineales utilizando métodos iterativos (Gauss-Seidel).
    
    Args:
        A (list[list[float]]): Matriz de coeficientes del sistema
        B (list[float]): Vector de términos independientes
    
    Returns:
        list[float]: Vector solución del sistema

    Note:
        La función verifica si la matriz es diagonalmente dominante antes de proceder.
    """
    esDiagDom(A)

    n = len(A)
    Xn = [0] * n  # Vector de soluciones nuevas
    Xv = [1] * n  # Vector de soluciones viejas

    tipo_error = int(input("Tipo de error (1=absoluto, 2=porcentual): ") or "1")
    gauss_seidel(A,B, Xn, Xv, tipo_error=tipo_error)
    
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
        print("⚠️  ADVERTENCIA: La matriz no es diagonalmente dominante")
        print("   La convergencia del método no está garantizada")
        return
    for i in range(1, n):
        suma = 0
        for j in range (n):
            if i != j:
                suma = suma + abs(A[i][j])
        if abs(A[i][i]) < suma:
            print("⚠️  ADVERTENCIA: La matriz no es diagonalmente dominante")
            print("   La convergencia del método no está garantizada")
            return
    print("✓ La matriz es diagonalmente dominante - convergencia garantizada")
def gauss_seidel(A, B, Xn, Xv, omega=1, tipo_error=1):
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
        tipo_error (int): 1=absoluto, 2=porcentual

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
    if tipo_error == 2: tolerancia = tolerancia/100
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
        norma = 0
        for i in range(n):
            error = error + (Xn[i]-Xv[i])**2
            if tipo_error == 2: norma += Xn[i]**2
        error = math.sqrt(error)
        if tipo_error == 2 and norma > 0: error = error / math.sqrt(norma)
        if error > errorV:
            print("El metodo no converge")
            return
        errorV = error
        for i in range(n):
            Xv[i]=Xn[i]
        if error < tolerancia:
            break;
    print("\nSOLUCIÓN DEL SISTEMA (Método Gauss-Seidel):")
    for i in range(n):
        print(f"  X{i+1} = {Xn[i]:12.8f}")
    print(f"\nError {'porcentual' if tipo_error==2 else 'absoluto'} final: {error*100 if tipo_error==2 else error:.8f}{'%' if tipo_error==2 else ''}")
    print(f"Convergencia alcanzada en {count} iteraciones")
main()