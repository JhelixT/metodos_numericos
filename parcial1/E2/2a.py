def main():
    print("=" * 70)
    print("RESOLUCIÓN DE SISTEMA LINEAL - ELIMINACIÓN GAUSSIANA CON PIVOTEO")
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
    
    gauss_pivot(A, B)
    print("=" * 70)

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
    print("SOLUCIÓN DEL SISTEMA:")
    for i in range(n):
        print(f"  X{i+1} = {X[i]:12.8f}")
    return X

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
    banderaPivot = False
    for i in range(n-1):
        p=i
        if abs(A[i][i]) < 1e-2:
            banderaPivot = True #Variable para chequear algun pivoteo
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
    if(banderaPivot):
        print("✓ Se aplicó pivoteo parcial para mejorar estabilidad numérica")
    else:
        print("✓ No fue necesario aplicar pivoteo")
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
    #print("Determinante = ",prod,"\n")

main()