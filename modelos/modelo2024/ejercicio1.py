import math
from metodos.integracion import trapecio
def main():
    f = lambda x: math.exp(x**2)
    intgauss= gauss_legendre(f,0, 1,n_puntos=6 ,verbose=False)
    print(f"El resultado de la integral con el metodo Gauss 6 puntos es: {intgauss:.10f}")
    iteracion = 0
    error_relativo = 1
    while error_relativo > 0.1:
        iteracion+=1
        inttrapecio = trapecio(f, 0, 1, n=iteracion, verbose=False)
        #print(f"El resultado de la integral con el metodo Trapecio y 18 intervalos es: {inttrapecio:.10f}")
        error_relativo = abs(inttrapecio-intgauss)/intgauss*100
        #print(f"El error relativo porcentual entre ambos metodos en la iteracion {iteracion} es {error_relativo:.6f}%")
    
    print(f"El error relativo menor a 0.1% se obtuvo en la iteracion {iteracion} con un error de {error_relativo:.6f}%")
    print(f"El resultado de la integral con el metodo Trapecio y {iteracion} intervalos es: {inttrapecio:.10f}")

    

def gauss_legendre(f, a, b, n_puntos=2, verbose=True):
    """
    Calcula la integral numérica usando cuadratura de Gauss-Legendre.
    
    La cuadratura de Gauss-Legendre aproxima la integral usando puntos de cuadratura
    óptimamente ubicados y sus factores de ponderación correspondientes.
    
    Transforma la integral ∫[a,b] f(x)dx a ∫[-1,1] g(x')dx' mediante el cambio:
        x = ((b-a)x' + (b+a))/2
        dx = (b-a)/2 dx'
    
    Args:
        f (callable): Función a integrar
        a (float): Límite inferior de integración
        b (float): Límite superior de integración
        n_puntos (int): Número de puntos de cuadratura (2, 3, 4, 5 o 6).
                       Por defecto 2.
        verbose (bool, optional): Si True, imprime información. Por defecto True.
        
    Returns:
        float: Aproximación de la integral
        
    Raises:
        ValueError: Si n_puntos no está en el rango [2, 6]
        
    Examples:
        >>> def f(x): return x**2
        >>> # Usando 2 puntos
        >>> resultado = gauss_legendre(f, 0, 3, n_puntos=2)
        >>> # Usando 4 puntos (mayor precisión)
        >>> resultado = gauss_legendre(f, 0, 3, n_puntos=4)
        
    Notes:
        - Mayor número de puntos → mayor precisión
        - Exacto para polinomios de grado ≤ 2n-1 (n = número de puntos)
        - 2 puntos: exacto hasta grado 3
        - 3 puntos: exacto hasta grado 5
        - 4 puntos: exacto hasta grado 7
        - 5 puntos: exacto hasta grado 9
        - 6 puntos: exacto hasta grado 11
    """
    
    # Tabla 22.1: Factores de ponderación y argumentos de la función
    # Fuente: Chapra & Canale, "Numerical Methods for Engineers"
    tablas_gauss = {
        2: {
            'c': [1.0000000, 1.0000000],
            'x': [-0.5773502692, 0.5773502692]
        },
        3: {
            'c': [0.5555556, 0.8888889, 0.5555556],
            'x': [-0.7745966692, 0.0, 0.7745966692]
        },
        4: {
            'c': [0.3478548, 0.6521452, 0.6521452, 0.3478548],
            'x': [-0.8611363116, -0.3399810435, 0.3399810435, 0.8611363116]
        },
        5: {
            'c': [0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269],
            'x': [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
        },
        6: {
            'c': [0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245],
            'x': [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142]
        }
    }
    
    if n_puntos not in tablas_gauss:
        raise ValueError(f"n_puntos debe ser 2, 3, 4, 5 o 6. Recibido: {n_puntos}")
    
    if verbose:
        print(f"Cuadratura de Gauss-Legendre con {n_puntos} puntos")
        print(f"Intervalo: [{a}, {b}]")
    
    # Obtener coeficientes y argumentos para n_puntos
    coeficientes = tablas_gauss[n_puntos]['c']
    argumentos = tablas_gauss[n_puntos]['x']
    
    # Cambio de variable: [a,b] → [-1,1]
    # x = ((b-a)x' + (b+a))/2
    # dx = (b-a)/2 dx'
    
    suma = 0.0
    if verbose:
        print(f"\n{'i':<5} {'c_i':<15} {'x_i':<15} {'x (transf.)':<15} {'f(x)':<15}")
        print("-" * 70)
    
    for i in range(n_puntos):
        ci = coeficientes[i]
        x_prima = argumentos[i]  # x' en [-1, 1] (dominio canónico)
        
        # Transformar x' → x en [a, b]
        xi = ((b - a) * x_prima + (b + a)) / 2
        
        # Evaluar f(x)
        fi = f(xi)
        
        if verbose:
            print(f"{i:<5} {ci:<15.10f} {x_prima:<15.10f} {xi:<15.10f} {fi:<15.10f}")
        
        suma += ci * fi
    
    # Aplicar factor de escala
    integral = ((b - a) / 2) * suma
    
    if verbose:
        print("-" * 70)
        print(f"\nFactor de escala: (b-a)/2 = ({b}-{a})/2 = {(b-a)/2}")
        print(f"Resultado: ∫f(x)dx ≈ {integral}")
    
    return integral

main()