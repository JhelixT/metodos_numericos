def main():
    print("=" * 60)
    print("BÚSQUEDA DE RAÍCES - MÉTODO DE NEWTON-RAPHSON")
    print("=" * 60)
    print("Función: f(x) = (x+1)/(x+4) - 0.25*x")
    print("Derivada: f'(x) = 3/(x+4)² - 0.25")
    print("Aproximación inicial: x₀ = 1")
    print("Tolerancia: 1×10⁻¹²")
    print("Tipo de error: Porcentual")
    print("-" * 60)
    
    f = lambda x: (x+1)/(x+4) - 0.25*x
    f_prime = lambda x: 3/((x+4)**2) - 0.25
    raiz = newton_raphson(f, f_prime, 1, 1e-12, 2)
    
    print("-" * 60)
    print("VERIFICACIÓN CON RAÍZ EXACTA")
    print(f"Raíz exacta conocida: 2.0000000000")
    print(f"Error porcentual exacto: {abs(2-raiz)*100/2:.10f}%")
    print("=" * 60)
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
        error = abs(x1-x0) if tipo_error==1 else abs(x1-x0)/((1/2)*abs(x1+x0)) #Definicion del tipo de error porcentual (Tipo 2)
        x0=x1
        if(error<tolerancia) or contador == 4:  #Criterio de corte para mostrar lo obtenido en la 4ta iteracion
            break
    print(f"\nRESULTADOS DEL MÉTODO DE NEWTON-RAPHSON")
    print(f"Raíz encontrada: {x1:.10f}")
    print(f"Error {'porcentual' if tipo_error==2 else 'absoluto'}: {error*100 if tipo_error==2 else error:.10f}{'%' if tipo_error==2 else ''}")
    print(f"Número de iteraciones: {contador}")
    return x1
main()