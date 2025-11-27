from metodos import trapecio, simpson
from scipy import integrate
import math

if __name__ == "__main__":

    # Definir funciones
    f1 = lambda x: (1+x**2)**(-1)
    f2 = lambda x: (x**2)*math.exp(-x)
    f3 = lambda x: (2+(math.sin(2*math.sqrt(x))))
    f4 = lambda x: (math.sin(2*x))*math.exp(-x)

    # Información de las funciones
    funciones = [
        (f1, "f(x) = 1/(1+x²)", -1, 1),
        (f2, "f(x) = x²·e^(-x)", 0, 4),
        (f3, "f(x) = 2 + sin(2√x)", 0, 1),
        (f4, "f(x) = sin(2x)·e^(-x)", 0, math.pi)
    ]
    
    print("=" * 90)
    print("COMPARACIÓN DE MÉTODOS DE INTEGRACIÓN NUMÉRICA")
    print("=" * 90)
    
    for i, (f, nombre, a, b) in enumerate(funciones, 1):
        print(f"\n{i}. Función a integrar: {nombre}")
        print(f"   Intervalo: [{a}, {b}]")
        print("-" * 90)
        
        # Calcular integrales
        int_exacta, error_quad = integrate.quad(f, a, b)
        int_trapecio = trapecio(f=f, a=a, b=b, n=10, verbose=False)
        int_simpson = simpson(f=f, a=a, b=b, n=10, verbose=False)
        
        # Calcular errores
        error_trapecio = abs(int_trapecio - int_exacta)
        error_simpson = abs(int_simpson - int_exacta)
        
        # Imprimir resultados
        print(f"   Integral exacta (SciPy):    {int_exacta:.12f}")
        print(f"   Integral trapecio (n=10):   {int_trapecio:.12f}")
        print(f"   Integral Simpson (n=10):    {int_simpson:.12f}")
        print(f"   Error trapecio:             {error_trapecio:.12e}")
        print(f"   Error Simpson:              {error_simpson:.12e}")
        
        # Ratio de mejora
        if error_simpson > 0:
            mejora = error_trapecio / error_simpson
            print(f"   Mejora Simpson vs Trapecio: {mejora:.2f}x más preciso")
        
        print("-" * 90)
    
    print("\n✅ Análisis completado")

