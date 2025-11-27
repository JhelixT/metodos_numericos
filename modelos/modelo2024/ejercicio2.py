import math
from metodos.aproximacion import interpolacion, curvas_spline, evaluar_spline, visualizar_polinomio, visualizar_spline

def main():
    X = [0, 1, 0.5]
    Y = [0.5, 1.35, 0.83]
    
    # Interpolación polinómica
    polinomio, coef, X, Y = interpolacion(X=X, Y=Y)
    print(f"La aproximacion de f(0.6) con polinomio interpolador es {polinomio(0.6)}")
    
    # Visualizar el polinomio y el punto de evaluación (modo simplificado)
    print("\n--- Modo simplificado (verbose=False) ---")
    visualizar_polinomio(polinomio, coef, X, Y, x_eval=0.6, 
                        titulo="Interpolación Polinómica - Evaluación en x=0.6",
                        verbose=False)
    
    # Spline cúbico
    funciones_spline, coef_spline, X, Y = curvas_spline(X=X, Y=Y, verbose=False)
    print(f"\nLa aproximacion de f(0.6) con Spline Cubica Normal es {evaluar_spline(0.6, funciones_spline, X)}")
    
    # Visualizar el spline y el punto de evaluación (modo simplificado)
    print("\n--- Modo simplificado (verbose=False) ---")
    visualizar_spline(funciones_spline, coef_spline, X, Y, x_eval=0.6,
                     titulo="Spline Cúbico Natural - Evaluación en x=0.6",
                     verbose=False)

main()