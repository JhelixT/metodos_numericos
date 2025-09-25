import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f
import numpy as np

def main():
    Re = [
        0.2, 2, 20, 200, 2000, 20000
    ]
    lnRe = np.log(Re)
    Cd = [
        103, 13.9, 2.72, 0.800, 0.401, 0.433
    ]

    print("Coeficiente de Resistencia en funcion del numero de Reynolds")
    print(f"Número de puntos: {len(Re)}")
    print(f"Rango X (Re): {min(Re)} a {max(Re)}")
    print(f"Rango Y (Cd): {min(Cd)} a {max(Cd)}")
    print()

    print("Seleccione el método a utilizar:")
    print("1 - Interpolación polinómica")
    print("2 - Splines cúbicos naturales")
    print("3 - Regresión polinómica")
    print("4 - Mostrar solo los datos")
    
    try:
        opcion = int(input("Ingrese su opción (1-4): "))
        
        if opcion == 1:
            print("\n--- INTERPOLACIÓN POLINÓMICA ---")
            p, coef, X, Y = f.interpolacion(lnRe, Cd)
            f.graficar_interpolacion(p, coef, X, Y)
            
        elif opcion == 2:
            print("\n--- SPLINES CÚBICOS NATURALES ---")
            funciones_spline, coeficientes, X, Y = f.curvas_spline(lnRe, Cd)
            f.graficar_splines(funciones_spline, coeficientes, X, Y)
            
            # Calcular coeficientes para Reynolds específicos
            reynolds_calc = [5, 50, 500, 5000]
            print("\nCoeficientes de resistencia calculados:")
            for re_val in reynolds_calc:
                ln_re_val = np.log(re_val)
                cd_calc = f.evaluar_spline(ln_re_val, funciones_spline, lnRe)
                print(f"Re = {re_val}: Cd = {cd_calc:.4f}")
            
        elif opcion == 3:
            print("\n--- REGRESIÓN POLINÓMICA ---")
            grado = int(input("Ingrese el grado del polinomio (recomendado 3-5): "))
            p, coef, X, Y, r = f.regresion_polinomica(lnRe, Cd, grado=grado)
            f.graficar_regresion(p, coef, X, Y, r)
            
        elif opcion == 4:
            print("\n--- DATOS TABULADOS ---")
            print("--- ESCALA LOGARITMICA ---")
            print("lnRe\tCd")
            print("-" * 20)
            for i in range(len(Re)):
                print(f"{lnRe[i]}\t{Cd[i]}")
                
        else:
            print("Opción no válida")
            
    except ValueError:
        print("Error: Ingrese un número válido")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main()