import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    # Datos de superficie superior (Upper surface)
    station = [
        0, 0.658, 0.920, 1.441, 2.717, 5.248, 7.753, 10.254, 15.243, 20.219, 
        25.189, 30.154, 35.116, 40.077, 45.088, 50.000, 54.955, 59.986, 
        64.914, 69.899, 74.898, 79.897, 84.910, 89.984, 94.967, 100.000
    ]
    
    ordinate = [
        0, -0.810, -0.956, -1.160, -1.490, -1.988, -2.314, -2.604, -3.049, 
        -3.378, -3.618, -3.770, -3.851, -3.855, -3.759, -3.551, -3.222, -2.801, -2.820, 
        -1.798, -1.267, -0.751, -0.282, 0.089, 0.278, 0
    ]
    
    print("=== EJERCICIO 1 - GUÍA 6 ===")
    print("Datos de superficie superior")
    print(f"Número de puntos: {len(station)}")
    print(f"Rango X (Station): {min(station)} a {max(station)}")
    print(f"Rango Y (Ordinate): {min(ordinate)} a {max(ordinate)}")
    print()
    
    # Ejemplo de uso con diferentes métodos
    print("Seleccione el método a utilizar:")
    print("1 - Interpolación polinómica")
    print("2 - Splines cúbicos naturales")
    print("3 - Regresión polinómica")
    print("4 - Mostrar solo los datos")
    
    try:
        opcion = int(input("Ingrese su opción (1-4): "))
        
        if opcion == 1:
            print("\n--- INTERPOLACIÓN POLINÓMICA ---")
            p, coef, X, Y = f.interpolacion(station, ordinate)
            f.graficar_interpolacion(p, coef, X, Y)
            
        elif opcion == 2:
            print("\n--- SPLINES CÚBICOS NATURALES ---")
            funciones_spline, coeficientes, X, Y = f.curvas_spline(station, ordinate)
            f.graficar_splines(funciones_spline, coeficientes, X, Y)
            
        elif opcion == 3:
            print("\n--- REGRESIÓN POLINÓMICA ---")
            grado = int(input("Ingrese el grado del polinomio (recomendado 3-5): "))
            p, coef, X, Y, r = f.regresion_polinomica(station, ordinate, grado=grado)
            f.graficar_regresion(p, coef, X, Y, r)
            
        elif opcion == 4:
            print("\n--- DATOS TABULADOS ---")
            print("Station\tOrdinate")
            print("-" * 20)
            for i in range(len(station)):
                print(f"{station[i]}\t{ordinate[i]}")
                
        else:
            print("Opción no válida")
            
    except ValueError:
        print("Error: Ingrese un número válido")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()