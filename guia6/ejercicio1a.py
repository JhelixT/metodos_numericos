import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    # Datos de superficie superior (Upper surface)
    station = [
        0, 0.347, 0.580, 1.059, 2.288, 4.757, 7.247, 9.746, 14.757, 19.781, 
        24.811, 29.840, 34.884, 39.028, 44.952, 50.000, 55.035, 60.064, 
        65.086, 70.101, 75.107, 80.108, 85.090, 90.060, 95.088, 100.000
    ]
    
    ordinate = [
        0, 1.010, 1.286, 1.588, 2.284, 3.227, 4.010, 4.672, 5.741, 6.562, 
        7.193, 7.658, 7.971, 8.189, 8.189, 7.968, 7.602, 7.085, 6.440, 5.686, 
        4.847, 3.935, 2.974, 1.979, 0.956, 0
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