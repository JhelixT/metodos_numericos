import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    T = [-260.15, -200, -100, 0, 100, 300]
    Cp= [0.1, 0.45, 0.699, 0.87, 0.941, 1.04]
    p, coef, T, Cp = f.interpolacion(T, Cp)
    f.graficar_interpolacion(p, coef, T, Cp, a = -273.15, b = 500)
    print(f"Valor aproximado para T= 500[C]: {p(500)}") 

    # Conclusion el metodo de aproximacion por polinomios de Lagrange no es bueno para aproximar los valores de este ensayo, ya que los polinomios 
    # a medida que x->inf |P(x)|->inf y no a un valor constante que represente una asintota horizontal tal como se ve en el grafico
    
main()