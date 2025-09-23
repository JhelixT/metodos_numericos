import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f
import numpy as np
import math

def main():
    t = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    Y = [1, 0.994, 0.990, 0.985, 0.979, 0.977, 0.972, 0.969, 0.967, 0.960, 0.956, 0.952]
    lnY = np.log(Y)

    # Sistema de ecuaciones normales para ln(y) = c0 + c1*t
    # donde c0 = ln(a) y c1 = -b
    A = [[len(t), sum(t)], 
         [sum(t), sum(np.array(t)**2)]]
    
    B = [sum(lnY), sum(np.array(t)*lnY)] 

    coef = f.gauss_pivot(A, B) # Devuelve [ln(a), -b]

    a = np.exp(coef[0])  # coef[0] = ln(a) -> a = exp(ln(a))
    b = -coef[1]         # coef[1] = -b -> b = -(-b)

    reg = lambda x: a*np.exp(-b*x)

    y_prom = sum(Y)/len(Y)
    Sr = sum((reg(t) - y)**2 for t, y in zip(t, Y))
    St = sum((y-y_prom)**2 for y in Y)

    r = math.sqrt((St - Sr)/St) if St > 0 else 0

    f.limpiar_terminal()

    f.graficar_regresion(reg, [a,b], t, Y, r)

    
    print(f"La funcion de ajuste es {a:.4f}e^(-{b:.4f}t)")
main()