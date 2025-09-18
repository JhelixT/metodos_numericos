import pprint as pp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f
if __name__ == "__main__":
    f.limpiar_terminal()
    A = [ [4,-1, 2, 3],
          [0,-2, 7,-4],
          [0, 0, 6, 5],
          [0, 0, 0, 3] ]
    B = [20, -7, 4, 6]

    f.gauss_pivot(A, B)
 

