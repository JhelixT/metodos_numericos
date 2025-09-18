import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    f.limpiar_terminal()
    A = [
        [1,2,1,4],
        [0,2,4,3],
        [4,2,2,1],
        [-3,1,3,2]
    ]
    B = [13,28,20,6]
    f.gauss_pivot(A,B)
    
main()