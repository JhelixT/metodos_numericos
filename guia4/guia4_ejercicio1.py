import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main ():
    f.limpiar_terminal()
    A =[ [4,-1,1],
         [4,-8,1],
         [-2,1,5] ]
    B = [7,-21,15]

    X = f.resolverJG(A,B)

main()