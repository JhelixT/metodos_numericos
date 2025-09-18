import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metodos import funciones as f

def main():
    A = [
        [1,0,0,0],
        [1,1,1,1],
        [1,2,4,8],
        [1,3,9,27]
    ]

    B = [0,1,3,4]

    X = f.gauss_pivot(A, B)

main()