from metodos.sistemas_lineales import gauss_seidel, gauss_pivot
def main():
    A = [[4, 1, 1, 1],
         [0, 5, 0, 1],
         [1, -2, 4, 0],
         [-2, 0, 0, 4]
         ]
    B =[13, 14, 9, 2]
    Xpivot = gauss_pivot(A, B, verbose=True)
    Xseidel = gauss_seidel(A, B, X0=[0,0,0,0], tolerancia=1e-8, verbose=True)

main()