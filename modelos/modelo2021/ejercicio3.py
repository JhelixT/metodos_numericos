from metodos.edo1 import runge_kutta4, euler
def main():
    f = lambda t, y: y*(4*y-t**2)
    y0 = 0.1  # Condición inicial intermedia
    t0 = 0
    Xrk, Yrk = runge_kutta4(f, x0=t0, y0=y0, xf=3, n=30, verbose=False)  # n muy grande para estabilidad
    Xeu, Yeu = euler(f, x0=t0, y0=y0, xf=3, n=30, verbose=False)

    errorRelativo = [abs((Yrk[i]-Yeu[i])/Yrk[i]) for i in range(len(Yrk))]
    # Mostrar resultados
    print("Tabla de valores equiespaciados (h = 0.1):")
    print(f"{'t':>8} {'y(t) RK4':>15} {'y(t) Euler':>15} {'Error Relativo':>15}")
    print("-" * 55)
    for x, yrk, yeu, error in zip(Xrk, Yrk, Yeu, errorRelativo):
        if(x==1 or x==2 or x==3):
            print(f"{x:8.2f} {yrk:15.6f} {yeu:15.6f} {error:15.8e}")
main()  