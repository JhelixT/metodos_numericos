import math
from metodos.sistemas_edo import euler_sistema, runge_kutta4_sistema
import matplotlib.pyplot as plt
def main():
    f1 = lambda t, X: X[1]
    f2 = lambda t, X: -(5*X[0]*X[1] + (X[0] + 7)*math.sin(t))
    t0 = 0
    tf = 1
    x0_0 = 6
    x1_0 = 1.5
    Teu, Xeu = euler_sistema([f1, f2], x0=t0, y0=[x0_0, x1_0], xf=tf, n=100, verbose=False)
    Trk, Xrk = runge_kutta4_sistema([f1, f2], x0=t0, y0=[x0_0, x1_0], xf=tf, n=100, verbose=False)

    #Arreglos con t espaciado en 0.2
    Teu_02 = [Teu[i] for i in range(0, len(Teu), 20)]
    Xeu_02 = [Xeu[i] for i in range(0, len(Xeu), 20)]
    Trk_02 = [Trk[i] for i in range(0, len(Trk), 20)]
    Xrk_02 = [Xrk[i] for i in range(0, len(Xrk), 20)]

    print("Tabla de valores equiespaciados (h = 0.2):")
    print(f"{'t':>8} {'X0 Euler':>15} {'X0 RK4':>15} {'X1 Euler':>15} {'X1 RK4':>15}")
    print("-" * 95)
    for i in range(len(Teu_02)):
        t = Teu_02[i]
        x0_eu = Xeu_02[i][0]
        x0_rk = Xrk_02[i][0]
        x1_eu = Xeu_02[i][1]
        x1_rk = Xrk_02[i][1]
        print(f"{t:8.2f} {x0_eu:15.6f} {x0_rk:15.6f} {x1_eu:15.6f} {x1_rk:15.6f}")
    


    #Grafico
    plt.plot(Teu, [Xeu[i][0] for i in range(len(Xeu))], label="X0 Euler", color='red')
    plt.plot(Trk, [Xrk[i][0] for i in range(len(Xrk))], label="X0 RK4", color='black')

    plt.xlabel("t")
    plt.ylabel("X0")
    plt.title("Soluciones del sistema de EDO")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Teu, [Xeu[i][1] for i in range(len(Xeu))], label="X1 Euler", color='blue')
    plt.plot(Trk, [Xrk[i][1] for i in range(len(Xrk))], label="X1 RK4", color='green')

    plt.xlabel("t")
    plt.ylabel("X1")
    plt.title("Soluciones del sistema de EDO")
    plt.legend()
    plt.grid(True)
    plt.show()


main()