from metodos.edo_orden_superior import runge_kutta4_orden_superior, euler_orden_superior
import matplotlib.pyplot as plt
import math
def main():
    f1 = lambda x, Y0, Y1: Y1
    f2 = lambda x, Y0, Y1: -5*Y0 -2*Y1 
    fexacta = lambda x: math.exp(-x)*math.sin(2*x)
    y0_0 = 0  # Y(0) = y0
    y1_0 = 2  # Y'(0) = y1
    x0 = 0 # punto inicial
    xf = 1 # punto final
    Xrk, Yrk = runge_kutta4_orden_superior(f2, x0, [y0_0, y1_0], xf, n=100, verbose=False)
    Xeu, Yeu = euler_orden_superior(f2, x0, [y0_0, y1_0], xf, n=100, verbose=False)

    # Extraer Y0 (y) de cada paso
    Y0_rk = [y[0] for y in Yrk]  # y(x)
    Y0_eu = [y[0] for y in Yeu]  # y(x)
    # Calcular Error Cuadrático Medio
    ecm_rk = sum((Y0_rk[i] - fexacta(Xrk[i]))**2 for i in range(len(Xrk))) / len(Xrk)
    ecm_eu = sum((Y0_eu[i] - fexacta(Xeu[i]))**2 for i in range(len(Xeu))) / len(Xeu)

    print(f"ECM Método RK4 Orden Superior: {ecm_rk:.6e}")
    print(f"ECM Método Euler Orden Superior: {ecm_eu:.6e}")

    plt.plot(Xrk, Y0_rk, label="RK4 Orden Superior", color='black')
    plt.plot(Xeu, Y0_eu, label="Euler Orden Superior", color='red')
    plt.plot(Xrk, [fexacta(x) for x in Xrk], label="Solución Exacta", linestyle='dashed', color='blue')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Y0 (y)")
    plt.title("Comparación Método RK4 vs Euler para EDO de 2do Orden")
    plt.show()
    

main()