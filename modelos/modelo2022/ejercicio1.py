
import math
from metodos.aproximacion import curvas_spline, evaluar_spline, visualizar_spline, graficar_splines
def main():
    X = [1, 1.3, 1.4, 1.45, 1.6, 1.72, 1.8, 1.93, 2]
    Y = [5.381, 8.672, 9.592, 9.988, 10.853, 11.146, 11.134, 10.780, 10.435]
    f = lambda x: math.exp(2*math.sin(x))*(1+math.log(x))
    Xeq = [X[0] + i*0.1 for i in range(int((X[-1]-X[0])/0.1)+1)] #Tabla de valores equiespaciados hasta 2

    # Spline cúbico
    funciones_spline, coef_spline, X, Y = curvas_spline(X=X, Y=Y, verbose=False)
    Yspline = [evaluar_spline(x, funciones_spline, X) for x in Xeq]
    
    # Calcular valores de la función original
    Yoriginal = [f(x) for x in Xeq]
    
    # Calcular errores
    errores = [abs(yspline - yorig) for yspline, yorig in zip(Yspline, Yoriginal)]
    
    # Mostrar la tabla
    print("Tabla de valores equiespaciados (h = 0.1):")
    print(f"{'x':>8} {'f(x) spline':>15} {'f(x) original':>15} {'Error':>15}")
    print("-" * 55)
    for x, yspline, yorig, error in zip(Xeq, Yspline, Yoriginal, errores):
        print(f"{x:8.2f} {yspline:15.6f} {yorig:15.6f} {error:15.8e}")

    #Integral Simpson 1/3
    from metodos.integracion import simpson
    a, b = 1, 2 
    integralSpline = simpson(X = Xeq, Y = Yspline, a = a, b = b, verbose=True)
    integralExacta = simpson(f=f, a = a, b = b, n = 10, verbose=True)
main()