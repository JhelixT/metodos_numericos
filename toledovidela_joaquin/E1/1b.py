import numpy as np
def main():
    print("=" * 60)
    print("BÚSQUEDA DE RAÍCES - MÉTODO DE BISECCIÓN/REGULA FALSI")
    print("=" * 60)
    print("Función: f(x) = (x+1)/(x+4) - 0.25*x")
    print("Intervalo: [1.75, 3.2]")
    print("Tolerancia: 0.01")
    print("Tipo de error: Porcentual")
    print("-" * 60)
    
    f = lambda x : (x+1)/(x+4) - 0.25*x
    raiz = buscar_raiz(f, 1.75, 3.2, 2, 0.01, 1)
    
    print("-" * 60)
    print("VERIFICACIÓN CON RAÍZ EXACTA")
    print(f"Raíz exacta conocida: 2.0000000000")
    print(f"Error porcentual exacto: {abs(2-raiz)*100/2:.6f}%")
    print("=" * 60)
def buscar_raiz(f, a, b, tipo_error, tolerancia, metodo=None):
    """
    Encuentra una raíz de una función en un intervalo dado usando el método de Bisección o Regula Falsi.

    Args:
        f (callable): Función continua de la cual se busca la raíz
        a (float): Límite inferior del intervalo
        b (float): Límite superior del intervalo
        tipo_error (int): Tipo de error a utilizar
            1: Error absoluto
            2: Error porcentual
        tolerancia (float): Tolerancia deseada para el error
        metodo (int, optional): Método a utilizar
            1: Bisección - divide el intervalo por la mitad
            2: Regula Falsi - usa interpolación lineal
            None: Permite al usuario elegir el método

    Returns:
        float: Raíz encontrada

    Note:
        - Requiere que f(a) y f(b) tengan signos opuestos (teorema del valor intermedio)
        - Bisección: convergencia garantizada pero lenta
        - Regula Falsi: convergencia más rápida pero puede ser lenta en algunos casos
    """
    if f(a)*f(b) > 0:
        print("No hay raiz dentro de este intervalo")
        return None, None
    
    if metodo is None:
        metodo = int(input("Seleccione método:\n1- Bisección\n2- Regula-Falsi\n"))

    anterior = a  #Definicion del Xviejo como b = 1.75
    contador = 0
    
    # Convertir la tolerancia a decimal si es porcentual
    tolerancia = tolerancia/100 if tipo_error == 2 else tolerancia
    
    while True:
        c = (a+b)/2 if metodo == 1 else (a*f(b)-b*f(a))/(f(b)-f(a))
        contador += 1

        if f(c) == 0:
            error = 0
            break
        elif f(a)*f(c) > 0:
            a = c
        else:
            b = c

        error = abs(c-anterior)/((1/2)*abs(c+anterior)) if tipo_error == 2 else abs(c-anterior)  #Definicion del criterio de error porcentual (Tipo 2)
        anterior = c

        if error < tolerancia:
            break
    
    metodo_nombre = "Bisección" if metodo == 1 else "Regula Falsi"
    print(f"\nRESULTADOS DEL MÉTODO DE {metodo_nombre.upper()}")
    print(f"Raíz encontrada: {c:.10f}")
    print(f"Error {'porcentual' if tipo_error==2 else 'absoluto'}: {error*100 if tipo_error==2 else error:.6f}{'%' if tipo_error==2 else ''}")
    print(f"Número de iteraciones: {contador}")
    return c
main()