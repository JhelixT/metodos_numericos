import matplotlib.pyplot as plt
import numpy as np
from metodos import funciones as func
"""
Problema n°1: Dada la función f(x) = ln(x² + 1) - sin(x),

a) Grafique la función (adjunte el plot por aula virtual). A partir de su gráfico, ¿Puede afirmar que 
   existe una raíz de f(x) en el intervalo [1.0, 2.0]? (Justifique).

b1) Realice a mano, paso a paso y en papel, la primera iteración del método de bisección, tomando 
    como intervalo inicial [1.0, 2.0]. Dé el valor de la raíz obtenida, así como el error absoluto 
    estimado.

b2) Usando el código apropiado, dé el valor de la raíz y el error estimado, correspondiente a la 
    iteración número 10. (Adjunte por aula virtual captura de pantalla de la consola donde figuren 
    los resultados pedidos).

c) Repita el inciso b) (completo), para el método de Regula Falsi.

d) ¿Se puede aplicar el método de punto fijo para encontrar la raíz de f(x)? (Justifique)

e1) Realice a mano, paso a paso y en papel, la primera iteración del Método de Newton-Raphson 
    para localizar la raíz de f(x), usando como valor inicial x₀ = 1.0. Escriba el valor obtenido de la 
    raíz así como el error estimado.

e2) Usando el código apropiado, escriba el valor de la raíz obtenida y el error estimado, para la 
    cuarta iteración. (Nuevamente, adjunte captura de pantalla de la consola)
"""

def main():
    f = lambda x: np.log(x**2+1) - np.sin(x)
    func.limpiar_terminal()
    print("METODO BISECCION")
    buscar_raiz(f, 1, 2, 1, 1e-8, 1)
    print("METODO FALSA POSICION")
    buscar_raiz(f, 1, 2, 1, 1e-8, 2)

    print("METODO DE PUNTO FIJO")
    g = lambda x: np.sqrt(np.exp(np.sin(x))-1)
    g_prime = lambda x: (g(x+0.01)-g(x))/0.01
    func.raiz_punto_fijo(g, g_prime, 2.7 , 1e-8, 1)
def graficar_funciones(*funciones, nombres=None, x_min=-10, x_max=10, n_puntos=1000):
    """
    Crea un gráfico interactivo y zoomeable de una o más funciones matemáticas.

    Args:
        *funciones (callable): Una o más funciones matemáticas para graficar
        nombres (list[str], optional): Lista con los nombres de las funciones para la leyenda.
            Si no se proporciona, se usarán nombres genéricos (f1, f2, etc.)
        x_min (float, optional): Límite inferior del dominio. Por defecto -10
        x_max (float, optional): Límite superior del dominio. Por defecto 10
        n_puntos (int, optional): Número de puntos para el gráfico. Por defecto 1000

    Example:
        # Graficar una función
        graficar_funciones(lambda x: x**2)

        # Graficar múltiples funciones con nombres personalizados
        graficar_funciones(
            lambda x: x**2,
            lambda x: math.sin(x),
            nombres=['Parábola', 'Seno']
        )
    
    Note:
        - Los colores se asignan automáticamente para cada función
    """
    # Validar entradas
    if not funciones:
        raise ValueError("Debe proporcionar al menos una función")
    
    if nombres is None:
        nombres = [f'f{i+1}' for i in range(len(funciones))]
    elif len(nombres) != len(funciones):
        raise ValueError("La cantidad de nombres debe coincidir con la cantidad de funciones")

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generar los datos
    x = np.linspace(x_min, x_max, n_puntos)
    
    # Graficar cada función
    lines = []
    for f, nombre in zip(funciones, nombres):
        try:
            y = [f(xi) for xi in x]
            line, = ax.plot(x, y, label=nombre)
            lines.append(line)
        except Exception as e:
            print(f"Error al graficar {nombre}: {e}")

    # Configurar el gráfico
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

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
        tuple[float, list[float]]: Tupla con la raíz encontrada y lista de errores en cada iteración

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

    anterior = a
    contador = 0
    errores = []  # Lista para almacenar errores
    
    # Convertir la tolerancia a decimal si es porcentual
    tolerancia = tolerancia/100 if tipo_error == 2 else tolerancia
    
    while True:
        c = (a+b)/2 if metodo == 1 else (a*f(b)-b*f(a))/(f(b)-f(a))
        contador += 1

        if f(c) == 0:
            error = 0
            errores.append(error)
            break
        elif f(a)*f(c) > 0:
            a = c
        else:
            b = c

        error = abs(c-anterior)/abs(c) if tipo_error == 2 else abs(c-anterior)  # Sin multiplicar por 100
        errores.append(error*100 if tipo_error == 2 else error)  # Convertir a porcentaje solo para almacenar
        anterior = c

        if error < tolerancia or contador == 10:
            break
    
    print(f"La raiz es {c} con un error de {error*100 if tipo_error==2 else error}{'%' if tipo_error==2 else ''}")
    print(f"Le tomó {contador} iteraciones")
    return c, errores
main()