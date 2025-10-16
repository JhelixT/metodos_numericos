"""
Funciones utilitarias para el módulo de métodos numéricos.

Este módulo contiene funciones auxiliares de propósito general que se utilizan
en diferentes partes de la biblioteca.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def limpiar_terminal():
    """
    Limpia la pantalla de la terminal de manera compatible con múltiples sistemas operativos.
    
    Utiliza el comando 'clear' en sistemas Unix/Linux/Mac y 'cls' en Windows.
    
    Example:
        >>> from metodos.utils import limpiar_terminal
        >>> limpiar_terminal()  # Limpia la pantalla
    """
    os.system('clear' if os.name == 'posix' else 'cls')


def graficar_funciones(*funciones, nombres=None, x_min=-10, x_max=10, n_puntos=1000):
    """
    Crea un gráfico interactivo de una o más funciones matemáticas.

    Esta función permite visualizar múltiples funciones en el mismo gráfico
    para compararlas fácilmente.

    Args:
        *funciones (callable): Una o más funciones matemáticas para graficar
        nombres (list[str], optional): Lista con los nombres de las funciones para la leyenda.
            Si no se proporciona, se usarán nombres genéricos (f1, f2, etc.)
        x_min (float, optional): Límite inferior del dominio. Por defecto -10
        x_max (float, optional): Límite superior del dominio. Por defecto 10
        n_puntos (int, optional): Número de puntos para el gráfico. Por defecto 1000

    Raises:
        ValueError: Si no se proporciona ninguna función o si el número de nombres
                   no coincide con el número de funciones

    Examples:
        >>> # Graficar una función
        >>> graficar_funciones(lambda x: x**2)

        >>> # Graficar múltiples funciones con nombres personalizados
        >>> import math
        >>> graficar_funciones(
        ...     lambda x: x**2,
        ...     lambda x: math.sin(x),
        ...     nombres=['Parábola', 'Seno']
        ... )
    
    Note:
        - Los colores se asignan automáticamente para cada función
        - Se incluyen ejes en x=0 e y=0 para referencia
        - La grilla está activada para facilitar la lectura
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
            line, = ax.plot(x, y, label=nombre, linewidth=2)
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
    ax.set_title('Gráfico de Funciones')
    
    plt.tight_layout()
    plt.show()
