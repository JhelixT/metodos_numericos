# Módulos refactorizados - Exportar todas las funciones

# Métodos de localización de raíces
from .raices import (
    buscar_raiz,
    raiz_punto_fijo,
    newton_raphson,
    metodo_secante
)

# Métodos de resolución de sistemas de ecuaciones lineales
from .sistemas_lineales import (
    esDiagDom,
    jacobi,
    gauss_seidel,
    resolverJG,
    triangulacion,
    determinante,
    gauss_pivot
)

# Métodos de aproximación de datos (interpolación, regresión, splines)
from .aproximacion import (
    leer_puntos_xy,
    interpolacion_lagrange,
    interpolacion,
    regresion_polinomica,
    curvas_spline,
    evaluar_spline,
    graficar_interpolacion,
    graficar_regresion,
    graficar_splines
)

# Métodos de integración numérica
from .integracion import (
    trapecio,
    simpson,
    gauss_legendre
)

# Métodos de diferenciación numérica
from .diferenciacion import (
    diferenciacion
)

# Métodos de resolución de EDOs de orden 1
from .edo1 import (
    euler,
    heun,
    punto_medio,
    runge_kutta4
)

# Métodos de resolución de sistemas de EDOs
from .sistemas_edo import (
    euler_sistema,
    heun_sistema,
    punto_medio_sistema,
    runge_kutta4_sistema
)

# Métodos de resolución de EDOs de orden superior
from .edo_orden_superior import (
    euler_orden_superior,
    heun_orden_superior,
    punto_medio_orden_superior,
    runge_kutta4_orden_superior
)

# Análisis de convergencia para EDOs
from .convergencia import (
    calcular_factor_convergencia_euler,
    calcular_factor_convergencia_heun,
    calcular_factor_convergencia_punto_medio,
    calcular_factor_convergencia_rk4
)

# Funciones utilitarias
from .utils import (
    limpiar_terminal,
    graficar_funciones
)

# Para compatibilidad con código legacy, también importamos de funciones.py
# (Puedes comentar esta línea si quieres forzar el uso de la nueva estructura)
from .funciones import *

__all__ = [
    # Raíces
    'buscar_raiz',
    'raiz_punto_fijo', 
    'newton_raphson',
    'metodo_secante',
    # Sistemas lineales
    'esDiagDom',
    'jacobi',
    'gauss_seidel',
    'resolverJG',
    'triangulacion',
    'determinante',
    'gauss_pivot',
    # Aproximación
    'leer_puntos_xy',
    'interpolacion_lagrange',
    'interpolacion',
    'regresion_polinomica',
    'curvas_spline',
    'evaluar_spline',
    'graficar_interpolacion',
    'graficar_regresion',
    'graficar_splines',
    # Integración
    'trapecio',
    'simpson',
    'gauss_legendre',
    # Diferenciación
    'diferenciacion',
    # EDOs de orden 1
    'euler',
    'heun',
    'punto_medio',
    'runge_kutta4',
    # Sistemas de EDOs
    'euler_sistema',
    'heun_sistema',
    'punto_medio_sistema',
    'runge_kutta4_sistema',
    # EDOs de orden superior
    'euler_orden_superior',
    'heun_orden_superior',
    'punto_medio_orden_superior',
    'runge_kutta4_orden_superior',
    # Convergencia
    'calcular_factor_convergencia_euler',
    'calcular_factor_convergencia_heun',
    'calcular_factor_convergencia_punto_medio',
    'calcular_factor_convergencia_rk4',
    # Utilidades
    'limpiar_terminal',
    'graficar_funciones',
]
