"""
Guía 8 - Ejercicio 4
Ecuación diferencial de circuito electrónico

E(t) = L(dI/dt) + RI(t)

Datos: L = 0.05 [H], R = 2 [Ω]
Tabla de valores I(t) dados

1. Determinar E(1.2) con precisión O(h²)
2. Comparar con I(t) = 10e^(-t/10)sin(2t)
"""

import math
from metodos import diferenciacion_tabulada


# Datos tabulados
t_datos = [1.0, 1.1, 1.2, 1.3, 1.4]
I_datos = [8.2277, 7.2428, 5.9908, 4.5260, 2.9122]

# Parámetros del circuito
L = 0.05  # Inductancia [H]
R = 2     # Resistencia [Ω]


def I_exacta(t):
    """Función exacta: I(t) = 10e^(-t/10)sin(2t)"""
    return 10 * math.exp(-t/10) * math.sin(2*t)


def E_exacta(t):
    """
    E(t) = L(dI/dt) + R*I(t)
    
    dI/dt = d/dt[10e^(-t/10)sin(2t)]
    dI/dt = 10e^(-t/10)[2cos(2t) - (1/10)sin(2t)]
    """
    exp_term = math.exp(-t/10)
    sin_term = math.sin(2*t)
    cos_term = math.cos(2*t)
    
    dI_dt = 10 * exp_term * (2*cos_term - 0.1*sin_term)
    I_t = 10 * exp_term * sin_term
    
    return L * dI_dt + R * I_t


if __name__ == "__main__":
    t_eval = 1.2
    
    print("\n" + "="*60)
    print("  Circuito Electrónico: E(t) = L(dI/dt) + RI(t)")
    print("="*60)
    print(f"L = {L} H,  R = {R} Ω,  t = {t_eval} s\n")
    
    # 1. Calcular dI/dt en t=1.2 usando diferencias finitas O(h²)
    _, dI_dt_array = diferenciacion_tabulada(t_datos, I_datos, verbose=False)
    
    # Obtener I(1.2) y dI/dt(1.2) de los datos
    idx = t_datos.index(t_eval)
    I_12 = I_datos[idx]
    dI_dt = dI_dt_array[idx]
    
    # Calcular E(1.2) = L*dI/dt + R*I(1.2)
    E_numerico = L * dI_dt + R * I_12
    
    print(f"{'Cálculo numérico (O(h²)):':<30}")
    print(f"  I(1.2) = {I_12:.4f} A")
    print(f"  dI/dt(1.2) = {dI_dt:.4f} A/s")
    print(f"  E(1.2) = {E_numerico:.4f} V\n")
    
    # 2. Comparar con función exacta
    I_exacta_12 = I_exacta(t_eval)
    E_exacta_12 = E_exacta(t_eval)
    
    error_I = abs(I_12 - I_exacta_12)
    error_E = abs(E_numerico - E_exacta_12)
    
    print(f"{'Solución exacta:':<30}")
    print(f"  I(1.2) = {I_exacta_12:.4f} A")
    print(f"  E(1.2) = {E_exacta_12:.4f} V\n")
    
    print(f"{'Error:':<30}")
    print(f"  Error I(1.2) = {error_I:.2e} A")
    print(f"  Error E(1.2) = {error_E:.2e} V")
    print("="*60 + "\n")
