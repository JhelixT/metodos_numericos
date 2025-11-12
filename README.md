# Métodos Numéricos

Este repositorio contiene implementaciones de diversos métodos numéricos organizados en guías prácticas y una biblioteca modular completa. El proyecto está estructurado para facilitar el aprendizaje y la aplicación de diferentes técnicas numéricas.

## 🎯 Características Principales

- ✅ **Biblioteca modular** organizada por categorías matemáticas
- 🎨 **Visualización de resultados** usando matplotlib
- 🔢 **Cálculos simbólicos** con sympy
- ⚡ **Operaciones eficientes** con numpy
- 📊 **Modo verbose opcional** para control de salida I/O
- 🔄 **Retornos estructurados** con información detallada (errores, iteraciones, convergencia)

## 📁 Estructura del Proyecto

```
metodos_numericos/
├── guia1/              # Ejercicios: Métodos de búsqueda de raíces básicos
├── guia2/              # Ejercicios: Métodos iterativos
├── guia3/              # Ejercicios: Sistemas de ecuaciones lineales
├── guia4/              # Ejercicios: Resolución de sistemas
├── guia5/              # Ejercicios: Interpolación y aproximación
├── guia6/              # Ejercicios: Interpolación segmentaria con curvas spline
├── guia7/              # Ejercicios: Integración numérica
├── ejemplos/           # 📂 Ejemplos de uso y casos de prueba
├── modelos/            # 🧪 Modelos matemáticos aplicados
└── metodos/            # 📚 Biblioteca principal (módulos especializados)
    ├── __init__.py           # Exportaciones y API pública
    ├── raices.py             # 🎯 Localización de raíces
    ├── sistemas_lineales.py  # 🔢 Sistemas de ecuaciones lineales
    ├── aproximacion.py       # 📈 Interpolación, regresión y splines
    ├── integracion.py        # ∫  Integración numérica
    ├── diferenciacion.py     # ∂  Diferenciación numérica
    ├── edo1.py               # 📊 EDOs de primer orden (individual)
    ├── sistemas_edo.py       # 🔗 Sistemas de EDOs de primer orden
    ├── edo_orden_superior.py # 📐 EDOs de orden m (m ≥ 2)
    ├── convergencia.py       # 🔬 Análisis de convergencia
    ├── utils.py              # 🛠️ Utilidades generales
    └── funciones.py          # ⚠️  Legacy (mantiene compatibilidad)
```

## 📚 Módulos de la Biblioteca

### 🎯 `metodos.raices` - Localización de Raíces
Métodos para encontrar ceros de funciones continuas:
- **`buscar_raiz()`** - Bisección y Regula Falsi
- **`raiz_punto_fijo()`** - Método del punto fijo
- **`newton_raphson()`** - Método de Newton-Raphson
- **`metodo_secante()`** - Método de la secante

### 🔢 `metodos.sistemas_lineales` - Sistemas de Ecuaciones
Métodos directos e iterativos para sistemas lineales Ax = B:
- **`gauss_pivot()`** - Eliminación gaussiana con pivoteo
- **`jacobi()`** - Método iterativo de Jacobi
- **`gauss_seidel()`** - Método de Gauss-Seidel con relajación (SOR)
- **`resolverJG()`** - Resolver con Jacobi o Gauss-Seidel
- **`triangulacion()`** - Triangulación de matrices
- **`determinante()`** - Cálculo de determinantes
- **`esDiagDom()`** - Verificación de diagonal dominante

### 📈 `metodos.aproximacion` - Interpolación y Regresión
Métodos de aproximación de datos:
- **`interpolacion()`** - Interpolación polinómica (Vandermonde)
- **`interpolacion_lagrange()`** - Interpolación de Lagrange
- **`regresion_polinomica()`** - Regresión por mínimos cuadrados
- **`curvas_spline()`** - Splines cúbicos naturales
- **`evaluar_spline()`** - Evaluación de splines
- **`graficar_interpolacion()`** - Visualización de interpolación
- **`graficar_regresion()`** - Visualización de regresión
- **`graficar_splines()`** - Visualización de splines

### ∫ `metodos.integracion` - Integración Numérica
Métodos de integración numérica:
- **`trapecio()`** - Regla del trapecio compuesta
  - Modo función continua: `trapecio(f, a, b, n)`
  - Modo datos tabulados: `trapecio(X=X, Y=Y)`
  - Soporte automático para datos no equiespaciados (usa splines)
- **`simpson()`** - Regla de Simpson 1/3 compuesta
  - Requiere número par de intervalos
  - Mayor precisión que trapecio para funciones suaves

### ∂ `metodos.diferenciacion` - Diferenciación Numérica
Métodos de derivación numérica:
- **`diferenciacion()`** - Cálculo de derivadas numéricas
  - Diferencias finitas progresivas, regresivas o centrales
  - Orden de precisión configurable (O(h), O(h²), O(h⁴))
  - Soporte para múltiples puntos simultáneos

### 📊 `metodos.edo1` - Ecuaciones Diferenciales Ordinarias de Primer Orden
Métodos numéricos para resolver EDOs individuales dy/dx = f(x,y):
- **`euler()`** - Método de Euler (orden 1)
- **`heun()`** - Método de Heun (orden 2)
- **`punto_medio()`** - Método del Punto Medio (orden 2)
- **`runge_kutta4()`** - Método de Runge-Kutta de 4to orden

Todos los métodos retornan: `(X, Y)` donde X son los puntos e Y las aproximaciones.

### 🔗 `metodos.sistemas_edo` - Sistemas de EDOs de Primer Orden
Métodos numéricos para resolver sistemas de n EDOs de primer orden:
```
dy₁/dx = f₁(x, y₁, y₂, ..., yₙ)
dy₂/dx = f₂(x, y₁, y₂, ..., yₙ)
...
dyₙ/dx = fₙ(x, y₁, y₂, ..., yₙ)
```

Métodos disponibles:
- **`euler_sistema(funciones, x0, y0, xf, n, verbose=True)`** - Euler para sistemas
- **`heun_sistema(funciones, x0, y0, xf, n, verbose=True)`** - Heun para sistemas
- **`punto_medio_sistema(funciones, x0, y0, xf, n, verbose=True)`** - Punto Medio para sistemas
- **`runge_kutta4_sistema(funciones, x0, y0, xf, n, verbose=True)`** - RK4 para sistemas

**Parámetros:**
- `funciones`: Lista de funciones `[f1, f2, ..., fn]` donde cada `fi(x, Y)` recibe el vector de estado Y
- `x0, xf`: Intervalo de integración
- `y0`: Lista con condiciones iniciales `[y1₀, y2₀, ..., yn₀]`
- `n`: Número de pasos

**Retorno:** `(X, Y)` donde X son los puntos e Y es una lista de listas, Y[i][j] = valor de yⱼ en el paso i.

### 📐 `metodos.edo_orden_superior` - EDOs de Orden m
Métodos para resolver EDOs de orden superior convirtiéndolas automáticamente a sistemas:
```
y⁽ᵐ⁾ = f(x, y, y', y'', ..., y⁽ᵐ⁻¹⁾)
```

Métodos disponibles:
- **`euler_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True)`**
- **`heun_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True)`**
- **`punto_medio_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True)`**
- **`runge_kutta4_orden_superior(f, x0, y0, xf, n, orden=2, verbose=True)`**

**Parámetros:**
- `f`: Función `f(x, y, y_prima, y_doble_prima, ...)` que retorna y⁽ᵐ⁾
- `orden`: Orden de la EDO (2 para segunda orden, 3 para tercera, etc.)
- `y0`: Lista con condiciones iniciales `[y(x₀), y'(x₀), y''(x₀), ..., y⁽ᵐ⁻¹⁾(x₀)]`

**Retorno:** `(X, Y)` donde Y es una lista de listas, Y[i] = `[y, y', y'', ..., y⁽ᵐ⁻¹⁾]` en el paso i.

**Conversión interna:** La EDO de orden m se convierte al sistema:
```
y₁' = y₂
y₂' = y₃
...
yₘ' = f(x, y₁, y₂, ..., yₘ)
```

### 🔬 `metodos.convergencia` - Análisis de Convergencia
Herramientas para analizar el orden de convergencia de métodos EDO:
- **`calcular_factor_convergencia_euler()`** - Análisis para Euler
- **`calcular_factor_convergencia_heun()`** - Análisis para Heun
- **`calcular_factor_convergencia_punto_medio()`** - Análisis para Punto Medio
- **`calcular_factor_convergencia_rk4()`** - Análisis para Runge-Kutta 4

Cada función ejecuta el método 3 veces con pasos h, h/2, h/4 y calcula el factor de convergencia punto a punto usando:
```
factor_i = ln(|y1_i - y2_i| / |y2_i - y3_i|) / ln(2)
```

Retornan: `(X, factores, factor_promedio)` - ideal para graficar y validar órdenes teóricos.

### 🛠️ `metodos.utils` - Utilidades
Funciones auxiliares de propósito general:
- **`limpiar_terminal()`** - Limpia la pantalla
- **`graficar_funciones()`** - Grafica múltiples funciones

## 💡 Ejemplos de Uso

### Ejemplo 1: Encontrar una raíz con Newton-Raphson

```python
from metodos import newton_raphson
import math

# Definir función y su derivada
f = lambda x: x**3 - 2*x - 5
f_prime = lambda x: 3*x**2 - 2

# Encontrar raíz con verbose=True (imprime resultados)
raiz, error, iteraciones = newton_raphson(
    f, f_prime, 
    x0=2.0, 
    tolerancia=1e-6, 
    tipo_error=1,
    verbose=True
)

# Uso programático con verbose=False
raiz, error, iteraciones = newton_raphson(
    f, f_prime, 
    x0=2.0, 
    tolerancia=1e-6, 
    tipo_error=1,
    verbose=False
)
print(f"Raíz encontrada: {raiz} en {iteraciones} iteraciones")
```

### Ejemplo 2: Resolver sistema de ecuaciones con Gauss-Seidel

```python
from metodos import gauss_seidel

# Sistema: 4x + y = 10, x + 3y = 9
A = [[4, 1], [1, 3]]
B = [10, 9]
Xn = [0, 0]
Xv = [0, 0]

# Resolver
solucion, error, iter, convergio = gauss_seidel(
    A, B, Xn, Xv, 
    tolerancia=1e-6,
    verbose=False
)
print(f"Solución: x={solucion[0]:.4f}, y={solucion[1]:.4f}")
```

### Ejemplo 3: Interpolación con splines cúbicos

```python
from metodos import curvas_spline, graficar_splines

# Datos
X = [0, 1, 2, 3, 4]
Y = [0, 1, 4, 9, 16]

# Crear splines
funciones, coef, X_ord, Y_ord = curvas_spline(X=X, Y=Y, verbose=False)

# Visualizar
graficar_splines(funciones, coef, X_ord, Y_ord)
```

### Ejemplo 4: Integración numérica con trapecio

```python
from metodos import trapecio
import math

# Modo 1: Función continua
def f(x):
    return x**2 + 1

resultado = trapecio(f=f, a=0, b=3, n=1000, verbose=False)
print(f"∫₀³ (x²+1)dx ≈ {resultado}")  # Exacto: 12.0

# Modo 2: Datos tabulados no equiespaciados
X = [0, 0.5, 2, 3]
Y = [1, 1.25, 5, 10]
resultado = trapecio(X=X, Y=Y, verbose=False)
print(f"Integral aproximada: {resultado}")
```

### Ejemplo 5: Resolver EDO con Runge-Kutta 4

```python
from metodos import runge_kutta4
import matplotlib.pyplot as plt

# Problema: dy/dx = -2xy, y(0) = 1
f = lambda x, y: -2*x*y

# Resolver de x=0 a x=2 con 20 pasos
X, Y = runge_kutta4(f, x0=0, y0=1, xf=2, n=20, verbose=False)

# Graficar solución
plt.plot(X, Y, 'o-', label='RK4')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución numérica de dy/dx = -2xy')
plt.legend()
plt.grid(True)
plt.show()
```

### Ejemplo 6: Análisis de convergencia

```python
from metodos import calcular_factor_convergencia_euler
import matplotlib.pyplot as plt

# Problema: dy/dx = y, y(0) = 1 (solución exacta: y = e^x)
f = lambda x, y: y

# Calcular factores de convergencia
X, factores, promedio = calcular_factor_convergencia_euler(
    f, x0=0, y0=1, xf=1, n=20, verbose=False
)

print(f"Factor promedio: {promedio:.4f}")  # Esperado ≈ 1.0 (orden 1)
print(f"Orden teórico confirmado: ✅" if abs(promedio - 1.0) < 0.3 else "⚠️")

# Graficar
plt.plot(X, factores, 'o-', label='Factor de convergencia')
plt.axhline(y=1.0, color='r', linestyle='--', label='Orden teórico = 1')
plt.axhline(y=promedio, color='g', linestyle=':', label=f'Promedio = {promedio:.3f}')
plt.xlabel('x')
plt.ylabel('Factor')
plt.title('Análisis de Convergencia - Método de Euler')
plt.legend()
plt.grid(True)
plt.show()
```

### Ejemplo 7: Resolver sistema de EDOs (Oscilador Armónico)

```python
from metodos import runge_kutta4_sistema
import matplotlib.pyplot as plt

# Sistema: y'' = -y (oscilador armónico simple)
# Conversión: y₁ = y, y₂ = y'
# dy₁/dx = y₂
# dy₂/dx = -y₁

f1 = lambda x, Y: Y[1]         # dy/dx = y'
f2 = lambda x, Y: -Y[0]        # dy'/dx = -y

# Condiciones iniciales: y(0)=1, y'(0)=0
X, Y = runge_kutta4_sistema(
    funciones=[f1, f2],
    x0=0,
    y0=[1.0, 0.0],
    xf=10,
    n=100,
    verbose=False
)

# Extraer y(x) y y'(x)
y_valores = [Y[i][0] for i in range(len(Y))]
y_prima_valores = [Y[i][1] for i in range(len(Y))]

# Graficar
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(X, y_valores, 'b-', label='y(x)')
plt.plot(X, y_prima_valores, 'r--', label="y'(x)")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Oscilador Armónico')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_valores, y_prima_valores, 'g-')
plt.xlabel('y')
plt.ylabel("y'")
plt.title('Diagrama de Fase')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Ejemplo 8: Resolver EDO de orden superior

```python
from metodos import runge_kutta4_orden_superior
import matplotlib.pyplot as plt
import math

# EDO de segundo orden: y'' + 2y' + 2y = 0
# Solución exacta: y = e^(-x) * cos(x)
def f(x, y, y_prima):
    return -2*y_prima - 2*y

# Condiciones iniciales: y(0)=1, y'(0)=-1
X, Y = runge_kutta4_orden_superior(
    f=f,
    x0=0,
    y0=[1.0, -1.0],  # [y(0), y'(0)]
    xf=5,
    n=100,
    orden=2,
    verbose=False
)

# Extraer soluciones
y_num = [Y[i][0] for i in range(len(Y))]
y_prima_num = [Y[i][1] for i in range(len(Y))]

# Solución exacta
y_exacta = [math.exp(-x) * math.cos(x) for x in X]

# Graficar comparación
plt.plot(X, y_num, 'b-', label='Numérica', linewidth=2)
plt.plot(X, y_exacta, 'r--', label='Exacta', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title("y'' + 2y' + 2y = 0")
plt.legend()
plt.grid(True)
plt.show()

# Calcular error
errores = [abs(y_num[i] - y_exacta[i]) for i in range(len(X))]
print(f"Error máximo: {max(errores):.2e}")
```

### Ejemplo 9: Comparar métodos en un sistema (Lotka-Volterra)

```python
from metodos import (euler_sistema, heun_sistema, 
                     punto_medio_sistema, runge_kutta4_sistema)
import matplotlib.pyplot as plt

# Sistema depredador-presa de Lotka-Volterra
# dx/dt = αx - βxy (presas)
# dy/dt = δxy - γy (depredadores)
alpha, beta, delta, gamma = 1.0, 0.5, 0.5, 1.0

f1 = lambda t, Y: alpha*Y[0] - beta*Y[0]*Y[1]
f2 = lambda t, Y: delta*Y[0]*Y[1] - gamma*Y[1]

# Condiciones iniciales
funciones = [f1, f2]
t0, tf, n = 0, 20, 100
y0 = [2.0, 1.0]  # poblaciones iniciales

# Resolver con cada método
X_e, Y_e = euler_sistema(funciones, t0, y0, tf, n, verbose=False)
X_h, Y_h = heun_sistema(funciones, t0, y0, tf, n, verbose=False)
X_pm, Y_pm = punto_medio_sistema(funciones, t0, y0, tf, n, verbose=False)
X_rk, Y_rk = runge_kutta4_sistema(funciones, t0, y0, tf, n, verbose=False)

# Graficar comparación
plt.figure(figsize=(12, 5))

# Evolución temporal
plt.subplot(1, 2, 1)
plt.plot([Y_e[i][0] for i in range(len(Y_e))], label='Presas (Euler)', alpha=0.6)
plt.plot([Y_rk[i][0] for i in range(len(Y_rk))], label='Presas (RK4)', linewidth=2)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Evolución Temporal')
plt.legend()
plt.grid(True)

# Diagrama de fase
plt.subplot(1, 2, 2)
plt.plot([Y_e[i][0] for i in range(len(Y_e))], 
         [Y_e[i][1] for i in range(len(Y_e))], 
         label='Euler', alpha=0.5)
plt.plot([Y_rk[i][0] for i in range(len(Y_rk))], 
         [Y_rk[i][1] for i in range(len(Y_rk))], 
         label='RK4', linewidth=2)
plt.xlabel('Presas')
plt.ylabel('Depredadores')
plt.title('Diagrama de Fase')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## ⚙️ Requisitos

Python 3.8 o superior

### Dependencias principales
- numpy>=1.24.3
- matplotlib>=3.7.1
- sympy>=1.12
- scipy>=1.10.1

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/JhelixT/metodos_numericos.git
cd metodos_numericos
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📖 Uso de la Biblioteca

### Importación Simple

```python
# Importar funciones individuales
from metodos import newton_raphson, gauss_pivot, trapecio

# Importar módulo completo
import metodos

# Usar funciones
raiz, error, iter = metodos.newton_raphson(f, f_prime, x0, tol, tipo_error)
```

### Importación por Módulo

```python
# Importar módulos específicos
from metodos.raices import newton_raphson, metodo_secante
from metodos.sistemas_lineales import jacobi, gauss_seidel
from metodos.aproximacion import curvas_spline, interpolacion
from metodos.integracion import trapecio, simpson
from metodos.diferenciacion import diferenciacion
from metodos.edo1 import euler, heun, punto_medio, runge_kutta4
from metodos.sistemas_edo import euler_sistema, heun_sistema, punto_medio_sistema, runge_kutta4_sistema
from metodos.edo_orden_superior import euler_orden_superior, runge_kutta4_orden_superior
from metodos.convergencia import calcular_factor_convergencia_euler
```

### Parámetro `verbose`

Todos los métodos soportan el parámetro `verbose` para controlar la salida:

- **`verbose=True`** (por defecto): Modo interactivo
  - Imprime resultados en pantalla
  - Permite input del usuario cuando es necesario
  - Ideal para uso en scripts y jupyter notebooks

- **`verbose=False`**: Modo programático
  - Sin salida por consola
  - Retorna resultados como tuplas
  - Requiere todos los parámetros
  - Ideal para integraciones y automatización

```python
# Modo interactivo
newton_raphson(f, f_prime, x0, tol, tipo_error, verbose=True)
# Salida: "La raiz es 2.094551... con un error de 1.23e-07"

# Modo programático
raiz, error, iteraciones = newton_raphson(f, f_prime, x0, tol, tipo_error, verbose=False)
# Sin salida, solo retorna valores
```

## 🎓 Guías de Ejercicios

Cada guía (`guia1/`, `guia2/`, etc.) contiene ejercicios específicos que implementan diferentes métodos numéricos. Los archivos están organizados por tema y numerados según el ejercicio correspondiente.

### Contenido de las Guías
- **Guía 1**: Métodos de búsqueda de raíces (bisección, regula falsi)
- **Guía 2**: Métodos iterativos (punto fijo, Newton-Raphson, secante)
- **Guía 3**: Sistemas de ecuaciones lineales
- **Guía 4**: Métodos directos (eliminación gaussiana)
- **Guía 5**: Interpolación y regresión
- **Guía 6**: Interpolación segmentaria (splines cúbicos)
- **Guía 7**: Integración numérica (trapecio, Simpson)

## 📂 Carpeta de Ejemplos

La carpeta `ejemplos/` contiene scripts listos para ejecutar que demuestran el uso de diferentes métodos:

- **`ejemplo_euler_sistema.py`**: Ejemplo básico de uso de euler_sistema con oscilador armónico
- **`ejemplo_comparacion_metodos.py`**: Comparación exhaustiva de los 4 métodos (Euler, Heun, Punto Medio, RK4) con análisis de error
- **`ejemplo_edo_orden_superior.py`**: Tutorial completo sobre EDOs de orden superior con múltiples ejemplos
- **`ejemplo_convergencia_completo.py`**: Análisis de convergencia de métodos EDO
- **`test_integracion.py`**: Tests de integración para verificar funcionamiento

Para ejecutar cualquier ejemplo:
```bash
cd ejemplos
python ejemplo_euler_sistema.py
```

O desde la raíz del proyecto:
```bash
python ejemplos/ejemplo_euler_sistema.py
```

## 🔄 Compatibilidad y Migración

### Código Legacy

El archivo `metodos/funciones.py` se mantiene para compatibilidad con código existente. Sin embargo, se recomienda migrar a la nueva estructura modular.

```python
# ⚠️ Forma antigua (funciona pero deprecated)
from metodos.funciones import newton_raphson

# ✅ Forma nueva (recomendada)
from metodos import newton_raphson
# o
from metodos.raices import newton_raphson
```

### Ventajas de la Nueva Estructura

1. **Organización**: Métodos agrupados por categoría matemática
2. **Mantenibilidad**: Código más fácil de mantener y actualizar
3. **Documentación**: Cada módulo con su propósito específico
4. **Reutilización**: Importar solo lo que necesitas
5. **Testing**: Tests más específicos por módulo
6. **Versatilidad**: Control total con parámetro `verbose`

## 🧪 Testing

```python
# Ejemplo de test simple
from metodos import newton_raphson

def test_newton_raphson():
    f = lambda x: x**2 - 4
    f_prime = lambda x: 2*x
    raiz, error, iter = newton_raphson(f, f_prime, 1.0, 1e-6, 1, verbose=False)
    assert abs(raiz - 2.0) < 1e-6, "La raíz debería ser 2.0"
    print("✅ Test pasado")

test_newton_raphson()
```

## 🤝 Desarrollo

Para contribuir al proyecto:

1. Crear un fork del repositorio
2. Crear una rama para la nueva característica (`git checkout -b feature/nueva-funcionalidad`)
3. Realizar los cambios siguiendo el estilo del código
4. Asegurarse de que todos los métodos tengan:
   - Parámetro `verbose` opcional
   - Docstrings completos
   - Retornos estructurados (tuplas con información detallada)
5. Hacer commit de los cambios (`git commit -m 'Descripción'`)
6. Push a la rama (`git push origin feature/nueva-funcionalidad`)
7. Crear un Pull Request

### Estructura de Commits

```bash
# Ejemplo de buen commit
git commit -m "Agregar método de Simpson para integración numérica

- Implementar simpson() en integracion.py
- Agregar parámetro verbose
- Incluir ejemplos en docstring
- Actualizar tests"
```

## 📝 Notas Importantes

- ⚠️ **Diagonal Dominante**: La verificación en `esDiagDom()` usa comparación estricta (`<`) para garantizar convergencia de métodos iterativos.
- 📊 **Splines con Datos No Equiespaciados**: El método `trapecio()` automáticamente construye splines cúbicos cuando detecta datos no equiespaciados.
- 🔄 **Retornos Consistentes**: Todos los métodos iterativos retornan tuplas con `(resultado, error, iteraciones)` o similar.
- 🎯 **Orden de Convergencia**: Los métodos EDO tienen órdenes teóricos: Euler (1), Heun (2), Punto Medio (2), RK4 (4). Usa el módulo `convergencia` para validarlos experimentalmente.
- 📐 **Diferenciación Numérica**: Las diferencias centrales (O(h²)) son más precisas que las progresivas/regresivas (O(h)) para el mismo paso h.
- 🔗 **Sistemas de EDOs**: Las funciones del sistema reciben el vector completo de estado Y como parámetro. Y[0] es la primera variable, Y[1] la segunda, etc.
- 📐 **EDOs de Orden Superior**: Se convierten automáticamente a sistemas de primer orden. El vector y0 debe contener `[y(x₀), y'(x₀), ..., y⁽ᵐ⁻¹⁾(x₀)]`.
- 🎲 **Estructura de Y en Sistemas**: Para sistemas, Y[i][j] representa el valor de la variable j en el paso i. Para acceder a toda la solución de la variable k: `[Y[i][k] for i in range(len(Y))]`.

## 📚 Referencias

- Burden, R.L., & Faires, J.D. (2010). *Numerical Analysis* (9th ed.)
- Chapra, S.C., & Canale, R.P. (2015). *Numerical Methods for Engineers* (7th ed.)

## 📧 Contacto

Para preguntas o sugerencias, abrir un issue en el repositorio.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

**Desarrollado con ❤️ para el aprendizaje de métodos numéricos**