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
└── metodos/            # 📚 Biblioteca principal (módulos especializados)
    ├── __init__.py           # Exportaciones y API pública
    ├── raices.py             # 🎯 Localización de raíces
    ├── sistemas_lineales.py  # 🔢 Sistemas de ecuaciones lineales
    ├── aproximacion.py       # 📈 Interpolación, regresión y splines
    ├── integracion.py        # ∫  Integración numérica
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
from metodos.integracion import trapecio
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
- **Guía 7**: Integración numérica

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

## 📚 Referencias

- Burden, R.L., & Faires, J.D. (2010). *Numerical Analysis* (9th ed.)
- Chapra, S.C., & Canale, R.P. (2015). *Numerical Methods for Engineers* (7th ed.)

## 📧 Contacto

Para preguntas o sugerencias, abrir un issue en el repositorio.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

**Desarrollado con ❤️ para el aprendizaje de métodos numéricos**