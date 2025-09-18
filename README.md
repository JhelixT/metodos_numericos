# Métodos Numéricos

Este repositorio contiene implementaciones de diversos métodos numéricos organizados en guías prácticas. El proyecto está estructurado para facilitar el aprendizaje y la aplicación de diferentes técnicas numéricas.

## Estructura del Proyecto

```
metodos_numericos/
├── guia1/         # Métodos de búsqueda de raíces básicos
├── guia2/         # Métodos iterativos
├── guia3/         # Sistemas de ecuaciones lineales
├── guia4/         # Resolución de sistemas
├── guia5/         # Interpolación
└── metodos/       # Módulo principal con funciones compartidas
```

## Características

- Implementación de métodos numéricos clásicos
- Visualización de resultados usando matplotlib
- Soporte para cálculos simbólicos con sympy
- Operaciones numéricas eficientes con numpy

## Requisitos

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

## Uso

Cada guía contiene ejercicios específicos que implementan diferentes métodos numéricos. Los archivos están organizados por tema y numerados según el ejercicio correspondiente.

## Desarrollo

Para contribuir al proyecto:

1. Crear un fork del repositorio
2. Crear una rama para la nueva característica
3. Realizar los cambios
4. Enviar un pull request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.