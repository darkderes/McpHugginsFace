# Ejemplos de Hugging Face en Python

Este proyecto contiene ejemplos prácticos de cómo usar la biblioteca Hugging Face (y utilidades relacionadas) para tareas de NLP y generación de imágenes.

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
McpHugginsFace/
├── README.md                     # Documentación principal
├── requirements.txt              # Dependencias del proyecto
├── run_all_examples.py           # Script para ejecutar todos los ejemplos
├── sentiment_analysis.py         # Análisis de sentimientos
├── text_generation.py            # Generación de texto
├── text_classification.py        # Clasificación de texto
├── image_generation.py           # Generación de imágenes (demo)
├── image_generation_simple.py    # Versión simplificada de generación de imágenes
├── image_generation_api.py       # Ejemplo de servicio/endpoint para generación de imágenes
├── classification_results.png    # Ejemplo de salida (imagen)
├── imagenes_generadas/           # Carpeta con imágenes generadas
└── examples/                      # Ejemplos adicionales
    ├── __init__.py
    ├── question_answering.py     # Respuesta a preguntas
    └── translation.py            # Traducción automática
```

## 🎯 Ejemplos Incluidos

### 1. Análisis de Sentimientos

Clasifica texto como positivo, negativo o neutral usando modelos preentrenados.

### 2. Generación de Texto

Genera texto coherente a partir de un prompt inicial.

### 3. Clasificación de Texto

Clasifica textos en diferentes categorías usando zero-shot classification.

### 4. Question Answering

Responde preguntas basadas en un contexto dado.

### 5. Traducción Automática

Traduce texto entre diferentes idiomas.

### 6. Generación de Imágenes (nuevo)

Este proyecto incluye ejemplos para generar imágenes a partir de prompts (demos y utilidades):

- `image_generation.py` - Demo completo que utiliza un backend/modelo para generar imágenes y guardarlas en `imagenes_generadas/`.
- `image_generation_simple.py` - Versión simplificada para pruebas rápidas.
- `image_generation_api.py` - Ejemplo de cómo exponer la funcionalidad como un endpoint o servicio local.

Las imágenes generadas se almacenan en el directorio `imagenes_generadas/` y puedes revisar `classification_results.png` como ejemplo de salida incluida.

## 🏃‍♂️ Cómo Ejecutar

### Ejecutar Todos los Ejemplos

```bash
python run_all_examples.py
```

### Ejecutar Ejemplos Individuales

```bash
# Análisis de sentimientos
python sentiment_analysis.py

# Generación de texto
python text_generation.py

# Clasificación de texto
python text_classification.py

# Question Answering
python examples/question_answering.py

# Traducción
python examples/translation.py

# Generación de imágenes (demo)
python image_generation.py

# Versión simple de generación de imágenes
python image_generation_simple.py

# Iniciar ejemplo de API para generación de imágenes (ver código para detalles de puerto/entorno)
python image_generation_api.py
```

## 📝 Notas y Recomendaciones

- Revisa `requirements.txt` para dependencias necesarias (puede incluir librerías de visión, clientes HTTP, o SDKs de servicios externos).
- Algunos ejemplos pueden requerir claves de API o conexión a internet para descargar modelos o consumir servicios externos; lee los comentarios de cada script.

## 📚 Recursos Adicionales

- [Documentación de Hugging Face](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Model Hub](https://huggingface.co/models)
