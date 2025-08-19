# Ejemplos de Hugging Face en Python

Este proyecto contiene ejemplos prácticos de cómo usar la biblioteca Hugging Face Transformers para diferentes tareas de procesamiento de lenguaje natural (NLP).

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
McpHugginsFace/
├── README.md                     # Documentación principal
├── requirements.txt              # Dependencias del proyecto
├── run_all_examples.py          # Script para ejecutar todos los ejemplos
├── sentiment_analysis.py        # Análisis de sentimientos
├── text_generation.py           # Generación de texto
├── text_classification.py       # Clasificación de texto
└── examples/                    # Ejemplos adicionales
    ├── __init__.py
    ├── question_answering.py    # Respuesta a preguntas
    └── translation.py           # Traducción automática
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
```

## 📚 Recursos Adicionales

- [Documentación de Hugging Face](https://huggingface.co/docs)
- [Transformers Library](https://github.com/huggingface/transformers)
- [Model Hub](https://huggingface.co/models)
