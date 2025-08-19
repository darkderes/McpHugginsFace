#!/usr/bin/env python3
"""
Ejemplo de Clasificación de Texto con Hugging Face
==================================================

Este script demuestra diferentes técnicas de clasificación de texto
usando modelos preentrenados y fine-tuning con Hugging Face.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def clasificacion_basica_zero_shot():
    """
    Ejemplo de clasificación zero-shot (sin entrenamiento previo)
    """
    print("🚀 Clasificación Zero-Shot...")
    
    # Crear pipeline de clasificación zero-shot
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Textos de ejemplo
    textos = [
        "Me encanta programar en Python, es muy divertido",
        "El partido de fútbol fue emocionante hasta el final",
        "La nueva película de Marvel es increíble",
        "Necesito comprar ingredientes para hacer una pizza",
        "El mercado de valores subió significativamente hoy",
        "Mi gato se perdió y estoy muy preocupado",
        "La conferencia sobre inteligencia artificial fue muy educativa"
    ]
    
    # Categorías candidatas
    categorias = [
        "tecnología", "deportes", "entretenimiento", 
        "cocina", "finanzas", "mascotas", "educación"
    ]
    
    print("\n📝 Clasificando textos en categorías:")
    print(f"Categorías: {', '.join(categorias)}")
    print("-" * 70)
    
    resultados = []
    
    for i, texto in enumerate(textos, 1):
        resultado = classifier(texto, categorias)
        
        categoria_principal = resultado['labels'][0]
        confianza_principal = resultado['scores'][0] * 100
        
        print(f"{i}. Texto: {texto}")
        print(f"   Categoría: {categoria_principal} ({confianza_principal:.1f}%)")
        
        # Mostrar top 3 categorías
        print("   Top 3 categorías:")
        for j in range(min(3, len(resultado['labels']))):
            cat = resultado['labels'][j]
            conf = resultado['scores'][j] * 100
            print(f"      {j+1}. {cat}: {conf:.1f}%")
        print()
        
        resultados.append({
            'texto': texto,
            'categoria': categoria_principal,
            'confianza': confianza_principal,
            'todas_categorias': resultado['labels'],
            'todas_confianzas': resultado['scores']
        })
    
    return resultados

def clasificacion_emociones():
    """
    Ejemplo específico de clasificación de emociones
    """
    print("😊 Clasificación de Emociones...")
    
    # Pipeline especializado en emociones
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Textos con diferentes emociones
    textos_emocionales = [
        "I'm so excited about my vacation next week!",
        "I can't believe they cancelled the concert, I'm devastated.",
        "This traffic is making me so angry and frustrated.",
        "I feel anxious about the job interview tomorrow.",
        "Spending time with my family makes me feel so happy and grateful.",
        "I'm really surprised by how well the project turned out.",
        "I feel disgusted by the way they treated the animals."
    ]
    
    print("\n🎭 Analizando emociones en textos:")
    print("-" * 60)
    
    emociones_detectadas = []
    
    for i, texto in enumerate(textos_emocionales, 1):
        resultado = emotion_classifier(texto)
        
        # Obtener todas las emociones con sus scores
        emociones = [(r['label'], r['score']) for r in resultado]
        emocion_principal = emociones[0]
        
        print(f"{i}. Texto: {texto}")
        print(f"   Emoción principal: {emocion_principal[0]} ({emocion_principal[1]*100:.1f}%)")
        
        # Mostrar todas las emociones si hay múltiples
        if len(emociones) > 1:
            print("   Otras emociones detectadas:")
            for emocion, score in emociones[1:3]:  # Top 3
                print(f"      • {emocion}: {score*100:.1f}%")
        print()
        
        emociones_detectadas.append({
            'texto': texto,
            'emocion': emocion_principal[0],
            'confianza': emocion_principal[1] * 100,
            'todas_emociones': emociones
        })
    
    return emociones_detectadas

def clasificacion_idioma():
    """
    Ejemplo de detección de idioma
    """
    print("🌍 Detección de Idioma...")
    
    # Textos en diferentes idiomas
    textos_multiidioma = [
        "Hello, how are you today?",
        "Hola, ¿cómo estás hoy?",
        "Bonjour, comment allez-vous aujourd'hui?",
        "Guten Tag, wie geht es Ihnen heute?",
        "Ciao, come stai oggi?",
        "こんにちは、今日はいかがですか？",
        "Привет, как дела сегодня?",
        "你好，你今天怎么样？"
    ]
    
    # Usar pipeline de clasificación con modelo de detección de idioma
    language_detector = pipeline(
        "text-classification",
        model="papluca/xlm-roberta-base-language-detection",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("\n🔍 Detectando idiomas:")
    print("-" * 50)
    
    # Mapeo de códigos de idioma a nombres
    idiomas = {
        'en': 'Inglés', 'es': 'Español', 'fr': 'Francés', 'de': 'Alemán',
        'it': 'Italiano', 'ja': 'Japonés', 'ru': 'Ruso', 'zh': 'Chino',
        'pt': 'Portugués', 'ar': 'Árabe', 'hi': 'Hindi', 'ko': 'Coreano'
    }
    
    for i, texto in enumerate(textos_multiidioma, 1):
        resultado = language_detector(texto)
        
        codigo_idioma = resultado[0]['label']
        confianza = resultado[0]['score'] * 100
        nombre_idioma = idiomas.get(codigo_idioma, codigo_idioma)
        
        print(f"{i}. '{texto}'")
        print(f"   → {nombre_idioma} ({codigo_idioma}) - {confianza:.1f}%")
        print()

def clasificacion_personalizada_con_datos():
    """
    Ejemplo de clasificación con datos personalizados
    """
    print("🎯 Clasificación con Datos Personalizados...")
    
    # Crear dataset de ejemplo para clasificación de reviews
    datos_ejemplo = [
        # Reviews positivos
        ("Este producto es fantástico, lo recomiendo totalmente", "positivo"),
        ("Excelente calidad, superó mis expectativas", "positivo"),
        ("Me encanta, definitivamente volveré a comprar", "positivo"),
        ("Muy buena experiencia de compra, rápido y eficiente", "positivo"),
        ("Producto de alta calidad, vale la pena el precio", "positivo"),
        
        # Reviews negativos
        ("Terrible producto, no funciona como se describe", "negativo"),
        ("Muy decepcionante, perdí mi dinero", "negativo"),
        ("Pésima calidad, se rompió al primer uso", "negativo"),
        ("No lo recomiendo para nada, muy malo", "negativo"),
        ("Servicio al cliente horrible, nunca más compro aquí", "negativo"),
        
        # Reviews neutros
        ("El producto está bien, nada especial", "neutral"),
        ("Cumple su función básica, precio justo", "neutral"),
        ("No está mal pero tampoco es extraordinario", "neutral"),
        ("Producto promedio, hay mejores opciones", "neutral"),
        ("Funciona correctamente, sin más", "neutral")
    ]
    
    # Separar textos y etiquetas
    textos = [dato[0] for dato in datos_ejemplo]
    etiquetas = [dato[1] for dato in datos_ejemplo]
    
    print(f"\n📊 Dataset de ejemplo creado:")
    print(f"   Total de muestras: {len(datos_ejemplo)}")
    
    # Contar etiquetas
    from collections import Counter
    conteo_etiquetas = Counter(etiquetas)
    for etiqueta, cantidad in conteo_etiquetas.items():
        print(f"   {etiqueta}: {cantidad} muestras")
    
    # Usar modelo preentrenado para clasificar estos textos
    classifier = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("\n🔬 Clasificando con modelo preentrenado:")
    print("-" * 60)
    
    predicciones = []
    etiquetas_reales = []
    
    for texto, etiqueta_real in datos_ejemplo:
        resultado = classifier(texto)
        prediccion = resultado[0]
        
        # Mapear etiquetas del modelo a nuestras etiquetas
        mapeo_etiquetas = {
            'LABEL_1': 'muy_negativo', 'LABEL_2': 'negativo', 'LABEL_3': 'neutral',
            'LABEL_4': 'positivo', 'LABEL_5': 'muy_positivo',
            'NEGATIVE': 'negativo', 'POSITIVE': 'positivo'
        }
        
        etiqueta_predicha = mapeo_etiquetas.get(prediccion['label'], prediccion['label'])
        confianza = prediccion['score'] * 100
        
        print(f"Texto: {texto[:50]}...")
        print(f"Real: {etiqueta_real} | Predicho: {etiqueta_predicha} ({confianza:.1f}%)")
        print()
        
        predicciones.append(etiqueta_predicha)
        etiquetas_reales.append(etiqueta_real)
    
    return predicciones, etiquetas_reales

def visualizar_resultados_clasificacion(resultados):
    """
    Crear visualizaciones de los resultados de clasificación
    """
    print("📈 Creando visualizaciones...")
    
    # Preparar datos
    df = pd.DataFrame(resultados)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribución de categorías
    categoria_counts = df['categoria'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(categoria_counts)))
    
    wedges, texts, autotexts = ax1.pie(categoria_counts.values, 
                                      labels=categoria_counts.index,
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90)
    ax1.set_title('Distribución de Categorías', fontsize=14, fontweight='bold')
    
    # 2. Confianza promedio por categoría
    confianza_por_categoria = df.groupby('categoria')['confianza'].mean().sort_values(ascending=True)
    
    bars = ax2.barh(confianza_por_categoria.index, confianza_por_categoria.values)
    ax2.set_xlabel('Confianza Promedio (%)')
    ax2.set_title('Confianza Promedio por Categoría', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 3. Distribución de confianza
    ax3.hist(df['confianza'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Nivel de Confianza (%)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Niveles de Confianza', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Scatter plot: longitud del texto vs confianza
    df['longitud_texto'] = df['texto'].str.len()
    scatter = ax4.scatter(df['longitud_texto'], df['confianza'], 
                         c=pd.Categorical(df['categoria']).codes, 
                         cmap='tab10', alpha=0.7, s=100)
    ax4.set_xlabel('Longitud del Texto (caracteres)')
    ax4.set_ylabel('Confianza (%)')
    ax4.set_title('Longitud del Texto vs Confianza', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # Añadir leyenda para el scatter plot
    categorias_unicas = df['categoria'].unique()
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=plt.cm.tab10(i), markersize=8, label=cat)
              for i, cat in enumerate(categorias_unicas)]
    ax4.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado como 'classification_results.png'")
    plt.show()

def ejemplo_interactivo_clasificacion():
    """
    Modo interactivo para clasificación personalizada
    """
    print("\n🎮 Modo Interactivo - Clasificación de Texto")
    print("Clasifica tus propios textos en categorías personalizadas")
    print("Escribe 'salir' para terminar")
    print("-" * 60)
    
    # Configurar categorías personalizadas
    print("📋 Primero, define las categorías para clasificar:")
    categorias_input = input("Ingresa categorías separadas por comas: ").strip()
    
    if not categorias_input:
        categorias = ["positivo", "negativo", "neutral"]
        print(f"Usando categorías por defecto: {categorias}")
    else:
        categorias = [cat.strip() for cat in categorias_input.split(',')]
        print(f"Categorías definidas: {categorias}")
    
    # Inicializar clasificador
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )
    
    print(f"\n🔄 Clasificador listo con {len(categorias)} categorías")
    print("Ahora ingresa textos para clasificar:")
    
    while True:
        texto = input("\n📝 Ingresa tu texto: ").strip()
        
        if texto.lower() in ['salir', 'exit', 'quit', '']:
            print("👋 ¡Hasta luego!")
            break
        
        try:
            print("🔄 Clasificando...")
            resultado = classifier(texto, categorias)
            
            print(f"\n🎯 Resultados para: '{texto}'")
            print("-" * 50)
            
            # Mostrar todas las categorías con sus scores
            for i, (categoria, score) in enumerate(zip(resultado['labels'], resultado['scores'])):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                print(f"{emoji} {categoria}: {score*100:.1f}%")
            
            # Barra visual para la categoría principal
            score_principal = resultado['scores'][0] * 100
            barra_length = int(score_principal / 5)  # Escalar a 20 caracteres max
            barra = "█" * barra_length + "░" * (20 - barra_length)
            print(f"\n📈 Confianza: [{barra}] {score_principal:.1f}%")
            
        except Exception as e:
            print(f"❌ Error al clasificar: {e}")

def main():
    """
    Función principal que ejecuta todos los ejemplos
    """
    print("🤗 Ejemplos de Clasificación de Texto con Hugging Face")
    print("=" * 70)
    
    try:
        # Clasificación zero-shot
        print("1️⃣  Ejecutando clasificación zero-shot...")
        resultados_zero_shot = clasificacion_basica_zero_shot()
        
        # Clasificación de emociones
        print("\n2️⃣  Ejecutando clasificación de emociones...")
        clasificacion_emociones()
        
        # Detección de idioma
        print("\n3️⃣  Ejecutando detección de idioma...")
        clasificacion_idioma()
        
        # Clasificación con datos personalizados
        print("\n4️⃣  Ejecutando clasificación personalizada...")
        clasificacion_personalizada_con_datos()
        
        # Visualizaciones
        print("\n5️⃣  Creando visualizaciones...")
        visualizar_resultados_clasificacion(resultados_zero_shot)
        
        # Modo interactivo
        respuesta = input("\n¿Quieres probar el modo interactivo? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
            ejemplo_interactivo_clasificacion()
        
        print("\n✅ ¡Ejemplos de clasificación completados exitosamente!")
        
        # Consejos finales
        print("\n💡 Consejos para mejorar la clasificación:")
        print("   • Zero-shot funciona bien para categorías generales")
        print("   • Para dominios específicos, considera fine-tuning")
        print("   • Usa modelos multilingües para textos en español")
        print("   • Combina múltiples modelos para mejor precisión")
        print("   • Evalúa siempre con datos de prueba representativos")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        print("💡 Asegúrate de haber instalado todas las dependencias con:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
