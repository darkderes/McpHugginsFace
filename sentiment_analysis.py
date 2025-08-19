#!/usr/bin/env python3
"""
Ejemplo de Análisis de Sentimientos con Hugging Face
===================================================

Este script demuestra cómo usar modelos preentrenados de Hugging Face
para realizar análisis de sentimientos en texto en español e inglés.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analisis_sentimientos_basico():
    """
    Ejemplo básico de análisis de sentimientos usando pipeline
    """
    print("🚀 Iniciando análisis de sentimientos básico...")
    
    # Crear pipeline para análisis de sentimientos
    # Usando modelo multilingüe que soporta español
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )
    
    # Textos de ejemplo en español
    textos_ejemplo = [
        "¡Me encanta este producto! Es fantástico.",
        "Este servicio es terrible, muy decepcionante.",
        "El clima está bien hoy, ni muy bueno ni muy malo.",
        "¡Qué día tan maravilloso! El sol brilla hermoso.",
        "Estoy muy triste por lo que pasó ayer.",
        "La comida estaba deliciosa, definitivamente volveré.",
        "No me gustó nada la película, fue aburrida"
    ]
    
    print("\n📝 Analizando textos:")
    print("-" * 60)
    
    resultados = []
    for i, texto in enumerate(textos_ejemplo, 1):
        resultado = sentiment_pipeline(texto)
        sentimiento = resultado[0]
        
        # Mapear etiquetas a español
        etiquetas_es = {
            'LABEL_0': 'Negativo',
            'LABEL_1': 'Neutral', 
            'LABEL_2': 'Positivo',
            'NEGATIVE': 'Negativo',
            'NEUTRAL': 'Neutral',
            'POSITIVE': 'Positivo'
        }
        
        etiqueta = etiquetas_es.get(sentimiento['label'], sentimiento['label'])
        confianza = sentimiento['score'] * 100
        
        print(f"{i}. Texto: {texto}")
        print(f"   Sentimiento: {etiqueta} ({confianza:.1f}% confianza)")
        print()
        
        resultados.append({
            'texto': texto,
            'sentimiento': etiqueta,
            'confianza': confianza
        })
    
    return resultados

def analisis_avanzado_con_modelo_personalizado():
    """
    Ejemplo avanzado usando modelo específico para español
    """
    print("🔬 Análisis avanzado con modelo personalizado...")
    
    # Modelo específicamente entrenado para español
    model_name = "pysentimiento/robertuito-sentiment-analysis"
    
    try:
        # Cargar modelo y tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Crear pipeline personalizado
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Textos más complejos en español
        textos_complejos = [
            "Aunque el producto tiene algunas fallas menores, en general estoy satisfecho con la compra.",
            "La atención al cliente fue excelente, pero el producto llegó dañado.",
            "No sé qué pensar sobre esta situación, es muy confusa.",
            "¡Increíble! Superó todas mis expectativas por completo.",
            "Meh, está bien pero nada del otro mundo."
        ]
        
        print("\n📊 Análisis de textos complejos:")
        print("-" * 60)
        
        for i, texto in enumerate(textos_complejos, 1):
            resultado = sentiment_pipeline(texto)
            sentimiento = resultado[0]
            
            print(f"{i}. {texto}")
            print(f"   → {sentimiento['label']}: {sentimiento['score']:.3f}")
            print()
            
    except Exception as e:
        print(f"⚠️  Error al cargar el modelo personalizado: {e}")
        print("Usando modelo por defecto...")
        return analisis_sentimientos_basico()

def visualizar_resultados(resultados):
    """
    Crear visualizaciones de los resultados
    """
    print("📈 Creando visualizaciones...")
    
    df = pd.DataFrame(resultados)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras - Distribución de sentimientos
    sentiment_counts = df['sentimiento'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Rojo, Amarillo, Verde
    
    ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    ax1.set_title('Distribución de Sentimientos', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentimiento')
    ax1.set_ylabel('Cantidad')
    
    # Añadir valores en las barras
    for i, v in enumerate(sentiment_counts.values):
        ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Gráfico de dispersión - Confianza por sentimiento
    sentiment_colors = {'Positivo': '#6bcf7f', 'Negativo': '#ff6b6b', 'Neutral': '#ffd93d'}
    
    for sentiment in df['sentimiento'].unique():
        data = df[df['sentimiento'] == sentiment]
        ax2.scatter(range(len(data)), data['confianza'], 
                   c=sentiment_colors[sentiment], label=sentiment, s=100, alpha=0.7)
    
    ax2.set_title('Nivel de Confianza por Predicción', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Índice de Texto')
    ax2.set_ylabel('Confianza (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    print("💾 Gráfico guardado como 'sentiment_analysis_results.png'")
    plt.show()

def ejemplo_interactivo():
    """
    Ejemplo interactivo para que el usuario pruebe sus propios textos
    """
    print("\n🎮 Modo Interactivo")
    print("Escribe textos para analizar (escribe 'salir' para terminar)")
    print("-" * 50)
    
    # Inicializar pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
    
    while True:
        texto = input("\n📝 Ingresa tu texto: ").strip()
        
        if texto.lower() in ['salir', 'exit', 'quit', '']:
            print("👋 ¡Hasta luego!")
            break
            
        try:
            resultado = sentiment_pipeline(texto)
            sentimiento = resultado[0]
            
            # Mapear a español
            etiquetas_es = {
                'LABEL_0': 'Negativo 😞',
                'LABEL_1': 'Neutral 😐', 
                'LABEL_2': 'Positivo 😊'
            }
            
            etiqueta = etiquetas_es.get(sentimiento['label'], sentimiento['label'])
            confianza = sentimiento['score'] * 100
            
            print(f"🎯 Resultado: {etiqueta}")
            print(f"📊 Confianza: {confianza:.1f}%")
            
            # Barra de confianza visual
            barra_length = int(confianza / 5)  # Escalar a 20 caracteres max
            barra = "█" * barra_length + "░" * (20 - barra_length)
            print(f"📈 [{barra}] {confianza:.1f}%")
            
        except Exception as e:
            print(f"❌ Error al procesar el texto: {e}")

def main():
    """
    Función principal que ejecuta todos los ejemplos
    """
    print("🤗 Ejemplos de Análisis de Sentimientos con Hugging Face")
    print("=" * 60)
    
    try:
        # Ejemplo básico
        resultados = analisis_sentimientos_basico()
        
        # Análisis avanzado
        analisis_avanzado_con_modelo_personalizado()
        
        # Visualizaciones
        visualizar_resultados(resultados)
        
        # Modo interactivo
        respuesta = input("\n¿Quieres probar el modo interactivo? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
            ejemplo_interactivo()
        
        print("\n✅ ¡Ejemplos completados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        print("💡 Asegúrate de haber instalado todas las dependencias con:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
