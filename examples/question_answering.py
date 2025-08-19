#!/usr/bin/env python3
"""
Ejemplo de Question Answering (Respuesta a Preguntas) con Hugging Face
======================================================================

Este script demuestra cómo usar modelos de QA para responder preguntas
basadas en un contexto dado.
"""

from transformers import pipeline
import torch

def qa_basico():
    """
    Ejemplo básico de Question Answering
    """
    print("❓ Question Answering Básico...")
    
    # Crear pipeline de QA
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Contexto de ejemplo
    contexto = """
    La inteligencia artificial (IA) es una rama de las ciencias de la computación 
    que se ocupa de la creación de sistemas capaces de realizar tareas que 
    típicamente requieren inteligencia humana. Estos sistemas pueden aprender, 
    razonar, percibir, y en algunos casos, actuar de manera autónoma. 
    
    El aprendizaje automático es un subcampo de la IA que se centra en el 
    desarrollo de algoritmos que pueden aprender y hacer predicciones o 
    decisiones basadas en datos. Los modelos de lenguaje como GPT y BERT 
    son ejemplos de sistemas de IA que han revolucionado el procesamiento 
    de lenguaje natural.
    
    La IA tiene aplicaciones en muchos campos, incluyendo medicina, 
    transporte autónomo, reconocimiento de imágenes, y asistentes virtuales.
    """
    
    # Preguntas de ejemplo
    preguntas = [
        "¿Qué es la inteligencia artificial?",
        "¿Cuáles son algunos ejemplos de modelos de lenguaje?",
        "¿En qué campos tiene aplicaciones la IA?",
        "¿Qué es el aprendizaje automático?",
        "¿Pueden los sistemas de IA actuar de manera autónoma?"
    ]
    
    print(f"\n📖 Contexto: {contexto[:100]}...")
    print("\n❓ Respondiendo preguntas:")
    print("-" * 60)
    
    for i, pregunta in enumerate(preguntas, 1):
        resultado = qa_pipeline(question=pregunta, context=contexto)
        
        respuesta = resultado['answer']
        confianza = resultado['score'] * 100
        
        print(f"{i}. Pregunta: {pregunta}")
        print(f"   Respuesta: {respuesta}")
        print(f"   Confianza: {confianza:.1f}%")
        print()

def main():
    """
    Función principal que ejecuta todos los ejemplos de Question Answering
    """
    print("🤗 QUESTION ANSWERING CON HUGGING FACE")
    print("=" * 60)
    print("Este script demuestra cómo responder preguntas basadas en contexto")
    
    try:
        qa_basico()
        print("\n✅ ¡Ejemplos de Question Answering completados exitosamente!")
        print("💡 Consejos:")
        print("   • Usa contextos más específicos para mejores respuestas")
        print("   • Experimenta con diferentes modelos de QA")
        print("   • Las preguntas deben estar relacionadas con el contexto")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("💡 Verifica que todas las dependencias estén instaladas")

if __name__ == "__main__":
    main()
