#!/usr/bin/env python3
"""
Ejemplo de Traducción con Hugging Face
======================================

Este script demuestra cómo usar modelos de traducción automática.
"""

from transformers import pipeline
import torch

def traduccion_multiidioma():
    """
    Ejemplo de traducción entre múltiples idiomas
    """
    print("🌍 Traducción Multiidioma...")
    
    # Crear pipeline de traducción
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-en-es",  # Inglés a Español
        device=0 if torch.cuda.is_available() else -1
    )
    
    textos_en_ingles = [
        "Hello, how are you today?",
        "The weather is beautiful.",
        "I love programming in Python.",
        "Artificial intelligence is fascinating.",
        "Thank you for your help."
    ]
    
    print("\n🔄 Traduciendo de inglés a español:")
    print("-" * 50)
    
    for i, texto in enumerate(textos_en_ingles, 1):
        traduccion = translator(texto)
        resultado = traduccion[0]['translation_text']
        
        print(f"{i}. EN: {texto}")
        print(f"   ES: {resultado}")
        print()

def main():
    """
    Función principal que ejecuta todos los ejemplos de traducción
    """
    print("🤗 TRADUCCIÓN CON HUGGING FACE")
    print("=" * 60)
    print("Este script demuestra cómo traducir texto entre diferentes idiomas")
    
    try:
        traduccion_multiidioma()
        print("\n✅ ¡Ejemplos de traducción completados exitosamente!")
        print("💡 Consejos:")
        print("   • Usa modelos específicos para pares de idiomas")
        print("   • Helsinki-NLP tiene excelentes modelos de traducción")
        print("   • Para mejor calidad, usa modelos más grandes")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("💡 Verifica que todas las dependencias estén instaladas")

if __name__ == "__main__":
    main()
