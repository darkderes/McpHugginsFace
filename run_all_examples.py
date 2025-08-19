#!/usr/bin/env python3
"""
Script Principal para Ejecutar Todos los Ejemplos
=================================================

Este script ejecuta todos los ejemplos de Hugging Face de manera secuencial.
"""

import sys
import importlib
import traceback
from pathlib import Path

def ejecutar_ejemplo(nombre_modulo, descripcion):
    """
    Ejecuta un ejemplo específico con manejo de errores
    """
    print(f"\n{'='*60}")
    print(f"🚀 EJECUTANDO: {descripcion}")
    print(f"{'='*60}")
    
    try:
        # Importar y ejecutar el módulo
        modulo = importlib.import_module(nombre_modulo)
        if hasattr(modulo, 'main'):
            modulo.main()
        else:
            print(f"⚠️  El módulo {nombre_modulo} no tiene función main()")
        
        print(f"✅ {descripcion} completado exitosamente!")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación en {nombre_modulo}: {e}")
        print("💡 Verifica que todas las dependencias estén instaladas")
        return False
        
    except Exception as e:
        print(f"❌ Error ejecutando {descripcion}: {e}")
        print(f"📍 Detalles del error:")
        traceback.print_exc()
        return False

def main():
    """
    Función principal que ejecuta todos los ejemplos
    """
    print("🤗 EJECUTOR DE EJEMPLOS DE HUGGING FACE")
    print("=" * 60)
    print("Este script ejecutará todos los ejemplos disponibles")
    
    # Lista de ejemplos a ejecutar
    ejemplos = [
        ("sentiment_analysis", "Análisis de Sentimientos"),
        ("text_generation", "Generación de Texto"),
        ("text_classification", "Clasificación de Texto"),
        ("examples.question_answering", "Question Answering"),
        ("examples.translation", "Traducción"),
        ("image_generation", "Generación de Imágenes")
    ]
    
    # Verificar que los archivos existan
    archivos_necesarios = [
        "sentiment_analysis.py",
        "text_generation.py", 
        "text_classification.py",
        "examples/question_answering.py",
        "examples/translation.py",
        "image_generation.py"
    ]
    
    archivos_faltantes = []
    for archivo in archivos_necesarios:
        if not Path(archivo).exists():
            archivos_faltantes.append(archivo)
    
    if archivos_faltantes:
        print(f"❌ Archivos faltantes: {', '.join(archivos_faltantes)}")
        return
    
    print(f"\n📋 Se ejecutarán {len(ejemplos)} ejemplos:")
    for i, (_, descripcion) in enumerate(ejemplos, 1):
        print(f"   {i}. {descripcion}")
    
    # Preguntar al usuario si quiere continuar
    respuesta = input("\n¿Continuar con la ejecución? (s/n): ").lower()
    if respuesta not in ['s', 'si', 'sí', 'yes', 'y']:
        print("👋 Ejecución cancelada por el usuario")
        return
    
    # Ejecutar ejemplos
    exitosos = 0
    fallidos = 0
    
    for modulo, descripcion in ejemplos:
        if ejecutar_ejemplo(modulo, descripcion):
            exitosos += 1
        else:
            fallidos += 1
        
        # Pausa entre ejemplos
        if modulo != ejemplos[-1][0]:  # No pausar después del último
            input("\n⏸️  Presiona Enter para continuar con el siguiente ejemplo...")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE EJECUCIÓN")
    print(f"{'='*60}")
    print(f"✅ Ejemplos exitosos: {exitosos}")
    print(f"❌ Ejemplos fallidos: {fallidos}")
    print(f"📈 Tasa de éxito: {(exitosos/(exitosos+fallidos)*100):.1f}%")
    
    if fallidos > 0:
        print("\n💡 Si hubo errores, verifica:")
        print("   • Que todas las dependencias estén instaladas: pip install -r requirements.txt")
        print("   • Que tengas conexión a internet para descargar modelos")
        print("   • Que tengas suficiente espacio en disco")
        print("   • Los logs de error arriba para más detalles")
    else:
        print("\n🎉 ¡Todos los ejemplos se ejecutaron exitosamente!")
        print("💡 Ahora puedes:")
        print("   • Ejecutar ejemplos individuales: python sentiment_analysis.py")
        print("   • Modificar los códigos para tus necesidades específicas")
        print("   • Explorar más modelos en https://huggingface.co/models")

if __name__ == "__main__":
    main()
