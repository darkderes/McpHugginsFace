#!/usr/bin/env python3
"""
Ejemplo Simplificado de Generación de Imágenes con Hugging Face
===============================================================

Este script demuestra cómo generar imágenes usando modelos de diffusion
de manera simple y robusta, evitando problemas de compatibilidad.
"""

import os
import warnings
from datetime import datetime

# Suprimir warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def verificar_dependencias():
    """
    Verifica que las dependencias necesarias estén instaladas
    """
    dependencias_faltantes = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        dependencias_faltantes.append("torch")
    
    try:
        from PIL import Image
        print("✅ PIL (Pillow): Disponible")
    except ImportError:
        dependencias_faltantes.append("pillow")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib: Disponible")
    except ImportError:
        dependencias_faltantes.append("matplotlib")
    
    try:
        # Intentar importar diffusers de manera segura
        from diffusers import StableDiffusionPipeline
        print("✅ Diffusers: Disponible")
        return True, None
    except ImportError as e:
        print(f"❌ Error con diffusers: {e}")
        dependencias_faltantes.append("diffusers")
    
    if dependencias_faltantes:
        print(f"\n❌ Dependencias faltantes: {', '.join(dependencias_faltantes)}")
        print("💡 Instala con: pip install torch pillow matplotlib diffusers")
        return False, dependencias_faltantes
    
    return True, None

def crear_directorio_salida():
    """
    Crea el directorio para guardar las imágenes
    """
    directorio = "imagenes_generadas"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
        print(f"📁 Directorio creado: {directorio}")
    return directorio

def generar_imagen_simple():
    """
    Genera una imagen usando un modelo simple y confiable
    """
    print("🎨 Generando imagen con Stable Diffusion...")
    print("-" * 60)
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        
        # Detectar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Usando dispositivo: {device}")
        
        # Usar modelo más pequeño y confiable
        model_id = "runwayml/stable-diffusion-v1-5"
        
        print("📥 Cargando modelo (esto puede tomar unos minutos la primera vez)...")
        
        # Configuración más conservadora para evitar errores
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Usar float32 para compatibilidad
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        pipe = pipe.to(device)
        
        # Optimizaciones para CPU
        if device == "cpu":
            pipe.enable_attention_slicing()
            print("🔧 Optimizaciones para CPU habilitadas")
        
        print("✅ Modelo cargado exitosamente!")
        
        # Directorio de salida
        directorio_salida = crear_directorio_salida()
        
        # Prompt simple de prueba
        prompt = "a beautiful landscape with mountains and a lake, digital art"
        print(f"\n🎭 Generando imagen con prompt: '{prompt}'")
        
        # Generar imagen con parámetros conservadores
        print("🔄 Generando... (esto puede tomar 1-2 minutos)")
        
        image = pipe(
            prompt,
            num_inference_steps=20,  # Pocos pasos para rapidez
            guidance_scale=7.5,
            width=512,
            height=512,
            num_images_per_prompt=1
        ).images[0]
        
        # Guardar imagen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"imagen_test_{timestamp}.png"
        ruta_completa = os.path.join(directorio_salida, nombre_archivo)
        
        image.save(ruta_completa)
        print(f"✅ Imagen guardada como: {ruta_completa}")
        
        # Mostrar información de la imagen
        print(f"📏 Dimensiones: {image.size}")
        print(f"🎨 Formato: {image.format}")
        
        return ruta_completa, image
        
    except Exception as e:
        print(f"❌ Error generando imagen: {e}")
        print("\n💡 Posibles soluciones:")
        print("   1. Verifica que tengas suficiente memoria RAM (mínimo 4GB)")
        print("   2. Cierra otras aplicaciones que consuman memoria")
        print("   3. Si persiste, reinicia el script")
        return None, None

def modo_interactivo():
    """
    Permite al usuario generar imágenes con sus propios prompts
    """
    print("\n🎮 MODO INTERACTIVO")
    print("=" * 60)
    print("Ingresa tus propios prompts para generar imágenes")
    print("Escribe 'salir' para terminar")
    print("\n💡 Consejos para mejores resultados:")
    print("   • Usa descripciones específicas en inglés")
    print("   • Añade estilos: 'digital art', 'oil painting', 'photorealistic'")
    print("   • Incluye detalles: 'highly detailed', 'beautiful lighting'")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "runwayml/stable-diffusion-v1-5"
        
        print(f"\n📥 Cargando modelo para modo interactivo...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        
        if device == "cpu":
            pipe.enable_attention_slicing()
        
        directorio_salida = crear_directorio_salida()
        contador = 1
        
        while True:
            print(f"\n🎨 Imagen #{contador}")
            prompt = input("Ingresa tu prompt: ").strip()
            
            if prompt.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if not prompt:
                print("⚠️  Por favor ingresa un prompt válido")
                continue
            
            print(f"🔄 Generando imagen para: '{prompt}'...")
            
            try:
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"custom_{contador}_{timestamp}.png"
                ruta_archivo = os.path.join(directorio_salida, nombre_archivo)
                
                image.save(ruta_archivo)
                print(f"✅ Imagen guardada: {ruta_archivo}")
                
                contador += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Error en modo interactivo: {e}")

def main():
    """
    Función principal
    """
    print("🤗 GENERADOR DE IMÁGENES SIMPLE CON HUGGING FACE")
    print("=" * 60)
    print("Versión simplificada para evitar problemas de compatibilidad")
    
    # Verificar dependencias
    dependencias_ok, faltantes = verificar_dependencias()
    if not dependencias_ok:
        return
    
    print(f"\n🚀 Iniciando generación de imágenes...")
    
    try:
        # 1. Generar imagen de prueba
        print("\n" + "="*60)
        print("1️⃣  GENERACIÓN DE PRUEBA")
        print("="*60)
        
        ruta_imagen, imagen = generar_imagen_simple()
        
        if ruta_imagen:
            print(f"\n🎉 ¡Primera imagen generada exitosamente!")
            
            # 2. Preguntar si quiere modo interactivo
            continuar = input("\n¿Quieres probar el modo interactivo? (s/n): ").lower()
            if continuar in ['s', 'si', 'sí', 'y', 'yes']:
                modo_interactivo()
        else:
            print("\n❌ No se pudo generar la imagen de prueba")
            print("💡 Revisa los errores anteriores")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Proceso cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("💡 Intenta reiniciar el script")
    
    print("\n📝 Notas finales:")
    print("   • Las imágenes se guardan en 'imagenes_generadas/'")
    print("   • Usa prompts en inglés para mejores resultados")
    print("   • La primera ejecución es más lenta (descarga el modelo)")
    print("   • Necesitas al menos 4GB de RAM libre")

if __name__ == "__main__":
    main()
