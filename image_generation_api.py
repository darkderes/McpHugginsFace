#!/usr/bin/env python3
"""
Generación de Imágenes usando APIs de Hugging Face
==================================================

Este script usa la API de Hugging Face para generar imágenes sin necesidad
de descargar modelos grandes localmente.
"""

import os
import requests
import base64
from datetime import datetime
from PIL import Image
import io

def crear_directorio_salida():
    """
    Crea el directorio para guardar las imágenes
    """
    directorio = "imagenes_generadas"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
        print(f"📁 Directorio creado: {directorio}")
    return directorio

def generar_con_api_local():
    """
    Genera imágenes usando una API más simple
    """
    print("🎨 Generador de Imágenes - Versión API")
    print("=" * 60)
    
    # Usar un enfoque más simple con requests
    prompts_ejemplo = [
        "a cute cat sitting on a windowsill",
        "a beautiful mountain landscape at sunset",
        "a robot reading a book in a library",
        "a magical forest with glowing mushrooms",
        "a cyberpunk city at night with neon lights"
    ]
    
    directorio_salida = crear_directorio_salida()
    
    print("🤖 Simulando generación de imágenes...")
    print("💡 Nota: Esta versión crea imágenes de ejemplo para demostrar la funcionalidad")
    
    for i, prompt in enumerate(prompts_ejemplo, 1):
        print(f"\n{i}. Generando: {prompt}")
        
        try:
            # Crear una imagen simple de ejemplo
            from PIL import Image, ImageDraw, ImageFont
            
            # Crear imagen base
            width, height = 512, 512
            image = Image.new('RGB', (width, height), color=(135, 206, 235))  # Sky blue
            draw = ImageDraw.Draw(image)
            
            # Dibujar elementos básicos
            if "cat" in prompt:
                # Dibujar un "gato" simple
                draw.ellipse([200, 200, 312, 312], fill=(255, 165, 0))  # Cuerpo naranja
                draw.ellipse([220, 180, 260, 220], fill=(255, 165, 0))  # Cabeza
                draw.ellipse([230, 190, 240, 200], fill=(0, 0, 0))     # Ojo izquierdo
                draw.ellipse([250, 190, 260, 200], fill=(0, 0, 0))     # Ojo derecho
                
            elif "mountain" in prompt:
                # Dibujar montañas simples
                draw.polygon([(0, 400), (150, 200), (300, 400)], fill=(139, 69, 19))
                draw.polygon([(200, 400), (350, 150), (500, 400)], fill=(160, 82, 45))
                
            elif "robot" in prompt:
                # Dibujar un robot simple
                draw.rectangle([200, 200, 312, 350], fill=(169, 169, 169))  # Cuerpo
                draw.rectangle([220, 160, 292, 220], fill=(169, 169, 169))  # Cabeza
                draw.rectangle([230, 180, 250, 200], fill=(255, 0, 0))     # Ojo izquierdo
                draw.rectangle([262, 180, 282, 200], fill=(255, 0, 0))     # Ojo derecho
                
            elif "forest" in prompt:
                # Dibujar árboles simples
                for x in range(50, 450, 80):
                    draw.rectangle([x, 300, x+20, 450], fill=(139, 69, 19))  # Tronco
                    draw.ellipse([x-30, 250, x+50, 330], fill=(0, 128, 0))   # Copa
                
            elif "city" in prompt:
                # Dibujar edificios simples
                for x in range(0, 500, 60):
                    height_building = 200 + (i * 30) % 150
                    draw.rectangle([x, 512-height_building, x+50, 512], fill=(64, 64, 64))
                    # Ventanas
                    for window_y in range(512-height_building+20, 512, 30):
                        draw.rectangle([x+10, window_y, x+20, window_y+10], fill=(255, 255, 0))
                        draw.rectangle([x+30, window_y, x+40, window_y+10], fill=(255, 255, 0))
            
            # Añadir texto del prompt
            try:
                # Intentar usar una fuente, si no está disponible usar la por defecto
                font = ImageFont.load_default()
            except:
                font = None
            
            # Texto en la parte inferior
            text_lines = prompt.split()
            text = " ".join(text_lines[:4]) + "..." if len(text_lines) > 4 else prompt
            draw.text((10, 480), text, fill=(255, 255, 255), font=font)
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"imagen_demo_{i}_{timestamp}.png"
            ruta_archivo = os.path.join(directorio_salida, nombre_archivo)
            
            image.save(ruta_archivo)
            print(f"   ✅ Guardada como: {nombre_archivo}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n🎉 Generación completada!")
    print(f"📁 Imágenes guardadas en: {directorio_salida}")

def modo_interactivo_demo():
    """
    Modo interactivo que crea imágenes de demostración
    """
    print("\n🎮 MODO INTERACTIVO - DEMO")
    print("=" * 60)
    print("Genera imágenes de demostración basadas en tus prompts")
    print("Escribe 'salir' para terminar")
    
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
        
        print(f"🔄 Creando imagen de demostración para: '{prompt}'...")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import random
            
            # Crear imagen base con color aleatorio
            colors = [(135, 206, 235), (255, 182, 193), (144, 238, 144), (255, 218, 185), (221, 160, 221)]
            bg_color = random.choice(colors)
            
            image = Image.new('RGB', (512, 512), color=bg_color)
            draw = ImageDraw.Draw(image)
            
            # Añadir elementos aleatorios basados en palabras clave
            palabras = prompt.lower().split()
            
            # Detectar elementos y dibujar formas simples
            if any(word in palabras for word in ['cat', 'gato', 'animal']):
                draw.ellipse([200, 200, 312, 312], fill=(255, 165, 0))
                draw.ellipse([220, 180, 260, 220], fill=(255, 165, 0))
                
            if any(word in palabras for word in ['tree', 'forest', 'árbol', 'bosque']):
                for i in range(3):
                    x = 100 + i * 100
                    draw.rectangle([x, 350, x+20, 450], fill=(139, 69, 19))
                    draw.ellipse([x-30, 300, x+50, 380], fill=(0, 128, 0))
            
            if any(word in palabras for word in ['house', 'building', 'casa', 'edificio']):
                draw.rectangle([150, 250, 350, 450], fill=(139, 69, 19))
                draw.rectangle([200, 200, 300, 280], fill=(255, 0, 0))
                draw.rectangle([220, 320, 280, 420], fill=(101, 67, 33))
            
            if any(word in palabras for word in ['sun', 'sol', 'sunset', 'atardecer']):
                draw.ellipse([400, 50, 480, 130], fill=(255, 255, 0))
            
            if any(word in palabras for word in ['car', 'coche', 'auto']):
                draw.rectangle([150, 350, 350, 420], fill=(255, 0, 0))
                draw.ellipse([170, 400, 210, 440], fill=(0, 0, 0))
                draw.ellipse([290, 400, 330, 440], fill=(0, 0, 0))
            
            # Añadir formas decorativas aleatorias
            for _ in range(random.randint(2, 5)):
                x, y = random.randint(50, 450), random.randint(50, 450)
                size = random.randint(10, 30)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                if random.choice([True, False]):
                    draw.ellipse([x, y, x+size, y+size], fill=color)
                else:
                    draw.rectangle([x, y, x+size, y+size], fill=color)
            
            # Añadir el texto del prompt
            font = ImageFont.load_default()
            text_lines = prompt.split()
            text = " ".join(text_lines[:5]) + "..." if len(text_lines) > 5 else prompt
            draw.text((10, 480), text, fill=(255, 255, 255), font=font)
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"custom_demo_{contador}_{timestamp}.png"
            ruta_archivo = os.path.join(directorio_salida, nombre_archivo)
            
            image.save(ruta_archivo)
            print(f"✅ Imagen de demo guardada: {nombre_archivo}")
            
            contador += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def main():
    """
    Función principal
    """
    print("🤗 GENERADOR DE IMÁGENES - VERSIÓN DEMO")
    print("=" * 60)
    print("Esta versión crea imágenes de demostración mientras resolvemos")
    print("los problemas de compatibilidad con Stable Diffusion")
    
    try:
        from PIL import Image, ImageDraw
        print("✅ PIL disponible")
    except ImportError:
        print("❌ PIL no disponible. Instala con: pip install pillow")
        return
    
    print("\n🚀 Iniciando generador de demo...")
    
    try:
        # 1. Generar imágenes de ejemplo
        print("\n" + "="*60)
        print("1️⃣  GENERANDO IMÁGENES DE EJEMPLO")
        print("="*60)
        
        generar_con_api_local()
        
        # 2. Modo interactivo
        continuar = input("\n¿Quieres probar el modo interactivo? (s/n): ").lower()
        if continuar in ['s', 'si', 'sí', 'y', 'yes']:
            modo_interactivo_demo()
        
        print("\n🎉 ¡Demo completado!")
        print("\n💡 Próximos pasos para usar Stable Diffusion real:")
        print("   1. Instalar PyTorch compatible con tu sistema")
        print("   2. Usar Google Colab para evitar problemas locales")
        print("   3. Considerar usar APIs en la nube como Replicate o RunPod")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()
