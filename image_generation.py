#!/usr/bin/env python3
"""
Ejemplo de Generación de Imágenes con Hugging Face
==================================================

Este script demuestra cómo usar modelos de diffusion para generar imágenes
a partir de prompts de texto usando Stable Diffusion.
"""

import os
import torch
import warnings
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# Suprimir warnings de xformers
warnings.filterwarnings("ignore", category=UserWarning, module="xformers")
os.environ["XFORMERS_MORE_DETAILS"] = "0"

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError as e:
    print(f"❌ Error importando diffusers: {e}")
    print("💡 Solucionando problema de dependencias...")
    
    # Desinstalar xformers problemático
    import subprocess
    import sys
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "xformers", "-y"], 
                      capture_output=True, check=False)
        print("✅ xformers desinstalado")
    except:
        pass
    
    # Intentar importar de nuevo
    try:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        print("✅ Diffusers importado exitosamente")
    except ImportError as e2:
        print(f"❌ Error persistente: {e2}")
        print("💡 Instalando versión compatible...")
        subprocess.run([sys.executable, "-m", "pip", "install", "diffusers==0.21.4"], 
                      capture_output=True)
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def crear_directorio_salida():
    """
    Crea el directorio para guardar las imágenes generadas
    """
    directorio = "imagenes_generadas"
    if not os.path.exists(directorio):
        os.makedirs(directorio)
        print(f"📁 Directorio creado: {directorio}")
    return directorio

def generar_imagen_basica():
    """
    Genera imágenes usando Stable Diffusion con prompts básicos
    """
    print("🎨 Generación Básica de Imágenes...")
    print("=" * 60)
    
    # Configurar el dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Dispositivo: {device}")
    
    # Cargar el modelo Stable Diffusion
    print("📥 Cargando modelo Stable Diffusion...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # Optimizar para velocidad
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Habilitar optimizaciones de memoria si es necesario
        if device == "cpu":
            pipe.enable_attention_slicing()
        
        print("✅ Modelo cargado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        print("💡 Intentando con modelo alternativo más ligero...")
        
        # Modelo alternativo más pequeño
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
    
    # Crear directorio de salida
    directorio_salida = crear_directorio_salida()
    
    # Prompts de ejemplo en español e inglés
    prompts_ejemplo = [
        {
            "prompt": "a beautiful sunset over a mountain landscape, digital art, highly detailed",
            "nombre": "atardecer_montanas"
        },
        {
            "prompt": "a cute robot reading a book in a cozy library, cartoon style, warm lighting",
            "nombre": "robot_biblioteca"
        },
        {
            "prompt": "a magical forest with glowing mushrooms and fairy lights, fantasy art",
            "nombre": "bosque_magico"
        },
        {
            "prompt": "a steampunk airship flying through clouds, vintage style, detailed",
            "nombre": "dirigible_steampunk"
        },
        {
            "prompt": "a cyberpunk cityscape at night with neon lights, futuristic, high contrast",
            "nombre": "ciudad_cyberpunk"
        }
    ]
    
    print(f"\n🖼️  Generando {len(prompts_ejemplo)} imágenes...")
    print("-" * 60)
    
    imagenes_generadas = []
    
    for i, ejemplo in enumerate(prompts_ejemplo, 1):
        prompt = ejemplo["prompt"]
        nombre_archivo = ejemplo["nombre"]
        
        print(f"\n{i}. Generando: {prompt[:50]}...")
        
        try:
            # Generar imagen
            with torch.autocast(device):
                imagen = pipe(
                    prompt,
                    num_inference_steps=20,  # Menos pasos para velocidad
                    guidance_scale=7.5,
                    width=512,
                    height=512
                ).images[0]
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_archivo = os.path.join(directorio_salida, f"{nombre_archivo}_{timestamp}.png")
            imagen.save(ruta_archivo)
            
            imagenes_generadas.append({
                "imagen": imagen,
                "prompt": prompt,
                "archivo": ruta_archivo
            })
            
            print(f"   ✅ Guardada como: {ruta_archivo}")
            
        except Exception as e:
            print(f"   ❌ Error generando imagen: {e}")
            continue
    
    return imagenes_generadas

def generar_imagen_personalizada():
    """
    Permite al usuario ingresar sus propios prompts
    """
    print("\n🎭 Generación Personalizada de Imágenes...")
    print("=" * 60)
    
    # Configurar el dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargar modelo (reutilizar si ya está cargado)
    print("📥 Cargando modelo...")
    model_id = "runwayml/stable-diffusion-v1-5"
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        if device == "cpu":
            pipe.enable_attention_slicing()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    directorio_salida = crear_directorio_salida()
    
    print("\n💡 Consejos para mejores prompts:")
    print("   • Sé específico: 'gato naranja durmiendo en un sofá azul'")
    print("   • Añade estilo: 'digital art', 'oil painting', 'cartoon style'")
    print("   • Incluye detalles: 'highly detailed', 'beautiful lighting'")
    print("   • Usa inglés para mejores resultados")
    print("\n📝 Escribe 'salir' para terminar")
    
    contador = 1
    while True:
        print(f"\n🎨 Imagen #{contador}")
        prompt_usuario = input("Ingresa tu prompt: ").strip()
        
        if prompt_usuario.lower() in ['salir', 'exit', 'quit']:
            print("👋 ¡Hasta luego!")
            break
        
        if not prompt_usuario:
            print("⚠️  Por favor ingresa un prompt válido")
            continue
        
        print(f"🔄 Generando imagen para: '{prompt_usuario}'...")
        
        try:
            # Parámetros personalizables
            print("⚙️  Configuración (presiona Enter para usar valores por defecto):")
            
            # Pasos de inferencia
            pasos_input = input("   Pasos de inferencia (20): ").strip()
            pasos = int(pasos_input) if pasos_input.isdigit() else 20
            
            # Guidance scale
            guidance_input = input("   Guidance scale (7.5): ").strip()
            guidance = float(guidance_input) if guidance_input else 7.5
            
            # Dimensiones
            width_input = input("   Ancho (512): ").strip()
            width = int(width_input) if width_input.isdigit() else 512
            
            height_input = input("   Alto (512): ").strip()
            height = int(height_input) if height_input.isdigit() else 512
            
            # Generar imagen
            with torch.autocast(device):
                imagen = pipe(
                    prompt_usuario,
                    num_inference_steps=pasos,
                    guidance_scale=guidance,
                    width=width,
                    height=height
                ).images[0]
            
            # Guardar imagen
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"custom_{contador}_{timestamp}.png"
            ruta_archivo = os.path.join(directorio_salida, nombre_archivo)
            imagen.save(ruta_archivo)
            
            print(f"✅ Imagen guardada como: {ruta_archivo}")
            
            # Mostrar imagen (opcional)
            mostrar = input("¿Mostrar imagen? (s/n): ").lower()
            if mostrar in ['s', 'si', 'sí', 'y', 'yes']:
                try:
                    imagen.show()
                except Exception:
                    print("⚠️  No se pudo mostrar la imagen automáticamente")
            
            contador += 1
            
        except Exception as e:
            print(f"❌ Error generando imagen: {e}")
            continue

def crear_galeria(imagenes):
    """
    Crea una galería visual de las imágenes generadas
    """
    if not imagenes:
        print("⚠️  No hay imágenes para mostrar")
        return
    
    print(f"\n🖼️  Creando galería de {len(imagenes)} imágenes...")
    
    # Calcular dimensiones de la grilla
    n_imagenes = len(imagenes)
    cols = min(3, n_imagenes)
    rows = (n_imagenes + cols - 1) // cols
    
    # Crear figura
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_imagenes == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img_data in enumerate(imagenes):
        if i < len(axes):
            axes[i].imshow(img_data["imagen"])
            axes[i].set_title(f"Prompt: {img_data['prompt'][:30]}...", fontsize=10)
            axes[i].axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(len(imagenes), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Guardar galería
    directorio_salida = crear_directorio_salida()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_galeria = os.path.join(directorio_salida, f"galeria_{timestamp}.png")
    plt.savefig(ruta_galeria, dpi=300, bbox_inches='tight')
    
    print(f"📸 Galería guardada como: {ruta_galeria}")
    
    # Mostrar galería
    try:
        plt.show()
    except Exception:
        print("⚠️  No se pudo mostrar la galería automáticamente")

def main():
    """
    Función principal que ejecuta todos los ejemplos
    """
    print("🤗 GENERACIÓN DE IMÁGENES CON HUGGING FACE")
    print("=" * 60)
    print("Este script demuestra cómo generar imágenes usando Stable Diffusion")
    
    # Verificar dependencias
    try:
        import diffusers
        print(f"✅ Diffusers versión: {diffusers.__version__}")
    except ImportError:
        print("❌ Error: diffusers no está instalado")
        print("💡 Instala con: pip install diffusers")
        return
    
    print(f"✅ PyTorch versión: {torch.__version__}")
    print(f"✅ CUDA disponible: {'Sí' if torch.cuda.is_available() else 'No'}")
    
    try:
        # 1. Generación básica
        print("\n" + "="*60)
        print("1️⃣  GENERACIÓN BÁSICA")
        print("="*60)
        imagenes_generadas = generar_imagen_basica()
        
        if imagenes_generadas:
            # 2. Crear galería
            print("\n" + "="*60)
            print("2️⃣  CREANDO GALERÍA")
            print("="*60)
            crear_galeria(imagenes_generadas)
        
        # 3. Generación personalizada (opcional)
        print("\n" + "="*60)
        print("3️⃣  GENERACIÓN PERSONALIZADA")
        print("="*60)
        
        continuar = input("¿Quieres probar la generación personalizada? (s/n): ").lower()
        if continuar in ['s', 'si', 'sí', 'y', 'yes']:
            generar_imagen_personalizada()
        
        print("\n🎉 ¡Generación de imágenes completada!")
        print("💡 Consejos:")
        print("   • Las imágenes se guardan en el directorio 'imagenes_generadas'")
        print("   • Usa prompts en inglés para mejores resultados")
        print("   • Experimenta con diferentes estilos y parámetros")
        print("   • Para mejor calidad, usa más pasos de inferencia (50-100)")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Generación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("💡 Verifica que todas las dependencias estén instaladas:")
        print("   pip install diffusers transformers accelerate")

if __name__ == "__main__":
    main()
