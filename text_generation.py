#!/usr/bin/env python3
"""
Ejemplo de Generación de Texto con Hugging Face
===============================================

Este script demuestra diferentes técnicas de generación de texto
usando modelos preentrenados de Hugging Face.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import time
from typing import List, Dict

def generacion_basica():
    """
    Ejemplo básico de generación de texto usando pipeline
    """
    print("🚀 Iniciando generación básica de texto...")
    
    # Configurar semilla para reproducibilidad
    set_seed(42)
    
    # Crear pipeline de generación de texto
    # Usando GPT-2 en español
    generator = pipeline(
        "text-generation",
        model="datificate/gpt2-small-spanish",
        tokenizer="datificate/gpt2-small-spanish",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Prompts de ejemplo
    prompts = [
        "En un futuro no muy lejano, la inteligencia artificial",
        "La receta secreta de la abuela incluía",
        "El explorador encontró en la cueva antigua",
        "La ciudad del futuro tendrá",
        "El científico descubrió que"
    ]
    
    print("\n📝 Generando textos a partir de prompts:")
    print("-" * 60)
    
    resultados = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        print("   Generando...")
        
        # Generar texto
        resultado = generator(
            prompt,
            max_length=100,
            num_return_sequences=2,
            temperature=0.8,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        for j, generacion in enumerate(resultado, 1):
            texto_generado = generacion['generated_text']
            texto_nuevo = texto_generado[len(prompt):].strip()
            
            print(f"   Opción {j}: {prompt}{texto_nuevo}")
            
            resultados.append({
                'prompt': prompt,
                'texto_generado': texto_generado,
                'texto_nuevo': texto_nuevo
            })
        
        print()
    
    return resultados

def generacion_avanzada_con_parametros():
    """
    Ejemplo avanzado mostrando diferentes parámetros de generación
    """
    print("🔬 Generación avanzada con diferentes parámetros...")
    
    try:
        # Cargar modelo y tokenizer específicos
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Añadir pad_token si no existe
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        prompt = "The future of artificial intelligence is"
        
        # Diferentes configuraciones de generación
        configuraciones = [
            {
                'nombre': 'Creativo (alta temperatura)',
                'params': {'temperature': 1.2, 'top_p': 0.9, 'do_sample': True}
            },
            {
                'nombre': 'Conservador (baja temperatura)',
                'params': {'temperature': 0.3, 'top_p': 0.8, 'do_sample': True}
            },
            {
                'nombre': 'Determinístico (greedy)',
                'params': {'do_sample': False, 'num_beams': 1}
            },
            {
                'nombre': 'Beam Search',
                'params': {'do_sample': False, 'num_beams': 4, 'early_stopping': True}
            }
        ]
        
        print(f"\n📊 Comparando diferentes estrategias con prompt: '{prompt}'")
        print("-" * 80)
        
        for config in configuraciones:
            print(f"\n🎯 {config['nombre']}:")
            
            params = {
                'max_length': 80,
                'num_return_sequences': 1,
                'pad_token_id': tokenizer.eos_token_id,
                **config['params']
            }
            
            start_time = time.time()
            resultado = generator(prompt, **params)
            end_time = time.time()
            
            texto_generado = resultado[0]['generated_text']
            tiempo = end_time - start_time
            
            print(f"   Resultado: {texto_generado}")
            print(f"   Tiempo: {tiempo:.2f}s")
            print()
            
    except Exception as e:
        print(f"⚠️  Error con modelo avanzado: {e}")
        print("Continuando con modelo básico...")

def generacion_conversacional():
    """
    Ejemplo de generación conversacional/diálogo
    """
    print("💬 Generación conversacional...")
    
    try:
        # Usar modelo conversacional
        chatbot = pipeline(
            "conversational",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("\n🤖 Simulando conversación:")
        print("-" * 40)
        
        # Simular una conversación
        conversaciones = [
            "Hello! How are you today?",
            "What's your favorite programming language?",
            "Can you tell me a joke?",
            "What do you think about artificial intelligence?"
        ]
        
        from transformers import Conversation
        
        conversacion = Conversation()
        
        for i, mensaje in enumerate(conversaciones, 1):
            conversacion.add_user_input(mensaje)
            resultado = chatbot(conversacion)
            
            respuesta = resultado.generated_responses[-1]
            
            print(f"👤 Usuario: {mensaje}")
            print(f"🤖 Bot: {respuesta}")
            print()
            
    except Exception as e:
        print(f"⚠️  Error en generación conversacional: {e}")

def generacion_con_control_de_estilo():
    """
    Ejemplo de generación con control de estilo y formato
    """
    print("🎨 Generación con control de estilo...")
    
    # Usar modelo que permite mejor control
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Prompts con diferentes estilos
    estilos = [
        {
            'estilo': 'Científico',
            'prompt': "According to recent research in neuroscience,",
            'params': {'temperature': 0.6, 'top_p': 0.9}
        },
        {
            'estilo': 'Narrativo',
            'prompt': "Once upon a time, in a magical forest,",
            'params': {'temperature': 1.0, 'top_p': 0.95}
        },
        {
            'estilo': 'Técnico',
            'prompt': "To implement this algorithm, we need to",
            'params': {'temperature': 0.4, 'top_p': 0.8}
        },
        {
            'estilo': 'Poético',
            'prompt': "The moonlight dances on the water,",
            'params': {'temperature': 1.2, 'top_p': 0.9}
        }
    ]
    
    print("\n🎭 Generando textos con diferentes estilos:")
    print("-" * 60)
    
    for estilo_info in estilos:
        print(f"\n📝 Estilo: {estilo_info['estilo']}")
        print(f"Prompt: '{estilo_info['prompt']}'")
        
        resultado = generator(
            estilo_info['prompt'],
            max_length=120,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
            **estilo_info['params']
        )
        
        texto_completo = resultado[0]['generated_text']
        print(f"Resultado: {texto_completo}")
        print()

def ejemplo_interactivo_generacion():
    """
    Modo interactivo para que el usuario genere sus propios textos
    """
    print("\n🎮 Modo Interactivo - Generación de Texto")
    print("Ingresa un prompt y genera texto personalizado")
    print("Escribe 'salir' para terminar")
    print("-" * 50)
    
    # Inicializar generador
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    while True:
        prompt = input("\n📝 Ingresa tu prompt: ").strip()
        
        if prompt.lower() in ['salir', 'exit', 'quit', '']:
            print("👋 ¡Hasta luego!")
            break
        
        # Configuración personalizable
        print("\n⚙️  Configuración (presiona Enter para usar valores por defecto):")
        
        try:
            max_length = input("Longitud máxima (50-200, default=100): ").strip()
            max_length = int(max_length) if max_length else 100
            max_length = max(50, min(200, max_length))  # Limitar rango
            
            temperature = input("Creatividad/Temperatura (0.1-2.0, default=0.8): ").strip()
            temperature = float(temperature) if temperature else 0.8
            temperature = max(0.1, min(2.0, temperature))  # Limitar rango
            
            num_sequences = input("Número de variaciones (1-3, default=1): ").strip()
            num_sequences = int(num_sequences) if num_sequences else 1
            num_sequences = max(1, min(3, num_sequences))  # Limitar rango
            
        except ValueError:
            print("⚠️  Usando valores por defecto...")
            max_length, temperature, num_sequences = 100, 0.8, 1
        
        print(f"\n🔄 Generando texto... (longitud: {max_length}, creatividad: {temperature})")
        
        try:
            start_time = time.time()
            resultados = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_sequences,
                temperature=temperature,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id,
                top_p=0.9
            )
            end_time = time.time()
            
            print(f"\n✨ Resultados generados en {end_time - start_time:.2f}s:")
            print("=" * 60)
            
            for i, resultado in enumerate(resultados, 1):
                texto_generado = resultado['generated_text']
                print(f"\n{i}. {texto_generado}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error al generar texto: {e}")

def main():
    """
    Función principal que ejecuta todos los ejemplos
    """
    print("🤗 Ejemplos de Generación de Texto con Hugging Face")
    print("=" * 60)
    
    try:
        # Ejemplo básico
        print("1️⃣  Ejecutando generación básica...")
        generacion_basica()
        
        # Ejemplo avanzado
        print("\n2️⃣  Ejecutando generación avanzada...")
        generacion_avanzada_con_parametros()
        
        # Generación conversacional
        print("\n3️⃣  Ejecutando generación conversacional...")
        generacion_conversacional()
        
        # Control de estilo
        print("\n4️⃣  Ejecutando control de estilo...")
        generacion_con_control_de_estilo()
        
        # Modo interactivo
        respuesta = input("\n¿Quieres probar el modo interactivo? (s/n): ").lower()
        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
            ejemplo_interactivo_generacion()
        
        print("\n✅ ¡Ejemplos de generación completados exitosamente!")
        
        # Consejos finales
        print("\n💡 Consejos para mejorar la generación:")
        print("   • Usa prompts más específicos para mejores resultados")
        print("   • Ajusta la temperatura: baja (0.3) = conservador, alta (1.2) = creativo")
        print("   • Experimenta con top_p y top_k para controlar la diversidad")
        print("   • Considera usar modelos específicos para tu dominio")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        print("💡 Asegúrate de haber instalado todas las dependencias con:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
