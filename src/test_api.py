#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cliente de prueba para la API de reconocimiento de lenguaje de señas.
Este script envía un video a la API y muestra los resultados.
"""

import requests
import argparse
import os
import json
from pprint import pprint

def parse_arguments():
    """Define y procesa los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Cliente de prueba para la API de SignConnect')
    parser.add_argument('--video', type=str, required=True, 
                        help='Ruta al archivo de video para clasificar')
    parser.add_argument('--url', type=str, default='http://localhost:4001/predict', 
                        help='URL del endpoint de la API')
    return parser.parse_args()

def send_video(video_path, api_url):
    """
    Envía un video a la API y devuelve la respuesta.
    
    Args:
        video_path: Ruta al archivo de video
        api_url: URL del endpoint de la API
        
    Returns:
        dict: Respuesta de la API en formato JSON
    """
    if not os.path.exists(video_path):
        print(f"Error: El archivo '{video_path}' no existe.")
        return None
    
    try:
        print(f"Enviando video '{video_path}' a {api_url}...")
        
        # Preparar archivo para envío
        with open(video_path, 'rb') as video_file:
            files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
            
            # Enviar solicitud POST
            response = requests.post(api_url, files=files)
        
        # Verificar respuesta
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: La API respondió con código {response.status_code}")
            print(f"Mensaje: {response.text}")
            return None
    
    except Exception as e:
        print(f"Error al enviar el video: {str(e)}")
        return None

def main():
    """Función principal del cliente de prueba."""
    # Procesar argumentos
    args = parse_arguments()
    
    # Enviar video a la API
    result = send_video(args.video, args.url)
    
    # Mostrar resultados
    if result:
        print("\n=== Resultados de la predicción ===")
        
        if 'predictions' in result:
            predictions = result['predictions']
            
            print("\nTop 3 predicciones:")
            for pred in predictions:
                print(f"{pred['rank']}. {pred['class']}: {pred['probability']:.4f} ({pred['probability']*100:.1f}%)")
            
            # Crear visualización sencilla para terminal
            top_class = predictions[0]['class']
            top_prob = predictions[0]['probability']
            
            print("\nResultado principal:")
            print(f"🔍 Seña detectada: {top_class}")
            
            # Barra de probabilidad ASCII
            bar_length = 40
            filled_length = int(top_prob * bar_length)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"Confianza: {bar} {top_prob*100:.1f}%")
            
            # Guardar resultado en archivo JSON
            output_file = 'ultima_prediccion.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResultados guardados en {output_file}")
        else:
            print("Formato de respuesta desconocido:", result)
    
    else:
        print("No se recibieron resultados de la API.")

if __name__ == "__main__":
    main()