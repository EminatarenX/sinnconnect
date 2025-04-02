import os
import numpy as np
import tensorflow as tf
import cv2
import time
from threading import Thread
import queue

class WebcamVideoStream:
    """
    Clase para capturar frames de la webcam en un hilo separado
    para reducir el lag y mejorar el rendimiento.
    """
    def __init__(self, src=0, name="WebcamVideoStream"):
        # Inicializar la captura de video
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        # Verificar si la cámara se abrió correctamente
        if not self.stream.isOpened():
            raise ValueError("No se pudo abrir la webcam")
            
        # Leer el primer frame
        (self.grabbed, self.frame) = self.stream.read()
        
        # Variables de control
        self.name = name
        self.stopped = False
    
    def start(self):
        # Iniciar el hilo para leer frames
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        # Mantener el bucle hasta que se detenga
        while not self.stopped:
            # Leer el siguiente frame
            (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        # Devolver el frame más reciente
        return self.frame
    
    def stop(self):
        # Indicar que el hilo debe detenerse
        self.stopped = True
        # Liberar recursos
        self.stream.release()

def load_keras_model(model_path):
    """
    Cargar el modelo de Keras.
    
    Args:
        model_path: Ruta al modelo de Keras (.h5)
        
    Returns:
        model: Modelo cargado
    """
    try:
        # Configurar para usar GPU si está disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU encontrada y configurada")
            except RuntimeError as e:
                print(f"Error configurando GPU: {e}")
        
        # Cargar el modelo
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Obtener y mostrar la forma de entrada esperada
        input_shape = model.input_shape
        print(f"Forma de entrada esperada: {input_shape}")
        
        # Compilar el modelo para mejorar el rendimiento de inferencia
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Realizar una inferencia ficticia para inicializar el modelo (reduce el lag del primer uso)
        dummy_input = np.zeros((1,) + input_shape[1:], dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        print("Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def load_metadata(file_path="model_metadata.txt"):
    """
    Cargar metadatos del modelo, incluyendo los nombres de las clases y la forma de entrada.
    
    Args:
        file_path: Ruta al archivo de metadatos
        
    Returns:
        metadata: Diccionario con los metadatos del modelo
    """
    default_classes = [
        "Hola", "Cómo estás?", "Comida", "Qué haces?", 
        "Bien", "Adiós", "Muchas gracias", "Perdón", 
        "Por favor", "Sí", "No", "Ayuda", "Qué hora es?"
    ]
    
    # Valores por defecto
    metadata = {
        "class_names": default_classes,
        "input_shape": (5, 128, 128, 3),
        "num_frames": 5,
        "frame_height": 128,
        "frame_width": 128,
        "channels": 3
    }
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "class_names":
                        # Procesar la lista de nombres de clase
                        # Eliminar corchetes, comillas y dividir por comas
                        value = value.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                        class_names = [c.strip() for c in value.split(",")]
                        metadata[key] = class_names
                    elif key in ["num_frames", "frame_height", "frame_width", "channels"]:
                        # Convertir valores numéricos a enteros
                        metadata[key] = int(value)
                    elif key == "input_shape":
                        # Convertir tupla de forma a enteros
                        value = value.replace("(", "").replace(")", "")
                        shape = tuple(int(x.strip()) for x in value.split(","))
                        metadata[key] = shape
            
            print(f"Metadatos cargados del archivo: {file_path}")
            print(f"Forma de entrada: {metadata['input_shape']}")
            print(f"Clases: {metadata['class_names']}")
            return metadata
        else:
            print(f"Archivo {file_path} no encontrado. Usando valores por defecto.")
            return metadata
    except Exception as e:
        print(f"Error cargando metadatos: {e}")
        return metadata

def get_input_shape(model, metadata=None):
    """
    Obtener la forma de entrada esperada por el modelo.
    Primero intenta usar los metadatos, si están disponibles.
    Si no, obtiene la forma directamente del modelo.
    
    Args:
        model: Modelo de Keras
        metadata: Diccionario con metadatos del modelo
        
    Returns:
        tuple: (num_frames, height, width, channels)
    """
    if metadata and 'input_shape' in metadata:
        return metadata['input_shape']
    
    # Fallback: obtener del modelo
    input_shape = model.input_shape
    # Eliminamos la dimensión del batch (None)
    if input_shape[0] is None:
        input_shape = input_shape[1:]
    return input_shape

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocesar un frame para la entrada al modelo.
    
    Args:
        frame: Frame de la cámara
        target_size: Tamaño objetivo (ancho, alto)
        
    Returns:
        processed_frame: Frame procesado
    """
    # Redimensionar
    resized = cv2.resize(frame, target_size)
    # Convertir a RGB (el modelo espera RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalizar los valores de píxeles a [0,1]
    normalized = rgb.astype(np.float32) / 255.0
    return normalized

def run_inference(model, frames_sequence):
    """
    Ejecutar la inferencia con el modelo.
    
    Args:
        model: Modelo de Keras
        frames_sequence: Secuencia de frames preprocesados
        
    Returns:
        output: Salida del modelo
        inference_time: Tiempo de inferencia en segundos
    """
    try:
        # Añadir dimensión de batch
        input_data = np.expand_dims(frames_sequence, axis=0)
        
        # Ejecutar inferencia
        start_time = time.time()
        output = model.predict(input_data, verbose=0)
        inference_time = time.time() - start_time
        
        return output, inference_time
    except Exception as e:
        print(f"Error durante la inferencia: {e}")
        return None, 0

def draw_ui(frame, prediction, confidence, inference_time, buffer_status, top_predictions=None):
    """
    Dibujar la interfaz de usuario con la predicción y estadísticas.
    Muestra las 3 predicciones principales.
    
    Args:
        frame: Frame actual
        prediction: Clase predicha
        confidence: Confianza de la predicción
        inference_time: Tiempo de inferencia
        buffer_status: Estado del buffer
        top_predictions: Lista de tuplas (clase, confianza) para las top 3 predicciones
    """
    # Fondo semi-transparente para texto (más alto para mostrar top 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Información de predicción principal
    cv2.putText(frame, f"Predicción: {prediction}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Barra de confianza (reducida al 50%)
    bar_length = 150
    filled_length = int(bar_length * confidence)
    cv2.rectangle(frame, (20, 50), (20 + bar_length, 65), (100, 100, 100), -1)
    cv2.rectangle(frame, (20, 50), (20 + filled_length, 65), (0, 255, 0), -1)
    cv2.putText(frame, f"{confidence:.2f}", (20 + bar_length + 10, 62), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Mostrar top 3 predicciones si están disponibles
    if top_predictions and len(top_predictions) > 0:
        cv2.putText(frame, "Top 3:", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_pos = 110
        for i, (pred_class, pred_conf) in enumerate(top_predictions[:3]):
            # Simplificar nombre de clase (quitar 'class_' si existe)
            short_name = pred_class.replace('class_', '')
            # Formatear confianza como porcentaje
            conf_text = f"{pred_conf*100:.1f}%"
            # Mostrar en formato: "1. nombre: 95.5%"
            cv2.putText(frame, f"{i+1}. {short_name}: {conf_text}", (30, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
    
    # Información adicional (opcional - en la parte inferior)
    cv2.putText(frame, f"{buffer_status} | {inference_time*1000:.0f}ms", 
                (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def run_webcam_demo(model, metadata):
    """
    Ejecutar demo con webcam para reconocimiento de lenguaje de señas en tiempo real.
    
    Args:
        model: Modelo de Keras
        metadata: Diccionario con metadatos del modelo
    """
    # Obtener forma de entrada del modelo y nombres de clases
    class_names = metadata['class_names']
    input_shape = get_input_shape(model, metadata)
    num_frames = input_shape[0]  # Número de frames por secuencia
    frame_height = input_shape[1]
    frame_width = input_shape[2]
    print(f"El modelo espera {num_frames} frames de {frame_width}x{frame_height}")
    
    # Iniciar captura de video en hilo separado
    print("Iniciando captura de webcam...")
    webcam = WebcamVideoStream().start()
    
    # Buffer circular para almacenar frames
    buffer_size = 20  # Mantenemos un buffer más grande que el requerido
    frame_buffer = []
    
    # Cola para predicciones para suavizar resultado
    prediction_queue = queue.Queue(maxsize=5)
    last_prediction = "Esperando..."
    confidence = 0.0
    
    # Temporizador para controlar la frecuencia de predicciones
    last_prediction_time = time.time()
    prediction_interval = 0.5  # Segundos entre predicciones
    
    print("Demo iniciada. Presiona 'q' para salir.")
    
    try:
        while True:
            # Leer frame
            frame = webcam.read()
            if frame is None:
                print("Error al leer frame")
                break
            
            # Crear copia para mostrar
            display_frame = frame.copy()
            
            # Preprocesar frame y añadirlo al buffer
            processed_frame = preprocess_frame(frame, (frame_width, frame_height))
            frame_buffer.append(processed_frame)
            
            # Mantener buffer al tamaño requerido
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Estado actual del buffer
            buffer_status = f"Buffer: {len(frame_buffer)}/{buffer_size}"
            
            # Realizar predicción si tenemos suficientes frames y ha pasado suficiente tiempo
            current_time = time.time()
            if len(frame_buffer) >= num_frames and (current_time - last_prediction_time) >= prediction_interval:
                # Seleccionar frames estratégicamente (similar al entrenamiento)
                indices = np.linspace(0, len(frame_buffer) - 1, num_frames, dtype=int)
                sequence = np.array([frame_buffer[i] for i in indices])
                
                # Ejecutar inferencia
                output, inference_time = run_inference(model, sequence)
                
                if output is not None:
                    # Obtener predicción principal
                    prediction_idx = np.argmax(output[0])
                    new_confidence = float(output[0][prediction_idx])
                    new_prediction = class_names[prediction_idx]
                    
                    # Añadir a la cola de predicciones
                    if prediction_queue.full():
                        prediction_queue.get()
                    prediction_queue.put((new_prediction, new_confidence))
                    
                    # Actualizar predicción basada en la moda de la cola
                    predictions = []
                    confidences = {}
                    for _ in range(prediction_queue.qsize()):
                        p, c = prediction_queue.get()
                        predictions.append(p)
                        if p not in confidences or c > confidences[p]:
                            confidences[p] = c
                        prediction_queue.put((p, c))
                    
                    # Obtener la predicción más frecuente
                    from collections import Counter
                    prediction_counter = Counter(predictions)
                    last_prediction = prediction_counter.most_common(1)[0][0]
                    confidence = confidences[last_prediction]
                    
                    # Obtener top-3 predicciones para mostrar en pantalla y consola
                    indices = np.argsort(output[0])[::-1][:3]  # Top 3 índices
                    top_predictions = [(class_names[idx], float(output[0][idx])) for idx in indices]
                    
                    # Mostrar en consola
                    print("\nTop 3 predicciones:")
                    for i, (class_name, conf) in enumerate(top_predictions):
                        print(f"{i+1}. {class_name}: {conf:.4f}")
                    
                    # Actualizar tiempo de última predicción
                    last_prediction_time = current_time
                else:
                    inference_time = 0
            else:
                inference_time = 0
            
            # Mostrar resultados en frame
            draw_ui(display_frame, last_prediction, confidence, inference_time, buffer_status, 
                   top_predictions if 'top_predictions' in locals() else None)
            
            # Mostrar frame
            cv2.imshow("SignConnect - Reconocimiento de Lenguaje de Señas", display_frame)
            
            # Salir al presionar 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Pequeña pausa para reducir uso de CPU
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
    except Exception as e:
        print(f"Error en la demo: {e}")
    finally:
        # Liberar recursos
        webcam.stop()
        cv2.destroyAllWindows()
        print("Demo finalizada")

if __name__ == "__main__":
    print("SignConnect - Demo de Reconocimiento de Lenguaje de Señas")
    print("=" * 60)
    
    # Definir rutas a archivos
    # Buscar en el directorio actual (src)
    model_path = "best_model.h5"
    metadata_path = "model_metadata.txt"
    
    # Verificar si existe el archivo del modelo
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        # Buscar archivos .h5 en directorio actual y superior
        h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if len(h5_files) == 0 and os.path.exists('..'):
            h5_files = [f for f in os.listdir('..') if f.endswith('.h5')]
        
        if h5_files:
            model_path = h5_files[0]
            print(f"Se encontró el modelo: {model_path}")
        else:
            print("No se encontraron archivos .h5")
            print("Por favor, proporciona la ruta al modelo:")
            model_path = input("> ")
    
    # Cargar el modelo
    model = load_keras_model(model_path)
    if model is None:
        print("No se pudo cargar el modelo. Saliendo.")
        exit(1)
    
    # Cargar metadatos del modelo
    metadata = load_metadata(metadata_path)
    
    # Ejecutar demo
    run_webcam_demo(model, metadata)