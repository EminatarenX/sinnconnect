from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
import cv2
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

def load_model(model_path, metadata_path):
    """Carga el modelo entrenado y los metadatos asociados."""
    # Cargar el modelo
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Cargar metadatos
    metadata = {}
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "class_names":
                # Procesar lista de nombres de clase
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
    
    return model, metadata

def normalize_video(video_path, output_path, target_size=(128, 128)):
    """Normaliza el video redimensionando los frames al tamaño objetivo."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # Deshabilitar auto-orientación
    
    if not cap.isOpened():
        raise ValueError(f"Error al abrir el archivo de video: {video_path}")
    
    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Crear objeto VideoWriter para la salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionar frame
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Escribir frame al video de salida
        out.write(resized_frame)
    
    # Liberar recursos
    cap.release()
    out.release()
    
    return output_path

def extract_frames(video_path, num_frames=5, target_size=(128, 128)):
    """Extrae frames del video para clasificación usando las mismas proporciones que script_dataset.py."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error al abrir el archivo de video: {video_path}")
    
    # Obtener número total de frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Usar proporciones base como en script_dataset.py (45%, 55%, 65%, 75%, 85%)
    base_proportions = [0.45, 0.55, 0.65, 0.75, 0.85]
    
    # Calcular índices de frames
    frame_indices = [min(int(p * frame_count), frame_count - 1) for p in base_proportions]
    frame_indices.sort()  # Asegurar orden cronológico
    
    # Leer frames en las posiciones calculadas
    frames = []
    for frame_idx in frame_indices:
        # Establecer posición del frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Leer frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"No se pudo leer el frame en la posición {frame_idx}")
        
        # Redimensionar frame
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Obtener dimensiones de la imagen para rotación
        height, width = resized.shape[:2]
        
        # Crear matriz de rotación para -90 grados
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), -90, 1)
        
        # Realizar rotación
        rotated = cv2.warpAffine(resized, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        
        # Añadir a la lista de frames
        frames.append(rgb_frame)
    
    # Liberar recursos
    cap.release()
    
    # Asegurar que tenemos suficientes frames
    if len(frames) < num_frames:
        raise ValueError(f"No se pudieron extraer {num_frames} frames del video. Se obtuvieron {len(frames)} frames.")
    
    # Convertir a array numpy y normalizar
    frames = np.array(frames, dtype=np.float32) / 255.0
    
    return frames

# Cargar modelo y metadatos al inicio
try:
    model, metadata = load_model("best_model.h5", "model_metadata.txt")
    class_names = metadata["class_names"]
    input_shape = metadata["input_shape"]
    num_frames = metadata["num_frames"]
    frame_height = metadata["frame_height"]
    frame_width = metadata["frame_width"]
    print(f"Modelo cargado exitosamente. Forma de entrada esperada: {input_shape}")
    print(f"Clases: {class_names}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # Establecer valores predeterminados en caso de error
    model = None
    class_names = []
    num_frames = 5
    frame_height = 128
    frame_width = 128

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para recibir un video y devolver las 3 clases más probables."""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    # Verificar si se incluyó un archivo de video en la solicitud
    if 'video' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo de video'}), 400
    
    video_file = request.files['video']
    
    # Verificar si se envió un archivo válido
    if video_file.filename == '':
        return jsonify({'error': 'No se seleccionó archivo de video'}), 400
    
    # Crear directorio temporal para procesamiento
    with tempfile.TemporaryDirectory() as temp_dir:
        # Guardar el archivo subido
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(temp_dir, filename)
        video_file.save(video_path)
        
        try:
            # Normalizar el video
            normalized_path = os.path.join(temp_dir, f"normalized_{filename}")
            normalize_video(video_path, normalized_path, (frame_width, frame_height))
            
            # Extraer frames para predicción
            frames = extract_frames(normalized_path, num_frames, (frame_width, frame_height))
            
            # Preparar entrada para el modelo
            input_data = np.expand_dims(frames, axis=0)
            
            # Realizar predicción
            predictions = model.predict(input_data)[0]
            
            # Obtener las 3 principales predicciones
            top_indices = np.argsort(predictions)[::-1][:3]
            
            # Formatear resultados
            results = []
            for i, idx in enumerate(top_indices):
                class_name = class_names[idx]
                probability = float(predictions[idx])
                # Formatear nombre de clase para visualización (eliminar prefijo 'class_')
                display_name = class_name.replace('class_', '')
                results.append({
                    'rank': i + 1,
                    'class': display_name,
                    'probability': probability
                })
                print(f"Clase {i+1}: {display_name}, Probabilidad: {probability}")

            print("Predicción final:", results) 
            return jsonify({
                'predictions': results
            })
        
        except Exception as e:
            return jsonify({'error': f'Error al procesar el video: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4001)